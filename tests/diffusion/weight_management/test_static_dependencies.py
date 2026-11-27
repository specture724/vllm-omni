# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Static dependency, coexistence registry, and frozen allowlist checks (RFC 01-7: T1-T8)."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.diffusion.weight_management.coexistence import registry

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

REPO_ROOT = Path(__file__).resolve().parents[3]
VLLM_OMNI_ROOT = REPO_ROOT / "vllm_omni"
DIFFUSION_ROOT = VLLM_OMNI_ROOT / "diffusion"
WM_ROOT = DIFFUSION_ROOT / "weight_management"
SITES_DIR = Path(__file__).resolve().parent / "coexistence" / "sites"

FROZEN_ALLOWLIST_SHA256 = {
    "OFFLOADER_IMPORT_ALLOWLIST": "9d63ee112ac2ecab555b1f4d4935645018de0623fc26f8875ca6770600d2b119",
    "HOOK_NAME_ALLOWLIST": "f7c02fce26542bd46b821a7099e9c7da3e36fd34035806296e93b1fc90f67b90",
    "CONFIG_FIELD_ALLOWLIST": "98a69cfd5927425d898ce9012f326136391c995aaaf0ff38d55cb29602a1abe2",
    "QUERY_CALLER_ALLOWLIST": "30a7d3f1872f0906e26a8b54a9807d384180d13afd2eee87eea738403e2c247d",
}

CONFIG_FIELDS = frozenset(
    {
        "enable_cpu_offload",
        "enable_layerwise_offload",
        "enable_distributed_layerwise_offload",
        "dlo_use_allgather",
        "dlo_resident_layers",
        "dlo_host_registration_limit_gib",
        "diffusion_offload_config",
        "pin_cpu_memory",
        "host_weight_runtime_mode",
        "host_weight_runtime_root",
        "diffusion_weight_stack",
    }
)

LEGACY_OFFLOAD_QUERIES = frozenset(
    {
        "resolve_offload",
        "resolve_offload_strategy",
        "offload_enabled",
        "offload_streams_blocks",
        "should_offload_component",
        "any_selected_component_uses_allgather",
        "selected_offload_components",
        "component_uses_allgather",
        "materialize_legacy_offload_flags",
    }
)

SELECTOR_FUNCS = frozenset({"is_unified_stack", "is_unified", "process_weight_stack", "is_building_structure"})


def _is_type_checking_guard(node: ast.If) -> bool:
    test = node.test
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _runtime_imports(tree: ast.AST, qual: str = "") -> list[tuple[str, str]]:
    """Return (qualname, imported_module) pairs outside ``if TYPE_CHECKING:``."""
    out: list[tuple[str, str]] = []

    def _visit(node: ast.AST, scope: list[str]) -> None:
        if isinstance(node, ast.If) and _is_type_checking_guard(node):
            for stmt in node.orelse:
                _visit(stmt, scope)
            return
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            scope.append(node.name)
            for child in ast.iter_child_nodes(node):
                _visit(child, scope)
            scope.pop()
            return
        where = ".".join(scope) if scope else qual
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append((where, alias.name))
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level + (node.module or "")
            out.append((where, prefix))
        for child in ast.iter_child_nodes(node):
            _visit(child, scope)

    _visit(tree, [])
    return out


def _defined_qualnames(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()

    def _walk(node: ast.AST, scope: list[str]) -> None:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            scope.append(node.name)
            found.add(".".join(scope))
            for child in ast.iter_child_nodes(node):
                _walk(child, scope)
            scope.pop()
            return
        for child in ast.iter_child_nodes(node):
            _walk(child, scope)

    _walk(tree, [])
    return found


def _legacy_hook_names() -> set[str]:
    names: set[str] = set()
    for path in (DIFFUSION_ROOT / "offloader").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Name)
                        and target.id == "_HOOK_NAME"
                        and isinstance(node.value, ast.Constant)
                        and isinstance(node.value.value, str)
                    ):
                        names.add(node.value.value)
    return names


def _check_t1_file(path: Path) -> list[str]:
    rel = path.relative_to(WM_ROOT).as_posix()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    violations: list[str] = []
    for scope, mod in _runtime_imports(tree):
        if mod == "vllm_omni.host_weight_runtime" or mod.startswith("vllm_omni.host_weight_runtime."):
            violations.append(f"{rel}:{scope} imports forbidden {mod}")
        if mod == "vllm_omni.diffusion.offloader.config" or mod.startswith("vllm_omni.diffusion.offloader."):
            if not (rel == "startup.py" and scope == "execution_offload_facts"):
                violations.append(f"{rel}:{scope} imports legacy offloader {mod}")
        if rel in ("group.py", "startup.py"):
            for concrete in (
                "vllm_omni.diffusion.weight_management.loader",
                "vllm_omni.diffusion.weight_management.host",
                "vllm_omni.diffusion.weight_management.offload",
            ):
                if mod == concrete or mod.startswith(concrete + "."):
                    violations.append(f"{rel}:{scope} imports concrete implementation {mod}")
    return violations


def _scan_model_tree(root: Path, hook_names: set[str]) -> tuple[set[str], set[str], set[str], set[str]]:
    offloader_files: set[str] = set()
    hook_files: set[str] = set()
    config_files: set[str] = set()
    stack_field_files: set[str] = set()
    for sub in ("models", "lora", "cache"):
        base = root / sub
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            rel = path.relative_to(root).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "vllm_omni.diffusion.offloader" or alias.name.startswith(
                            "vllm_omni.diffusion.offloader."
                        ):
                            offloader_files.add(rel)
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    if mod == "vllm_omni.diffusion.offloader" or mod.startswith("vllm_omni.diffusion.offloader."):
                        offloader_files.add(rel)
                    elif node.level > 0 and (mod == "offloader" or mod.startswith("offloader.")):
                        offloader_files.add(rel)
                elif isinstance(node, ast.Attribute) and node.attr in CONFIG_FIELDS:
                    config_files.add(rel)
                    if node.attr == "diffusion_weight_stack":
                        stack_field_files.add(rel)
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "getattr"
                    and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value in CONFIG_FIELDS
                ):
                    config_files.add(rel)
                    if node.args[1].value == "diffusion_weight_stack":
                        stack_field_files.add(rel)
                elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in hook_names:
                    hook_files.add(rel)
    return offloader_files, hook_files, config_files, stack_field_files


def _scan_selector_sites(root: Path) -> set[str]:
    sites: set[str] = set()
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root.parent).as_posix()
        if rel.startswith("vllm_omni/diffusion/weight_management/") or rel in (
            "vllm_omni/diffusion/model_loader/construct.py",
            "vllm_omni/diffusion/model_loader/hub_prefetch.py",
        ):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))

        def _walk(node: ast.AST, scope: list[str]) -> None:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                scope.append(node.name)
                for child in ast.iter_child_nodes(node):
                    _walk(child, scope)
                scope.pop()
                return
            if isinstance(node, ast.Call):
                name = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else (node.func.attr if isinstance(node.func, ast.Attribute) else None)
                )
                if name in SELECTOR_FUNCS:
                    sites.add(f"{rel}::{'.'.join(scope) if scope else '<module>'}")
            for child in ast.iter_child_nodes(node):
                _walk(child, scope)

        _walk(tree, [])
    return sites


def _scan_query_callers(root: Path) -> set[str]:
    callers: set[str] = set()
    skip = (
        "vllm_omni/diffusion/offloader/",
        "vllm_omni/diffusion/models/",
        "vllm_omni/diffusion/lora/",
        "vllm_omni/diffusion/cache/",
    )
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root.parent).as_posix()
        if rel.startswith(skip):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # Only track files that actually import vllm_omni.diffusion.offloader.config.
        imports_legacy = any(
            mod == "vllm_omni.diffusion.offloader.config" or mod.startswith("vllm_omni.diffusion.offloader.config.")
            for _, mod in _runtime_imports(tree)
        )
        if not imports_legacy:
            continue

        def _walk(node: ast.AST, scope: list[str]) -> None:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                scope.append(node.name)
                for child in ast.iter_child_nodes(node):
                    _walk(child, scope)
                scope.pop()
                return
            if isinstance(node, ast.Call):
                name = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else (node.func.attr if isinstance(node.func, ast.Attribute) else None)
                )
                if name in LEGACY_OFFLOAD_QUERIES:
                    callers.add(f"{rel}::{'.'.join(scope) if scope else '<module>'}")
            for child in ast.iter_child_nodes(node):
                _walk(child, scope)

        _walk(tree, [])
    return callers


def _load_site_files(sites_dir: Path) -> tuple[dict[str, registry.CoexistenceRow], set[str]]:
    rows = {row.row_id: row for row in registry.ROWS}
    registered_sites: set[str] = set()
    for path in sorted(sites_dir.glob("*.py")):
        if path.name.startswith("test_") or path.name == "__init__.py":
            continue
        spec = importlib.util.spec_from_file_location(f"_site_{path.stem}", path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        row_id = mod.ROW
        assert isinstance(mod.PR, str) and mod.PR.strip(), f"{path.name}: PR must be nonempty"
        new_row = getattr(mod, "NEW_ROW", None)
        if new_row is not None:
            assert isinstance(new_row, registry.CoexistenceRow)
            assert new_row.row_id == row_id, f"{path.name}: NEW_ROW.row_id must match ROW"
            assert new_row.row_id not in rows, f"{path.name}: duplicate row_id {new_row.row_id}"
            rows[new_row.row_id] = new_row
        assert row_id in rows, f"{path.name}: unknown ROW {row_id!r}"
        for site in mod.SITES:
            file_rel, qual = site.split("::", 1)
            target_file = REPO_ROOT / file_rel
            assert target_file.exists(), f"{path.name}: site file {file_rel} does not exist"
            assert qual in _defined_qualnames(target_file), f"{path.name}: {site} not found in AST"
            registered_sites.add(site)
    return rows, registered_sites


def test_t1_internal_import_direction():
    violations = [v for path in sorted(WM_ROOT.rglob("*.py")) for v in _check_t1_file(path)]
    assert violations == []


def test_t2_runtime_module_graph_isolation(tmp_path):
    out = tmp_path / "loaded.json"
    code = """
import json, sys
import vllm_omni  # noqa: F401
baseline = set(sys.modules)
from vllm_omni.diffusion.weight_management import errors, group, interfaces, startup  # noqa: F401
forbidden = (
    "vllm_omni.host_weight_runtime",
    "vllm_omni.diffusion.offloader",
    "vllm_omni.diffusion.weight_management.loader",
    "vllm_omni.diffusion.weight_management.host",
    "vllm_omni.diffusion.weight_management.offload",
)
loaded = sorted(m for m in (set(sys.modules) - baseline) if any(m == p or m.startswith(p + ".") for p in forbidden))
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    json.dump(loaded, fh)
"""
    res = subprocess.run([sys.executable, "-c", code, str(out)], capture_output=True, text=True, timeout=180)
    assert res.returncode == 0, res.stderr
    assert json.loads(out.read_text(encoding="utf-8")) == []


def test_t3_t4_t5_model_lora_cache_boundaries():
    hook_names = _legacy_hook_names()
    assert hook_names == {"distributed_layerwise_offload", "layerwise_offload", "sequential_offload"}
    offloader_files, hook_files, config_files, stack_field_files = _scan_model_tree(DIFFUSION_ROOT, hook_names)

    assert offloader_files <= registry.OFFLOADER_IMPORT_ALLOWLIST, sorted(
        offloader_files - registry.OFFLOADER_IMPORT_ALLOWLIST
    )
    assert hook_files <= registry.HOOK_NAME_ALLOWLIST, sorted(hook_files - registry.HOOK_NAME_ALLOWLIST)
    assert config_files <= registry.CONFIG_FIELD_ALLOWLIST, sorted(config_files - registry.CONFIG_FIELD_ALLOWLIST)
    assert stack_field_files == set(), f"model code must not read diffusion_weight_stack directly: {stack_field_files}"


def test_t6_coexistence_registry_and_selector_sites():
    rows, registered = _load_site_files(SITES_DIR)
    for row_id in ("D01", "D02", "D03", "D04", "D05"):
        assert row_id in rows
    actual = _scan_selector_sites(VLLM_OMNI_ROOT)
    assert actual <= registered, f"unregistered selector sites: {sorted(actual - registered)}"


def test_t7_nonmodel_legacy_query_callers():
    actual = _scan_query_callers(VLLM_OMNI_ROOT)
    assert actual <= registry.QUERY_CALLER_ALLOWLIST, sorted(actual - registry.QUERY_CALLER_ALLOWLIST)


def test_t8_frozen_allowlist_digests():
    for name, expected_sha in FROZEN_ALLOWLIST_SHA256.items():
        items = sorted(getattr(registry, name))
        actual_sha = hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()
        assert actual_sha == expected_sha, f"{name} changed without updating frozen digest"


def test_synthetic_violations_are_detected(tmp_path):
    # Synthetic T3/T4/T5 violation inside a fake diffusion root.
    fake_model = tmp_path / "models" / "bad" / "pipeline_bad.py"
    fake_model.parent.mkdir(parents=True)
    fake_model.write_text(
        "from vllm_omni.diffusion.offloader import config\n"
        "HOOK = 'layerwise_offload'\n"
        "def f(cfg):\n"
        "    return cfg.diffusion_weight_stack\n",
        encoding="utf-8",
    )
    offloader_f, hook_f, cfg_f, stack_f = _scan_model_tree(tmp_path, _legacy_hook_names())
    assert "models/bad/pipeline_bad.py" in offloader_f
    assert "models/bad/pipeline_bad.py" in hook_f
    assert "models/bad/pipeline_bad.py" in cfg_f
    assert "models/bad/pipeline_bad.py" in stack_f

    # Synthetic T6 invalid row detection.
    bad_sites = tmp_path / "sites"
    bad_sites.mkdir()
    (bad_sites / "BAD_99.py").write_text('ROW = "NO_SUCH_ROW"\nPR = "99"\nSITES = ()\n', encoding="utf-8")
    with pytest.raises(AssertionError, match="unknown ROW"):
        _load_site_files(bad_sites)

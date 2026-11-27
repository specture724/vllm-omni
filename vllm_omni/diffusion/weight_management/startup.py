# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Selector, process state, resource ledger, fault plan, and startup driver (RFC 01)."""

from __future__ import annotations

import enum
import gc
import importlib
import logging
import os
import threading
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Any, Protocol

from vllm_omni.diffusion.weight_management.errors import (
    ConfigError,
    LifecycleError,
    StackUnavailableError,
    WeightStackError,
    wrap_step_error,
)
from vllm_omni.diffusion.weight_management.interfaces import (
    ExecutionOffloadFacts,
    LoadState,
    ParallelCoordinate,
    RankCoordinates,
    RankFingerprints,
    WeightStack,
)

if TYPE_CHECKING:
    from torch import nn

    from vllm_omni.diffusion.weight_management.components import ComponentCatalog
    from vllm_omni.diffusion.weight_management.group import GroupConfirmation
    from vllm_omni.diffusion.weight_management.interfaces import (
        Elections,
        HostStorage,
        HostWeightManager,
        Loader,
        LoadTransaction,
        Offloader,
        OffloadRuntime,
        StartupContext,
        Topology,
    )

logger = logging.getLogger(__name__)


def parse_weight_stack(value: object) -> WeightStack:
    if value is None:
        return WeightStack.LEGACY
    if isinstance(value, str):
        try:
            return WeightStack(value)
        except ValueError:
            pass
    raise ConfigError(
        f"diffusion_weight_stack must be 'legacy' or 'unified', got {value!r}",
        fix="use --diffusion-weight-stack legacy",
    )


def is_unified(config: object) -> bool:
    value = getattr(config, "diffusion_weight_stack", None)
    return isinstance(value, str) and parse_weight_stack(value) is WeightStack.UNIFIED


def execution_offload_facts(config: object) -> ExecutionOffloadFacts:
    """Return engine/executor offload facts for legacy or unified configuration."""
    if is_unified(config):
        # Before RFC 05 policy parsing exists, unified mode rejects offload so all facts are false.
        return ExecutionOffloadFacts(
            weight_collectives=False,
            text_encoder_allgather=False,
            whole_component_offload=False,
        )
    from vllm_omni.diffusion.offloader.config import (
        TEXT_ENCODER_COMPONENT,
        OffloadStrategy,
        resolve_offload,
    )

    resolved = resolve_offload(config)
    return ExecutionOffloadFacts(
        weight_collectives=resolved.any_allgather,
        text_encoder_allgather=resolved.offloads(TEXT_ENCODER_COMPONENT)
        and resolved.uses_allgather(TEXT_ENCODER_COMPONENT),
        whole_component_offload=resolved.strategy is OffloadStrategy.MODEL_LEVEL,
    )


# Identity keys support unhashable owners; weakrefs avoid leaking unclosed workers.
_process_owners: dict[int, tuple[weakref.ref[object] | None, int | None, WeightStack]] = {}
_process_lock = Lock()


def _prune_dead_owners() -> None:
    for owner_id in [k for k, (ref, _, _) in _process_owners.items() if ref is not None and ref() is None]:
        _process_owners.pop(owner_id, None)


def set_process_weight_stack(value: WeightStack, *, owner: object) -> None:
    value = parse_weight_stack(value)
    owner_id = id(owner)
    stage_id = getattr(owner, "stage_id", None)
    try:
        ref: weakref.ref[object] | None = weakref.ref(owner)
    except TypeError:
        ref = None
    with _process_lock:
        _prune_dead_owners()
        for current_id, (_, current_stage, current_value) in _process_owners.items():
            if current_id != owner_id and current_value is not value:
                raise ConfigError(
                    f"stage {stage_id} selects {value.value!r}, but "
                    f"stage {current_stage} already owns {current_value.value!r} "
                    "in this process",
                    fix="use the same diffusion_weight_stack for all workers in one process",
                )
        _process_owners[owner_id] = (ref, stage_id, value)


def clear_process_weight_stack(*, owner: object) -> None:
    with _process_lock:
        _process_owners.pop(id(owner), None)


def process_weight_stack() -> WeightStack:
    with _process_lock:
        _prune_dead_owners()
        return next((value for _, _, value in _process_owners.values()), WeightStack.LEGACY)


class ResourceKind(enum.IntEnum):
    """Resource categories, closed in ascending order (RFC 01 section 5.2)."""

    EXECUTION = 1  # stop accepting new executions
    INFLIGHT = 2  # wait for device copies and collectives to finish
    HOOKS = 3  # remove hooks and model references
    DEVICE = 4  # device slots, events, streams, collective workspaces
    REGISTRATION = 5  # undo memory registration
    STAGING = 6  # release the staging area
    PLACEHOLDER = 7  # repoint parameters at placeholders
    HOST_STORAGE = 8  # close this process's host storage mapping


@dataclass
class _LedgerEntry:
    handle: object
    kind: ResourceKind
    close: Callable[[], None]


class ResourceLedger:
    """Resources owned by one startup transaction or one ``OffloadRuntime``.

    Every resource handle is registered exactly once (by object identity).  The
    ledger owns the handle until it is handed over to an owner outside the
    ledger (:meth:`hand_over`) or moved to another ledger (:meth:`transfer`).

    :meth:`close` releases the remaining entries in ``ResourceKind`` order and,
    within one kind, in reverse registration order.  Closing is idempotent: the
    second call does nothing, and every callback runs at most once even when one
    of them fails.  A failing close callback does not stop the other callbacks;
    after all of them have run, one :class:`LifecycleError` reports the ordinary
    failures with the first one as ``__cause__``.  A cancellation
    (``KeyboardInterrupt``, ``SystemExit`` or another ``BaseException`` that is
    not an ``Exception``) is re-raised in its own type once the remaining
    callbacks have run, and takes precedence over ordinary failures.
    """

    def __init__(self) -> None:
        self._entries: list[_LedgerEntry] = []
        self._closed = False
        self._transferred = False

    def register(self, handle: object, kind: ResourceKind, close: Callable[[], None]) -> None:
        """Register one resource handle; registering the same handle twice is an error."""
        self._check_mutable("register")
        if any(entry.handle is handle for entry in self._entries):
            raise LifecycleError(
                "resource handle is already registered in this ledger",
                where="ResourceLedger.register",
            )
        self._entries.append(_LedgerEntry(handle=handle, kind=kind, close=close))

    def hand_over(self, handle: object) -> None:
        """Remove one entry and hand the handle over to an owner outside the ledger."""
        self._check_mutable("hand_over")
        for index, entry in enumerate(self._entries):
            if entry.handle is handle:
                del self._entries[index]
                return
        raise LifecycleError(
            "resource handle is not registered in this ledger",
            where="ResourceLedger.hand_over",
        )

    def transfer(self) -> ResourceLedger:
        """Move every entry into a new ledger and leave this ledger empty."""
        if self._closed:
            raise LifecycleError(
                "cannot transfer resources after the ledger is closed",
                where="ResourceLedger.transfer",
            )
        if self._transferred:
            raise LifecycleError(
                "resource ledger was already transferred",
                where="ResourceLedger.transfer",
            )
        target = ResourceLedger()
        target._entries = self._entries
        self._entries = []
        self._transferred = True
        return target

    def close(self) -> None:
        """Close every remaining resource; safe to call more than once.

        The ledger is marked closed and emptied before the first callback runs, so
        every registered callback runs exactly once even when one of them raises or
        is interrupted, and a retried ``close()`` cannot run a callback twice.
        """
        if self._closed:
            return
        self._closed = True
        entries = self._entries
        self._entries = []

        # ResourceKind ascending, then reverse registration order inside one kind.
        ordered = sorted(enumerate(entries), key=lambda item: (int(item[1].kind), -item[0]))
        failures: list[tuple[_LedgerEntry, BaseException]] = []
        for _, entry in ordered:
            try:
                entry.close()
            except BaseException as exc:  # noqa: BLE001 - every callback must run
                failures.append((entry, exc))

        if not failures:
            return

        # A cancellation keeps its own type and wins over ordinary cleanup failures:
        # RFC 01 section 6.2 requires ``KeyboardInterrupt``/``SystemExit`` to bypass
        # confirmation and stop the worker immediately, which a ``LifecycleError``
        # wrapper would prevent.  Ordinary failures are logged instead of raised.
        cancellation = next(((entry, exc) for entry, exc in failures if not isinstance(exc, Exception)), None)
        first_entry, first_error = failures[0] if cancellation is None else cancellation
        for entry, exc in failures:
            if exc is first_error:
                continue
            logger.error(
                "resource close callback failed (kind=%s): %s",
                entry.kind.name,
                exc,
                exc_info=exc,
            )
        if cancellation is not None:
            logger.error(
                "resource close was interrupted by %s while closing %s; "
                "the remaining callbacks were attempted and the interruption is re-raised",
                type(first_error).__name__,
                first_entry.kind.name,
                exc_info=first_error,
            )
            raise first_error
        raise LifecycleError(
            f"{len(failures)} resource close callback(s) failed; "
            f"first failure was {first_entry.kind.name}: {type(first_error).__name__}: {first_error}",
            where="ResourceLedger.close",
        ) from first_error

    def _check_mutable(self, action: str) -> None:
        if self._closed:
            raise LifecycleError(
                f"cannot {action} resources after the ledger is closed",
                where=f"ResourceLedger.{action}",
            )
        if self._transferred:
            raise LifecycleError(
                f"cannot {action} resources after the ledger was transferred",
                where=f"ResourceLedger.{action}",
            )


FAULT_ENV_VAR = "VLLM_OMNI_WEIGHT_STACK_FAULTS"

NAMED_FAULT_POINTS = frozenset(
    {
        "construct_modules",
        "build_structure",
        "point0_topology",
        "plan",
        "preflight",
        "point1_fingerprints",
        "confirm",
        "point1_elections",
        "admit",
        "allocate",
        "prepare",
        "point2_ready",
        "connect_writer",
        "write",
        "finalize",
        "commit",
        "point3_commit",
        "compile_setup",
        "close",
    }
)


class InjectedFault(RuntimeError):  # noqa: N818 - RFC 01 fixes this public name
    """Synthetic failure raised by :class:`FaultPlan` in fault-injection tests."""


class FaultPlan:
    """Test-only fault injection table parsed from ``VLLM_OMNI_WEIGHT_STACK_FAULTS``."""

    def __init__(self, entries: Mapping[str, tuple[tuple[int | None, str], ...]]) -> None:
        self._entries = {point: tuple(rules) for point, rules in entries.items()}

    @classmethod
    def parse(cls, spec: str) -> FaultPlan:
        """Parse comma-separated ``<point>[@<rank>][:raise|:hang]`` entries."""
        raw_items = [item.strip() for item in spec.split(",")]
        if not spec.strip() or any(not item for item in raw_items):
            raise ConfigError(f"invalid {FAULT_ENV_VAR} specification {spec!r}")
        parsed: dict[str, list[tuple[int | None, str]]] = {}
        for item in raw_items:
            action = "raise"
            head = item
            if ":" in item:
                head, action = item.split(":", 1)
                if action not in ("raise", "hang"):
                    raise ConfigError(f"invalid fault action {action!r} in {item!r}; use 'raise' or 'hang'")
            rank: int | None = None
            point = head
            if "@" in head:
                point, rank_text = head.split("@", 1)
                if not rank_text.isdigit():
                    raise ConfigError(f"invalid fault rank {rank_text!r} in {item!r}")
                rank = int(rank_text)
            if not point or (point not in NAMED_FAULT_POINTS and not point.startswith(("hwm_", "offload_"))):
                raise ConfigError(f"unknown fault point {point!r} in {item!r}")
            parsed.setdefault(point, []).append((rank, action))
        return cls({point: tuple(rules) for point, rules in parsed.items()})

    @classmethod
    def from_environ(cls, env: Mapping[str, str] | None = None) -> FaultPlan | None:
        raw = (os.environ if env is None else env).get(FAULT_ENV_VAR)
        if raw is None:
            return None
        return cls.parse(raw)

    def hit(self, point: str, rank: int = 0) -> None:
        for target_rank, action in self._entries.get(point, ()):
            if target_rank is None or target_rank == rank:
                if action == "hang":
                    threading.Event().wait()
                raise InjectedFault(f"injected fault at {point} on rank {rank}")


class StackFactory(Protocol):
    """Creates Loader, HostWeightManager, and Offloader for one unified startup."""

    def create(self, ctx: StartupContext) -> tuple[Loader, HostWeightManager, Offloader]: ...


class EntryPointFactory:
    """Default factory that resolves concrete implementations lazily via importlib."""

    LOADER = "vllm_omni.diffusion.weight_management.loader:create_loader"
    HOST = "vllm_omni.diffusion.weight_management.host:create_host_weight_manager"
    OFFLOADER = "vllm_omni.diffusion.weight_management.offload:create_offloader"

    def create(self, ctx: StartupContext) -> tuple[Loader, HostWeightManager, Offloader]:
        specs = (self.LOADER, self.HOST, self.OFFLOADER)
        resolved: list[Callable[[StartupContext], Any]] = []
        missing: list[str] = []
        for spec in specs:
            module_name, func_name = spec.split(":", 1)
            try:
                module = importlib.import_module(module_name)
                func = getattr(module, func_name)
            except (ModuleNotFoundError, AttributeError):
                missing.append(spec)
                continue
            resolved.append(func)
        if missing:
            raise StackUnavailableError(
                f"unified weight-stack factory not available: {', '.join(missing)}",
                fix="use --diffusion-weight-stack legacy",
            )
        return resolved[0](ctx), resolved[1](ctx), resolved[2](ctx)


_VALID_TRANSITIONS: dict[LoadState | None, frozenset[LoadState]] = {
    None: frozenset({LoadState.STRUCTURE_READY, LoadState.FAILED}),
    LoadState.STRUCTURE_READY: frozenset({LoadState.PLANNED, LoadState.FAILED}),
    LoadState.PLANNED: frozenset({LoadState.SOURCES_PREPARED, LoadState.FAILED}),
    LoadState.SOURCES_PREPARED: frozenset({LoadState.ADMITTED, LoadState.FAILED}),
    LoadState.ADMITTED: frozenset({LoadState.TRANSACTION_READY, LoadState.FAILED}),
    LoadState.TRANSACTION_READY: frozenset({LoadState.WEIGHTS_READY, LoadState.FAILED}),
    LoadState.WEIGHTS_READY: frozenset({LoadState.FINALIZED, LoadState.FAILED}),
    LoadState.FINALIZED: frozenset({LoadState.COMMITTED, LoadState.FAILED}),
    LoadState.COMMITTED: frozenset({LoadState.IN_SERVICE, LoadState.FAILED, LoadState.CLOSED}),
    LoadState.IN_SERVICE: frozenset({LoadState.CLOSED}),
    LoadState.FAILED: frozenset(),
    LoadState.CLOSED: frozenset(),
}


def _transition(current: LoadState | None, target: LoadState) -> LoadState:
    if target not in _VALID_TRANSITIONS[current]:
        raise LifecycleError(
            f"illegal weight-stack state transition {current} -> {target}",
            where="WeightStackState",
        )
    return target


class WeightStackHandle:
    """Service-time handle returned by :func:`run_unified_startup`."""

    def __init__(
        self,
        *,
        state: LoadState,
        model: nn.Module,
        catalog: ComponentCatalog,
        runtime: OffloadRuntime,
        faults: FaultPlan | None = None,
        rank: int = 0,
    ) -> None:
        self.state = state
        self.model: nn.Module | None = model
        self.catalog: ComponentCatalog | None = catalog
        self.runtime: OffloadRuntime | None = runtime
        self._faults = faults
        self._rank = rank

    def mark_in_service(self) -> None:
        self.state = _transition(self.state, LoadState.IN_SERVICE)

    def close(self) -> None:
        """Close runtime locally, discard references, and enter ``CLOSED``. Idempotent."""
        if self.state is LoadState.CLOSED:
            return
        injected: BaseException | None = None
        if self._faults is not None:
            try:
                self._faults.hit("close", self._rank)
            except BaseException as exc:
                injected = exc
        runtime = self.runtime
        self.runtime = None
        self.model = None
        self.catalog = None
        try:
            if runtime is not None:
                runtime.close()
        finally:
            self.state = LoadState.CLOSED
        if injected is not None:
            raise LifecycleError(
                f"close interrupted by {type(injected).__name__}: {injected}",
                phase="close",
            ) from injected


def _attach_context(err: WeightStackError, ctx: StartupContext) -> WeightStackError:
    if err.stage_id is None:
        err.stage_id = getattr(ctx.od_config, "stage_id", None)
    if err.rank is None:
        err.rank = ctx.rank
    if err.world_size is None:
        err.world_size = ctx.world_size
    return err


def _close_owner(owner: object | None) -> None:
    if owner is None:
        return
    if hasattr(owner, "abort"):
        owner.abort()
    elif hasattr(owner, "close"):
        owner.close()


def _local_coordinates(ctx: StartupContext) -> RankCoordinates:
    from vllm_omni.diffusion.distributed import parallel_state as diff_ps
    from vllm_omni.diffusion.weight_management.group import read_rank_coordinates

    if not diff_ps.model_parallel_is_initialized():
        one = ParallelCoordinate(1, 0)
        return RankCoordinates(
            tp=one,
            sp=one,
            ulysses=one,
            ring=one,
            cfg=one,
            dp=ParallelCoordinate(ctx.world_size, ctx.rank),
            pp=one,
            ep=None,
            fs=None,
        )
    return read_rank_coordinates()


class _Startup:
    """Explicit state and ownership holder for one unified startup transaction."""

    def __init__(
        self,
        ctx: StartupContext,
        factory: StackFactory,
        group: GroupConfirmation,
    ) -> None:
        self.ctx = ctx
        self.factory = factory
        self.group = group
        self.state: LoadState | None = None
        self.owner: Any = None
        self.pending: WeightStackError | None = None
        self.loader: Loader | None = None
        self.hwm: HostWeightManager | None = None
        self.offloader: Offloader | None = None
        self.model: nn.Module | None = None
        self.catalog: Any = None
        self.topology: Topology | None = None
        self.plan_: Any = None
        self.preflight_: Any = None
        self.layout: Any = None
        self.fingerprints: RankFingerprints | None = None
        self.share_plan: Any = None
        self.election: Any = None
        self.elections: Elections | None = None
        self.storage: HostStorage | None = None
        self.transaction: LoadTransaction | None = None
        self.final_manifest: Any = None
        self.runtime: OffloadRuntime | None = None

    def enter(self, target: LoadState) -> None:
        self.state = _transition(self.state, target)

    def local(self, phase: str, fn: Callable[[], None]) -> None:
        if self.pending is not None:
            return
        try:
            if self.ctx.faults is not None:
                self.ctx.faults.hit(phase, self.ctx.rank)
            fn()
        except Exception as exc:
            self.fail(exc, phase)

    def round(self, point: str, compute: Callable[[], object] | None = None) -> tuple[Any, ...]:
        from vllm_omni.diffusion.weight_management.group import RoundMessage, raise_if_any_failed

        payload: object | None = None
        if self.pending is None:
            try:
                if self.ctx.faults is not None:
                    self.ctx.faults.hit(point, self.ctx.rank)
                payload = compute() if compute is not None else None
            except Exception as exc:
                self.fail(exc, point)
        message = RoundMessage.of(point, self.ctx.rank, self.pending, payload)
        gathered = self.group.exchange(point, message)
        try:
            raise_if_any_failed(gathered, self.pending)
        except WeightStackError as exc:
            raise _attach_context(exc, self.ctx) from exc.__cause__
        return tuple(m.payload for m in gathered)

    def fail(self, exc: Exception, phase: str) -> None:
        self.pending = _attach_context(wrap_step_error(phase, exc), self.ctx)
        self.release()

    def release(self) -> None:
        owner = self.owner
        self.owner = None
        try:
            _close_owner(owner)
        finally:
            loader = self.loader
            self.loader = None
            if loader is not None:
                loader.close()
            self.model = None
            self.catalog = None
            self.transaction = None
            self.storage = None
            self.runtime = None
            gc.collect()
            from vllm_omni.platforms import current_omni_platform

            current_omni_platform.empty_cache()
            if self.state is not LoadState.FAILED:
                self.state = LoadState.FAILED

    def construct_modules(self) -> None:
        self.loader, self.hwm, self.offloader = self.factory.create(self.ctx)

    def build_structure(self) -> None:
        assert self.loader is not None
        self.model, self.catalog = self.loader.build_structure()
        self.enter(LoadState.STRUCTURE_READY)

    def confirm_topology(self) -> None:
        from vllm_omni.diffusion.weight_management.group import Point0Payload, assemble_topology

        gathered = self.round(
            "point0_topology",
            lambda: Point0Payload(
                record=self.hwm.local_record(),  # type: ignore[union-attr]
                coordinates=_local_coordinates(self.ctx),
                deployment=self.hwm.deployment_id(),  # type: ignore[union-attr]
            ),
        )
        try:
            self.topology = assemble_topology(gathered, self.ctx)
        except WeightStackError as exc:
            raise _attach_context(exc, self.ctx) from exc.__cause__

    def plan(self) -> None:
        assert self.offloader is not None and self.topology is not None
        self.plan_ = self.offloader.plan(self.catalog, self.topology, self.ctx.policy)
        self.enter(LoadState.PLANNED)

    def preflight(self) -> None:
        assert self.loader is not None and self.hwm is not None
        self.preflight_ = self.loader.preflight(self.catalog, self.plan_)
        self.layout = self.hwm.layout(self.plan_, self.preflight_.expected_manifest)
        self.fingerprints = RankFingerprints(
            rank=self.ctx.rank,
            content_identities=self.preflight_.content_identities,
            plan_fingerprint=self.layout.plan_fingerprint,
            collective=self.plan_.collective_signature(),
        )
        self.enter(LoadState.SOURCES_PREPARED)

    def confirm_admission(self) -> None:
        from vllm_omni.diffusion.weight_management.group import (
            Point1Payload,
            check_collective_parts,
            check_unmatched_policy_keys,
            create_allgather_groups,
            group_table_keys,
        )

        gathered = self.round(
            "point1_fingerprints",
            lambda: Point1Payload(
                fingerprints=self.fingerprints,  # type: ignore[arg-type]
                group_keys=group_table_keys(),
                unmatched_policy_keys=self.plan_.unmatched_policy_keys(),
            ),
        )
        all_fps = tuple(p.fingerprints for p in gathered)
        try:
            check_unmatched_policy_keys(gathered)
            check_collective_parts(all_fps)
            groups = create_allgather_groups(all_fps, gathered)
        except WeightStackError as exc:
            raise _attach_context(exc, self.ctx) from exc.__cause__
        self.local("confirm", lambda: self._bind_and_elect(groups, all_fps))
        records = self.round("point1_elections", lambda: self.election.record())
        self.local("admit", lambda: self._admit(records))

    def _bind_and_elect(self, groups: Any, all_fps: tuple[RankFingerprints, ...]) -> None:
        assert self.hwm is not None and self.topology is not None
        self.layout.bind_groups(groups)
        self.plan_.bind_groups(groups)
        self.share_plan = self.hwm.share_plan(self.topology, all_fps)
        self.election = self.hwm.elect(self.share_plan)
        self.owner = self.election

    def _admit(self, records: tuple[Any, ...]) -> None:
        from vllm_omni.diffusion.weight_management.interfaces import Elections

        assert self.hwm is not None and self.topology is not None
        self.elections = Elections(local=self.election, records=records)
        self.hwm.admit(self.topology, self.share_plan, self.elections, self.layout)
        self.enter(LoadState.ADMITTED)

    def allocate(self) -> None:
        assert self.hwm is not None and self.elections is not None
        self.storage = self.hwm.allocate(self.layout, self.share_plan, self.elections)
        self.owner = self.storage

    def prepare(self) -> None:
        assert self.offloader is not None and self.model is not None and self.storage is not None
        self.transaction = self.offloader.prepare_materialization(self.model, self.plan_, self.storage)
        self.owner = self.transaction

    def confirm_ready(self) -> None:
        self.round("point2_ready")
        self.enter(LoadState.TRANSACTION_READY)

    def connect_writer(self) -> None:
        assert self.storage is not None
        self.storage.connect_writer()

    def write(self) -> None:
        assert self.loader is not None and self.model is not None
        assert self.storage is not None and self.transaction is not None
        self.loader.write(self.model, self.storage, self.transaction)
        self.enter(LoadState.WEIGHTS_READY)

    def finalize(self) -> None:
        assert self.loader is not None and self.model is not None and self.storage is not None
        self.final_manifest = self.loader.finalize(self.model, self.storage)
        loader = self.loader
        self.loader = None
        loader.close()
        self.enter(LoadState.FINALIZED)

    def commit(self) -> None:
        assert self.transaction is not None
        self.runtime = self.transaction.commit(self.final_manifest)
        self.owner = self.runtime
        self.enter(LoadState.COMMITTED)

    def confirm_commit(self) -> None:
        self.round("point3_commit")

    def compile_setup(self) -> None:
        if not self.ctx.compile_enabled:
            return
        assert self.offloader is not None and self.model is not None and self.runtime is not None
        try:
            if self.ctx.faults is not None:
                self.ctx.faults.hit("compile_setup", self.ctx.rank)
            self.offloader.setup_compile(self.model, self.catalog, self.runtime)
        except Exception as exc:
            logger.warning("setup_compile failed (%s); running without compile", exc)

    def handle(self) -> WeightStackHandle:
        assert self.state is LoadState.COMMITTED
        assert self.model is not None and self.runtime is not None
        return WeightStackHandle(
            state=self.state,
            model=self.model,
            catalog=self.catalog,
            runtime=self.runtime,
            faults=self.ctx.faults,
            rank=self.ctx.rank,
        )


def run_unified_startup(
    ctx: StartupContext,
    factory: StackFactory | None = None,
    *,
    group: GroupConfirmation | None = None,
) -> WeightStackHandle:
    """Run the unified startup transaction (RFC 01 section 6)."""
    if group is None:
        from vllm_omni.diffusion.weight_management.group import GroupConfirmation

        group = (
            GroupConfirmation(world_size=1, timeout_seconds=getattr(ctx.od_config, "dist_timeout", None))
            if ctx.world_size == 1
            else GroupConfirmation.from_config(ctx.od_config)
        )
    driver = _Startup(ctx, factory or EntryPointFactory(), group)
    try:
        driver.local("construct_modules", driver.construct_modules)
        driver.local("build_structure", driver.build_structure)
        driver.confirm_topology()
        driver.local("plan", driver.plan)
        driver.local("preflight", driver.preflight)
        driver.confirm_admission()
        driver.local("allocate", driver.allocate)
        driver.local("prepare", driver.prepare)
        driver.confirm_ready()
        driver.local("connect_writer", driver.connect_writer)
        driver.local("write", driver.write)
        driver.local("finalize", driver.finalize)
        driver.local("commit", driver.commit)
        driver.confirm_commit()
        driver.compile_setup()
        return driver.handle()
    except BaseException:
        driver.release()
        raise

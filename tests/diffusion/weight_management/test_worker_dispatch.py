# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for unified worker and model-runner dispatch (RFC 01-6)."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from tests.diffusion.weight_management.fakes import FakeMemoryPools, FakeStackFactory
from vllm_omni.diffusion.data import CuMemTag, OmniDiffusionConfig
from vllm_omni.diffusion.weight_management.errors import LifecycleError
from vllm_omni.diffusion.weight_management.interfaces import LoadState
from vllm_omni.diffusion.weight_management.startup import WeightStack, process_weight_stack

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _make_runner(od_config: OmniDiffusionConfig):
    from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner

    runner = object.__new__(DiffusionModelRunner)
    runner.vllm_config = SimpleNamespace()
    runner.od_config = od_config
    runner.device = torch.device("cpu")
    runner.pipeline = None
    runner.weight_stack = None
    runner._weight_stack_factory = None
    runner.cache_backend = None
    runner.offload_backend = None
    runner.prompt_embed_cache = None
    runner._interaction_coordinator = None
    runner.model_memory_usage = 0
    runner._validate_service_attention_schedule = lambda: None
    return runner


def test_runner_unified_load_installs_services_and_rejects_reload_without_close(monkeypatch):
    from vllm_omni.diffusion.worker import diffusion_model_runner as runner_mod

    monkeypatch.setattr(runner_mod.current_omni_platform, "init_diffusion_model_runner_runtime", lambda **kw: None)
    monkeypatch.setattr(runner_mod.current_omni_platform, "supports_torch_inductor", lambda: False)
    monkeypatch.setattr(runner_mod, "get_cache_backend", lambda *a, **kw: None)
    monkeypatch.setattr(runner_mod.InteractionCoordinator, "build", lambda pipeline, cfg: SimpleNamespace(p=pipeline))

    od_config = OmniDiffusionConfig(diffusion_weight_stack="unified")
    runner = _make_runner(od_config)
    factory = FakeStackFactory()
    pools = FakeMemoryPools()
    runner._weight_stack_factory = factory

    runner.load_model(memory_pools=pools)
    assert runner.weight_stack is not None
    assert runner.weight_stack.state is LoadState.IN_SERVICE
    assert runner.pipeline is runner.weight_stack.model
    assert runner.offload_backend is None
    assert runner._interaction_coordinator is not None

    with pytest.raises(LifecycleError, match="closed before reloading"):
        runner.load_model(memory_pools=pools)

    runner.close_weight_stack()
    assert runner.weight_stack is None
    assert runner.pipeline is None
    assert factory.offloader is not None and factory.offloader.slot_closed == 1
    # Idempotent close.
    runner.close_weight_stack()
    assert factory.offloader.slot_closed == 1


def test_runner_legacy_load_keeps_existing_call_sequence(monkeypatch):
    from vllm_omni.diffusion.worker import diffusion_model_runner as runner_mod

    calls: list[str] = []
    fake_pipeline = SimpleNamespace(setup_compile=lambda: calls.append("setup_compile"))

    class _FakeLegacyLoader:
        def __init__(self, *args, **kwargs):
            pass

        def load_model(self, **kwargs):
            calls.append("legacy.load_model")
            return fake_pipeline

    monkeypatch.setattr(runner_mod.current_omni_platform, "init_diffusion_model_runner_runtime", lambda **kw: None)
    monkeypatch.setattr(runner_mod.current_omni_platform, "supports_torch_inductor", lambda: True)
    monkeypatch.setattr(runner_mod, "DiffusersPipelineLoader", _FakeLegacyLoader)
    monkeypatch.setattr(
        runner_mod,
        "enable_offload_backend",
        lambda cfg, pipe, device: (calls.append("enable_offload_backend") or pipe, "legacy-backend"),
    )
    monkeypatch.setattr(runner_mod, "get_cache_backend", lambda *a, **kw: None)
    monkeypatch.setattr(runner_mod.InteractionCoordinator, "build", lambda p, c: None)

    od_config = OmniDiffusionConfig(diffusion_weight_stack="legacy", enforce_eager=False)
    runner = _make_runner(od_config)
    runner.load_model()

    assert calls == ["legacy.load_model", "enable_offload_backend", "setup_compile"]
    assert runner.pipeline is fake_pipeline
    assert runner.offload_backend == "legacy-backend"
    assert runner.weight_stack is None


def test_worker_unified_shutdown_order_and_reinit(monkeypatch):
    from vllm_omni.diffusion.weight_management import group as group_mod
    from vllm_omni.diffusion.worker import diffusion_worker as worker_mod

    events: list[str] = []
    monkeypatch.setattr(worker_mod.DiffusionWorker, "init_device", lambda self: None)
    monkeypatch.setattr(worker_mod.DiffusionWorker, "_create_profiler", lambda self: None)
    monkeypatch.setattr(worker_mod.DiffusionWorker, "init_lora_manager", lambda self: events.append("init_lora"))
    monkeypatch.setattr(worker_mod, "destroy_distributed_env", lambda: events.append("destroy_distributed_env"))
    monkeypatch.setattr(group_mod, "destroy_allgather_groups", lambda: events.append("destroy_allgather_groups"))
    monkeypatch.setattr(worker_mod.current_omni_platform, "empty_cache", lambda: None)

    runner_stub = SimpleNamespace(
        offload_backend=None,
        kv_transfer_manager=None,
        pipeline="initial",
        close_weight_stack=lambda: events.append("close_weight_stack"),
    )
    monkeypatch.setattr(worker_mod, "resolve_obj_by_qualname", lambda name: lambda **kw: runner_stub)

    od_config = OmniDiffusionConfig(
        stage_id=0,
        diffusion_weight_stack="unified",
        diffusion_model_runner_cls="test.Runner",
    )
    worker = worker_mod.DiffusionWorker(0, 0, od_config, skip_load_model=True)
    assert process_weight_stack() is WeightStack.UNIFIED

    # Custom-pipeline reinit closes previous weight stack before reloading.
    monkeypatch.setattr(worker, "load_model", lambda **kw: events.append(f"load:{kw['load_format']}"))
    monkeypatch.setattr(worker, "_init_weight_transfer_engine", lambda: None)
    worker_mod.CustomPipelineWorkerExtension.re_init_pipeline(worker, {"pipeline_class": "my.CustomPipeline"})
    assert events[:3] == ["close_weight_stack", "load:custom_pipeline", "init_lora"]

    events.clear()
    worker.shutdown()
    assert events == ["close_weight_stack", "destroy_allgather_groups", "destroy_distributed_env"]
    assert process_weight_stack() is WeightStack.LEGACY


def test_worker_unified_sleep_wake_hooks_and_weight_slots_tag(monkeypatch):
    from vllm_omni.diffusion.worker import diffusion_worker as worker_mod

    assert CuMemTag.WEIGHT_SLOTS.value == "weight_slots"

    events: list[object] = []

    class _FakeAllocator:
        def get_current_usage(self) -> int:
            return 0

        def use_memory_pool(self, tag: str):
            events.append(("pool", tag))
            return nullcontext()

        def sleep(self, offload_tags=()):
            events.append(("sleep", offload_tags))

        def wake_up(self, tags=None):
            events.append(("wake_up", list(tags) if tags is not None else None))

    fake_alloc = _FakeAllocator()
    monkeypatch.setattr(worker_mod, "_get_mem_allocator_instance", lambda: fake_alloc)
    monkeypatch.setattr(worker_mod.current_omni_platform, "synchronize", lambda: None)
    monkeypatch.setattr(worker_mod.current_omni_platform, "empty_cache", lambda: None)
    monkeypatch.setattr(worker_mod.current_omni_platform, "get_free_memory", lambda *a: 1024)
    monkeypatch.setattr(worker_mod.current_omni_platform, "get_device_total_memory", lambda: 2048)

    worker = object.__new__(worker_mod.DiffusionWorker)
    worker.rank = 0
    worker.device = torch.device("cpu")
    worker.od_config = OmniDiffusionConfig(diffusion_weight_stack="unified", enable_sleep_mode=True)
    worker._owns_sleep_pool = False
    worker._sleep_saved_buffers = {}
    runtime = SimpleNamespace(
        before_sleep=lambda: events.append("before_sleep"),
        after_wake=lambda: events.append("after_wake"),
    )
    worker.model_runner = SimpleNamespace(
        weight_stack=SimpleNamespace(runtime=runtime),
        pipeline=None,
        _kv_receive_progress=None,
        refresh_diffusion_kv_block_table_layout=lambda: None,
    )

    pools = worker._build_unified_memory_pools()
    assert worker._owns_sleep_pool is True
    with pools.weights(), pools.slots():
        pass
    assert events[:2] == [("pool", "weights"), ("pool", "weight_slots")]

    events.clear()
    worker.sleep(level=1)
    assert events == ["before_sleep", ("sleep", ("weights",))]

    events.clear()
    worker.wake_up(tags=["weights"])
    assert events == [("wake_up", ["weights", "weight_slots"]), "after_wake"]

    events.clear()
    worker.wake_up(tags=["kv_cache"])
    assert events == [("wake_up", ["kv_cache"]), "after_wake"]

    events.clear()
    worker.wake_up(tags=None)
    assert events == [("wake_up", None), "after_wake"]

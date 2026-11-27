# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the unified startup driver (RFC 01-5)."""

from __future__ import annotations

import pytest
import torch.distributed as dist

from tests.diffusion.weight_management.fakes import (
    FakeHostLayout,
    FakeMemoryPools,
    FakePlacementPlan,
    FakeSharePlan,
    FakeStackFactory,
    make_startup_context,
)
from vllm_omni.diffusion.weight_management.errors import LifecycleError, StackUnavailableError
from vllm_omni.diffusion.weight_management.interfaces import (
    LoadState,
    RankFingerprints,
    Topology,
)
from vllm_omni.diffusion.weight_management.startup import EntryPointFactory, run_unified_startup

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_unified_startup_happy_path_states_pools_and_close(monkeypatch):
    pools = FakeMemoryPools()
    factory = FakeStackFactory()
    ctx = make_startup_context(compile_enabled=True, pools=pools)

    handle = run_unified_startup(ctx, factory=factory)
    assert handle.state is LoadState.COMMITTED
    assert handle.model is not None
    assert handle.catalog is not None
    assert handle.runtime is not None
    assert factory.loader is not None and factory.loader.writes == 1 and factory.loader.close_count == 1
    assert factory.offloader is not None and factory.offloader.compile_calls == 1
    assert factory.hwm is not None and factory.hwm.storage is not None and factory.hwm.storage.sealed

    # Driver never enters pools globally; Offloader enters slots during prepare and Loader enters weights during write.
    assert pools.events == ["slots:enter", "slots:exit", "weights:enter", "weights:exit"]

    # No startup state leaked onto model/submodules or returned handle.
    forbidden_types = (Topology, FakeSharePlan, RankFingerprints, FakePlacementPlan, FakeHostLayout)
    for module in handle.model.modules():
        for value in vars(module).values():
            assert not isinstance(value, forbidden_types)
    for value in vars(handle).values():
        assert not isinstance(value, forbidden_types)

    handle.mark_in_service()
    assert handle.state is LoadState.IN_SERVICE
    with pytest.raises(LifecycleError):
        handle.mark_in_service()

    # Local close must never invoke torch.distributed collectives.
    def _forbidden(*args, **kwargs):
        pytest.fail("handle.close() must not call torch.distributed")

    for name in ("barrier", "monitored_barrier", "all_gather_object", "destroy_process_group"):
        monkeypatch.setattr(dist, name, _forbidden)

    handle.close()
    assert handle.state is LoadState.CLOSED
    assert handle.model is None and handle.runtime is None
    assert factory.offloader.slot_closed == 1
    assert factory.hwm.storage.close_count == 1
    # Idempotent close.
    handle.close()
    assert factory.offloader.slot_closed == 1


def test_default_entry_point_factory_reports_missing_modules_and_legacy_fix():
    ctx = make_startup_context(stage_id=2)
    with pytest.raises(StackUnavailableError) as exc_info:
        run_unified_startup(ctx, factory=EntryPointFactory())
    message = str(exc_info.value)
    assert "[stage 2][rank 0/1][construct_modules]" in message
    assert "create_loader" in message
    assert "create_host_weight_manager" in message
    assert "create_offloader" in message
    assert "--diffusion-weight-stack legacy" in message

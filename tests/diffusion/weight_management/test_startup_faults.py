# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fault-injection and failure-cleanup tests for RFC 01-5."""

from __future__ import annotations

import logging

import pytest

from tests.diffusion.weight_management.fakes import FakeStackFactory, make_startup_context
from vllm_omni.diffusion.weight_management.errors import (
    STEP_ERROR_TYPES,
    ConfigError,
    LifecycleError,
)
from vllm_omni.diffusion.weight_management.interfaces import LoadState
from vllm_omni.diffusion.weight_management.startup import (
    FAULT_ENV_VAR,
    FaultPlan,
    InjectedFault,
    run_unified_startup,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_fault_plan_from_environ_and_syntax():
    assert FaultPlan.from_environ({}) is None

    plan = FaultPlan.from_environ({FAULT_ENV_VAR: "write, prepare@1:raise, point0_topology@0:hang, hwm_seal@2"})
    assert plan is not None
    with pytest.raises(InjectedFault, match="write"):
        plan.hit("write", rank=0)
    plan.hit("prepare", rank=0)  # rank mismatch -> no-op
    with pytest.raises(InjectedFault, match="prepare"):
        plan.hit("prepare", rank=1)
    with pytest.raises(InjectedFault, match="hwm_seal"):
        plan.hit("hwm_seal", rank=2)


@pytest.mark.parametrize("spec", ["", "   ", "unknown_point", "write@abc", "write:-1", "write:explode", "write,"])
def test_fault_plan_rejects_invalid_syntax(spec):
    with pytest.raises(ConfigError):
        FaultPlan.parse(spec)


PRE_WRITE_POINTS = {
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
}


@pytest.mark.parametrize("point", sorted(STEP_ERROR_TYPES))
def test_fault_at_every_startup_step_cleans_up_once(point):
    factory = FakeStackFactory()
    ctx = make_startup_context(stage_id=1, faults=FaultPlan.parse(point))
    expected_type = STEP_ERROR_TYPES[point]

    with pytest.raises(expected_type) as exc_info:
        run_unified_startup(ctx, factory=factory)

    err = exc_info.value
    assert err.stage_id == 1
    assert err.rank == 0
    assert err.world_size == 1
    assert isinstance(err.__cause__, InjectedFault)

    if point in PRE_WRITE_POINTS:
        assert factory.loader is None or factory.loader.writes == 0
    if factory.loader is not None:
        assert factory.loader.close_count == 1

    # Ownership cleanup: whoever owned resources at failure closed exactly once.
    if point in ("point1_elections", "admit", "allocate"):
        assert factory.hwm is not None and factory.hwm.election is not None
        assert factory.hwm.election.close_count == 1
    elif point == "prepare":
        assert factory.hwm is not None and factory.hwm.storage is not None
        assert factory.hwm.storage.close_count == 1
        assert factory.hwm.election is not None and factory.hwm.election.close_count == 0
    elif point in ("point2_ready", "connect_writer", "write", "finalize", "commit"):
        assert factory.offloader is not None and factory.offloader.transaction is not None
        assert factory.offloader.transaction.aborted == 1
        assert factory.offloader.slot_closed == 1
        assert factory.hwm is not None and factory.hwm.storage is not None
        assert factory.hwm.storage.close_count == 1
    elif point == "point3_commit":
        assert factory.offloader is not None and factory.offloader.transaction is not None
        assert factory.offloader.transaction.runtime is not None
        assert factory.offloader.transaction.runtime.close_count == 1
        assert factory.offloader.slot_closed == 1
        assert factory.hwm is not None and factory.hwm.storage is not None
        assert factory.hwm.storage.close_count == 1


def test_compile_setup_fault_warns_and_returns_committed_handle(caplog):
    factory = FakeStackFactory()
    ctx = make_startup_context(compile_enabled=True, faults=FaultPlan.parse("compile_setup"))
    with caplog.at_level(logging.WARNING):
        handle = run_unified_startup(ctx, factory=factory)
    assert handle.state is LoadState.COMMITTED
    assert any("setup_compile failed" in rec.getMessage() for rec in caplog.records)
    handle.close()


def test_close_fault_finishes_cleanup_before_raising_lifecycle_error():
    factory = FakeStackFactory()
    ctx = make_startup_context(faults=FaultPlan.parse("close"))
    handle = run_unified_startup(ctx, factory=factory)

    with pytest.raises(LifecycleError) as exc_info:
        handle.close()

    assert isinstance(exc_info.value.__cause__, InjectedFault)
    assert handle.state is LoadState.CLOSED
    assert handle.model is None and handle.runtime is None
    assert factory.offloader is not None and factory.offloader.slot_closed == 1
    assert factory.hwm is not None and factory.hwm.storage is not None and factory.hwm.storage.close_count == 1


def test_keyboard_interrupt_bypasses_confirmation_and_releases_resources():
    factory = FakeStackFactory()
    ctx = make_startup_context()

    def _interrupt_write(model, storage, transaction):
        raise KeyboardInterrupt("stop now")

    orig_create = factory.create

    def _create(c):
        loader, hwm, offloader = orig_create(c)
        loader.write = _interrupt_write  # type: ignore[method-assign]
        return loader, hwm, offloader

    factory.create = _create  # type: ignore[method-assign]

    with pytest.raises(KeyboardInterrupt, match="stop now"):
        run_unified_startup(ctx, factory=factory)

    assert factory.offloader is not None and factory.offloader.slot_closed == 1
    assert factory.hwm is not None and factory.hwm.storage is not None and factory.hwm.storage.close_count == 1
    assert factory.loader is not None and factory.loader.close_count == 1

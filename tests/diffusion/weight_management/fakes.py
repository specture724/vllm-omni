# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Minimal test doubles for RFC 01-5 startup driver and 01-6/02/03/04 tests."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace

import torch
from torch import nn

from vllm_omni.diffusion.weight_management.interfaces import (
    AllGatherGroups,
    Elections,
    Fingerprints,
    StartupContext,
    Topology,
)
from vllm_omni.diffusion.weight_management.startup import ResourceKind, ResourceLedger


class FakeMemoryPools:
    """Tracks pool context entries without requiring a GPU allocator."""

    def __init__(self) -> None:
        self.events: list[str] = []

    @contextmanager
    def weights(self) -> Iterator[None]:
        self.events.append("weights:enter")
        try:
            yield
        finally:
            self.events.append("weights:exit")

    @contextmanager
    def slots(self) -> Iterator[None]:
        self.events.append("slots:enter")
        try:
            yield
        finally:
            self.events.append("slots:exit")


@dataclass
class FakeCatalog:
    components: dict[str, nn.Module] = field(default_factory=dict)


@dataclass
class FakePlacementPlan:
    unmatched: frozenset[str] = frozenset()
    collective: dict[str, str] = field(default_factory=dict)
    bound_groups: AllGatherGroups | None = None

    def collective_signature(self) -> Mapping[str, str]:
        return dict(self.collective)

    def unmatched_policy_keys(self) -> frozenset[str]:
        return self.unmatched

    def bind_groups(self, groups: AllGatherGroups) -> None:
        self.bound_groups = groups


@dataclass
class FakePreflight:
    content_identities: dict[str, str]
    expected_manifest: dict[str, tuple[tuple[int, ...], torch.dtype]]


@dataclass
class FakeHostLayout:
    plan_fingerprint: str = "fake-plan-fp"
    bound_groups: AllGatherGroups | None = None

    def bind_groups(self, groups: AllGatherGroups) -> None:
        self.bound_groups = groups


@dataclass
class FakeSharePlan:
    members: tuple[int, ...] = (0,)


class FakeElection:
    def __init__(self, rank: int = 0) -> None:
        self.rank = rank
        self.close_count = 0

    def record(self) -> dict[str, int]:
        return {"rank": self.rank}

    def close(self) -> None:
        self.close_count += 1


class FakeHostStorage:
    """Empty host storage for no-offload startup doubles."""

    def __init__(self, election: FakeElection | None = None) -> None:
        self.election = election
        self.connected = False
        self.sealed = False
        self.close_count = 0
        self.written_keys: list[str] = []

    def view(self, tensor_key: str) -> torch.Tensor:
        return torch.empty(0)

    def ranges(self, unit: object) -> list[object]:
        return []

    def seal(self, final_manifest: object) -> None:
        self.sealed = True

    def connect_writer(self) -> None:
        self.connected = True

    def note_written(self, tensor_key: str) -> None:
        self.written_keys.append(tensor_key)

    def close(self) -> None:
        self.close_count += 1


class FakeOffloadRuntime:
    def __init__(self, ledger: ResourceLedger, events: list[str]) -> None:
        self._ledger = ledger
        self._events = events
        self.close_count = 0
        self.slept = False

    def close(self) -> None:
        if self.close_count == 0:
            self._ledger.close()
            self._events.append("runtime.close")
        self.close_count += 1

    @contextmanager
    def grad_mode(self) -> Iterator[None]:
        with torch.inference_mode():
            yield

    @contextmanager
    def execution(self) -> Iterator[None]:
        yield

    def before_sleep(self) -> None:
        self.slept = True

    def after_wake(self) -> None:
        self.slept = False


class FakeLoadTransaction:
    def __init__(self, ledger: ResourceLedger, events: list[str]) -> None:
        self._ledger = ledger
        self._events = events
        self.aborted = 0
        self.committed = False
        self.runtime: FakeOffloadRuntime | None = None

    def target(self, tensor_key: str) -> torch.Tensor | None:
        # All weights are permanently resident in the no-offload double.
        return None

    def commit(self, final_manifest: object) -> FakeOffloadRuntime:
        self._events.append("transaction.commit")
        self.committed = True
        self.runtime = FakeOffloadRuntime(self._ledger.transfer(), self._events)
        return self.runtime

    def abort(self) -> None:
        if not self.committed and self.aborted == 0:
            self._ledger.close()
            self._events.append("transaction.abort")
        self.aborted += 1


class FakeLoader:
    def __init__(self, ctx: StartupContext, events: list[str]) -> None:
        self.ctx = ctx
        self.events = events
        self.writes = 0
        self.close_count = 0

    def build_structure(self) -> tuple[nn.Module, FakeCatalog]:
        self.events.append("loader.build_structure")
        model = nn.Sequential(nn.Linear(4, 4, bias=False))
        return model, FakeCatalog(components={"dit": model})

    def preflight(self, catalog: FakeCatalog, plan: FakePlacementPlan) -> FakePreflight:
        self.events.append("loader.preflight")
        return FakePreflight(
            content_identities={"dit": "fake-dit-id"},
            expected_manifest={"0.weight": ((4, 4), torch.float32)},
        )

    def write(self, model: nn.Module, storage: FakeHostStorage, transaction: FakeLoadTransaction) -> None:
        self.events.append("loader.write")
        self.writes += 1
        with self.ctx.memory_pools.weights():
            for name, param in model.named_parameters():
                if transaction.target(name) is None:
                    param.data = torch.ones_like(param.data, device=self.ctx.device)

    def finalize(self, model: nn.Module, storage: FakeHostStorage) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
        self.events.append("loader.finalize")
        storage.seal({"0.weight": ((4, 4), torch.float32)})
        model.eval()
        return {"0.weight": ((4, 4), torch.float32)}

    def close(self) -> None:
        self.close_count += 1
        self.events.append("loader.close")


class FakeHostWeightManager:
    def __init__(self, ctx: StartupContext, events: list[str]) -> None:
        self.ctx = ctx
        self.events = events
        self.election: FakeElection | None = None
        self.storage: FakeHostStorage | None = None

    def local_record(self) -> dict[str, int]:
        return {"rank": self.ctx.rank}

    def deployment_id(self) -> tuple[str, bool]:
        return ("a" * 32, False)

    def layout(self, plan: FakePlacementPlan, expected_manifest: object) -> FakeHostLayout:
        self.events.append("hwm.layout")
        return FakeHostLayout()

    def share_plan(self, topology: Topology, fingerprints: Fingerprints) -> FakeSharePlan:
        self.events.append("hwm.share_plan")
        return FakeSharePlan(members=tuple(range(topology.world_size)))

    def elect(self, share_plan: FakeSharePlan) -> FakeElection:
        self.events.append("hwm.elect")
        self.election = FakeElection(self.ctx.rank)
        return self.election

    def admit(
        self, topology: Topology, share_plan: FakeSharePlan, elections: Elections, layout: FakeHostLayout
    ) -> None:
        self.events.append("hwm.admit")

    def allocate(self, layout: FakeHostLayout, share_plan: FakeSharePlan, elections: Elections) -> FakeHostStorage:
        self.events.append("hwm.allocate")
        self.storage = FakeHostStorage(election=self.election)
        return self.storage


class FakeOffloader:
    def __init__(self, ctx: StartupContext, events: list[str]) -> None:
        self.ctx = ctx
        self.events = events
        self.transaction: FakeLoadTransaction | None = None
        self.slot_closed = 0
        self.compile_calls = 0

    def plan(self, catalog: FakeCatalog, topology: Topology, policy: object) -> FakePlacementPlan:
        self.events.append("offloader.plan")
        return FakePlacementPlan()

    def prepare_materialization(
        self,
        model: nn.Module,
        plan: FakePlacementPlan,
        storage: FakeHostStorage,
    ) -> FakeLoadTransaction:
        self.events.append("offloader.prepare_materialization")
        ledger = ResourceLedger()
        with self.ctx.memory_pools.slots():
            slot = torch.empty(1, device=self.ctx.device)

        def _close_slot() -> None:
            self.slot_closed += 1

        ledger.register(slot, ResourceKind.DEVICE, _close_slot)
        ledger.register(storage, ResourceKind.HOST_STORAGE, storage.close)
        self.transaction = FakeLoadTransaction(ledger, self.events)
        return self.transaction

    def setup_compile(self, model: nn.Module, catalog: FakeCatalog, runtime: FakeOffloadRuntime) -> None:
        self.events.append("offloader.setup_compile")
        self.compile_calls += 1


class FakeStackFactory:
    """Factory that builds and exposes the three fake modules for assertions."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.loader: FakeLoader | None = None
        self.hwm: FakeHostWeightManager | None = None
        self.offloader: FakeOffloader | None = None

    def create(self, ctx: StartupContext) -> tuple[FakeLoader, FakeHostWeightManager, FakeOffloader]:
        self.events.append("factory.create")
        self.loader = FakeLoader(ctx, self.events)
        self.hwm = FakeHostWeightManager(ctx, self.events)
        self.offloader = FakeOffloader(ctx, self.events)
        return self.loader, self.hwm, self.offloader


def make_startup_context(
    *,
    rank: int = 0,
    world_size: int = 1,
    stage_id: int = 0,
    compile_enabled: bool = False,
    faults: object = None,
    pools: FakeMemoryPools | None = None,
) -> StartupContext:
    return StartupContext(
        od_config=SimpleNamespace(stage_id=stage_id, dist_timeout=5),  # type: ignore[arg-type]
        vllm_config=object(),  # type: ignore[arg-type]
        device=torch.device("cpu"),
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        load_format="default",
        custom_pipeline_name=None,
        memory_pools=pools if pools is not None else FakeMemoryPools(),
        policy=None,
        compile_enabled=compile_enabled,
        faults=faults,  # type: ignore[arg-type]
    )

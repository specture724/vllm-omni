# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Executable output, residency and teardown contracts for the J2 backends."""

import gc
import weakref
from typing import ClassVar

import pytest
import torch
from torch import nn

from tests.diffusion.offloader.helpers import patch_offload_runtime
from vllm_omni.diffusion.offloader import layerwise_backend, sequential_backend
from vllm_omni.diffusion.offloader.base import OffloadConfig, OffloadStrategy
from vllm_omni.diffusion.offloader.layerwise_backend import LayerWiseOffloadBackend
from vllm_omni.diffusion.offloader.offload_plan import OffloadPlan
from vllm_omni.diffusion.offloader.plan_resolver import resolve_offload_plan
from vllm_omni.diffusion.offloader.sequential_backend import ModelLevelOffloadBackend
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]


class _Block(nn.Linear):
    def __init__(self):
        super().__init__(4, 4)
        self.register_buffer("scale", torch.tensor(0.5))

    def forward(self, x):
        return super().forward(x).tanh() * self.scale


class _Stack(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_Block() for _ in range(3)])
        self.tail = nn.ModuleList([_Block() for _ in range(2)])
        self.bias = nn.Parameter(torch.ones(4))
        self.register_buffer("scale", torch.tensor(0.25))
        self.proj = nn.Linear(4, 4)

    def forward(self, x):
        for block in (*self.blocks, *self.tail):
            x = block(x)
        return self.proj(x) * self.scale + self.bias


class _Pipeline(nn.Module):
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder", "image_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    _resident_modules: ClassVar[list[str]] = ["resident"]
    _offload_plan = OffloadPlan(
        block_attrs={"transformer": ("blocks", "tail")},
        encoder_block_attrs={"text_encoder": ("blocks", "tail")},
    )

    def __init__(self):
        super().__init__()
        self.transformer = _Stack()
        self.text_encoder = _Stack()
        self.image_encoder = nn.Linear(4, 4)
        self.vae = nn.Linear(4, 4)
        self.resident = nn.Linear(4, 4)

    def forward(self, x):
        return self.resident(self.vae(self.transformer(self.text_encoder(self.image_encoder(x)))))


@pytest.fixture(params=[pytest.param("cpu", marks=pytest.mark.cpu), pytest.param("cuda", marks=pytest.mark.cuda)])
def execution_device(request, monkeypatch):
    if request.param == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for real transfer/residency coverage")
        return torch.device("cuda:0")
    patch_offload_runtime(monkeypatch, current_omni_platform, synchronize=True)
    monkeypatch.setattr(current_omni_platform, "get_free_memory", lambda: 0)
    return torch.device("cpu")


def _tensors(module):
    return (*module.parameters(), *module.buffers())


def _assert_device(module, device):
    assert all(tensor.device == device for tensor in _tensors(module))


def _assert_ring_residency(blocks, device):
    # The final block prefetches block zero for the next iteration. Each
    # encoder stack has its own ring; the DiT's containers form one ring.
    _assert_device(blocks[0], device)
    assert all(t.numel() > 0 for t in _tensors(blocks[0]))
    assert all(t.numel() == 0 for block in blocks[1:] for t in _tensors(block))


def _assert_no_offload_hooks(pipeline):
    for module in pipeline.modules():
        registry = getattr(module, "_hook_registry", None)
        if registry is not None:
            assert registry.get_hook("sequential_offload") is None
            assert registry.get_hook("layerwise_offload") is None
        assert not getattr(module, "_omni_layerwise_enabled", False)


@pytest.mark.parametrize("strategy", [OffloadStrategy.MODEL_LEVEL, OffloadStrategy.LAYER_WISE])
@pytest.mark.parametrize(
    "components", [None, frozenset({"dit"}), frozenset({"text_encoder"}), frozenset({"dit", "text_encoder"})]
)
@torch.inference_mode()
def test_output_residency_and_reenable(execution_device, strategy, components):
    torch.manual_seed(42)
    device = execution_device
    pipeline = _Pipeline().to(device)
    x = torch.randn(2, 4, device=device)
    expected = pipeline(x)
    if strategy is OffloadStrategy.LAYER_WISE:
        # Exercise placement of real CPU-loaded non-block state as well as
        # streaming. The module swap baseline starts device-resident.
        pipeline.to("cpu")
    original = {name: value.cpu().clone() for name, value in pipeline.state_dict().items()}
    parameter_ids = [id(parameter) for parameter in pipeline.parameters()]
    backend_type = ModelLevelOffloadBackend if strategy is OffloadStrategy.MODEL_LEVEL else LayerWiseOffloadBackend
    backend = backend_type(OffloadConfig(strategy=strategy, components=components, pin_cpu_memory=False), device)
    dit_selected = components is None or "dit" in components
    encoder_selected = components is None or "text_encoder" in components

    for _cycle in range(2):
        backend.enable(pipeline)
        assert backend.enabled
        for _iteration in range(2):
            image = pipeline.image_encoder(x)
            encoded = pipeline.text_encoder(image)
            if strategy is OffloadStrategy.MODEL_LEVEL:
                _assert_device(pipeline.transformer, torch.device("cpu") if dit_selected else device)
                _assert_device(pipeline.text_encoder, device)
            else:
                if encoder_selected:
                    for blocks in (pipeline.text_encoder.blocks, pipeline.text_encoder.tail):
                        _assert_ring_residency(blocks, device)
                else:
                    _assert_device(pipeline.text_encoder, device)
            denoised = pipeline.transformer(encoded)
            if strategy is OffloadStrategy.MODEL_LEVEL:
                _assert_device(pipeline.text_encoder, torch.device("cpu") if encoder_selected else device)
                _assert_device(pipeline.image_encoder, torch.device("cpu") if components is None else device)
                _assert_device(pipeline.transformer, device)
            elif dit_selected:
                _assert_ring_residency((*pipeline.transformer.blocks, *pipeline.transformer.tail), device)
            else:
                _assert_device(pipeline.transformer, device)
            _assert_device(pipeline.vae, device)
            _assert_device(pipeline.resident, device)
            actual = pipeline.resident(pipeline.vae(denoised))
            torch.testing.assert_close(actual, expected)

        hook_refs = [
            weakref.ref(hook)
            for module in pipeline.modules()
            if (registry := getattr(module, "_hook_registry", None)) is not None
            for name in ("sequential_offload", "layerwise_offload")
            if (hook := registry.get_hook(name)) is not None
        ]
        del hook  # Assignment expressions keep the last hook alive.
        backend.disable()
        gc.collect()
        assert all(ref() is None for ref in hook_refs)
        assert not backend.enabled
        _assert_no_offload_hooks(pipeline)
        assert [id(parameter) for parameter in pipeline.parameters()] == parameter_ids
        for name, value in pipeline.state_dict().items():
            torch.testing.assert_close(value.cpu(), original[name])
        backend.disable()  # Idempotent cleanup must leave usable storage.
        pipeline.to(device)
        torch.testing.assert_close(pipeline(x), expected)


@pytest.mark.parametrize("strategy", [OffloadStrategy.MODEL_LEVEL, OffloadStrategy.LAYER_WISE])
def test_invalid_later_component_does_not_mutate_pipeline(execution_device, strategy):
    pipeline = _Pipeline()
    # The first encoder is valid. Reject the second before any placement/hook.
    pipeline._offload_plan = OffloadPlan(
        block_attrs={"transformer": ("blocks", "tail")},
        encoder_block_attrs={"text_encoder": ("blocks", "tail")},
        encoder_component_types={"image_encoder": "unsupported"},
    )
    original = {name: value.clone() for name, value in pipeline.state_dict().items()}
    backend_type = ModelLevelOffloadBackend if strategy is OffloadStrategy.MODEL_LEVEL else LayerWiseOffloadBackend
    backend = backend_type(
        OffloadConfig(strategy=strategy, components=frozenset({"dit", "text_encoder"}), pin_cpu_memory=False),
        execution_device,
    )
    with pytest.raises(ValueError, match="unknown component"):
        backend.enable(pipeline)
    assert not backend.enabled
    _assert_no_offload_hooks(pipeline)
    _assert_device(pipeline, torch.device("cpu"))
    for name, value in pipeline.state_dict().items():
        torch.testing.assert_close(value, original[name])


@pytest.mark.parametrize("strategy", [OffloadStrategy.MODEL_LEVEL, OffloadStrategy.LAYER_WISE])
@torch.inference_mode()
def test_backends_use_resolved_selection_and_blocks(execution_device, strategy, monkeypatch):
    pipeline = _Pipeline().to(execution_device)
    x = torch.randn(2, 4, device=execution_device)
    expected = pipeline(x)
    config = OffloadConfig(strategy=strategy, components=frozenset({"text_encoder"}), pin_cpu_memory=False)
    resolved = resolve_offload_plan(pipeline, config)

    def forbidden_read(*args, **kwargs):
        pytest.fail("Backend reinterpreted topology after plan resolution")

    # Once resolved, neither declaration paths nor selector helpers are an
    # input to backend execution. Keep ordinary transport options available.
    monkeypatch.setattr(config, "offloads", forbidden_read)
    monkeypatch.setattr(config, "should_offload_encoder", forbidden_read)
    monkeypatch.setattr(pipeline, "_offload_plan", None)
    for module in (pipeline.transformer, pipeline.text_encoder):
        monkeypatch.setattr(module, "_layerwise_offload_blocks_attrs", ["missing"], raising=False)
    backend_module = sequential_backend if strategy is OffloadStrategy.MODEL_LEVEL else layerwise_backend
    monkeypatch.setattr(backend_module, "resolve_offload_plan", lambda *_: resolved)
    backend_type = ModelLevelOffloadBackend if strategy is OffloadStrategy.MODEL_LEVEL else LayerWiseOffloadBackend
    backend = backend_type(config, execution_device)
    try:
        backend.enable(pipeline)
        torch.testing.assert_close(pipeline(x), expected)
    finally:
        backend.disable()
    _assert_no_offload_hooks(pipeline)

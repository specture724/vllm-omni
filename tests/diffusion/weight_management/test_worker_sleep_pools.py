# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""L2 CUDA sleep-pool verification for unified weight stack (RFC 01-6)."""

from __future__ import annotations

import gc

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.data import CuMemTag, OmniDiffusionConfig
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker, _get_mem_allocator_instance

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    hardware_test(res={"cuda": "L4"}, num_cards=1),
]


def test_unified_sleep_pools_preserve_weights_address_and_release_on_shutdown():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for CuMem sleep pool test")

    worker = object.__new__(DiffusionWorker)
    worker.rank = 0
    worker.device = torch.device("cuda", 0)
    worker.od_config = OmniDiffusionConfig(diffusion_weight_stack="unified", enable_sleep_mode=True)
    worker._owns_sleep_pool = False
    worker._sleep_saved_buffers = {}
    worker._shutdown_complete = False
    worker.model_runner = None
    worker.lora_manager = None
    worker.weight_transfer_engine = None
    worker.profiler = None

    pools = worker._build_unified_memory_pools()
    assert CuMemTag.WEIGHT_SLOTS.value == "weight_slots"
    with pools.weights():
        resident = torch.arange(16, device=worker.device, dtype=torch.float32)
    with pools.slots():
        slot = torch.zeros(16, device=worker.device, dtype=torch.float32)

    addr_before = resident.data_ptr()
    worker.sleep(level=1)
    worker.wake_up(tags=[CuMemTag.WEIGHTS.value])
    assert resident.data_ptr() == addr_before
    assert torch.equal(resident.cpu(), torch.arange(16, dtype=torch.float32))

    del resident, slot
    gc.collect()
    worker.shutdown()
    assert _get_mem_allocator_instance().get_current_usage() == 0

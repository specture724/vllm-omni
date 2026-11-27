# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Coexistence site registration for D04 (RFC 01-6)."""

from __future__ import annotations

ROW = "D04"
PR = "01-6"
SITES = (
    "vllm_omni/diffusion/worker/diffusion_worker.py::DiffusionWorker.load_model",
    "vllm_omni/diffusion/worker/diffusion_worker.py::DiffusionWorker.shutdown",
    "vllm_omni/diffusion/worker/diffusion_worker.py::DiffusionWorker.sleep",
    "vllm_omni/diffusion/worker/diffusion_worker.py::DiffusionWorker.wake_up",
)
NEW_ROW = None

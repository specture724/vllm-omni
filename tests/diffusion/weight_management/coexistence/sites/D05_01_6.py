# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Coexistence site registration for D05 (RFC 01-6)."""

from __future__ import annotations

ROW = "D05"
PR = "01-6"
SITES = ("vllm_omni/diffusion/worker/diffusion_worker.py::CustomPipelineWorkerExtension.re_init_pipeline",)
NEW_ROW = None

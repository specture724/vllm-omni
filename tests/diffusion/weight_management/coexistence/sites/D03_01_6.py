# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Coexistence site registration for D03 (RFC 01-6)."""

from __future__ import annotations

ROW = "D03"
PR = "01-6"
SITES = ("vllm_omni/diffusion/worker/diffusion_model_runner.py::DiffusionModelRunner.load_model",)
NEW_ROW = None

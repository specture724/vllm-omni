# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Coexistence site registration for D02 (RFC 01-3)."""

from __future__ import annotations

ROW = "D02"
PR = "01-3"
SITES = (
    "vllm_omni/diffusion/offloader/config.py::resolve_offload",
    "vllm_omni/diffusion/offloader/config.py::materialize_legacy_offload_flags",
)
NEW_ROW = None

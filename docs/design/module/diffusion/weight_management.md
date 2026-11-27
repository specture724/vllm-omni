---
title: Diffusion Weight Management
kind: module
status: draft
owners:
  - "@specture724"
  - "@david6666666"
  - "@yuanheng-zhao"
primary_code_paths:
  - vllm_omni/diffusion/weight_management/**
related_code_paths:
  - vllm_omni/diffusion/worker/diffusion_worker.py
  - vllm_omni/diffusion/worker/diffusion_model_runner.py
  - vllm_omni/diffusion/offloader/config.py
depends_on:
  - diffusion_runtime.md
  - offloader.md
validation_paths:
  - tests/diffusion/weight_management/**
upstream_refs:
  - torch.distributed
last_reviewed: 2026-11-27
---

# Diffusion weight management

The `vllm_omni/diffusion/weight_management/` package owns the unified startup
transaction, replica-wide coordination, implementation selector, error hierarchy,
and shared interfaces for the unified Loader, Host Weight Manager, and Offloader
(RFC #9536 / RFC #9874).

## Candidate invariants

### WM-INV-001: Explicit ownership chain

**Rule:** Host storage and runtime resources have a single owner at every
startup phase (`Election -> HostStorage -> LoadTransaction -> OffloadRuntime`)
and are released locally without collectives on failure or shutdown.

### WM-INV-002: Coordinated replica failure

**Rule:** Any rank failing before a confirmation point reports its error summary
at the next round so all replica ranks fail together within bounded timeout.

### WM-INV-003: Coexistence isolation

**Rule:** Legacy mode remains unchanged until release gates pass; new unified
branches must be registered under
`tests/diffusion/weight_management/coexistence/sites/` without expanding frozen
allowlists.

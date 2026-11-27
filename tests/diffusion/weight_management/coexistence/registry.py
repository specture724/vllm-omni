# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Frozen coexistence registry and allowlists for unified weight management (RFC 01-7)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CoexistenceRow:
    row_id: str
    title: str
    carrier: str
    legacy_removal: str


ROWS: tuple[CoexistenceRow, ...] = (
    CoexistenceRow(
        "C01", "MAGI-2 offload checks, staged components, memory window", "04", "Remove legacy offload branches"
    ),
    CoexistenceRow(
        "C02", "MAGI-2 MoE rebinding W_gate/W_up and merged W13 structure", "02", "Remove legacy MoE rebinding"
    ),
    CoexistenceRow(
        "C03", "MiniMax H3 SM120 AdaLN defaults and upscaler transfers", "04", "Remove legacy upscaler transfers"
    ),
    CoexistenceRow("C04", "MiniMax H3 AdaLN sidecar nonpersistent buffers", "02", "Remove legacy sidecar loader"),
    CoexistenceRow(
        "C05", "MiniMax H3 video VAE device lookup and float32 cast", "02", "Remove legacy VAE constructor cast"
    ),
    CoexistenceRow(
        "C06", "MiniMax H3 multiple-output CPU staging", "04", "Remove legacy _offload_model_cpu_stage_output"
    ),
    CoexistenceRow("C07", "Exact-projection and AdaLN cache signatures", "04", "Remove legacy signature branches"),
    CoexistenceRow("C08", "LingBot-Video offload_vae_during_denoise", "04", "Remove legacy VAE offload branch"),
    CoexistenceRow("C09", "SANA-WM request-time _ensure_*", "02", "Remove legacy _ensure_* calls"),
    CoexistenceRow(
        "C10", "HunyuanImage3 scattered-module .to(device) in load_weights", "02", "Remove legacy .to(device) calls"
    ),
    CoexistenceRow("C11", "Kandinsky6 self.device", "02", "Remove legacy device lookup"),
    CoexistenceRow("C12", "Model CUDA graph gating via fully_resident", "04", "Remove legacy offload graph guards"),
    CoexistenceRow("C13", "Wan VAE tensor.data= restoration", "02", "Remove legacy tensor.data= restoration"),
    CoexistenceRow("C14", "Cosmos3 mixed-precision NVFP4 lookup tables", "04", "Remove legacy NVFP4 offload handling"),
    CoexistenceRow(
        "C15", "LTX2 vocoder .float() round trips on mps/xpu/cpu", "02", "Remove legacy vocoder round trips"
    ),
    CoexistenceRow(
        "C16", "Post-commit PEFT wrapping and distilled-LoRA fusion", "02", "Remove legacy LoRA fusion/wrapping"
    ),
    CoexistenceRow("C17", "Runner gradient mode and execution scope", "04", "Remove legacy runner no_grad branch"),
    CoexistenceRow(
        "C18", "DiffusersAdapterPipeline _raise_unsupported_features", "02", "Remove legacy offload_config refusal"
    ),
    CoexistenceRow(
        "C19", "DreamZero extension VAE decode RPC execution scope", "04", "Remove legacy extension wrapper branch"
    ),
    CoexistenceRow(
        "D01",
        "OmniDiffusionConfig.__post_init__ checks and skipped legacy materialization",
        "01",
        "Remove legacy materialize_legacy_offload_flags branch in OmniDiffusionConfig.__post_init__",
    ),
    CoexistenceRow(
        "D02",
        "resolve_offload and materialize_legacy_offload_flags unified branches",
        "01",
        "Delete legacy offloader/config.py with legacy backends",
    ),
    CoexistenceRow(
        "D03",
        "DiffusionModelRunner.load_model unified dispatch",
        "01",
        "Remove legacy DiffusersPipelineLoader/enable_offload_backend branch in DiffusionModelRunner.load_model",
    ),
    CoexistenceRow(
        "D04",
        "DiffusionWorker lifecycle unified dispatch (load/shutdown/sleep/wake)",
        "01",
        "Remove legacy memory pool and offload_backend shutdown branches in DiffusionWorker",
    ),
    CoexistenceRow(
        "D05",
        "CustomPipelineWorkerExtension.re_init_pipeline unified cleanup",
        "01",
        "Remove legacy del model_runner.pipeline branch in CustomPipelineWorkerExtension.re_init_pipeline",
    ),
)

OFFLOADER_IMPORT_ALLOWLIST: frozenset[str] = frozenset(
    {
        "models/abot_world/pipeline_abot_world.py",
        "models/anima/pipeline_anima.py",
        "models/bagel/pipeline_bagel.py",
        "models/cosmos3/pipeline_cosmos3.py",
        "models/cosmos3/transformer_cosmos3.py",
        "models/diffusers_adapter/pipeline_diffusers_adapter.py",
        "models/dreamzero/pipeline_dreamzero.py",
        "models/flux2/pipeline_flux2.py",
        "models/flux2_klein/flux2_klein_transformer.py",
        "models/hy_nomas/pipeline_hy_nomas.py",
        "models/kandinsky6/pipeline_kandinsky6.py",
        "models/kandinsky6/super_resolution.py",
        "models/kandinsky6_sr/pipeline_kandinsky6_sr.py",
        "models/lance/pipeline_lance.py",
        "models/lingbot_video/pipeline_lingbot_video.py",
        "models/lingbot_world/pipeline.py",
        "models/ltx2/ltx2_components.py",
        "models/magi2/pipeline_magi2.py",
        "models/minimax_h3/encoder.py",
        "models/minimax_h3/fasth3.py",
        "models/minimax_h3/minimax_h3_blocks.py",
        "models/minimax_h3/minimax_h3_transformer.py",
        "models/minimax_h3/pipeline_minimax_h3.py",
        "models/minimax_h3/pipeline_minimax_h3_image.py",
        "models/minimax_h3/regenerate.py",
        "models/minimax_h3/vae.py",
        "models/minimax_h3/vdnh3.py",
        "models/mossimage/pipeline_mossimage.py",
        "models/pan2/pipeline_pan2.py",
        "models/praxis1/pipeline_praxis1.py",
        "models/qwen_image/pipeline_qwen_image.py",
        "models/qwen_image_21/decode_graph.py",
        "models/qwen_image_21/npu_attention.py",
        "models/qwen_image_21/pipeline_qwen_image_21.py",
        "models/qwen_image_21/qwen_image_21_transformer.py",
        "models/sana_video/pipeline_sana_video.py",
        "models/sana_wm/pipeline_sana_wm.py",
        "models/seedvr2/pipeline_seedvr2.py",
        "models/wan2_2/pipeline_wan2_2.py",
        "models/wan2_2/pipeline_wan2_2_s2v.py",
    }
)

HOOK_NAME_ALLOWLIST: frozenset[str] = frozenset({"cache/seacache/hook.py"})

CONFIG_FIELD_ALLOWLIST: frozenset[str] = frozenset(
    {
        "models/diffusers_adapter/pipeline_diffusers_adapter.py",
        "models/magi2/pipeline_magi2.py",
        "models/minimax_h3/pipeline_minimax_h3.py",
    }
)

QUERY_CALLER_ALLOWLIST: frozenset[str] = frozenset(
    {
        "vllm_omni/diffusion/data.py::OmniDiffusionConfig.__post_init__",
        "vllm_omni/diffusion/graph.py::_refusal",
        "vllm_omni/diffusion/model_loader/diffusers_loader.py::DiffusersPipelineLoader._load_model_with_shared_backbone",
        "vllm_omni/diffusion/model_loader/diffusers_loader.py::DiffusersPipelineLoader.load_model",
        "vllm_omni/diffusion/model_loader/weight_reload.py::prepare_model_for_weight_reload",
        "vllm_omni/diffusion/weight_management/startup.py::execution_offload_facts",
        "vllm_omni/diffusion/worker/diffusion_model_runner.py::DiffusionModelRunner._execute_request_list",
        "vllm_omni/diffusion/worker/diffusion_model_runner.py::DiffusionModelRunner.load_model",
        "vllm_omni/engine/weight_sharing_runtime.py::StageSharingHost._validate",
    }
)

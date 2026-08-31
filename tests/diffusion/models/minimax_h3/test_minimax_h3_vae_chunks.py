# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path
from types import MethodType
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from vllm_omni.diffusion.models.minimax_h3.vae import (
    MiniMaxH3ChunkCallbackPeerError,
    MiniMaxH3ChunkedDecodeUnsupportedError,
    MiniMaxH3VideoVAE,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _IdentityProcessor:
    @staticmethod
    def revert_tensor(tensor: torch.Tensor) -> torch.Tensor:
        return tensor


class _FakeVendorTemporalModel(nn.Module):
    temporal_chunk_callback_api_version = 1
    vae_ratio = 1

    def __init__(
        self,
        *,
        collective: bool = False,
        metadata_mode: str = "valid",
    ) -> None:
        super().__init__()
        values = torch.linspace(0, 1, 124).view(1, 1, 124, 1, 1)
        self.expected = values.expand(1, 3, -1, 1, 1).contiguous()
        self.processor = _IdentityProcessor()
        self.parallel_tiling = collective
        self.collective = collective
        self.metadata_mode = metadata_mode
        self.decode_base_calls = 0
        self.vendor_callback_calls = 0

    @staticmethod
    def split_tiles(size: int, tiled: bool):
        del size, tiled
        return [0, 1], None, None

    def decode_base(
        self,
        latent: torch.Tensor,
        *,
        temporal_chunk_callback=None,
    ) -> torch.Tensor:
        del latent
        self.decode_base_calls += 1
        output = self.expected.clone()
        if temporal_chunk_callback is None:
            return output

        frame_start = 0
        for chunk_index, chunk_size in enumerate([17] * 7 + [5]):
            if self.collective:
                collective_probe = torch.tensor([chunk_index], dtype=torch.int32)
                dist.all_reduce(collective_probe)
            self.vendor_callback_calls += 1
            metadata_index = chunk_index
            metadata_total = 8
            metadata_start = frame_start
            is_final = chunk_index == 7
            if self.metadata_mode == "gap" and chunk_index == 1:
                metadata_start += 1
            elif self.metadata_mode == "bad_total" and chunk_index == 1:
                metadata_total += 1
            elif self.metadata_mode == "no_final":
                is_final = False
            temporal_chunk_callback(
                self.expected[:, :, frame_start : frame_start + chunk_size].clone(),
                chunk_index=metadata_index,
                total_chunks=metadata_total,
                frame_start=metadata_start,
                is_final=is_final,
            )
            frame_start += chunk_size
        return output


def _vae(
    *,
    parallel_size: int = 1,
    collective: bool = False,
    metadata_mode: str = "valid",
) -> tuple[MiniMaxH3VideoVAE, _FakeVendorTemporalModel]:
    vae = object.__new__(MiniMaxH3VideoVAE)
    nn.Module.__init__(vae)
    model = _FakeVendorTemporalModel(
        collective=collective,
        metadata_mode=metadata_mode,
    )
    vae.model = model
    vae.config_dict = {
        "latent_channels": 3,
        "latents_mean": [0.0, 0.0, 0.0],
        "latents_std": [1.0, 1.0, 1.0],
    }
    vae.parallel_size = parallel_size
    return vae, model


def _latent() -> torch.Tensor:
    return torch.zeros(1, 3, 37, 1, 1)


def test_default_decode_path_does_not_request_vendor_chunks() -> None:
    vae, model = _vae()

    output = vae.decode_latent(_latent())

    torch.testing.assert_close(output, model.expected, rtol=0, atol=0)
    assert model.decode_base_calls == 1
    assert model.vendor_callback_calls == 0


def test_callback_chunks_are_owned_and_reconstruct_the_full_decode() -> None:
    vae, model = _vae()
    chunks: list[torch.Tensor] = []
    snapshots: list[torch.Tensor] = []
    metadata: list[tuple[int, int, int, bool]] = []

    def consume(
        frames: torch.Tensor,
        *,
        chunk_index: int,
        total_chunks: int,
        frame_start: int,
        is_final: bool,
    ) -> None:
        assert frames.dtype is torch.float32
        assert frames.is_contiguous()
        assert torch.all((0 <= frames) & (frames <= 1))
        chunks.append(frames)
        snapshots.append(frames.clone())
        metadata.append((chunk_index, total_chunks, frame_start, is_final))

    output = vae.decode_latent_with_chunks(_latent(), consume)

    torch.testing.assert_close(output, model.expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.cat(chunks, dim=2), output, rtol=0, atol=0)
    for chunk, snapshot in zip(chunks, snapshots, strict=True):
        torch.testing.assert_close(chunk, snapshot, rtol=0, atol=0)
    assert len({chunk.untyped_storage().data_ptr() for chunk in chunks}) == len(chunks)
    assert metadata == [(index, 8, index * 17, index == 7) for index in range(8)]


def test_callback_mutation_does_not_change_returned_full_output() -> None:
    vae, model = _vae()

    def mutate(frames: torch.Tensor, **metadata: Any) -> None:
        del metadata
        frames.fill_(1234)

    output = vae.decode_latent_with_chunks(_latent(), mutate)

    torch.testing.assert_close(output, model.expected, rtol=0, atol=0)


def test_callback_failure_is_deferred_until_vendor_decode_finishes() -> None:
    vae, model = _vae()
    callback_calls = 0

    def fail(*args, **kwargs) -> None:
        nonlocal callback_calls
        del args, kwargs
        callback_calls += 1
        raise LookupError("sink failed")

    with pytest.raises(LookupError, match="sink failed"):
        vae.decode_latent_with_chunks(_latent(), fail)

    assert callback_calls == 1
    assert model.vendor_callback_calls == 8
    torch.testing.assert_close(vae.decode_latent(_latent()), model.expected, rtol=0, atol=0)


def test_terminal_callback_failure_preserves_the_original_exception() -> None:
    vae, model = _vae()
    seen: list[tuple[int, int, int, bool]] = []

    def fail_final(
        frames: torch.Tensor,
        *,
        chunk_index: int,
        total_chunks: int,
        frame_start: int,
        is_final: bool,
    ) -> None:
        del frames
        seen.append((chunk_index, total_chunks, frame_start, is_final))
        if is_final:
            raise LookupError("final sink failed")

    with pytest.raises(LookupError, match="final sink failed"):
        vae.decode_latent_with_chunks(_latent(), fail_final)

    assert seen == [(index, 8, index * 17, index == 7) for index in range(8)]
    assert model.vendor_callback_calls == 8


@pytest.mark.parametrize(
    ("metadata_mode", "message"),
    [
        ("gap", "discontinuous chunk metadata"),
        ("bad_total", "discontinuous chunk metadata"),
        ("no_final", "discontinuous chunk metadata"),
    ],
)
def test_adapter_rejects_a_broken_vendor_callback_contract(
    metadata_mode: str,
    message: str,
) -> None:
    vae, model = _vae(metadata_mode=metadata_mode)

    with pytest.raises(RuntimeError, match=message):
        vae.decode_latent_with_chunks(_latent(), lambda *args, **kwargs: None)

    assert model.vendor_callback_calls == 8


def test_callback_request_fails_closed_without_the_vendor_api() -> None:
    vae, model = _vae()
    model.temporal_chunk_callback_api_version = -1

    with pytest.raises(MiniMaxH3ChunkedDecodeUnsupportedError, match="official MiniMax"):
        vae.decode_latent_with_chunks(_latent(), lambda *args, **kwargs: None)

    torch.testing.assert_close(vae.decode_latent(_latent()), model.expected, rtol=0, atol=0)


def _distributed_worker(
    rank: int,
    init_file: str,
    result_queue: Any,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    try:
        vae, model = _vae(parallel_size=2, collective=True)
        vae._native_parallel_state = MethodType(
            lambda self: {"sp_process_group": dist.group.WORLD},
            vae,
        )

        try:
            vae.decode_latent_with_chunks(
                _latent(),
                lambda *args, **kwargs: None,
            )
        except BaseException as exc:  # noqa: BLE001
            invalid_owner_type = type(exc).__name__
        else:
            invalid_owner_type = "none"

        def fail(*args, **kwargs) -> None:
            del args, kwargs
            raise LookupError("distributed sink failed")

        try:
            vae.decode_latent_with_chunks(
                _latent(),
                fail if rank == 0 else None,
            )
        except BaseException as exc:  # noqa: BLE001
            failure_type = type(exc).__name__
        else:
            failure_type = "none"

        success_calls = 0

        def succeed(*args, **kwargs) -> None:
            nonlocal success_calls
            del args, kwargs
            success_calls += 1

        output = vae.decode_latent_with_chunks(
            _latent(),
            succeed if rank == 0 else None,
        )

        local_vae, local_model = _vae(parallel_size=1)
        local_calls = 0

        def consume_local(*args, **kwargs) -> None:
            nonlocal local_calls
            del args, kwargs
            local_calls += 1

        local_output = local_vae.decode_latent_with_chunks(
            _latent(),
            consume_local if rank == 0 else None,
        )

        recovery = torch.tensor([rank + 1], dtype=torch.int32)
        dist.all_reduce(recovery)
        result_queue.put(
            {
                "rank": rank,
                "invalid_owner_type": invalid_owner_type,
                "failure_type": failure_type,
                "vendor_calls": model.vendor_callback_calls,
                "success_calls": success_calls,
                "output_matches": torch.equal(output, model.expected),
                "local_calls": local_calls,
                "local_output_matches": torch.equal(local_output, local_model.expected),
                "recovery": int(recovery.item()),
            }
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parallel
def test_two_rank_callback_failure_is_symmetric_and_group_recovers(
    tmp_path: Path,
) -> None:
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    init_file = str(tmp_path / "gloo-init")
    processes = [
        context.Process(
            target=_distributed_worker,
            args=(rank, init_file, result_queue),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    deadline = time.monotonic() + 90
    for process in processes:
        process.join(timeout=max(0, deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            process.join(timeout=1)
    hung = [process for process in processes if process.is_alive()]
    for process in hung:
        process.terminate()
        process.join(timeout=5)
    assert not hung, "two-rank chunk callback test deadlocked"
    assert [process.exitcode for process in processes] == [0, 0]

    results = sorted(
        [result_queue.get(timeout=5) for _ in range(2)],
        key=lambda result: result["rank"],
    )
    assert [result["failure_type"] for result in results] == [
        "LookupError",
        MiniMaxH3ChunkCallbackPeerError.__name__,
    ]
    assert [result["invalid_owner_type"] for result in results] == [
        "ValueError",
        "ValueError",
    ]
    assert [result["vendor_calls"] for result in results] == [16, 16]
    assert [result["success_calls"] for result in results] == [8, 0]
    assert all(result["output_matches"] for result in results)
    # With VAE parallelism disabled, the caller still designates one request owner.
    assert [result["local_calls"] for result in results] == [8, 0]
    assert all(result["local_output_matches"] for result in results)
    assert [result["recovery"] for result in results] == [3, 3]

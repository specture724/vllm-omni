# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import gc
import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from vllm_omni.diffusion.models.minimax_h3.chunked_decode import decode_h3_chunks
from vllm_omni.diffusion.models.minimax_h3.temporal_chunks import decode_temporal_chunks

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture(scope="module")
def cpu_process_group():
    if dist.is_initialized():
        yield dist.group.WORLD
        return

    with tempfile.NamedTemporaryFile(prefix="h3_chunk_dist_") as rendezvous:
        init_method = f"file://{rendezvous.name}"
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=init_method)
    try:
        yield dist.group.WORLD
    finally:
        dist.destroy_process_group()
        gc.collect()


class _FakeTemporalModel:
    use_3d_conv = True
    token_drop = 3
    tokens_chunk_size = 5
    token_overlap = 2
    vae_ratio_t = 4
    frame_pre_padding = 3
    frame_overlap = 5
    isolated_first_frame = False
    isolated_last_frame = False

    def _decode_temporal_output_frame_plan(self, z, z_head, z_tail, num_chunks, pad_tokens):
        del z, z_head, z_tail, num_chunks, pad_tokens
        return 35, 0, 35

    def _adaptive_decode(self, clip):
        value = float(clip[:, :, 0].mean())
        return torch.full((1, 1, 24, 2, 2), value)

    @staticmethod
    def blend(overlap, part, frame_overlap, dim):
        del overlap, frame_overlap, dim
        return part


class _FakeHost:
    def __init__(self):
        self.model = _FakeTemporalModel()

    @staticmethod
    def _denormalize_latent(latent):
        return latent

    @staticmethod
    def _normalize_decoded_frames(frames):
        return frames.float()


def test_temporal_chunks_emit_ordered_frames_and_collect_when_unconsumed():
    model = _FakeTemporalModel()
    latent = torch.arange(8, dtype=torch.float32).view(1, 1, 8, 1, 1)
    chunks = []
    marker = decode_temporal_chunks(model, latent, chunks.append)

    assert marker.shape == (0,)
    assert [chunk.shape[2] for chunk in chunks] == [17, 17, 1]
    assert torch.equal(torch.cat(chunks, dim=2), decode_temporal_chunks(model, latent, None))


def test_h3_callback_failure_is_deferred_until_temporal_decode_finishes():
    host = _FakeHost()
    latent = torch.zeros(1, 1, 8, 1, 1)
    seen = []

    def fail_once(frames):
        seen.append(frames.shape[2])
        raise RuntimeError("sink failed")

    with pytest.raises(RuntimeError, match="sink failed"):
        decode_h3_chunks(host, latent, fail_once, group=None)
    assert seen == [17]


def test_h3_without_callback_is_a_plain_full_decode_on_a_vae_group(cpu_process_group):
    host = _FakeHost()
    latent = torch.arange(8, dtype=torch.float32).view(1, 1, 8, 1, 1)

    full = decode_h3_chunks(host, latent, None, group=cpu_process_group)

    chunks = []
    decode_h3_chunks(host, latent, chunks.append, group=cpu_process_group)
    assert torch.equal(torch.cat(chunks, dim=2), full)


class _SimulatedGroup:
    """Stand in for a VAE group so one process can check every rank's verdict.

    ``all_reduce`` is replaced with the sum the real collective would produce
    for a group of ``world_size`` ranks whose callback sits on ``owner_rank``,
    which is what makes a rank-local verdict observable from a single process.
    """

    def __init__(self, world_size: int, owner_rank: int | None):
        self.world_size = world_size
        self.owner_rank = owner_rank

    def census(self) -> list[int]:
        if self.owner_rank is None:
            return [0, 0]
        return [1, self.owner_rank]


def _patch_simulated_ranks(monkeypatch, group: _SimulatedGroup, rank: int) -> None:
    from vllm_omni.diffusion.models.minimax_h3 import chunked_decode as mod

    def fake_all_reduce(tensor, group=None):
        del group
        tensor.copy_(torch.tensor(group_state.census(), dtype=tensor.dtype))

    group_state = group
    monkeypatch.setattr(mod.dist, "get_rank", lambda _group: rank)
    monkeypatch.setattr(mod.dist, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(mod.dist, "broadcast", lambda tensor, src=0, group=None: None)
    monkeypatch.setattr(mod.dist, "get_global_rank", lambda _group, _rank: 0)


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_h3_every_rank_rejects_a_callback_off_rank_zero(monkeypatch, rank):
    """A rank-local verdict let a peer decode alone while the others raised."""
    host = _FakeHost()
    latent = torch.zeros(1, 1, 8, 1, 1)
    group = _SimulatedGroup(world_size=3, owner_rank=1)
    _patch_simulated_ranks(monkeypatch, group, rank)

    # Only the owning rank is handed a callback; every rank must still refuse.
    callback = (lambda _frames: None) if rank == group.owner_rank else None
    with pytest.raises(ValueError, match="exactly one rank"):
        decode_h3_chunks(host, latent, callback, group=object())


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_h3_every_rank_accepts_a_callback_on_rank_zero(monkeypatch, rank):
    host = _FakeHost()
    latent = torch.zeros(1, 1, 8, 1, 1)
    group = _SimulatedGroup(world_size=3, owner_rank=0)
    _patch_simulated_ranks(monkeypatch, group, rank)

    seen: list[int] = []
    callback = seen.append if rank == 0 else None
    decode_h3_chunks(host, latent, (lambda f: seen.append(int(f.shape[2]))) if callback else None, group=object())

    # Only rank 0 publishes; peers run the loop to stay in the collectives.
    assert (seen != []) is (rank == 0)


def _stub_video_vae(*, parallel_size: int, tile_count: int):
    """A MiniMaxH3VideoVAE with only the pieces the decode entry points read."""
    from contextlib import contextmanager

    from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE

    vae = object.__new__(MiniMaxH3VideoVAE)
    vae.parallel_size = parallel_size
    vae.model = SimpleNamespace(_adaptive_decode=lambda clip: clip)
    vae.entered_rank_local = False

    @contextmanager
    def rank_local():
        vae.entered_rank_local = True
        yield

    vae._decoder_tile_count = lambda latent: tile_count
    vae._rank_local_tiling = rank_local
    vae.is_distributed_enabled = lambda: False
    return vae


def test_chunked_decode_falls_back_when_tiles_are_fewer_than_ranks(monkeypatch):
    """Without the fallback, tileless ranks raise while their peers block."""
    from vllm_omni.diffusion.models.minimax_h3 import vae as vae_mod

    vae = _stub_video_vae(parallel_size=4, tile_count=1)
    monkeypatch.setattr(vae_mod, "decode_h3_chunks", lambda *args, **kwargs: torch.zeros(1))

    vae.decode_latent_with_chunks(torch.zeros(1, 1, 8, 1, 1), lambda _frames: None)

    assert vae.entered_rank_local, "chunked decode must share decode_latent's too-few-tiles fallback"


def test_chunked_decode_keeps_shared_tiling_when_every_rank_has_a_tile(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import vae as vae_mod

    vae = _stub_video_vae(parallel_size=4, tile_count=8)
    monkeypatch.setattr(vae_mod, "decode_h3_chunks", lambda *args, **kwargs: torch.zeros(1))

    vae.decode_latent_with_chunks(torch.zeros(1, 1, 8, 1, 1), lambda _frames: None)

    assert not vae.entered_rank_local

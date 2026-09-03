# SPDX-License-Identifier: Apache-2.0
"""Frames-only callback coordination for MiniMax-H3 VAE decoding."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.distributed as dist

from .temporal_chunks import decode_temporal_chunks


class MiniMaxH3ChunkedDecodeUnsupportedError(RuntimeError):
    """Raised when the loaded H3 VAE cannot provide temporal chunks."""


class MiniMaxH3ChunkCallbackPeerError(RuntimeError):
    """Raised on a peer rank when rank zero's callback fails."""


MiniMaxH3VideoChunkCallback = Callable[[torch.Tensor], None]


def decode_h3_chunks(
    host,
    latent: torch.Tensor,
    callback: MiniMaxH3VideoChunkCallback | None,
    *,
    group: dist.ProcessGroup | None,
) -> torch.Tensor:
    """Run the local temporal loop, synchronizing callback errors across ranks."""
    owner = callback is not None
    if group is not None:
        rank = dist.get_rank(group)
        owners = torch.tensor([int(owner)], dtype=torch.int32, device=latent.device)
        dist.all_reduce(owners, group=group)
        if int(owners.item()) != 1 or (rank == 0) != owner:
            raise ValueError("MiniMax-H3 chunk callback must be supplied only on VAE group rank 0")
        owner = rank == 0

    error: BaseException | None = None

    def publish(raw: torch.Tensor) -> None:
        nonlocal error
        if not owner or error is not None:
            return
        try:
            processor = getattr(host.model, "processor", None)
            decoded = raw if processor is None else processor.revert_tensor(raw)
            frames = host._normalize_decoded_frames(decoded).contiguous()
            assert callback is not None
            callback(frames)
        except BaseException as exc:  # noqa: BLE001
            exc.__traceback__ = None
            error = exc

    sink = publish if owner else (None if group is None else (lambda _frames: None))
    result = decode_temporal_chunks(
        host.model,
        host._denormalize_latent(latent),
        sink,
    )

    if group is not None:
        failed = torch.tensor([int(rank == 0 and error is not None)], dtype=torch.int32, device=latent.device)
        dist.broadcast(failed, src=dist.get_global_rank(group, 0), group=group)
        if int(failed.item()):
            if rank == 0:
                assert error is not None
                raise error
            raise MiniMaxH3ChunkCallbackPeerError("MiniMax-H3 video chunk callback failed on rank zero")
    elif error is not None:
        raise error

    if not result.numel():
        return result
    processor = getattr(host.model, "processor", None)
    decoded = result if processor is None else processor.revert_tensor(result)
    return host._normalize_decoded_frames(decoded)

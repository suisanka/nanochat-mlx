"""Recomputed linear cross entropy: O(tile*V), no retained full B*T*V logits."""

import math
import torch
from torch.nn import functional as F
from torch.autograd.function import once_differentiable


class _LinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, chunk_size):
        h, y = hidden.reshape(-1, hidden.shape[-1]), targets.reshape(-1)
        count = (y != -1).sum().clamp_min(1)
        loss = torch.zeros((), device=h.device, dtype=torch.float32)
        for start in range(0, h.shape[0], chunk_size):
            logits = F.linear(h[start : start + chunk_size], weight).float()
            loss += F.cross_entropy(
                logits, y[start : start + chunk_size], ignore_index=-1, reduction="sum"
            )
        ctx.save_for_backward(hidden, weight, targets, count)
        ctx.chunk_size = chunk_size
        return loss / count

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        hidden, weight, targets, count = ctx.saved_tensors
        h, y = hidden.reshape(-1, hidden.shape[-1]), targets.reshape(-1)
        dh = torch.empty_like(h)
        dw = torch.zeros_like(weight, dtype=torch.float32)
        for start in range(0, h.shape[0], ctx.chunk_size):
            ht, yt = (
                h[start : start + ctx.chunk_size],
                y[start : start + ctx.chunk_size],
            )
            dz = F.linear(ht, weight).float().softmax(-1)
            valid = yt != -1
            dz.scatter_add_(
                1,
                yt.clamp_min(0)[:, None],
                -torch.ones_like(yt[:, None], dtype=dz.dtype),
            )
            dz *= valid[:, None] * (grad_output / count)
            dz = dz.to(h.dtype)
            dh[start : start + ctx.chunk_size] = dz @ weight
            dw.add_((dz.T @ ht).float())
        return dh.reshape_as(hidden), dw.to(weight.dtype), None, None


@torch.compiler.disable
def linear_cross_entropy(hidden, weight, targets, chunk_size=64, fused=False):
    """Mean over non--1 labels; first derivatives only, including tied weights."""
    if chunk_size <= 0 or targets.shape != hidden.shape[:-1]:
        raise ValueError("Invalid loss tile or target shape")
    if fused:
        if not hidden.is_cuda or hidden.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("Fused cross entropy requires CUDA FP16/BF16")
        from fla.modules.fused_linear_cross_entropy import FusedLinearCrossEntropyLoss

        # FLA rounds the tile up to a power of two and caps the chunk count.
        # Fail explicitly rather than exceeding the requested memory bound.
        chunks = min(
            math.ceil(targets.numel() / chunk_size),
            math.ceil(weight.shape[0] / hidden.shape[-1]),
        )
        tile = 1 << (math.ceil(targets.numel() / chunks) - 1).bit_length()
        if tile > chunk_size:
            raise ValueError(f"FLA loss requires loss-chunk-size >= {tile}")
        return FusedLinearCrossEntropyLoss(
            ignore_index=-1, num_chunks=chunks, accumulate_grad_in_fp32=True
        )(hidden, targets.long(), weight)
    return _LinearCrossEntropy.apply(hidden, weight, targets.long(), chunk_size)

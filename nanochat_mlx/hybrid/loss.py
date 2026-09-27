"""Linear cross entropy with bounded logits and explicit recomputation VJP.

This MLX fused operation accepts hidden states and the tied embedding directly.
It never retains a [B,T,V] tensor: token tiles are evaluated/released one at a
time in forward and backward. This is an operation-level fusion, not a port
of FLA's CUDA kernel. First-order gradients are supported.
"""

from functools import lru_cache
import mlx.core as mx
import mlx.nn as nn


@lru_cache(maxsize=16)
def _linear_ce(chunk_size):
    @mx.custom_function
    def loss(hidden, weight, labels):
        h, y = hidden.reshape(-1, hidden.shape[-1]), labels.reshape(-1)
        count = mx.maximum(mx.sum(y != -1), 1)
        total = mx.array(0.0)
        for start in range(0, h.shape[0], chunk_size):
            targets = y[start : start + chunk_size]
            logits = (h[start : start + chunk_size] @ weight.T).astype(mx.float32)
            ce = nn.losses.cross_entropy(
                logits, mx.maximum(targets, 0), reduction="none"
            )
            total = total + mx.sum(mx.where(targets != -1, ce, 0))
            mx.eval(total)
        return total / count

    @loss.vjp
    def backward(primals, cotangent, output):
        hidden, weight, labels = primals
        h, y = hidden.reshape(-1, hidden.shape[-1]), labels.reshape(-1)
        count = mx.maximum(mx.sum(y != -1), 1)
        dw = mx.zeros(weight.shape, dtype=mx.float32)
        dh = []
        for start in range(0, h.shape[0], chunk_size):
            x, targets = h[start : start + chunk_size], y[start : start + chunk_size]
            logits = (x @ weight.T).astype(mx.float32)
            delta = mx.softmax(logits, axis=-1)
            delta = delta.at[mx.arange(targets.size), mx.maximum(targets, 0)].add(-1)
            delta = delta * mx.expand_dims((targets != -1) * cotangent / count, -1)
            dx = delta @ weight.astype(mx.float32)
            dw = dw + delta.T @ x.astype(mx.float32)
            mx.eval(dx, dw)
            dh.append(dx)
        return (
            mx.concatenate(dh).reshape(hidden.shape).astype(hidden.dtype),
            dw.astype(weight.dtype),
            mx.zeros_like(labels),
        )

    return loss


def linear_cross_entropy(hidden, weight, targets, chunk_size=64):
    if hidden.shape[:-1] != targets.shape:
        raise ValueError("targets must match hidden batch/sequence dimensions")
    if chunk_size <= 0:
        raise ValueError("loss chunk_size must be positive")
    return _linear_ce(chunk_size)(hidden, weight, targets)

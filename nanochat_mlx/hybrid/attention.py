"""Local-aware tiled GQA. No sequence-length squared mask for SWA."""

import math
import mlx.core as mx
import mlx.nn as nn
from .rope import RotaryEmbedding


def local_attention(
    q, k, v, query_offset=0, key_offset=0, window=1024, tile=128, diagnostics=None
):
    """[B,H,T,D] inputs; absolute positions survive cache eviction.

    Each tile only submits its local K/V slice to SDPA: O(T*(W+tile)),
    rather than a dense T*T SDPA with a cosmetic sliding mask.
    """
    outputs, entropies = [], []
    for start in range(0, q.shape[2], tile):
        end = min(start + tile, q.shape[2])
        q_first, q_end = query_offset + start, query_offset + end
        left = max(0, q_first - window + 1 - key_offset) if window is not None else 0
        right = min(k.shape[2], q_end - key_offset)
        kt, vt = k[:, :, left:right], v[:, :, left:right]
        queries = mx.arange(q_first, q_end)
        keys = mx.arange(key_offset + left, key_offset + right)
        allowed = keys[None, :] <= queries[:, None]
        if window is not None:
            allowed = allowed & (queries[:, None] - keys[None, :] < window)
        mask = mx.where(allowed, mx.array(0.0), mx.array(-mx.inf)).astype(q.dtype)
        qt = q[:, :, start:end]
        outputs.append(
            mx.fast.scaled_dot_product_attention(
                qt, kt, vt, scale=1 / math.sqrt(q.shape[-1]), mask=mask
            )
        )
        if diagnostics is not None:
            expanded = mx.repeat(kt, q.shape[1] // k.shape[1], axis=1)
            probs = mx.softmax(
                (qt.astype(mx.float32) @ expanded.astype(mx.float32).swapaxes(-1, -2))
                / math.sqrt(q.shape[-1])
                + mask,
                axis=-1,
            )
            entropies.append(-mx.sum(probs * mx.log(mx.maximum(probs, 1e-30)), axis=-1))
    if diagnostics is not None:
        diagnostics["attention_entropy"] = mx.concatenate(entropies, axis=-1)
    return mx.concatenate(outputs, axis=2)


class SlidingWindowAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head, self.n_kv_head, self.head_dim = (
            config.n_head,
            config.n_kv_head,
            config.head_dim,
        )
        self.window = config.window_size
        self.tile = config.attention_tile
        self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            config.n_embd, self.n_kv_head * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.n_embd, self.n_kv_head * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(self.n_head * self.head_dim, config.n_embd, bias=False)
        self.rope = RotaryEmbedding(config)

    def __call__(self, x, cache=None, offset=0, diagnostics=None):
        b, t, _ = x.shape
        q = (
            self.q_proj(x)
            .reshape(b, t, self.n_head, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(x)
            .reshape(b, t, self.n_kv_head, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(x)
            .reshape(b, t, self.n_kv_head, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        q, k = self.rope(q, offset), self.rope(k, offset)
        key_offset = offset
        if cache is not None and "keys" in cache:
            key_offset = offset - cache["keys"].shape[2]
            k = mx.concatenate([cache["keys"], k], axis=2)
            v = mx.concatenate([cache["values"], v], axis=2)
        y = local_attention(
            q, k, v, offset, key_offset, self.window, self.tile, diagnostics
        )
        # Evict AFTER all queries have attended, never before a multi-token prefill.
        if cache is not None:
            cache["keys"] = k if self.window is None else k[:, :, -self.window :]
            cache["values"] = v if self.window is None else v[:, :, -self.window :]
        return self.o_proj(y.transpose(0, 2, 1, 3).reshape(b, t, -1))

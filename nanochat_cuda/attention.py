"""Local GQA: FlashAttention-2 when requested, bounded tiled SDPA otherwise."""

from functools import lru_cache
import torch
from torch import nn
from torch.nn import functional as F
from .rope import RotaryEmbedding


@lru_cache(maxsize=1)
def flash_kernel():
    try:
        from flash_attn import flash_attn_func
    except ImportError as exc:
        raise RuntimeError(
            "Install FlashAttention-2 or select --attention-backend sdpa; see docs/cuda.md"
        ) from exc
    return flash_attn_func


def local_attention(
    q, k, v, query_offset=0, key_offset=0, window=1024, tile=128, backend="sdpa"
):
    """[B,H,T,D], using absolute positions; never allocates a full T*T mask."""
    if backend == "flash":
        if not q.is_cuda or q.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("FlashAttention requires CUDA FP16/BF16")
        if query_offset + q.shape[2] != key_offset + k.shape[2]:
            raise ValueError("FlashAttention local cache requires right-aligned Q/K")
        return flash_kernel()(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p=0.0,
            causal=True,
            window_size=(window - 1, 0),
        ).transpose(1, 2)
    if backend != "sdpa":
        raise ValueError(f"Unknown attention backend: {backend}")
    out = []
    for start in range(0, q.shape[2], tile):
        end = min(start + tile, q.shape[2])
        first, last = query_offset + start, query_offset + end
        left = max(0, first - window + 1 - key_offset)
        right = min(k.shape[2], last - key_offset)
        queries = torch.arange(first, last, device=q.device)
        keys = torch.arange(key_offset + left, key_offset + right, device=q.device)
        allowed = (keys[None, :] <= queries[:, None]) & (
            queries[:, None] - keys[None, :] < window
        )
        kt, vt = k[:, :, left:right], v[:, :, left:right]
        # Expand only the local tile, keeping CPU and CUDA math identical.
        groups = q.shape[1] // k.shape[1]
        kt, vt = (a.repeat_interleave(groups, dim=1) for a in (kt, vt))
        out.append(
            F.scaled_dot_product_attention(
                q[:, :, start:end], kt, vt, attn_mask=allowed, dropout_p=0.0
            )
        )
    return torch.cat(out, dim=2)


class SlidingWindowAttention(nn.Module):
    def __init__(self, config, backend="sdpa"):
        super().__init__()
        self.backend = backend
        self.n_head, self.n_kv_head, self.head_dim = (
            config.n_head,
            config.n_kv_head,
            config.head_dim,
        )
        self.window, self.tile = config.window_size, config.attention_tile
        self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            config.n_embd, self.n_kv_head * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.n_embd, self.n_kv_head * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(self.n_head * self.head_dim, config.n_embd, bias=False)
        self.rope = RotaryEmbedding(config)

    def forward(self, x, cache=None, offset=0):
        b, t, _ = x.shape
        q = self.q_proj(x).reshape(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, offset), self.rope(k, offset)
        key_offset = offset
        if cache is not None and "keys" in cache:
            key_offset -= cache["keys"].shape[2]
            k, v = (
                torch.cat((cache[name], a), 2)
                for name, a in (("keys", k), ("values", v))
            )
        y = local_attention(
            q, k, v, offset, key_offset, self.window, self.tile, self.backend
        )
        if cache is not None:
            # clone releases the storage of long prefills, not just its view.
            cache.update(
                keys=k[:, :, -self.window :].detach().clone(),
                values=v[:, :, -self.window :].detach().clone(),
            )
        return self.o_proj(y.transpose(1, 2).reshape(b, t, -1))

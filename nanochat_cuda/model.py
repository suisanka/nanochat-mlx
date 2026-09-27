"""Pre-norm GDN/GDN/SWA language model with tied embeddings and bounded caches."""

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .attention import SlidingWindowAttention
from .gdn import GatedDeltaNet
from .loss import linear_cross_entropy


class HybridCache:
    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.layers = [{} for _ in range(self.config.n_layer)]
        self.offset, self.batch_size = 0, None

    def repeat(self, count):
        if count < 1:
            raise ValueError("count must be positive")

        def repeat(value):
            if isinstance(value, torch.Tensor):
                return value.repeat_interleave(count, 0)
            if isinstance(value, tuple):
                return tuple(repeat(x) for x in value)
            return {k: repeat(v) for k, v in value.items()}

        result = HybridCache(self.config)
        result.layers = [repeat(layer) for layer in self.layers]
        result.offset = self.offset
        result.batch_size = None if self.batch_size is None else self.batch_size * count
        return result


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        d, f = config.n_embd, config.intermediate_size
        self.gate_proj, self.up_proj = (
            nn.Linear(d, f, bias=False),
            nn.Linear(d, f, bias=False),
        )
        self.down_proj = nn.Linear(f, d, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, config, kind, gdn_backend, attention_backend):
        super().__init__()
        self.kind = kind
        self.mixer_norm = nn.RMSNorm(config.n_embd, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.n_embd, eps=config.norm_eps)
        self.mixer = (
            GatedDeltaNet(config, gdn_backend)
            if kind == "gdn"
            else SlidingWindowAttention(config, attention_backend)
        )
        self.mlp = SwiGLU(config)

    def forward(self, x, cache=None, offset=0):
        normalized = self.mixer_norm(x)
        x = x + (
            self.mixer(normalized, cache)
            if self.kind == "gdn"
            else self.mixer(normalized, cache, offset)
        )
        return x + self.mlp(self.ffn_norm(x))


class HybridLM(nn.Module):
    def __init__(self, config, gdn_backend="fla", attention_backend="sdpa"):
        super().__init__()
        if gdn_backend not in ("fla", "reference") or attention_backend not in (
            "sdpa",
            "flash",
        ):
            raise ValueError("Invalid kernel backend")
        self.config = config
        self.gdn_backend, self.attention_backend = gdn_backend, attention_backend
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList(
            Block(config, kind, gdn_backend, attention_backend)
            for kind in config.layer_types
        )
        self.final_norm = nn.RMSNorm(config.n_embd, eps=config.norm_eps)
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name == "wte.weight" or "_proj.weight" in name:
                    sigma = (
                        0.02 / math.sqrt(2 * config.n_layer)
                        if name.endswith(("o_proj.weight", "down_proj.weight"))
                        else 0.02
                    )
                    nn.init.normal_(param, std=sigma)
                if not name.endswith(("A_log", "dt_bias")):
                    param.data = param.data.to(getattr(torch, config.dtype))
        self._compiled = False

    def compile_hotpaths(self, **kwargs):
        """Compile feed-forward blocks; FLA already uses Triton JIT kernels.

        Avoid tracing Python cache mutation, the tiled loss loop, and FLA's
        compiler-disabled wrapper into one unstable full training graph.
        """
        if not self._compiled:
            for block in self.blocks:
                block.mlp.forward = torch.compile(block.mlp.forward, **kwargs)
            self._compiled = True
        return self

    def make_cache(self):
        return HybridCache(self.config)

    def hidden(self, ids, cache=None):
        if ids.ndim != 2 or ids.shape[1] == 0:
            raise ValueError("ids must be non-empty [batch, sequence]")
        offset = 0 if cache is None else cache.offset
        if offset + ids.shape[1] > self.config.max_context:
            raise ValueError(
                f"Context exceeds configured maximum {self.config.max_context}"
            )
        if cache is not None:
            if torch.is_grad_enabled():
                raise ValueError("Caches require torch.no_grad() or inference_mode()")
            if cache.config != self.config or cache.batch_size not in (
                None,
                ids.shape[0],
            ):
                raise ValueError("Cache configuration/batch differs from model input")
        x = self.wte(ids)
        for i, block in enumerate(self.blocks):
            if (
                self.config.checkpoint_blocks
                and cache is None
                and torch.is_grad_enabled()
            ):
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x, None if cache is None else cache.layers[i], offset)
        if cache is not None:
            cache.offset += ids.shape[1]
            cache.batch_size = ids.shape[0]
        return self.final_norm(x)

    def forward(self, ids, targets=None, kv_cache=None, last_only=False):
        if targets is not None and kv_cache is not None:
            raise ValueError("Training samples must start with zero state (no cache)")
        h = self.hidden(ids, kv_cache)
        if targets is not None:
            return linear_cross_entropy(
                h, self.wte.weight, targets, self.config.loss_chunk_size
            )
        return F.linear(h[:, -1:] if last_only else h, self.wte.weight).float()

    @torch.no_grad()
    def prefill(self, ids, cache, chunk_size=256):
        if ids.ndim != 2 or ids.shape[1] == 0 or chunk_size <= 0:
            raise ValueError(
                "Prefill requires a non-empty batch and positive chunk size"
            )
        if cache.offset + ids.shape[1] > self.config.max_context:
            raise ValueError("Prefill exceeds configured maximum")
        for start in range(0, ids.shape[1], chunk_size):
            h = self.hidden(ids[:, start : start + chunk_size], cache)
        return F.linear(h[:, -1:], self.wte.weight).float()

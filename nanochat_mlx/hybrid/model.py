"""Pre-norm GDN/SWA backbone, tied embedding and bounded-vocabulary loss."""

import math
import mlx.core as mx
import mlx.nn as nn
from mlx.nn.utils import checkpoint
from mlx.utils import tree_flatten, tree_unflatten

from .attention import SlidingWindowAttention
from .gdn import GatedDeltaNet
from .loss import linear_cross_entropy


class HybridCache:
    def __init__(self, config):
        self.config = config
        self.layers = [{} for _ in range(config.n_layer)]
        self.offset = 0
        self.batch_size = None

    def reset(self):
        self.layers = [{} for _ in self.layers]
        self.offset = 0
        self.batch_size = None

    def arrays(self):
        return [array for _, array in tree_flatten(self.layers)]

    def repeat(self, count):
        if count < 1:
            raise ValueError("count must be positive")
        result = HybridCache(self.config)
        from mlx.utils import tree_map

        result.layers = tree_map(lambda x: mx.repeat(x, count, axis=0), self.layers)
        result.offset = self.offset
        result.batch_size = self.batch_size * count if self.batch_size else None
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

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, config, kind):
        super().__init__()
        self.kind = kind
        self.mixer_norm = nn.RMSNorm(config.n_embd, eps=config.norm_eps)
        self.ffn_norm = nn.RMSNorm(config.n_embd, eps=config.norm_eps)
        self.mixer = (
            GatedDeltaNet(config) if kind == "gdn" else SlidingWindowAttention(config)
        )
        self.mlp = SwiGLU(config)

    def __call__(self, x, cache=None, offset=0, diagnostics=None):
        normalized = self.mixer_norm(x)
        if self.kind == "gdn":
            x = x + self.mixer(normalized, cache, diagnostics)
        else:
            x = x + self.mixer(normalized, cache, offset, diagnostics)
        return x + self.mlp(self.ffn_norm(x))


class HybridLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = [Block(config, kind) for kind in config.layer_types]
        self.final_norm = nn.RMSNorm(config.n_embd, eps=config.norm_eps)
        self.init_weights()

    def init_weights(self):
        updated = []
        for path, value in tree_flatten(self.parameters()):
            if path == "wte.weight" or ("_proj.weight" in path):
                sigma = (
                    0.02 / math.sqrt(2 * self.config.n_layer)
                    if path.endswith(("o_proj.weight", "down_proj.weight"))
                    else 0.02
                )
                value = mx.random.normal(value.shape) * sigma
            # Preserve FLA A_log/dt_bias initialization and conv initialization.
            dtype = (
                mx.float32
                if path.endswith(("A_log", "dt_bias"))
                else getattr(mx, self.config.dtype)
            )
            updated.append((path, value.astype(dtype)))
        self.update(tree_unflatten(updated))

    def make_cache(self):
        return HybridCache(self.config)

    def hidden(self, ids, cache=None, diagnostics=None):
        if ids.ndim != 2 or ids.shape[1] == 0:
            raise ValueError("ids must be non-empty [batch, sequence]")
        offset = 0 if cache is None else cache.offset
        if offset + ids.shape[1] > self.config.max_context:
            raise ValueError(
                f"Context exceeds configured maximum {self.config.max_context}"
            )
        if cache is not None:
            if cache.config != self.config:
                raise ValueError("Cache configuration differs from model")
            if cache.batch_size is not None and cache.batch_size != ids.shape[0]:
                raise ValueError("Cache batch size changed; reset or repeat the cache")
        x = self.wte(ids)
        for i, block in enumerate(self.blocks):
            stats = None if diagnostics is None else {}
            if self.config.checkpoint_blocks and cache is None and diagnostics is None:
                x = checkpoint(block)(x)
            else:
                x = block(x, None if cache is None else cache.layers[i], offset, stats)
            if diagnostics is not None:
                # Reduce immediately so diagnostics do not retain all layers'
                # full activation tensors during long-sequence monitoring.
                summary = {}
                for name, value in stats.items():
                    a = value.astype(mx.float32)
                    item = {"min": mx.min(a), "mean": mx.mean(a), "max": mx.max(a)}
                    if name in ("alpha", "beta"):
                        item["histogram"] = mx.stack(
                            [
                                mx.mean(
                                    (
                                        (a >= j / 10)
                                        & (a < ((j + 1) / 10) if j < 9 else a <= 1)
                                    ).astype(mx.float32)
                                )
                                for j in range(10)
                            ]
                        )
                    summary[name] = item
                mx.eval(x, summary)
                diagnostics[str(i)] = summary
        if cache is not None:
            cache.offset += ids.shape[1]
            cache.batch_size = ids.shape[0]
        return self.final_norm(x)

    def __call__(
        self, ids, targets=None, kv_cache=None, last_only=False, diagnostics=None
    ):
        if targets is not None and kv_cache is not None:
            raise ValueError("Training samples must start with zero state (no cache)")
        h = self.hidden(ids, kv_cache, diagnostics)
        if targets is not None:
            return linear_cross_entropy(
                h, self.wte.weight, targets, self.config.loss_chunk_size
            )
        if last_only:
            h = h[:, -1:, :]
        return self.wte.as_linear(h).astype(mx.float32)

    def prefill(self, ids, cache, chunk_size=256):
        if ids.ndim != 2 or ids.shape[1] == 0:
            raise ValueError("Prefill requires a non-empty [batch, sequence] array")
        if chunk_size <= 0:
            raise ValueError("prefill chunk_size must be positive")
        if cache.offset + ids.shape[1] > self.config.max_context:
            raise ValueError(
                f"Context exceeds configured maximum {self.config.max_context}"
            )
        # Do not materialize vocabulary logits for earlier prefill chunks.
        for start in range(0, ids.shape[1], chunk_size):
            h = self.hidden(ids[:, start : start + chunk_size], cache)
            mx.eval(h, cache.arrays())
        return self.wte.as_linear(h[:, -1:]).astype(mx.float32)

    def num_scaling_params(self):
        return self.config.parameter_counts()

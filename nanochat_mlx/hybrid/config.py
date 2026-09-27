"""Serializable architecture and optimizer contracts (no MLX import required)."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class HybridConfig:
    architecture: str = "gdn_swa"
    n_layer: int = 12
    n_embd: int = 512
    intermediate_size: int = 1536
    vocab_size: int = 129280
    sequence_len: int = 4096
    max_context: int = 32768
    gdn_heads: int = 6
    key_head_dim: int = 64
    value_head_dim: int = 128
    conv_kernel: int = 4
    chunk_size: int = 64
    n_head: int = 4
    n_kv_head: int = 1
    head_dim: int = 128
    window_size: int = 1024
    attention_tile: int = 128
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    rope_scaling: str = "none"
    rope_factor: float = 1.0
    original_context: int = 4096
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    dtype: str = "bfloat16"
    loss_chunk_size: int = 64
    checkpoint_blocks: bool = False

    def __post_init__(self):
        if self.architecture != "gdn_swa":
            raise ValueError("Only the gdn_swa hybrid architecture is supported")
        for name in (
            "n_layer",
            "n_embd",
            "intermediate_size",
            "vocab_size",
            "sequence_len",
            "max_context",
            "gdn_heads",
            "key_head_dim",
            "value_head_dim",
            "conv_kernel",
            "n_head",
            "n_kv_head",
            "head_dim",
            "window_size",
            "attention_tile",
            "original_context",
            "loss_chunk_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.sequence_len > self.max_context or self.max_context > 32768:
            raise ValueError("Require sequence_len <= max_context <= 32768")
        if self.n_head % self.n_kv_head:
            raise ValueError("n_head must be divisible by n_kv_head")
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even head_dim")
        if self.chunk_size != 64:
            raise ValueError("The v0 GDN chunk size is fixed at 64")
        if self.rope_scaling not in ("none", "linear", "yarn"):
            raise ValueError("rope_scaling must be none, linear or yarn")
        if not math.isfinite(self.rope_factor) or self.rope_factor < 1:
            raise ValueError("rope_factor must be finite and >= 1")
        if self.rope_scaling == "none" and self.rope_factor != 1:
            raise ValueError("rope_factor requires linear or yarn scaling")
        if self.rope_theta <= 1 or not math.isfinite(self.rope_theta):
            raise ValueError("rope_theta must be finite and > 1")
        if not 0 < self.yarn_beta_slow < self.yarn_beta_fast:
            raise ValueError("Require 0 < yarn_beta_slow < yarn_beta_fast")
        if self.dtype not in ("float32", "bfloat16"):
            raise ValueError("dtype must be float32 or bfloat16")

    @property
    def layer_types(self):
        # Tile the specified pattern, including at depths not divisible by 3.
        return tuple(("gdn", "gdn", "swa")[i % 3] for i in range(self.n_layer))

    def to_dict(self):
        return asdict(self)

    def parameter_counts(self):
        d, h, k, v = self.n_embd, self.gdn_heads, self.key_head_dim, self.value_head_dim
        gdn = (
            d * h * (2 * k + 3 * v + 2) + h * (2 * k + v) * self.conv_kernel + 2 * h + v
        )
        attn = 2 * d * self.head_dim * (self.n_head + self.n_kv_head)
        counts = {
            "tied_embedding": self.vocab_size * d,
            "ffn": self.n_layer * 3 * d * self.intermediate_size,
            "gdn": self.layer_types.count("gdn") * gdn,
            "attention": sum(t != "gdn" for t in self.layer_types) * attn,
            "norm": (2 * self.n_layer + 1) * d,
        }
        return {**counts, "total": sum(counts.values())}


def config_for_depth(depth=12, **overrides):
    if depth <= 0:
        raise ValueError("depth must be positive")
    # d12 is exactly the frozen 110M design. Multiples of 256 preserve GDN's
    # 0.75*d Q/K allocation and 4:1 attention grouping at every supported depth.
    width = 256 * math.ceil(depth / 6)
    head_dim = 128 if width % 512 == 0 else 64
    values = dict(
        n_layer=depth,
        n_embd=width,
        intermediate_size=3 * width,
        gdn_heads=3 * width // 256,
        head_dim=head_dim,
        n_head=width // head_dim,
        n_kv_head=width // head_dim // 4,
    )
    return HybridConfig(**(values | overrides))


@dataclass(frozen=True)
class TrainingConfig:
    tokens_per_step: int = 131072
    muon_lr: float = 0.02
    adamw_lr: float = 3e-4
    momentum: float = 0.95
    ns_steps: int = 5
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-10
    warmup_ratio: float = 0.02
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    optimizer: str = "muon"

    def __post_init__(self):
        if self.optimizer not in ("muon", "adamw"):
            raise ValueError("optimizer must be muon or adamw")
        if self.tokens_per_step <= 0 or self.ns_steps <= 0 or self.grad_clip <= 0:
            raise ValueError("Training sizes and gradient clip must be positive")
        if not 0 <= self.warmup_ratio < 1 or not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("Invalid learning-rate schedule")
        if not all(0 <= x < 1 for x in (self.beta1, self.beta2, self.momentum)):
            raise ValueError("Invalid optimizer momentum/betas")
        if (
            self.muon_lr <= 0
            or self.adamw_lr <= 0
            or self.eps <= 0
            or self.weight_decay < 0
        ):
            raise ValueError("Invalid learning rate, epsilon or weight decay")

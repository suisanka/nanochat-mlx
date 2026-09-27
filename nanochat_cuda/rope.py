"""Fixed-frequency FP32 RoPE; positions remain absolute after KV eviction."""

import math
import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        dims = config.head_dim
        inverse = config.rope_theta ** (-torch.arange(0, dims, 2).float() / dims)
        self.scale = 1.0
        if config.rope_scaling == "linear":
            inverse /= config.rope_factor
        elif config.rope_scaling == "yarn":

            def correction(rotations):
                return (
                    dims
                    * math.log(config.original_context / (2 * math.pi * rotations))
                    / (2 * math.log(config.rope_theta))
                )

            low = max(math.floor(correction(config.yarn_beta_fast)), 0)
            high = min(math.ceil(correction(config.yarn_beta_slow)), dims - 1)
            ramp = (
                (torch.arange(dims // 2).float() - low) / max(high - low, 0.001)
            ).clamp(0, 1)
            inverse = inverse * (1 - ramp) + inverse / config.rope_factor * ramp
            self.scale = 1 + 0.1 * math.log(config.rope_factor)
        self.register_buffer("inverse", inverse, persistent=False)

    def forward(self, x, offset=0):
        positions = (
            torch.arange(x.shape[-2], device=x.device, dtype=torch.float32) + offset
        )
        angles = positions[:, None] * self.inverse.float()[None, :]
        cos, sin = angles.cos(), angles.sin()
        left, right = x.float().chunk(2, dim=-1)
        return (
            torch.cat((left * cos - right * sin, right * cos + left * sin), -1)
            * self.scale
        ).to(x.dtype)

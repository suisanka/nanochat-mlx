"""Fixed-frequency RoPE: cache-safe native, linear and YaRN context extension.

YaRN equations follow Apple's mlx-lm/models/rope_utils.py (MIT). Frequencies
never change mid-stream: cached keys and new queries share the same basis.
"""

import math
import mlx.core as mx
import mlx.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dims = config.head_dim
        self._scale = 1.0
        inverse = config.rope_theta ** (
            -mx.arange(0, self.dims, 2, dtype=mx.float32) / self.dims
        )
        if config.rope_scaling == "linear":
            inverse = inverse / config.rope_factor
        elif config.rope_scaling == "yarn":

            def correction(rotations):
                return (
                    self.dims
                    * math.log(config.original_context / (2 * math.pi * rotations))
                    / (2 * math.log(config.rope_theta))
                )

            low = max(math.floor(correction(config.yarn_beta_fast)), 0)
            high = min(math.ceil(correction(config.yarn_beta_slow)), self.dims - 1)
            ramp = mx.clip(
                (mx.arange(self.dims // 2, dtype=mx.float32) - low)
                / max(high - low, 0.001),
                0,
                1,
            )
            inverse = inverse * (1 - ramp) + (inverse / config.rope_factor) * ramp
            self._scale = 1 + 0.1 * math.log(config.rope_factor)
        self._inverse = inverse

    def __call__(self, x, offset=0):
        # Compute angles in FP32 even when model weights/activations are BF16.
        positions = mx.arange(x.shape[-2], dtype=mx.float32) + offset
        angles = positions[:, None] * self._inverse[None, :]
        cos, sin = mx.cos(angles), mx.sin(angles)
        left, right = mx.split(x.astype(mx.float32), 2, axis=-1)
        return (
            mx.concatenate(
                [left * cos - right * sin, right * cos + left * sin], axis=-1
            )
            * self._scale
        ).astype(x.dtype)

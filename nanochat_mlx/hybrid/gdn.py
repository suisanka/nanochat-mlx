"""Gated DeltaNet in MLX, following the FLA recurrence (MIT).

State is [B,H,K,V], with FP32 accumulation. The chunk algorithm solves the
strictly lower-triangular delta system with a finite matrix geometric series;
it parallelizes tokens within each 64-token chunk without a CUDA dependency.
"""

import math
import mlx.core as mx
import mlx.nn as nn


def l2norm(x):
    x = x.astype(mx.float32)
    return x * mx.rsqrt(mx.sum(x * x, axis=-1, keepdims=True) + 1e-6)


def naive_recurrent_gdn(q, k, v, g, beta, state=None):
    q, k, v, g, beta = (x.astype(mx.float32) for x in (q, k, v, g, beta))
    b, t, h, dk = q.shape
    if state is None:
        state = mx.zeros((b, h, dk, v.shape[-1]))
    out = []
    for i in range(t):
        state = state * mx.exp(g[:, i])[..., None, None]
        error = (v[:, i] - mx.sum(k[:, i, :, :, None] * state, axis=-2)) * beta[
            :, i, :, None
        ]
        state = state + k[:, i, :, :, None] * error[:, :, None, :]
        out.append(mx.sum(q[:, i, :, :, None] * state, axis=-2) / math.sqrt(dk))
    return mx.stack(out, axis=1), state


@mx.compile
def recurrent_decode(q, k, v, g, beta, state):
    """Compiled one-token MLX recurrence; same FP32 state contract as chunks."""
    return naive_recurrent_gdn(q, k, v, g, beta, state)


@mx.compile
def _chunk_step(q, k, v, g, beta, state):
    """Compile one fixed-size chunk, reusing its graph across the sequence."""
    cumulative = mx.cumsum(g, axis=-1)
    n, dk = q.shape[-2:]
    positions = mx.arange(n)
    causal = positions[:, None] >= positions[None, :]
    strict = positions[:, None] > positions[None, :]
    # Mask before exp: upper triangle can otherwise overflow for strong decay.
    decay = mx.exp(
        mx.where(causal, cumulative[..., :, None] - cumulative[..., None, :], -mx.inf)
    )
    lower = mx.where(strict, (k @ k.swapaxes(-1, -2)) * beta[..., :, None] * decay, 0)
    # (I+lower)^-1 = product_j (I+(-lower)^(2^j)); nilpotent in n steps.
    power = -lower
    inverse = mx.broadcast_to(mx.eye(n), lower.shape)
    for _ in range(math.ceil(math.log2(n)) if n > 1 else 0):
        inverse = inverse + power @ inverse
        power = power @ power
    rhs = beta[..., None] * (v - mx.exp(cumulative)[..., None] * (k @ state))
    updates = inverse @ rhs
    scores = (q @ k.swapaxes(-1, -2)) * decay
    out = (mx.exp(cumulative)[..., None] * (q @ state) + scores @ updates) / math.sqrt(
        dk
    )
    final_decay = mx.exp(cumulative[..., -1, None] - cumulative)
    state = (
        state * mx.exp(cumulative[..., -1, None, None])
        + (k * final_decay[..., None]).swapaxes(-1, -2) @ updates
    )
    return out, state


def chunk_gated_delta_rule(q, k, v, g, beta, state=None, chunk_size=64):
    q, k, v = (x.astype(mx.float32).transpose(0, 2, 1, 3) for x in (q, k, v))
    g, beta = (x.astype(mx.float32).transpose(0, 2, 1) for x in (g, beta))
    b, h, t, dk = q.shape
    if state is None:
        state = mx.zeros((b, h, dk, v.shape[-1]))
    out = []
    for start in range(0, t, chunk_size):
        end = min(start + chunk_size, t)
        result, state = _chunk_step(
            q[:, :, start:end],
            k[:, :, start:end],
            v[:, :, start:end],
            g[:, :, start:end],
            beta[:, :, start:end],
            state,
        )
        out.append(result)
    return mx.concatenate(out, axis=2).transpose(0, 2, 1, 3), state


class CausalDepthwiseConv(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = mx.random.uniform(
            -(kernel_size**-0.5), kernel_size**-0.5, (channels, kernel_size)
        )

    def __call__(self, x, state=None):
        if state is None:
            state = mx.zeros(
                (x.shape[0], self.kernel_size - 1, x.shape[-1]), dtype=x.dtype
            )
        history = mx.concatenate([state, x], axis=1)
        y = sum(
            history[:, i : i + x.shape[1]] * self.weight[:, i]
            for i in range(self.kernel_size)
        )
        return nn.silu(y), history[
            :, -(self.kernel_size - 1) :
        ] if self.kernel_size > 1 else history[:, :0]


class GatedDeltaNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.dk, self.dv = (
            config.gdn_heads,
            config.key_head_dim,
            config.value_head_dim,
        )
        self.chunk_size = config.chunk_size
        self.norm_eps = config.norm_eps
        d, h, k, v = config.n_embd, self.heads, self.dk, self.dv
        self.q_proj, self.k_proj = (
            nn.Linear(d, h * k, bias=False),
            nn.Linear(d, h * k, bias=False),
        )
        self.v_proj, self.g_proj = (
            nn.Linear(d, h * v, bias=False),
            nn.Linear(d, h * v, bias=False),
        )
        self.a_proj, self.b_proj = (
            nn.Linear(d, h, bias=False),
            nn.Linear(d, h, bias=False),
        )
        self.o_proj = nn.Linear(h * v, d, bias=False)
        self.q_conv = CausalDepthwiseConv(h * k, config.conv_kernel)
        self.k_conv = CausalDepthwiseConv(h * k, config.conv_kernel)
        self.v_conv = CausalDepthwiseConv(h * v, config.conv_kernel)
        self.o_norm = nn.RMSNorm(v, eps=config.norm_eps)
        # FLA initialization: A ~ U(0,16); dt log-uniform in [0.001,0.1].
        self.A_log = mx.log(mx.maximum(mx.random.uniform(0, 16, (h,)), 1e-6))
        dt = mx.exp(mx.random.uniform(math.log(0.001), math.log(0.1), (h,)))
        self.dt_bias = dt + mx.log(-mx.expm1(-dt))

    def __call__(self, x, cache=None, diagnostics=None):
        b, t, _ = x.shape
        conv = (
            (None, None, None)
            if cache is None
            else cache.get("conv", (None, None, None))
        )
        q, cq = self.q_conv(self.q_proj(x), conv[0])
        k, ck = self.k_conv(self.k_proj(x), conv[1])
        v, cv = self.v_conv(self.v_proj(x), conv[2])
        q, k = (l2norm(a.reshape(b, t, self.heads, self.dk)) for a in (q, k))
        v = v.reshape(b, t, self.heads, self.dv)
        g = -mx.exp(self.A_log.astype(mx.float32)) * nn.softplus(
            self.a_proj(x).astype(mx.float32) + self.dt_bias.astype(mx.float32)
        )
        beta = mx.sigmoid(self.b_proj(x).astype(mx.float32))
        initial = None if cache is None else cache.get("state")
        kernel = recurrent_decode if t == 1 else chunk_gated_delta_rule
        out, state = kernel(q, k, v, g, beta, initial)
        gate = nn.silu(self.g_proj(x).reshape(b, t, self.heads, self.dv))
        out = self.o_norm(out.astype(x.dtype)) * gate
        if cache is not None:
            cache.update(state=state, conv=(cq, ck, cv))
        if diagnostics is not None:
            diagnostics.update(
                alpha=mx.exp(g),
                beta=beta,
                state_norm=mx.sqrt(mx.sum(state * state)),
                q_norm=mx.sqrt(mx.sum(q * q, axis=-1)),
                k_norm=mx.sqrt(mx.sum(k * k, axis=-1)),
                v_norm=mx.sqrt(mx.sum(v.astype(mx.float32) ** 2, axis=-1)),
                output_gate=gate,
            )
        return self.o_proj(out.reshape(b, t, -1))

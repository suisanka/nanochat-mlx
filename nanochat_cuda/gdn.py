"""FLA 0.5.2 kernels with an explicit short-sequence reference for CPU tests."""

import math
from functools import lru_cache
import torch
from torch import nn
from torch.nn import functional as F


def l2norm(x):
    x = x.float()
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def reference_gated_delta_rule(
    q, k, v, g, beta, initial_state=None, output_final_state=True
):
    """Differentiable FP32 oracle, [B,T,H,K/V]; not a CUDA training fallback."""
    q, k, v, g, beta = (a.float() for a in (q, k, v, g, beta))
    b, t, h, dk = q.shape
    state = (
        q.new_zeros((b, h, dk, v.shape[-1]))
        if initial_state is None
        else initial_state.float()
    )
    out = []
    for i in range(t):
        state = state * g[:, i].exp()[..., None, None]
        error = (v[:, i] - (k[:, i, :, :, None] * state).sum(-2)) * beta[:, i, :, None]
        state = state + k[:, i, :, :, None] * error[:, :, None, :]
        out.append((q[:, i, :, :, None] * state).sum(-2) / math.sqrt(dk))
    return torch.stack(out, 1), state if output_final_state else None


@lru_cache(maxsize=1)
def fla_kernels():
    try:
        from fla.ops.gated_delta_rule import (
            chunk_gated_delta_rule,
            fused_recurrent_gated_delta_rule,
        )
    except ImportError as exc:
        raise RuntimeError(
            "FLA kernels are required: on Linux install with `uv sync --extra cuda`. Use --gdn-backend reference only for short CPU checks."
        ) from exc
    return chunk_gated_delta_rule, fused_recurrent_gated_delta_rule


def gated_delta_rule(
    q, k, v, g, beta, initial_state=None, output_final_state=False, backend="fla"
):
    if backend == "reference":
        return reference_gated_delta_rule(
            q, k, v, g, beta, initial_state, output_final_state
        )
    if backend != "fla":
        raise ValueError(f"Unknown GDN backend: {backend}")
    if not q.is_cuda or q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(
            "FLA requires CUDA FP16/BF16 inputs; select reference for CPU checks"
        )
    chunk, recurrent = fla_kernels()
    # FLA's fused recurrent operator is inference-only. Training, including T=1,
    # always uses the differentiable chunk operator.
    kernel = recurrent if q.shape[1] == 1 and not torch.is_grad_enabled() else chunk
    return kernel(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=False,
    )


class CausalDepthwiseConv(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        # Same layout as MLX to permit explicit weight-only interchange.
        self.weight = nn.Parameter(torch.empty(channels, kernel_size))
        nn.init.uniform_(
            self.weight, -1 / math.sqrt(kernel_size), 1 / math.sqrt(kernel_size)
        )

    def forward(self, x, previous=None):
        b, t, c = x.shape
        if previous is None:
            previous = x.new_zeros((b, self.kernel_size - 1, c))
        history = torch.cat((previous, x), dim=1)
        y = F.conv1d(
            history.transpose(1, 2), self.weight[:, None, :], groups=c
        ).transpose(1, 2)
        tail = (
            history[:, -(self.kernel_size - 1) :]
            if self.kernel_size > 1
            else history[:, :0]
        )
        return F.silu(y), tail


class GatedDeltaNet(nn.Module):
    def __init__(self, config, backend="fla"):
        super().__init__()
        self.backend = backend
        self.heads, self.dk, self.dv = (
            config.gdn_heads,
            config.key_head_dim,
            config.value_head_dim,
        )
        d, h, k, v = config.n_embd, self.heads, self.dk, self.dv
        for name, width in (
            ("q", h * k),
            ("k", h * k),
            ("v", h * v),
            ("g", h * v),
            ("a", h),
            ("b", h),
        ):
            setattr(self, name + "_proj", nn.Linear(d, width, bias=False))
        self.o_proj = nn.Linear(h * v, d, bias=False)
        self.q_conv = CausalDepthwiseConv(h * k, config.conv_kernel)
        self.k_conv = CausalDepthwiseConv(h * k, config.conv_kernel)
        self.v_conv = CausalDepthwiseConv(h * v, config.conv_kernel)
        self.o_norm = nn.RMSNorm(v, eps=config.norm_eps)
        self.A_log = nn.Parameter(torch.rand(h).mul(16).clamp_min(1e-6).log())
        dt = torch.empty(h).uniform_(math.log(0.001), math.log(0.1)).exp()
        self.dt_bias = nn.Parameter(dt + (-torch.expm1(-dt)).log())

    def forward(self, x, cache=None):
        b, t, _ = x.shape
        conv = (
            (None, None, None)
            if cache is None
            else cache.get("conv", (None, None, None))
        )
        q, cq = self.q_conv(self.q_proj(x), conv[0])
        k, ck = self.k_conv(self.k_proj(x), conv[1])
        v, cv = self.v_conv(self.v_proj(x), conv[2])
        q, k = (
            l2norm(a.reshape(b, t, self.heads, self.dk)).to(x.dtype) for a in (q, k)
        )
        v = v.reshape(b, t, self.heads, self.dv)
        g = -self.A_log.float().exp() * F.softplus(
            self.a_proj(x).float() + self.dt_bias.float()
        )
        beta = self.b_proj(x).float().sigmoid()
        out, state = gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            None if cache is None else cache.get("state"),
            cache is not None,
            self.backend,
        )
        gate = F.silu(self.g_proj(x).reshape(b, t, self.heads, self.dv))
        out = self.o_norm(out.to(x.dtype)) * gate
        if cache is not None:
            cache.update(
                state=state.detach(),
                conv=tuple(a.detach().clone() for a in (cq, ck, cv)),
            )
        return self.o_proj(out.reshape(b, t, -1))

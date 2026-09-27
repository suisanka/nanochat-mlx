"""FP32-master Muon/AdamW, matching the MLX optimizer equations/partition."""

import math
import torch


def parameter_group(path, config):
    no_decay = path.endswith(("A_log", "dt_bias")) or "norm.weight" in path
    representation = path.startswith("blocks.") and path.endswith(
        tuple(
            name + ".weight"
            for name in (
                "q_proj",
                "k_proj",
                "v_proj",
                "g_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            )
        )
    )
    kind = "muon" if representation and config.optimizer == "muon" else "adamw"
    return dict(
        kind=kind,
        lr=config.muon_lr if kind == "muon" else config.adamw_lr,
        weight_decay=0.0 if no_decay else config.weight_decay,
    )


def lr_multiplier(step, total_steps, warmup_ratio=0.02, min_lr_ratio=0.1):
    if total_steps <= 0 or not 0 <= step < total_steps:
        raise ValueError("Require 0 <= step < total_steps")
    warmup = min(math.ceil(total_steps * warmup_ratio), max(total_steps - 1, 0))
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(total_steps - 1 - warmup, 1)
    return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def newton_schulz(gradient, steps=5):
    x = gradient.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.square().sum().sqrt() + 1e-7)
    for _ in range(steps):
        a = x @ x.T
        x = 3.4445 * x + (-4.7750 * a + 2.0315 * (a @ a)) @ x
    return x.T if transposed else x


class HybridOptimizer:
    def __init__(self, model, config, compile=False):
        self.config = config
        self.params = dict(model.named_parameters())
        self.groups = {name: parameter_group(name, config) for name in self.params}
        self.state = {}
        self.step, self.multiplier = 0, 1.0
        self.orthogonalize = torch.compile(newton_schulz) if compile else newton_schulz
        self.last_metrics = {}

    def zero_grad(self):
        for param in self.params.values():
            param.grad = None

    @torch.no_grad()
    def update(self, gradients=None, scale=1.0):
        gradients = (
            {name: p.grad for name, p in self.params.items()}
            if gradients is None
            else gradients
        )
        if set(gradients) != set(self.params) or any(
            g is None for g in gradients.values()
        ):
            raise ValueError("Missing gradient for an optimizer parameter")
        norm = (
            torch.stack([g.float().square().sum() for g in gradients.values()])
            .sum()
            .sqrt()
            * scale
        )
        if not torch.isfinite(norm).item():
            raise RuntimeError("Non-finite gradient norm")
        scale = (self.config.grad_clip / (norm + 1e-6)).clamp(max=1.0) * scale
        self.step += 1
        c = self.config
        for name, param in self.params.items():
            g = gradients[name].float() * scale
            group = self.groups[name]
            if name not in self.state:
                self.state[name] = {"master": param.float().clone()}
                for key in ("momentum",) if group["kind"] == "muon" else ("m", "v"):
                    self.state[name][key] = torch.zeros_like(g)
            state = self.state[name]
            if group["kind"] == "muon":
                buf = state["momentum"].mul_(c.momentum).add_(g, alpha=1 - c.momentum)
                direction = self.orthogonalize(
                    (1 - c.momentum) * g + c.momentum * buf, c.ns_steps
                ) * math.sqrt(max(1, g.shape[0] / g.shape[1]))
            else:
                m = state["m"].mul_(c.beta1).add_(g, alpha=1 - c.beta1)
                v = state["v"].mul_(c.beta2).addcmul_(g, g, value=1 - c.beta2)
                direction = (m / (1 - c.beta1**self.step)) / (
                    (v / (1 - c.beta2**self.step)).sqrt() + c.eps
                )
            master = state["master"]
            lr = self.multiplier * group["lr"]
            master.mul_(1 - lr * group["weight_decay"]).add_(direction, alpha=-lr)
            param.copy_(master)
        self.last_metrics = {"gradient_norm": norm.item()}

    def state_dict(self):
        return {
            "_step": torch.tensor(self.step, dtype=torch.int64),
            **{
                f"{path}/{name}": v
                for path, state in self.state.items()
                for name, v in state.items()
            },
        }

    def load_state_dict(self, arrays):
        step = int(arrays["_step"].item())
        state = {}
        for key, value in arrays.items():
            if key == "_step":
                continue
            path, name = key.rsplit("/", 1)
            if path not in self.params:
                raise ValueError(f"Unknown optimizer parameter {path}")
            if value.shape != self.params[path].shape or value.dtype != torch.float32:
                raise ValueError(f"Invalid optimizer tensor {key}")
            state.setdefault(path, {})[name] = value.to(self.params[path].device)
        if step < 0 or (step and set(state) != set(self.params)):
            raise ValueError("Invalid/incomplete optimizer state")
        for path, values in state.items():
            expected = (
                {"master", "momentum"}
                if self.groups[path]["kind"] == "muon"
                else {"master", "m", "v"}
            )
            if set(values) != expected:
                raise ValueError(f"Invalid optimizer state for {path}")
        self.state, self.step = state, step

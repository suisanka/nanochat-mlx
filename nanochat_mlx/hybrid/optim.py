"""Jordan-style Muon for representation matrices; FP32 auxiliary AdamW.

Reference: https://github.com/KellerJordan/Muon/blob/master/muon.py
Memory controllers and short convolution weights deliberately stay on AdamW.
"""

import math
import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten


@mx.compile
def newton_schulz(gradient, steps=5):
    x = gradient.astype(mx.float32)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (mx.sqrt(mx.sum(x * x)) + 1e-7)
    for _ in range(steps):
        a = x @ x.T
        x = 3.4445 * x + (-4.7750 * a + 2.0315 * (a @ a)) @ x
    return x.T if transposed else x


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
    return {
        "kind": kind,
        "lr": config.muon_lr if kind == "muon" else config.adamw_lr,
        "weight_decay": 0.0 if no_decay else config.weight_decay,
    }


def lr_multiplier(step, total_steps, warmup_ratio=0.02, min_lr_ratio=0.1):
    if total_steps <= 0 or not 0 <= step < total_steps:
        raise ValueError("Require 0 <= step < total_steps")
    warmup = min(math.ceil(total_steps * warmup_ratio), max(total_steps - 1, 0))
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(total_steps - 1 - warmup, 1)
    return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


class HybridOptimizer:
    def __init__(self, model, config):
        self.config = config
        self.groups = {
            p: parameter_group(p, config) for p, _ in tree_flatten(model.parameters())
        }
        self.shapes = {p: v.shape for p, v in tree_flatten(model.parameters())}
        self.state = {}
        self.step = 0
        self.multiplier = 1.0
        self.last_metrics = {}

    def update(self, model, grads):
        flat_grads = dict(tree_flatten(grads))
        norm = mx.sqrt(
            sum(mx.sum(g.astype(mx.float32) ** 2) for g in flat_grads.values())
        )
        if not mx.isfinite(norm).item():
            raise RuntimeError("Non-finite gradient norm")
        scale = mx.minimum(1.0, self.config.grad_clip / (norm + 1e-6))
        updates, metrics = [], {"gradient_norm": norm}
        self.step += 1
        c = self.config
        for path, param in tree_flatten(model.parameters()):
            g = flat_grads[path].astype(mx.float32) * scale
            group = self.groups[path]
            state = self.state.setdefault(path, {})
            if group["kind"] == "muon":
                buf = state.get("momentum", mx.zeros_like(g))
                buf = c.momentum * buf + (1 - c.momentum) * g
                state["momentum"] = buf
                direction = (1 - c.momentum) * g + c.momentum * buf
                direction = newton_schulz(direction, c.ns_steps) * math.sqrt(
                    max(1, g.shape[0] / g.shape[1])
                )
            else:
                m = c.beta1 * state.get("m", mx.zeros_like(g)) + (1 - c.beta1) * g
                v = c.beta2 * state.get("v", mx.zeros_like(g)) + (1 - c.beta2) * g * g
                state.update(m=m, v=v)
                direction = (m / (1 - c.beta1**self.step)) / (
                    mx.sqrt(v / (1 - c.beta2**self.step)) + c.eps
                )
            # Keep an FP32 master copy so BF16 casts do not discard small updates.
            master = state.get("master", param.astype(mx.float32))
            delta = (
                self.multiplier
                * group["lr"]
                * (direction + group["weight_decay"] * master)
            )
            master = master - delta
            state["master"] = master
            updates.append((path, master.astype(param.dtype)))
            metrics[f"{group['kind']}_update_rms/{path}"] = mx.sqrt(
                mx.mean(delta * delta)
            )
            metrics[f"gradient_norm/{path}"] = mx.sqrt(mx.sum(g * g))
        model.update(tree_unflatten(updates))
        self.last_metrics = metrics

    def arrays(self):
        return [v for _, v in tree_flatten(self.state)]

    def save(self, filename):
        arrays = {
            f"{path}/{name}": v
            for path, state in self.state.items()
            for name, v in state.items()
        }
        arrays["_step"] = mx.array(self.step, dtype=mx.int32)
        mx.save_safetensors(str(filename), arrays)

    def load(self, filename):
        arrays = mx.load(str(filename))
        self.step = int(arrays.pop("_step").item())
        state = {}
        for key, value in arrays.items():
            path, name = key.rsplit("/", 1)
            if path not in self.groups:
                raise ValueError(f"Unknown optimizer parameter {path}")
            state.setdefault(path, {})[name] = value.astype(mx.float32)
        if self.step and set(state) != set(self.groups):
            raise ValueError("Incomplete optimizer checkpoint")
        if self.step < 0:
            raise ValueError("Invalid optimizer step")
        for path, values in state.items():
            expected = (
                {"master", "momentum"}
                if self.groups[path]["kind"] == "muon"
                else {"master", "m", "v"}
            )
            if set(values) != expected or any(
                v.shape != self.shapes[path] for v in values.values()
            ):
                raise ValueError(f"Invalid optimizer state for {path}")
        self.state = state

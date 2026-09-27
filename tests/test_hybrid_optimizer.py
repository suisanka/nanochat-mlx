"""Numerical optimizer/checkpoint contracts on a tiny synthetic parameter tree."""

from dataclasses import replace
import numpy as np
import pytest
import mlx.core as mx
from mlx.utils import tree_flatten, tree_map
from nanochat_mlx.hybrid.config import TrainingConfig
from nanochat_mlx.hybrid.model import HybridLM
from nanochat_mlx.hybrid.optim import HybridOptimizer, lr_multiplier, newton_schulz
from nanochat_mlx.hybrid.checkpoint import save_checkpoint, load_checkpoint
from tests.test_hybrid import tiny_config, assert_close


def test_optimizer_partition_tied_embedding_once():
    model = HybridLM(tiny_config())
    opt = HybridOptimizer(model, TrainingConfig())
    assert opt.groups["wte.weight"]["kind"] == "adamw"
    assert sum(p == "wte.weight" for p in opt.groups) == 1
    assert not any("lm_head" in p for p in opt.groups)
    for path, group in opt.groups.items():
        if any(
            x in path
            for x in (
                ".a_proj.",
                ".b_proj.",
                "_conv.",
                "norm.weight",
                "A_log",
                "dt_bias",
            )
        ):
            assert group["kind"] == "adamw"
        if "norm.weight" in path or path.endswith(("A_log", "dt_bias")):
            assert group["weight_decay"] == 0
    assert opt.groups["blocks.0.mixer.g_proj.weight"]["kind"] == "muon"
    assert opt.groups["blocks.2.mixer.q_proj.weight"]["kind"] == "muon"


def test_optimizer_formula_and_state_roundtrip(tmp_path):
    # Fixed synthetic gradients test the update equations; this is not a model
    # training loop and no corpus or forward/backward loss is used.
    model = HybridLM(tiny_config(dtype="bfloat16"))
    config = TrainingConfig(grad_clip=1e6)
    opt = HybridOptimizer(model, config)
    grads = tree_map(
        lambda p: mx.full(p.shape, 0.01, dtype=mx.float32), model.parameters()
    )
    before = dict(tree_flatten(model.parameters()))
    path = "blocks.0.mixer.q_proj.weight"
    grad = dict(tree_flatten(grads))[path]
    m = (1 - config.momentum) * grad
    direction = (1 - config.momentum) * grad + config.momentum * m
    update = newton_schulz(direction) * max(1, grad.shape[0] / grad.shape[1]) ** 0.5
    expected = before[path].astype(mx.float32) - config.muon_lr * (
        update + config.weight_decay * before[path].astype(mx.float32)
    )
    opt.update(model, grads)
    assert_close(opt.state[path]["master"], expected)
    assert all(v.dtype == mx.float32 for v in opt.arrays())
    file = save_checkpoint(
        tmp_path, model, 1, {"test": "synthetic"}, opt, {"cursor": 8}, config
    )
    restored, meta, loaded = load_checkpoint(file, {"test": "synthetic"}, True)
    assert loaded.step == 1 and meta["loader"]["cursor"] == 8
    for a, b in zip(opt.arrays(), loaded.arrays()):
        # Save/load ordering is tested below by names, not dict insertion order.
        assert a.dtype == b.dtype
    for param, state in opt.state.items():
        for key, value in state.items():
            assert_close(value, loaded.state[param][key], atol=0, rtol=0)
    ids = mx.array([[1, 2, 3]])
    assert_close(model(ids), restored(ids), atol=0, rtol=0)
    with pytest.raises(ValueError, match="tokenizer"):
        load_checkpoint(file, {"test": "wrong"})
    with pytest.raises(FileExistsError):
        save_checkpoint(tmp_path, model, 1, {"test": "synthetic"})


def test_cosine_schedule_endpoints():
    assert lr_multiplier(0, 100) == 0.5
    assert lr_multiplier(1, 100) == 1
    assert lr_multiplier(2, 100) == 1
    assert lr_multiplier(99, 100) == pytest.approx(0.1)


def test_adamw_control_all_parameters():
    opt = HybridOptimizer(HybridLM(tiny_config()), TrainingConfig(optimizer="adamw"))
    assert all(v["kind"] == "adamw" for v in opt.groups.values())

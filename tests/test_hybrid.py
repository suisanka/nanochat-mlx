"""Phase-0 synthetic numerical checks. No training runs or long-context runs."""

from dataclasses import replace
import numpy as np
import pytest
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from nanochat_mlx.hybrid.config import HybridConfig, config_for_depth
from nanochat_mlx.hybrid.gdn import l2norm, naive_recurrent_gdn, chunk_gated_delta_rule
from nanochat_mlx.hybrid.attention import local_attention
from nanochat_mlx.hybrid.loss import linear_cross_entropy
from nanochat_mlx.hybrid.model import HybridLM


def tiny_config(**kwargs):
    fields = dict(
        n_layer=3,
        n_embd=32,
        intermediate_size=64,
        vocab_size=97,
        sequence_len=64,
        max_context=256,
        gdn_heads=2,
        key_head_dim=8,
        value_head_dim=16,
        n_head=4,
        n_kv_head=1,
        head_dim=8,
        window_size=8,
        attention_tile=4,
        dtype="float32",
        loss_chunk_size=7,
    )
    return HybridConfig(**(fields | kwargs))


def assert_close(a, b, atol=2e-5, rtol=2e-4):
    np.testing.assert_allclose(
        np.array(a.astype(mx.float32)),
        np.array(b.astype(mx.float32)),
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize("depth", [4, 12, 20, 26])
def test_depth_contract(depth):
    c = config_for_depth(depth)
    assert c.n_layer == depth
    assert len(c.layer_types) == depth
    assert c.gdn_heads * c.key_head_dim == 3 * c.n_embd // 4
    assert c.n_head == 4 * c.n_kv_head
    assert c.intermediate_size == 3 * c.n_embd
    if depth == 12:
        assert c.n_embd == 512 and c.gdn_heads == 6
        assert c.layer_types == ("gdn", "gdn", "swa") * 4
        assert 109_000_000 < c.parameter_counts()["total"] < 111_000_000


def test_context_configuration_only():
    # The requested 32K support is checked as metadata, not by executing 32K.
    for scaling in ("none", "linear", "yarn"):
        c = config_for_depth(
            12,
            sequence_len=32768,
            rope_scaling=scaling,
            rope_factor=1 if scaling == "none" else 8,
        )
        assert HybridConfig(**c.to_dict()) == c
    with pytest.raises(ValueError, match="32768"):
        config_for_depth(12, max_context=131072)


@pytest.mark.parametrize("length", [16, 32, 64, 79])
def test_gdn_chunk_output_state_and_gradients(length):
    mx.random.seed(19)
    q, k = [l2norm(mx.random.normal((1, length, 2, 8))) for _ in range(2)]
    v = mx.random.normal((1, length, 2, 6))
    g = -mx.exp(mx.random.normal((1, length, 2))) * 0.03
    beta = mx.sigmoid(mx.random.normal((1, length, 2)))
    state = mx.random.normal((1, 2, 8, 6)) * 0.1
    args = (q, k, v, g, beta, state)
    for a, b in zip(naive_recurrent_gdn(*args), chunk_gated_delta_rule(*args)):
        assert_close(a, b)

    def objective(kernel, *inputs):
        out, final = kernel(*inputs)
        return mx.sum(out**2) + mx.sum(final**2)

    ref = mx.grad(
        lambda *x: objective(naive_recurrent_gdn, *x), argnums=tuple(range(6))
    )(*args)
    actual = mx.grad(
        lambda *x: objective(chunk_gated_delta_rule, *x), argnums=tuple(range(6))
    )(*args)
    for a, b in zip(actual, ref):
        assert_close(a, b, atol=5e-5, rtol=5e-4)


def test_gdn_strong_decay_is_finite():
    q = l2norm(mx.random.normal((1, 64, 1, 8)))
    v = mx.random.normal((1, 64, 1, 6))
    g, beta = mx.full((1, 64, 1), -20.0), mx.full((1, 64, 1), 0.7)
    a, state = chunk_gated_delta_rule(q, q, v, g, beta)
    b, _ = naive_recurrent_gdn(q, q, v, g, beta)
    assert mx.all(mx.isfinite(a)).item()
    assert_close(a, b)


def test_local_attention_matches_dense_reference_and_gradients():
    q = mx.random.normal((1, 4, 23, 8))
    k, v = [mx.random.normal((1, 1, 23, 8)) for _ in range(2)]
    p = mx.arange(23)
    mask = mx.where(
        (p[:, None] >= p[None, :]) & (p[:, None] - p[None, :] < 8), 0.0, -mx.inf
    )
    reference = lambda q, k, v: mx.fast.scaled_dot_product_attention(
        q, k, v, scale=8**-0.5, mask=mask
    )
    tiled = lambda q, k, v: local_attention(q, k, v, window=8, tile=4)
    assert_close(tiled(q, k, v), reference(q, k, v))
    for a, b in zip(
        mx.grad(lambda *x: mx.sum(tiled(*x) ** 2), argnums=(0, 1, 2))(q, k, v),
        mx.grad(lambda *x: mx.sum(reference(*x) ** 2), argnums=(0, 1, 2))(q, k, v),
    ):
        assert_close(a, b, atol=8e-5)


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_fused_loss_and_both_gradients(dtype):
    h = mx.random.normal((2, 11, 16)).astype(dtype)
    w = (mx.random.normal((101, 16)) * 0.1).astype(dtype)
    y = mx.random.randint(0, 101, (2, 11))
    y[:, ::3] = -1

    def vanilla(h, w):
        logits = (h @ w.T).astype(mx.float32)
        ce = nn.losses.cross_entropy(logits, mx.maximum(y, 0), reduction="none")
        return mx.sum(mx.where(y != -1, ce, 0)) / mx.maximum(mx.sum(y != -1), 1)

    f = lambda h, w: linear_cross_entropy(h, w, y, chunk_size=7)
    assert_close(f(h, w), vanilla(h, w), atol=2e-3 if dtype == mx.bfloat16 else 1e-6)
    for a, b in zip(
        mx.grad(f, argnums=(0, 1))(h, w), mx.grad(vanilla, argnums=(0, 1))(h, w)
    ):
        assert_close(a, b, atol=0.002 if dtype == mx.bfloat16 else 2e-6, rtol=0.02)
    assert linear_cross_entropy(h, w, mx.full(y.shape, -1), 7).item() == 0


def test_linear_ce_releases_tiles_during_autodiff():
    # A reduced vocabulary reproduces the real 4K failure without allocating
    # gigabytes: the previous VJP retained >100 MiB for this 16 MiB logits shape.
    import gc

    gc.collect()
    h = mx.random.normal((1, 512, 32))
    w = mx.random.normal((8192, 32))
    y = mx.zeros((1, 512), dtype=mx.int32)
    mx.eval(h, w, y)
    mx.clear_cache()
    baseline = mx.get_active_memory()
    mx.reset_peak_memory()
    loss, grads = mx.value_and_grad(
        lambda h, w: linear_cross_entropy(h, w, y, 64), argnums=(0, 1)
    )(h, w)
    mx.eval(loss, grads)
    assert mx.isfinite(loss).item()
    assert mx.get_peak_memory() - baseline < 32 * 1024**2


@pytest.mark.parametrize("rope_scaling", ["none", "linear", "yarn"])
def test_cache_prefill_decode_and_weight_roundtrip(rope_scaling, tmp_path):
    mx.random.seed(27)
    c = tiny_config(
        rope_scaling=rope_scaling,
        rope_factor=1 if rope_scaling == "none" else 8,
    )
    model = HybridLM(c)
    ids = mx.random.randint(0, c.vocab_size, (1, 29))
    full = model(ids)
    cache = model.make_cache()
    pieces = [model(ids[:, :11], kv_cache=cache), model(ids[:, 11:17], kv_cache=cache)]
    for t in range(17, 29):
        pieces.append(model(ids[:, t : t + 1], kv_cache=cache))
    assert_close(mx.concatenate(pieces, axis=1), full)
    assert cache.offset == 29
    for kind, layer in zip(c.layer_types, cache.layers):
        if kind == "swa":
            assert layer["keys"].shape[2] == c.window_size
        elif kind == "gdn":
            assert layer["state"].shape == (
                1,
                c.gdn_heads,
                c.key_head_dim,
                c.value_head_dim,
            )
            assert layer["conv"][0].shape[1] == c.conv_kernel - 1
    assert_close(model.prefill(ids, model.make_cache(), chunk_size=5), full[:, -1:])
    counts = sum(p.size for _, p in tree_flatten(model.parameters()))
    assert counts == c.parameter_counts()["total"]
    filename = str(tmp_path / "model.safetensors")
    model.save_weights(filename)
    restored = HybridLM(c)
    restored.load_weights(filename, strict=True)
    assert_close(restored(ids), full, atol=0, rtol=0)


def test_model_projection_and_gate_gradients_and_reset():
    model = HybridLM(tiny_config())
    ids = mx.random.randint(0, 97, (1, 17))
    loss, grads = nn.value_and_grad(model, lambda m: m(ids, targets=ids))(model)
    assert mx.isfinite(loss).item()
    flat = dict(tree_flatten(grads))
    for suffix in (
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "a_proj.weight",
        "b_proj.weight",
        "A_log",
        "dt_bias",
    ):
        g = flat["blocks.0.mixer." + suffix]
        assert mx.all(mx.isfinite(g)).item()
        assert mx.max(mx.abs(g)).item() > 0
    cache = model.make_cache()
    model(ids, kv_cache=cache)
    cache.reset()
    assert_close(model(ids, kv_cache=cache), model(ids))


def test_checkpointed_blocks():
    model = HybridLM(tiny_config(checkpoint_blocks=True))
    ids = mx.array([[1, 2, 3, 4]])
    loss, grad = nn.value_and_grad(model, lambda m: m(ids, targets=ids))(model)
    mx.eval(loss, grad)
    assert mx.isfinite(loss).item()


def test_diagnostics_distributions_are_reduced():
    model = HybridLM(tiny_config())
    stats = {}
    mx.eval(model.hidden(mx.array([[1, 2, 3, 4]]), diagnostics=stats))
    for index in ("0", "1"):
        assert stats[index]["alpha"]["histogram"].shape == (10,)
        assert mx.sum(stats[index]["alpha"]["histogram"]).item() == pytest.approx(1.0)
        assert "state_norm" in stats[index]
    assert "attention_entropy" in stats["2"]


def test_bfloat16_full_model_loss_and_cache():
    model = HybridLM(tiny_config(dtype="bfloat16"))
    ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])
    loss, grads = nn.value_and_grad(model, lambda m: m(ids, targets=ids))(model)
    mx.eval(loss, grads)
    assert mx.isfinite(loss).item()
    cache = model.make_cache()
    assert_close(
        model.prefill(ids, cache, chunk_size=4),
        model(ids)[:, -1:],
        atol=0.004,
        rtol=0.03,
    )

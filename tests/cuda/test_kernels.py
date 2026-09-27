"""Run explicitly on an NVIDIA host; these checks never train a model."""

import pytest

torch = pytest.importorskip("torch")

from nanochat_cuda.gdn import gated_delta_rule, reference_gated_delta_rule, l2norm
from nanochat_cuda.attention import local_attention

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires NVIDIA CUDA"),
]


@pytest.mark.parametrize("length", [1, 65])
def test_fla_forward_backward_and_state(length):
    torch.manual_seed(42)
    q, k = [
        l2norm(torch.randn(1, length, 2, 64, device="cuda")).bfloat16().requires_grad_()
        for _ in range(2)
    ]
    v = torch.randn(
        1, length, 2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    g = (-torch.rand(1, length, 2, device="cuda") * 0.1).requires_grad_()
    beta = torch.rand(1, length, 2, device="cuda", requires_grad=True)
    initial = torch.randn(1, 2, 64, 128, device="cuda", requires_grad=True)
    args = (q, k, v, g, beta, initial)
    out, state = gated_delta_rule(*args, output_final_state=True)
    expected, final = reference_gated_delta_rule(*args)
    torch.testing.assert_close(out.float(), expected, atol=0.02, rtol=0.03)
    torch.testing.assert_close(state, final, atol=0.04, rtol=0.04)
    actual_grads = torch.autograd.grad(
        out.float().square().mean() + state.square().mean(), args
    )
    expected_grads = torch.autograd.grad(
        expected.square().mean() + final.square().mean(), args
    )
    for a, b in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a.float(), b.float(), atol=0.003, rtol=0.06)
    with torch.no_grad():
        decoded, final = gated_delta_rule(
            q[:, :1], k[:, :1], v[:, :1], g[:, :1], beta[:, :1], initial, True
        )
        reference, ref_final = reference_gated_delta_rule(
            q[:, :1], k[:, :1], v[:, :1], g[:, :1], beta[:, :1], initial
        )
        torch.testing.assert_close(decoded.float(), reference, atol=0.02, rtol=0.03)
        torch.testing.assert_close(final, ref_final, atol=0.04, rtol=0.04)


def test_flash_local_attention_and_cached_alignment():
    pytest.importorskip("flash_attn")
    q = torch.randn(
        1, 4, 7, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    k, v = [
        torch.randn(
            1, 1, 23, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        for _ in range(2)
    ]
    actual = local_attention(q, k, v, 16, 0, window=8, tile=4, backend="flash")
    expected = local_attention(q, k, v, 16, 0, window=8, tile=4)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.03)
    for a, b in zip(
        torch.autograd.grad(actual.float().square().mean(), (q, k, v)),
        torch.autograd.grad(expected.float().square().mean(), (q, k, v)),
    ):
        torch.testing.assert_close(a, b, atol=0.002, rtol=0.05)


def test_cuda_compiled_model_loss_backward_and_cache():
    from nanochat_cuda.model import HybridLM
    from tests.cuda.test_model import tiny_config

    torch.manual_seed(5)
    config = tiny_config(
        n_embd=64,
        intermediate_size=128,
        key_head_dim=64,
        value_head_dim=128,
        head_dim=64,
        dtype="bfloat16",
        checkpoint_blocks=True,
    )
    model = HybridLM(config).cuda()
    ids, labels = (
        torch.randint(97, (1, 17), device="cuda"),
        torch.randint(97, (1, 17), device="cuda"),
    )
    model(ids, targets=labels).backward()
    expected = {n: p.grad.clone() for n, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    model.compile_hotpaths()
    loss = model(ids, targets=labels)
    loss.backward()
    assert torch.isfinite(loss)
    for name, param in model.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all()
        torch.testing.assert_close(param.grad, expected[name], atol=0.015, rtol=0.08)
    model.eval()
    with torch.no_grad():
        full = model(ids)
        cache = model.make_cache()
        model.prefill(ids[:, :-1], cache, 5)
        last = model(ids[:, -1:], kv_cache=cache)
        torch.testing.assert_close(last, full[:, -1:], atol=0.015, rtol=0.08)

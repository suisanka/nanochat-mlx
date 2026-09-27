from dataclasses import replace
import json
import subprocess
import sys
from types import SimpleNamespace
import pytest

torch = pytest.importorskip("torch")
from torch.nn import functional as F
from nanochat_cuda.config import HybridConfig, TrainingConfig, config_for_depth
from nanochat_cuda.model import HybridLM
from nanochat_cuda.loss import linear_cross_entropy
from nanochat_cuda.attention import local_attention
from nanochat_cuda.optim import HybridOptimizer, newton_schulz, lr_multiplier
from nanochat_cuda.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    initialize_weights,
)
from nanochat_cuda.engine import HybridEngine
from nanochat_cuda.training import build_parser, resolve_plan, accumulate_gradients


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    yield
    torch.set_num_threads(before)


def tiny_config(**kwargs):
    values = dict(
        n_layer=3,
        n_embd=32,
        intermediate_size=64,
        vocab_size=97,
        sequence_len=16,
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
    return HybridConfig(**(values | kwargs))


def test_offline_plan_imports_no_tensor_runtime():
    code = "from nanochat_cuda.training import main; import sys; assert main(['--recipe', 'configs/gdn_swa_32k_memory.json', '--depth', '4', '--dry-run']) == 0; assert not any(n in sys.modules for n in ('torch', 'mlx', 'fla', 'datasets'))"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert '"sequence_len": 32768' in result.stdout
    assert '"training_started": false' in result.stdout


@pytest.mark.parametrize("depth", [4, 12, 20, 26])
def test_depth_shapes(depth):
    c = config_for_depth(depth)
    assert c.layer_types == tuple(("gdn", "gdn", "swa")[i % 3] for i in range(depth))
    assert c.n_head == c.n_kv_head * 4
    if depth == 12:
        assert c.parameter_counts()["total"] == 109819488


def test_parameter_count_and_partition():
    c = tiny_config()
    m = HybridLM(c, "reference")
    assert sum(p.numel() for p in m.parameters()) == c.parameter_counts()["total"]
    opt = HybridOptimizer(m, TrainingConfig())
    for name, group in opt.groups.items():
        if any(
            s in name
            for s in (
                ".a_proj.",
                ".b_proj.",
                "_conv.",
                "norm.weight",
                "A_log",
                "dt_bias",
            )
        ):
            assert group["kind"] == "adamw"
        if "norm.weight" in name or name.endswith(("A_log", "dt_bias")):
            assert group["weight_decay"] == 0
    assert opt.groups["wte.weight"]["kind"] == "adamw"
    assert opt.groups["blocks.0.mixer.q_proj.weight"]["kind"] == "muon"
    assert len([name for name in opt.params if name == "wte.weight"]) == 1


@pytest.mark.parametrize("tile", [1, 7, 100])
@pytest.mark.parametrize("masked", [False, True])
def test_tiled_loss_and_gradients(tile, masked):
    h = torch.randn(2, 13, 16, requires_grad=True)
    w = torch.randn(97, 16, requires_grad=True)
    y = torch.randint(97, (2, 13))
    if masked:
        y[:, ::3] = -1
    loss = linear_cross_entropy(h, w, y, tile)
    dense = F.cross_entropy(
        F.linear(h, w).reshape(-1, 97), y.flatten(), ignore_index=-1
    )
    actual = torch.autograd.grad(loss * 2.7, (h, w))
    expected = torch.autograd.grad(dense * 2.7, (h, w))
    torch.testing.assert_close(loss, dense)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)


def test_loss_saved_tensors_are_bounded_and_all_masked_is_zero():
    h = torch.randn(2, 19, 16, requires_grad=True)
    w = torch.randn(97, 16, requires_grad=True)
    y = torch.full((2, 19), -1)
    saved = []

    def pack(t):
        saved.append(t.shape)
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss = linear_cross_entropy(h, w, y, 3)
    assert saved == [h.shape, w.shape, y.shape, torch.Size([])]
    loss.backward()
    assert loss.item() == 0 and h.grad.abs().sum() == 0 and w.grad.abs().sum() == 0


def test_local_attention_matches_dense_mask_and_gradients():
    q = torch.randn(2, 4, 17, 8, requires_grad=True)
    k = torch.randn(2, 1, 17, 8, requires_grad=True)
    v = torch.randn(2, 1, 17, 8, requires_grad=True)
    p = torch.arange(17)
    mask = (p[:, None] >= p[None, :]) & (p[:, None] - p[None, :] < 5)
    expected = F.scaled_dot_product_attention(
        q, k.repeat_interleave(4, 1), v.repeat_interleave(4, 1), attn_mask=mask
    )
    actual = local_attention(q, k, v, window=5, tile=3)
    torch.testing.assert_close(actual, expected)
    for a, b in zip(
        torch.autograd.grad(actual.square().sum(), (q, k, v)),
        torch.autograd.grad(expected.square().sum(), (q, k, v)),
    ):
        torch.testing.assert_close(a, b, atol=4e-6, rtol=3e-5)


@pytest.mark.parametrize(
    "scaling,factor", [("none", 1.0), ("linear", 4.0), ("yarn", 4.0)]
)
def test_cache_prefill_decode_eviction_and_causality(scaling, factor):
    model = HybridLM(
        tiny_config(rope_scaling=scaling, rope_factor=factor), "reference"
    ).eval()
    ids = torch.randint(97, (2, 31))
    with torch.no_grad():
        full = model(ids)
        cache = model.make_cache()
        prefill = model.prefill(ids[:, :23], cache, chunk_size=5)
        torch.testing.assert_close(prefill, full[:, 22:23], atol=2e-6, rtol=2e-5)
        decoded = torch.cat(
            [model(ids[:, i : i + 1], kv_cache=cache) for i in range(23, 31)], 1
        )
        torch.testing.assert_close(decoded, full[:, 23:], atol=2e-6, rtol=2e-5)
        changed = ids.clone()
        changed[:, 15:] = torch.randint(97, (2, 16))
        torch.testing.assert_close(model(changed)[:, :15], full[:, :15])
        assert cache.offset == 31
        assert cache.layers[2]["keys"].shape[2] == 8
        for name in ("keys", "values"):
            a = cache.layers[2][name]
            assert a.untyped_storage().nbytes() == a.numel() * a.element_size()
        assert cache.layers[0]["state"].dtype == torch.float32
        repeated = cache.repeat(2)
        assert repeated.batch_size == 4
        assert repeated.layers[0]["state"].shape[0] == 4
        cache.reset()
        assert cache.offset == 0 and cache.layers == [{}, {}, {}]


def test_checkpointed_backward_and_tied_embedding():
    a = HybridLM(tiny_config(), "reference")
    b = HybridLM(replace(a.config, checkpoint_blocks=True), "reference")
    b.load_state_dict(a.state_dict())
    ids, targets = torch.randint(97, (2, 9)), torch.randint(97, (2, 9))
    targets[:, :2] = -1
    a(ids, targets=targets).backward()
    b(ids, targets=targets).backward()
    for p, q in zip(a.parameters(), b.parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all()
        torch.testing.assert_close(p.grad, q.grad)
    # Dense tied head must also contribute its gradient to the embedding.
    c = HybridLM(a.config, "reference")
    c.load_state_dict(a.state_dict())
    F.cross_entropy(
        c(ids).reshape(-1, 97), targets.flatten(), ignore_index=-1
    ).backward()
    for p, q in zip(a.parameters(), c.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=3e-5)


def test_compile_forward_and_gradients():
    a = HybridLM(tiny_config(), "reference")
    b = HybridLM(a.config, "reference")
    b.load_state_dict(a.state_dict())
    b.compile_hotpaths(backend="aot_eager")
    ids, targets = torch.randint(97, (1, 5)), torch.randint(97, (1, 5))
    a(ids, targets=targets).backward()
    b(ids, targets=targets).backward()
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad)
    assert a.state_dict().keys() == b.state_dict().keys()


@pytest.mark.parametrize("kind", ["muon", "adamw"])
def test_optimizer_equations_and_resume(tmp_path, kind):
    model = HybridLM(tiny_config(dtype="bfloat16"), "reference")
    c = TrainingConfig(optimizer=kind, grad_clip=1e6)
    opt = HybridOptimizer(model, c)
    gradients = {n: torch.full(p.shape, 0.01) for n, p in model.named_parameters()}
    name = "blocks.0.mixer.q_proj.weight"
    before, g = opt.params[name].float().clone(), gradients[name]
    if kind == "muon":
        buf = (1 - c.momentum) * g
        direction = (
            newton_schulz((1 - c.momentum) * g + c.momentum * buf)
            * max(1, g.shape[0] / g.shape[1]) ** 0.5
        )
        lr = c.muon_lr
    else:
        direction, lr = g / (g.abs() + c.eps), c.adamw_lr
    expected = before - lr * (direction + c.weight_decay * before)
    opt.update(gradients)
    torch.testing.assert_close(opt.state[name]["master"], expected)
    assert all(
        v.dtype == torch.float32 for state in opt.state.values() for v in state.values()
    )
    meta = save_checkpoint(
        tmp_path, model, 1, {"test": True}, opt, {"cursor": 16}, c, {"steps": 8}
    )
    expected_rng = torch.rand(4)
    restored, info, resumed = load_checkpoint(
        meta, {"test": True}, gdn_backend="reference", load_optimizer=True
    )
    torch.testing.assert_close(torch.rand(4), expected_rng, rtol=0, atol=0)
    assert info["loader"] == {"cursor": 16}
    opt.update(gradients)
    resumed.update(gradients)
    for p, q in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        save_checkpoint(tmp_path, model, 1, {"test": True})
    with pytest.raises(ValueError, match="tokenizer"):
        load_checkpoint(meta, {"wrong": True})


def test_warm_start_explicit_mlx_format_and_context_change(tmp_path):
    model = HybridLM(tiny_config(), "reference")
    meta = save_checkpoint(tmp_path, model, 0, {"test": True})
    data = json.loads(meta.read_text())
    data["format"] = "nanochat-mlx-hybrid-v1"
    meta.write_text(json.dumps(data))
    extended = HybridLM(
        replace(model.config, sequence_len=128, rope_scaling="linear", rope_factor=4),
        "reference",
    )
    initialize_weights(extended, meta, {"test": True})
    for p, q in zip(model.parameters(), extended.parameters()):
        torch.testing.assert_close(p, q)
    with pytest.raises(ValueError, match="CUDA checkpoint"):
        load_checkpoint(meta)


def test_engine_seed_context_and_cache():
    model = HybridLM(tiny_config(), "reference")
    engine = HybridEngine(model, SimpleNamespace(contract={"eos": 1}))
    options = dict(num_samples=2, max_tokens=5, seed=17, prefill_chunk_size=2)
    a = list(engine.generate([3, 4, 5], **options))
    b = list(engine.generate([3, 4, 5], **options))
    assert a == b and 0 < len(a) <= 5 and len(a[0][0]) == 2
    with pytest.raises(ValueError, match="context"):
        list(engine.generate([3] * 250, max_tokens=10))


def test_invalid_plan_and_no_silent_kernel_fallback(monkeypatch):
    with pytest.raises(ValueError, match="CPU checks"):
        resolve_plan(build_parser().parse_args(["--device", "cpu"]))
    with pytest.raises(ValueError, match="CUDA"):
        HybridLM(tiny_config())(torch.ones((1, 2), dtype=torch.long))
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="single GPU"):
        resolve_plan(build_parser().parse_args([]))


def test_schedule():
    assert lr_multiplier(0, 100) == 0.5
    assert lr_multiplier(1, 100) == 1.0
    assert lr_multiplier(99, 100) == pytest.approx(0.1)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_weighted_accumulation_matches_full_batch(dtype):
    model = HybridLM(tiny_config(dtype=dtype), "reference")
    ids = torch.randint(97, (2, 7))
    targets = torch.randint(97, (2, 7))
    targets[0, :5] = -1
    targets[1, :1] = -1
    grads, loss, count = accumulate_gradients(
        model, [(ids[:1], targets[:1]), (ids[1:], targets[1:])], 14
    )
    assert count == 8 and all(g.dtype == torch.float32 for g in grads.values())
    dense = model(ids, targets=targets)
    dense.backward()
    assert loss == pytest.approx(dense.item(), abs=0.002)
    tolerance = 0.008 if dtype == "bfloat16" else 4e-6
    for name, p in model.named_parameters():
        torch.testing.assert_close(
            grads[name] * (14 / count),
            p.grad.float(),
            atol=tolerance,
            rtol=0.06 if dtype == "bfloat16" else 5e-5,
        )


def test_mmap_adapter_cursor_without_mlx(tmp_path):
    from nanochat_mlx.hybrid.data import TokenDataset, prepare_documents
    from nanochat_cuda.data import next_batch

    class Tokenizer:
        contract = {"eos": 1, "vocab_size": 256}

        def encode(self, text):
            return list(text.encode("ascii"))

        def get_vocab_size(self):
            return 256

    prepare_documents(tmp_path, ["abcdefg" * 12], ["xyz" * 30], Tokenizer())
    loader = TokenDataset(tmp_path, "train", 7, 2)
    x, y = next_batch(loader, torch.device("cpu"))
    assert x.dtype == y.dtype == torch.long and x.shape == (2, 7)
    state = loader.state_dict()
    expected = next_batch(loader, torch.device("cpu"))
    resumed = TokenDataset(tmp_path, "train", 7, 2, state)
    for a, b in zip(expected, next_batch(resumed, torch.device("cpu"))):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize("prefetch", [0, 2, "process"])
def test_stream_adapter_cursor(tmp_path, prefetch):
    import datasets
    from nanochat_mlx.hybrid.streaming import StreamingTokenDataset
    from nanochat_cuda.data import next_batch, PrefetchedDataset
    from tests.test_streaming_data import CharacterTokenizer as Tokenizer

    def create(state=None):
        stream = datasets.Dataset.from_dict(
            {"text": ["abcdefg" * 12, "hijklm" * 20]}
        ).to_iterable_dataset()
        source = {
            "request": {"text_column": "text"},
            "datasets_version": datasets.__version__,
            "revision": "fixture",
        }
        return StreamingTokenDataset(stream, Tokenizer(), source, "train", 7, 2, state)

    loader = create()
    if prefetch:
        loader = PrefetchedDataset(loader, 2, process=prefetch == "process")
    next_batch(loader, torch.device("cpu"))
    resumed = create(json.loads(json.dumps(loader.state_dict())))
    for _ in range(3):
        for a, b in zip(
            next_batch(loader, torch.device("cpu")),
            next_batch(resumed, torch.device("cpu")),
        ):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    loader.close()
    resumed.close()


def test_prefetch_cursor_ignores_lookahead_and_propagates_errors():
    import threading
    from nanochat_cuda.data import PrefetchedDataset

    ready = threading.Event()

    class Loader:
        meta = {}
        position = 0

        def state_dict(self):
            return {"position": self.position}

        def next_numpy(self):
            self.position += 1
            if self.position == 3:
                ready.set()
                raise RuntimeError("data failed")
            return self.position

    loader = PrefetchedDataset(Loader(), 2)
    try:
        assert ready.wait(2)
        assert loader.state_dict() == {"position": 0}
        assert loader.next_numpy() == 1
        assert loader.state_dict() == {"position": 1}
        assert loader.next_numpy() == 2
        with pytest.raises(RuntimeError, match="data failed"):
            loader.next_numpy()
        assert loader.state_dict() == {"position": 2}
    finally:
        loader.close()
    assert not loader.thread.is_alive()

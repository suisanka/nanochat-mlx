"""Weight interchange and likelihood accounting for local base-model evaluation."""

import json
import math

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from mlx.utils import tree_flatten

from nanochat_mlx.hybrid.checkpoint import import_cuda_checkpoint, load_checkpoint
from nanochat_mlx.hybrid.evaluation import score_ids, score_text, scoring_windows
from nanochat_mlx.hybrid.model import HybridLM
from tests.test_hybrid import tiny_config


@pytest.mark.parametrize("context,stride", [(1, 1), (8, 1), (8, 3), (8, 8)])
def test_windows_cover_targets_once_with_correct_causal_alignment(context, stride):
    seen = []
    for x, y in scoring_windows(list(range(25)), context, stride):
        assert 0 < len(x) == len(y) <= context
        for previous, target in zip(x, y):
            if target != -1:
                assert target == previous + 1
                seen.append(target)
    assert seen == list(range(1, 25))


def test_bounded_score_matches_dense_and_single_token_rolling_oracle():
    model = HybridLM(tiny_config())
    ids = [0, 4, 8, 1, 2, 5, 6, 9]
    logits = model(mx.array([ids[:-1]]))
    dense = nn.losses.cross_entropy(logits, mx.array([ids[1:]]), reduction="sum")
    result = score_ids(model, ids, context_length=8, stride=8, loss_chunk_size=2)
    np.testing.assert_allclose(result["nll_sum"], dense.item(), rtol=1e-6)
    assert result["tokens"] == len(ids) - 1
    oracle = 0.0
    for end in range(1, len(ids)):
        x = mx.array([ids[max(0, end - 3) : end]])
        oracle += nn.losses.cross_entropy(
            model(x)[:, -1], mx.array([ids[end]]), reduction="sum"
        ).item()
    rolling = score_ids(model, ids, context_length=3, stride=1, loss_chunk_size=2)
    np.testing.assert_allclose(rolling["nll_sum"], oracle, rtol=1e-6)


def test_bpb_counts_original_utf8_bytes_and_rejects_lossy_tokenization():
    class Tokenizer:
        def __init__(self):
            self.contract = {"eos": 0}

        def encode(self, text):
            return [5, 8]

        def decode(self, tokens):
            return "中a"

    result = score_text(
        HybridLM(tiny_config()), Tokenizer(), "中a", context_length=8, stride=4
    )
    assert result["tokens"] == 2
    assert result["bytes"] == 4
    assert result["bpb"] == result["nll_sum"] / (4 * math.log(2))
    with pytest.raises(ValueError, match="round-trip"):
        score_text(HybridLM(tiny_config()), Tokenizer(), "中b")


def test_cuda_weight_import_preserves_values_and_cannot_resume_optimizer(tmp_path):
    torch = pytest.importorskip("torch")
    from nanochat_cuda.checkpoint import save_checkpoint
    from nanochat_cuda.model import HybridLM as TorchModel

    torch.set_num_threads(1)
    config = tiny_config()
    source_model = TorchModel(config, "reference")
    source = save_checkpoint(tmp_path / "cuda", source_model, 7, {"fixture": True})
    original = source.read_bytes()
    dest = import_cuda_checkpoint(source, tmp_path / "mlx", {"fixture": True})
    model, meta, optimizer = load_checkpoint(dest, {"fixture": True})
    assert source.read_bytes() == original
    assert meta["step"] == 7 and not meta["optimizer"] and optimizer is None
    assert meta["loader"] is None and meta["training"] is None
    assert meta["run"]["source_format"] == "nanochat-cuda-hybrid-v1"
    for name, value in tree_flatten(model.parameters()):
        np.testing.assert_array_equal(
            np.array(value), source_model.state_dict()[name].numpy()
        )
    ids = np.array([[0, 7, 3, 8, 2]], dtype=np.int32)
    with torch.no_grad():
        expected = source_model(torch.tensor(ids, dtype=torch.long)).numpy()
    np.testing.assert_allclose(np.array(model(mx.array(ids))), expected, atol=3e-6)
    with pytest.raises(ValueError, match="optimizer"):
        load_checkpoint(dest, load_optimizer=True)
    with pytest.raises(FileExistsError):
        import_cuda_checkpoint(source, tmp_path / "mlx", {"fixture": True})
    with pytest.raises(ValueError, match="tokenizer"):
        import_cuda_checkpoint(source, tmp_path / "wrong", {"fixture": False})


@pytest.mark.parametrize("corruption", ["names", "dtype", "nan", "format"])
def test_import_rejects_corrupt_checkpoint(tmp_path, corruption):
    model = HybridLM(tiny_config())
    arrays = dict(tree_flatten(model.parameters()))
    meta = {
        "format": "nanochat-cuda-hybrid-v1",
        "step": 1,
        "model": model.config.to_dict(),
        "tokenizer": {},
    }
    if corruption == "names":
        del arrays["wte.weight"]
    elif corruption == "dtype":
        arrays["wte.weight"] = arrays["wte.weight"].astype(mx.bfloat16)
    elif corruption == "nan":
        arrays["wte.weight"] = mx.full(arrays["wte.weight"].shape, float("nan"))
    else:
        meta["format"] = "legacy"
    path = tmp_path / "source.json"
    path.write_text(json.dumps(meta))
    mx.save_safetensors(str(path.with_suffix(".safetensors")), arrays)
    with pytest.raises(ValueError):
        import_cuda_checkpoint(path, tmp_path / "out", {})

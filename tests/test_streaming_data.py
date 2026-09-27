"""Real HF iterable/parquet checks on tiny local corpora; no model or network."""

from dataclasses import asdict
import json
from types import SimpleNamespace

import datasets
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nanochat_mlx.hybrid.data import prepare_documents, TokenDataset
from nanochat_mlx.hybrid.streaming import (
    StreamConfig,
    StreamingTokenDataset,
    open_streaming_datasets,
)
from nanochat_mlx.hybrid.training import (
    build_parser,
    resolve_plan,
    build_datasets,
    main,
)


class CharacterTokenizer:
    contract = {"eos": 1, "vocab_size": 256, "fixture": "character"}

    def encode(self, text):
        return list(text.encode("ascii"))

    def get_vocab_size(self):
        return self.contract["vocab_size"]


@pytest.fixture
def corpus(tmp_path):
    # Several parquet shards / row groups exercise HF's resume implementation.
    texts = ["held out alpha" * 3, "held out beta" * 3] + [
        f"document {i}: " + "some words " * (i + 1) for i in range(6)
    ]
    files = []
    for i in range(0, len(texts), 2):
        path = tmp_path / f"shard-{i}.parquet"
        pq.write_table(pa.table({"text": texts[i : i + 2]}), path, row_group_size=1)
        files.append(str(path))
    return texts, files


def local_stream(files, tmp_path):
    return datasets.load_dataset(
        "parquet",
        data_files=files,
        split="train",
        streaming=True,
        cache_dir=str(tmp_path / "cache"),
    )


def source(config=None):
    return {
        "request": asdict(config or StreamConfig("test/corpus")),
        "revision": "a" * 40,
        "datasets_version": datasets.__version__,
    }


def test_stream_matches_mmap_eos_packing_across_epochs(corpus, tmp_path):
    texts, files = corpus
    tokenizer = CharacterTokenizer()
    prepare_documents(tmp_path / "prepared", texts, texts, tokenizer)
    mmap = TokenDataset(tmp_path / "prepared", "train", 17, batch_size=3)
    stream = StreamingTokenDataset(
        local_stream(files, tmp_path), tokenizer, source(), "train", 17, 3
    )
    for _ in range(30):
        for a, b in zip(mmap.next_numpy(), stream.next_numpy()):
            np.testing.assert_array_equal(a, b)
    assert stream.epoch == mmap.epoch > 0


def test_json_resume_preserves_pending_tokens_shards_and_epoch(corpus, tmp_path):
    _, files = corpus
    tokenizer = CharacterTokenizer()
    stream = StreamingTokenDataset(
        local_stream(files, tmp_path), tokenizer, source(), "train", 19, 2
    )
    for checkpoint_batch in (0, 1, 9, 25):
        stream.reset()
        for _ in range(checkpoint_batch):
            stream.next_numpy()
        state = json.loads(json.dumps(stream.state_dict()))
        restored = StreamingTokenDataset(
            local_stream(files, tmp_path), tokenizer, source(), "train", 19, 2, state
        )
        assert restored.state_dict() == state
        for _ in range(25):
            for a, b in zip(stream.next_numpy(), restored.next_numpy()):
                np.testing.assert_array_equal(a, b)
        assert restored.epoch == stream.epoch


def test_lazy_read_and_no_lost_lookahead_token():
    seen = []

    def documents():
        for i in range(100):
            seen.append(i)
            yield {"text": "a" * 100}

    stream = datasets.IterableDataset.from_generator(documents)
    loader = StreamingTokenDataset(stream, CharacterTokenizer(), source(), "train", 8)
    assert seen == []
    loader.next_numpy()
    assert seen == [0]
    assert len(loader.state_dict()["pending_tokens"]) == 101 - 8


@pytest.mark.parametrize(
    "change", ["sequence", "batch", "source", "tokenizer", "tokens", "epoch"]
)
def test_resume_rejects_incompatible_or_invalid_state(change, corpus, tmp_path):
    _, files = corpus
    stream = StreamingTokenDataset(
        local_stream(files, tmp_path), CharacterTokenizer(), source(), "train", 8
    )
    stream.next_numpy()
    state = stream.state_dict()
    if change == "sequence":
        state["contract"]["sequence_len"] = 9
    elif change == "batch":
        state["contract"]["batch_size"] = 2
    elif change == "source":
        state["contract"]["source"]["revision"] = "b" * 40
    elif change == "tokenizer":
        state["contract"]["tokenizer"]["eos"] = 2
    elif change == "tokens":
        state["pending_tokens"][0] = -1
    else:
        state["epoch"] = -1
    with pytest.raises(ValueError):
        StreamingTokenDataset(
            local_stream(files, tmp_path),
            CharacterTokenizer(),
            source(),
            "train",
            8,
            state=state,
        )


@pytest.mark.parametrize(
    "rows", [[], [{"text": "a"}], [{"wrong": "column"}], [{"text": None}]]
)
def test_empty_short_and_invalid_documents_fail(rows):
    stream = datasets.IterableDataset.from_generator(lambda: iter(rows))
    loader = StreamingTokenDataset(stream, CharacterTokenizer(), source(), "train", 8)
    with pytest.raises(ValueError, match="sequence_len|column"):
        loader.next_numpy()


def test_factory_holdout_revision_pin_resume_and_validation_reset(
    corpus, tmp_path, monkeypatch
):
    texts, files = corpus
    config = StreamConfig("test/corpus", val_documents=2)
    calls, revisions = [], []
    real_load = datasets.load_dataset

    def load(path, **kwargs):
        assert path == config.dataset and kwargs["streaming"] is True
        calls.append(kwargs)
        return real_load(
            "parquet",
            data_files=files,
            split="train",
            streaming=True,
            cache_dir=str(tmp_path / "cache"),
        )

    def info(self, path, revision):
        revisions.append(revision)
        return SimpleNamespace(sha="a" * 40)

    monkeypatch.setattr(datasets, "load_dataset", load)
    monkeypatch.setattr("huggingface_hub.HfApi.dataset_info", info)
    tokenizer = CharacterTokenizer()
    train, val = open_streaming_datasets(config, tokenizer, 8)
    first_val = val.next_numpy()
    expected_train = tokenizer.encode(texts[2])[:9]
    x, y = train.next_numpy()
    assert x[0].tolist() == expected_train[:-1]
    assert y[0].tolist() == expected_train[1:]
    state = json.loads(json.dumps(train.state_dict()))
    restored, _ = open_streaming_datasets(config, tokenizer, 8, state=state)
    assert revisions == [None]  # Resume never resolves moving main again.
    assert all(call["revision"] == "a" * 40 for call in calls)
    val.next_numpy()
    val.reset()
    for a, b in zip(first_val, val.next_numpy()):
        np.testing.assert_array_equal(a, b)
    for _ in range(20):
        val.next_numpy()  # Cannot corrupt the training cursor.
        for a, b in zip(train.next_numpy(), restored.next_numpy()):
            np.testing.assert_array_equal(a, b)
    state["contract"]["source"]["datasets_version"] = "changed"
    with pytest.raises(ValueError, match="version mismatch"):
        open_streaming_datasets(config, tokenizer, 8, state=state)


def test_explicit_splits_and_loader_integration(corpus, tmp_path, monkeypatch):
    _, files = corpus
    args = build_parser().parse_args(
        [
            "--stream-dataset",
            "test/corpus",
            "--stream-val-split",
            "validation",
            "--context-length",
            "8",
        ]
    )
    plan = resolve_plan(args)
    calls = []
    real_load = datasets.load_dataset

    def load(path, **kwargs):
        calls.append(kwargs["split"])
        return real_load(
            "parquet",
            data_files=files,
            split="train",
            streaming=True,
            cache_dir=str(tmp_path / "cache"),
        )

    monkeypatch.setattr(datasets, "load_dataset", load)
    monkeypatch.setattr(
        "huggingface_hub.HfApi.dataset_info",
        lambda *a, **k: SimpleNamespace(sha="a" * 40),
    )
    train, val = build_datasets(args, plan, CharacterTokenizer())
    assert calls == ["train", "validation"]
    assert train.next_numpy()[0].shape == val.next_numpy()[0].shape == (1, 8)
    # Input modes cannot be accidentally interchanged on resume.
    local_args = build_parser().parse_args(["--data-dir", str(tmp_path)])
    with pytest.raises(ValueError, match="original --stream-dataset"):
        build_datasets(
            local_args,
            resolve_plan(local_args),
            CharacterTokenizer(),
            train.state_dict(),
        )
    with pytest.raises(ValueError, match="streaming loader"):
        build_datasets(args, plan, CharacterTokenizer(), {"cursor": 0})


def test_stream_dry_run_offline_and_invalid_combinations(monkeypatch, capsys):
    def fail(*a, **k):
        pytest.fail("Dry-run must not access the network or load datasets")

    monkeypatch.setattr(datasets, "load_dataset", fail)
    monkeypatch.setattr("huggingface_hub.HfApi.dataset_info", fail)
    base = ["--stream-dataset", "karpathy/fineweb-edu-100b-shuffle"]
    assert main(base + ["--dry-run"]) == 0
    assert '"training_started": false' in capsys.readouterr().out
    for extra in (
        ["--data-dir", "/tmp/unused"],
        ["--recipe", "configs/gdn_swa_32k_memory.json"],
        ["--source", "sft", "--init-from", "/tmp/unused.json"],
        ["--stream-val-split", "train"],
        ["--stream-val-documents", "0"],
    ):
        with pytest.raises(ValueError):
            resolve_plan(build_parser().parse_args(base + extra))

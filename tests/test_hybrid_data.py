"""Short data/control-plane checks; no model training or 32K execution."""

import json
import os
from pathlib import Path
import numpy as np
import pytest

from nanochat_mlx.hybrid.data import (
    prepare_documents,
    prepare_conversations,
    TokenDataset,
)
from nanochat_mlx.hybrid.tokenizer import DeepSeekTokenizer
from nanochat_mlx.hybrid.config import HybridConfig
from nanochat_mlx.hybrid.synthetic import (
    make_record,
    prepare_synthetic,
    TASKS,
    PROFILES,
)
from nanochat_mlx.hybrid.training import build_parser, resolve_plan, main


@pytest.fixture
def tokenizer():
    filename = os.environ.get("NANOCHAT_TEST_TOKENIZER")
    if not filename:
        pytest.skip(
            "Set NANOCHAT_TEST_TOKENIZER to a locally installed official tokenizer"
        )
    return DeepSeekTokenizer(filename)


def test_all_recipes_dry_run_preserve_depths(capsys):
    for recipe in Path("configs").glob("*.json"):
        for depth in (4, 12, 20, 26):
            args = build_parser().parse_args(
                ["--recipe", str(recipe), "--depth", str(depth)]
            )
            plan = resolve_plan(args)
            assert plan["model"]["n_layer"] == depth
            assert plan["model"]["max_context"] == 32768
            assert (
                plan["gradient_accumulation"] * plan["model"]["sequence_len"] == 131072
            )
            assert plan["training_started"] is False
    assert main(["--recipe", "configs/gdn_swa_32k_memory.json", "--dry-run"]) == 0
    assert '"sequence_len": 32768' in capsys.readouterr().out


def test_invalid_batch_and_legacy_mode():
    with pytest.raises(ValueError, match="exact multiple"):
        resolve_plan(build_parser().parse_args(["--total-batch-size", "1"]))
    for mode in ("legacy", "gdn_only", "swa_only", "full_attn"):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--architecture", mode])
        with pytest.raises(ValueError, match="Only the gdn_swa"):
            HybridConfig(architecture=mode)


def test_uint32_data_eos_split_and_exact_resume(tmp_path, tokenizer):
    prepare_documents(
        tmp_path, ["Hello world.你好！" * 20], ["Validation only." * 30], tokenizer
    )
    tokens = np.fromfile(tmp_path / "train.bin", dtype="<u4")
    assert tokens[-1] == 1
    assert tokens.tolist() == tokenizer.encode("Hello world.你好！" * 20) + [1]
    dataset = TokenDataset(tmp_path, "train", 8, tokenizer_contract=tokenizer.contract)
    x, y = dataset.next_numpy()
    assert np.array_equal(x[:, 1:], y[:, :-1])
    state = dataset.state_dict()
    resumed = TokenDataset(
        tmp_path, "train", 8, state=state, tokenizer_contract=tokenizer.contract
    )
    for a, b in zip(dataset.next_numpy(), resumed.next_numpy()):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="mismatch"):
        TokenDataset(tmp_path, "train", 9, state=state)
    with (tmp_path / "train.bin").open("ab") as f:
        f.write(b"xxxx")
    with pytest.raises(ValueError, match="fingerprint"):
        TokenDataset(tmp_path, "train", 8)


def test_official_conversation_masks_and_sft_padding(tmp_path, tokenizer):
    conversation = {
        "messages": [
            {"role": "system", "content": "Answer briefly."},
            {"role": "user", "content": "你好，2+2=?"},
            {"role": "assistant", "content": "4。"},
        ]
    }
    ids, mask = tokenizer.render_conversation(conversation)
    assert (
        tokenizer.decode([i for i, m in zip(ids, mask) if m])
        == "4。<｜end▁of▁sentence｜>"
    )
    assert ids == tokenizer.apply_chat_template(conversation["messages"])
    prepare_conversations(tmp_path, [conversation], [conversation], tokenizer, 64)
    dataset = TokenDataset(tmp_path, "train", 64, tokenizer_contract=tokenizer.contract)
    x, y = dataset.next_numpy()
    assert tokenizer.decode(y[0][y[0] != -1].tolist()) == "4。<｜end▁of▁sentence｜>"
    assert x[0, -1] == tokenizer.contract["pad"]
    assert np.all(y[0, len(ids) - 1 :] == -1)


def test_thinking_and_tool_supervision(tokenizer):
    messages = [
        {
            "role": "system",
            "content": "Use tools.",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Lookup",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        },
        {"role": "user", "content": "Find value"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "Use lookup.",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "external-result-42"},
        {"role": "assistant", "content": "42", "reasoning_content": "Read result."},
    ]
    ids, mask = tokenizer.render_conversation(
        {"messages": messages, "thinking_mode": "thinking", "reasoning_effort": 75}
    )
    text = tokenizer.decode([i for i, m in zip(ids, mask) if m])
    assert "external-result-42" not in text
    assert "<｜DSML｜ calls>" in text
    assert "Use lookup." in text
    assert "Reasoning Effort:" not in text
    assert ids == tokenizer.apply_chat_template(
        messages, thinking_mode="thinking", reasoning_effort=75
    )


@pytest.mark.parametrize("kind", TASKS)
def test_memory_record_short_context(kind, tokenizer):
    ids, labels, info = make_record(tokenizer, 512, 128, kind, 42, key_count=4)
    assert ids.dtype == np.dtype("<u4") and labels.dtype == np.dtype("<i4")
    assert len(ids) == 513
    assert info["query_start"] - info["fact_end"] == 128
    assert np.all(labels[: info["answer_start"]] == -1)
    assert (
        tokenizer.decode(labels[labels != -1].tolist())
        == info["answer"] + "<｜end▁of▁sentence｜>"
    )
    assert tokenizer.decode(ids[:2].tolist()) == "<｜begin▁of▁sentence｜><｜User｜>"
    assert "</think>" in tokenizer.decode(
        ids[info["query_start"] : info["answer_start"]].tolist()
    )
    if kind == "overwrite":
        assert len(info["fact_spans"]) == 3
        assert info["fact_spans"][-1][0] > info["fact_spans"][-2][1] + 1


def test_32k_profile_metadata_only():
    assert PROFILES["32k"]["sequence_len"] == 32768
    assert all(d in PROFILES["32k"]["distances"] for d in (16384, 24576, 30720))


def test_synthetic_coverage_skips_infeasible_gaps(tmp_path, tokenizer, monkeypatch):
    monkeypatch.setitem(
        PROFILES, "short-test", {"sequence_len": 512, "distances": [512, 128]}
    )
    prepare_synthetic(
        tmp_path, tokenizer, "short-test", train_examples=12, val_examples=12
    )
    coverage = json.loads((tmp_path / "coverage.json").read_text())
    assert ["recall", 512, 1] in coverage["skipped_infeasible_combinations"]
    ledgers = {}
    for split in ("train", "val"):
        records = [
            json.loads(line)
            for line in (tmp_path / f"{split}.tasks.jsonl").read_text().splitlines()
        ]
        assert len(records) == 12
        assert all(
            r["distance"] == r["query_start"] - r["fact_end"] == 128 for r in records
        )
        assert np.fromfile(tmp_path / f"{split}.bin", dtype="<u4").size == 12 * 513
        ledgers[split] = {r["seed"] for r in records}
    assert ledgers["train"].isdisjoint(ledgers["val"])


def test_images_fail_before_tokenization(tokenizer):
    with pytest.raises(ValueError, match="text-only"):
        tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "unused.png"}}
                    ],
                }
            ]
        )


def test_oversized_sft_fails_instead_of_stalling(tmp_path, tokenizer):
    conversation = {
        "messages": [
            {"role": "user", "content": "a long message " * 40},
            {"role": "assistant", "content": "reply"},
        ]
    }
    with pytest.raises(ValueError, match="exceeding"):
        prepare_conversations(tmp_path, [conversation], [conversation], tokenizer, 16)

"""Hybrid-only API checks with untrained, short-context model fixtures."""

import os
import pytest
from fastapi.testclient import TestClient

from scripts import quickstart
from nanochat_mlx.hybrid.config import config_for_depth
from nanochat_mlx.hybrid.model import HybridLM
from nanochat_mlx.hybrid.checkpoint import save_checkpoint, checkpoint_directory
from nanochat_mlx.hybrid.tokenizer import DeepSeekTokenizer
from tests.test_hybrid import tiny_config


def test_config_and_job_commands_never_train_by_default(monkeypatch, tmp_path):
    monkeypatch.setattr(quickstart, "BASE", tmp_path)
    client = TestClient(quickstart.app)
    r = client.get("/config?depth=20&context=32768")
    assert r.status_code == 200
    assert r.json()["model"]["n_layer"] == 20
    assert r.json()["model"]["sequence_len"] == 32768
    assert r.json()["training_started"] is False
    assert client.get("/config?context=131072").status_code == 400
    for stage in ("train", "sft"):
        req = quickstart.JobRequest(
            stage=stage, init_from="fixture.json" if stage == "sft" else None
        )
        cmd = quickstart.job_command(req)
        assert "--dry-run" in cmd and "--start-training" not in cmd
    with pytest.raises(ValueError, match="required"):
        quickstart.job_command(quickstart.JobRequest(stage="train", execute=True))
    assert "Hybrid MLX Workbench" in client.get("/").text
    assert client.get("/status").json()["checkpoints"] == []


def test_api_load_short_chat_unload(monkeypatch, tmp_path):
    directory = os.environ.get("NANOCHAT_TEST_TOKENIZER")
    if not directory:
        pytest.skip("Official tokenizer must be installed locally")
    tokenizer = DeepSeekTokenizer(directory)
    model = HybridLM(tiny_config(vocab_size=tokenizer.get_vocab_size()))
    checkpoint = save_checkpoint(
        checkpoint_directory(tmp_path, model.config), model, 0, tokenizer.contract
    )
    monkeypatch.setattr(quickstart, "BASE", tmp_path)
    monkeypatch.setattr(quickstart, "loaded_engine", None)
    monkeypatch.setattr(quickstart, "loaded_metadata", None)
    with TestClient(quickstart.app) as client:
        assert (
            client.post(
                "/chat/completions",
                json={"messages": [{"role": "user", "content": "hello"}]},
            ).status_code
            == 400
        )
        r = client.post(
            "/chat/load",
            json={"checkpoint": str(checkpoint), "tokenizer_dir": directory},
        )
        assert r.status_code == 200, r.text
        assert client.get("/status").json()["chat"] is True
        assert len(client.get("/checkpoints").json()) == 1
        r = client.post(
            "/chat/completions",
            json={
                "messages": [{"role": "user", "content": "你好"}],
                "max_tokens": 3,
                "temperature": 0,
            },
        )
        assert r.status_code == 200, r.text
        assert '"done": true' in r.text
        r = client.post(
            "/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "thinking_mode": "invalid",
                "max_tokens": 3,
            },
        )
        assert r.status_code == 400
        assert client.post("/chat/unload", json={}).json()["status"] == "unloaded"


def test_streaming_job_can_inspect_without_prepared_data():
    from nanochat_mlx.hybrid.training import build_parser, resolve_plan

    req = quickstart.JobRequest(
        stage="train", stream_dataset="test/corpus", stream_val_documents=16
    )
    cmd = quickstart.job_command(req)
    assert "--data-dir" not in cmd and "--dry-run" in cmd
    plan = resolve_plan(build_parser().parse_args(cmd[3:]))
    assert plan["streaming"]["dataset"] == "test/corpus"
    assert plan["streaming"]["val_documents"] == 16
    req.execute = True
    assert "--start-training" in quickstart.job_command(req)
    req.data_dir = "/tmp/unused"
    with pytest.raises(ValueError, match="without data_dir"):
        quickstart.job_command(req)
    req.data_dir, req.stage, req.init_from = None, "sft", "/tmp/checkpoint.json"
    with pytest.raises(ValueError, match="pretraining"):
        quickstart.job_command(req)

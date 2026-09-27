"""Local hybrid model workbench: configuration, explicit jobs and DeepSeek chat."""

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from nanochat_mlx.hybrid.config import config_for_depth

BASE = Path(
    os.environ.get("NANOCHAT_BASE_DIR", os.path.expanduser("~/.cache/nanochat"))
)
ROOT = Path(__file__).resolve().parents[1]
app = FastAPI()
running_process = None
loaded_engine = None
loaded_metadata = None
job_lock = asyncio.Lock()
chat_lock = asyncio.Lock()
memory_limit_gb = 8.0


def available_checkpoints():
    items = []
    for filename in sorted((BASE / "hybrid_checkpoints").glob("*/*/*/step_*.json")):
        try:
            meta = json.loads(filename.read_text())
            if (
                meta.get("format") == "nanochat-mlx-hybrid-v1"
                and filename.with_suffix(".safetensors").is_file()
            ):
                items.append(
                    {
                        "checkpoint": str(filename),
                        "depth": meta["model"]["n_layer"],
                        "architecture": meta["model"]["architecture"],
                        "step": meta["step"],
                        "context": meta["model"]["max_context"],
                        "source": filename.parent.name,
                    }
                )
        except (OSError, ValueError, KeyError):
            continue
    return items


@app.get("/")
async def root():
    return HTMLResponse((ROOT / "nanochat_mlx" / "quickstart_ui.html").read_text())


@app.get("/status")
async def status():
    return {
        "tokenizer": (BASE / "deepseek_tokenizer" / "contract.json").is_file(),
        "checkpoints": available_checkpoints(),
        "chat": loaded_engine is not None,
        "model": loaded_metadata,
        "running": running_process is not None and running_process.returncode is None,
    }


@app.get("/checkpoints")
async def checkpoints():
    return available_checkpoints()


@app.get("/config")
async def config(depth: int = 12, context: int = 4096):
    try:
        model = config_for_depth(depth, sequence_len=context)
        return {
            "model": model.to_dict(),
            "parameter_counts": model.parameter_counts(),
            "training_started": False,
        }
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


class JobRequest(BaseModel):
    stage: str
    depth: int = Field(default=12, ge=1)
    context: int = Field(default=4096, ge=1, le=32768)
    data_dir: str | None = None
    stream_dataset: str | None = None
    stream_name: str | None = None
    stream_revision: str | None = None
    stream_train_split: str = "train"
    stream_val_split: str | None = None
    stream_val_documents: int = Field(default=1024, gt=0)
    stream_text_column: str = "text"
    output_dir: str | None = None
    train_text: str | None = None
    val_text: str | None = None
    train_chat: str | None = None
    val_chat: str | None = None
    init_from: str | None = None
    profile: str = "4k"
    iterations: int | None = Field(default=None, gt=0)
    execute: bool = False


def job_command(req):
    tokenizer = str(BASE / "deepseek_tokenizer")
    if req.stage == "tokenizer":
        return [
            sys.executable,
            "-m",
            "scripts.prepare_hybrid",
            "--tokenizer-dir",
            tokenizer,
            "--install-tokenizer",
        ]
    if req.stage == "prepare":
        if not req.output_dir:
            raise ValueError("output_dir is required")
        cmd = [
            sys.executable,
            "-m",
            "scripts.prepare_hybrid",
            "--tokenizer-dir",
            tokenizer,
            "--output",
            req.output_dir,
        ]
        if req.train_chat or req.val_chat:
            if not req.train_chat or not req.val_chat:
                raise ValueError(
                    "Both training and validation conversation paths are required"
                )
            cmd += [
                "--train-chat",
                req.train_chat,
                "--val-chat",
                req.val_chat,
                "--context-length",
                str(req.context),
            ]
        elif req.train_text or req.val_text:
            if not req.train_text or not req.val_text:
                raise ValueError("Both training and validation text paths are required")
            cmd += ["--train-text", req.train_text, "--val-text", req.val_text]
        else:
            if req.profile not in ("4k", "32k"):
                raise ValueError("profile must be 4k or 32k")
            cmd += ["--synthetic", req.profile]
        return cmd
    if req.stage in ("train", "sft"):
        if req.execute and not (req.data_dir or req.stream_dataset):
            raise ValueError("data_dir or stream_dataset is required for training")
        if req.stream_dataset:
            from nanochat_mlx.hybrid.streaming import StreamConfig

            if req.data_dir or req.stage == "sft":
                raise ValueError("Streaming text requires pretraining without data_dir")
            StreamConfig(
                dataset=req.stream_dataset,
                name=req.stream_name,
                revision=req.stream_revision,
                train_split=req.stream_train_split,
                val_split=req.stream_val_split,
                val_documents=req.stream_val_documents,
                text_column=req.stream_text_column,
            )
        if req.stage == "sft" and not req.init_from:
            raise ValueError("SFT requires an init_from checkpoint")
        cmd = [
            sys.executable,
            "-m",
            "scripts." + req.stage,
            "--depth",
            str(req.depth),
            "--context-length",
            str(req.context),
            "--tokenizer-dir",
            tokenizer,
            "--output-dir",
            str(BASE),
            "--memory-limit-gb",
            str(memory_limit_gb),
            "--checkpoint-blocks",
        ]
        if req.data_dir:
            cmd += ["--data-dir", req.data_dir]
        if req.stream_dataset:
            for field in (
                "dataset",
                "name",
                "revision",
                "train_split",
                "val_split",
                "val_documents",
                "text_column",
            ):
                value = getattr(req, "stream_" + field)
                if value is not None:
                    cmd += ["--stream-" + field.replace("_", "-"), str(value)]
        if req.init_from:
            cmd += ["--init-from", req.init_from]
        if req.iterations:
            cmd += ["--num-iterations", str(req.iterations)]
        cmd += ["--start-training" if req.execute else "--dry-run"]
        return cmd
    raise ValueError("Unknown stage")


@app.post("/run")
async def run(req: JobRequest):
    global running_process
    try:
        cmd = job_command(req)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    async with job_lock:
        if running_process is not None and running_process.returncode is None:
            raise HTTPException(409, "A process is already running")
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        running_process = process

    async def stream():
        global running_process
        try:
            async for line in process.stdout:
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "type": "output",
                            "text": line.decode(errors="replace").rstrip(),
                        }
                    )
                    + "\n\n"
                )
            code = await process.wait()
            yield (
                "data: "
                + json.dumps({"type": "done" if code == 0 else "error", "code": code})
                + "\n\n"
            )
        finally:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), 5)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
            if running_process is process:
                running_process = None

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/stop")
async def stop():
    process = running_process
    if process is not None and process.returncode is None:
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), 5)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
    return {"status": "stopped"}


class LoadRequest(BaseModel):
    checkpoint: str
    tokenizer_dir: str | None = None


@app.post("/chat/load")
async def load(req: LoadRequest):
    global loaded_engine, loaded_metadata
    from nanochat_mlx.hybrid.tokenizer import DeepSeekTokenizer
    from nanochat_mlx.hybrid.checkpoint import load_checkpoint
    from nanochat_mlx.hybrid.engine import HybridEngine
    from nanochat_mlx.common import set_memory_limit

    async with chat_lock:
        try:
            set_memory_limit(memory_limit_gb)
            tokenizer = DeepSeekTokenizer(
                req.tokenizer_dir or BASE / "deepseek_tokenizer"
            )
            model, metadata, _ = load_checkpoint(req.checkpoint, tokenizer.contract)
            loaded_engine = HybridEngine(model, tokenizer)
            loaded_metadata = {
                "checkpoint": req.checkpoint,
                "depth": model.config.n_layer,
                "max_context": model.config.max_context,
                "step": metadata["step"],
            }
        except (OSError, ValueError, KeyError) as exc:
            raise HTTPException(400, str(exc)) from exc
    return {"status": "loaded", **loaded_metadata}


@app.post("/chat/unload")
async def unload():
    global loaded_engine, loaded_metadata
    async with chat_lock:
        loaded_engine, loaded_metadata = None, None
    return {"status": "unloaded"}


class ChatRequest(BaseModel):
    messages: list[dict]
    temperature: float = Field(default=0.8, ge=0)
    max_tokens: int = Field(default=256, ge=1, le=32768)
    top_k: int = Field(default=50, ge=0)
    thinking_mode: str = "chat"
    reasoning_effort: int = Field(default=75, ge=1, le=100)


@app.post("/chat/completions")
async def chat(req: ChatRequest):
    engine = loaded_engine
    if engine is None:
        raise HTTPException(400, "Load a hybrid checkpoint first")
    try:
        tokens = engine.tokenizer.apply_chat_template(
            req.messages,
            thinking_mode=req.thinking_mode,
            reasoning_effort=req.reasoning_effort,
        )
        if len(tokens) + req.max_tokens > engine.model.config.max_context:
            raise ValueError(
                "Prompt and generation exceed the configured context limit"
            )
    except (ValueError, AssertionError, NotImplementedError) as exc:
        raise HTTPException(400, str(exc)) from exc

    async def stream():
        async with chat_lock:
            output, last = [], ""
            for column, _ in engine.generate(
                tokens,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
            ):
                if column[0] == engine.tokenizer.contract["eos"]:
                    break
                output.append(column[0])
                text = engine.tokenizer.decode(output)
                if not text.endswith("\ufffd"):
                    if text[len(last) :]:
                        yield (
                            "data: "
                            + json.dumps(
                                {"token": text[len(last) :]}, ensure_ascii=False
                            )
                            + "\n\n"
                        )
                    last = text
                await asyncio.sleep(0)
            yield 'data: {"done": true}\n\n'

    return StreamingResponse(stream(), media_type="text/event-stream")


def main(argv=None):
    global memory_limit_gb
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--memory-limit-gb", type=float, default=8)
    args = p.parse_args(argv)
    memory_limit_gb = args.memory_limit_gb
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

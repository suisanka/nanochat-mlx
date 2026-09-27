"""Evaluate an MLX base checkpoint on local raw-text JSONL, without a chat template."""

import argparse
import json
import math
import platform
import resource
import sys
import time
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokenizer-dir", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True, help='JSONL with a "text" field')
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--context-length", type=int, default=1024)
    p.add_argument("--stride", type=int, default=512)
    p.add_argument("--loss-chunk-size", type=int, default=128)
    p.add_argument("--max-documents", type=int, default=0, help="0 means all documents")
    p.add_argument("--memory-limit-gb", type=float, default=8)
    args = p.parse_args(argv)
    if args.max_documents < 0:
        p.error("max-documents must be nonnegative")
    if not 1 <= args.stride <= args.context_length or args.loss_chunk_size <= 0:
        p.error("Require 1 <= stride <= context-length and a positive loss chunk")
    if args.output.exists() or args.output.with_suffix(".samples.jsonl").exists():
        p.error("Evaluation output already exists; choose a new output path")
    import mlx.core as mx

    from nanochat_mlx.common import get_peak_memory_mb, set_memory_limit
    from nanochat_mlx.hybrid.checkpoint import file_sha256, load_checkpoint
    from nanochat_mlx.hybrid.evaluation import score_text
    from nanochat_mlx.hybrid.tokenizer import DeepSeekTokenizer

    set_memory_limit(args.memory_limit_gb, cache_gb=1)
    tokenizer = DeepSeekTokenizer(args.tokenizer_dir)
    model, meta, _ = load_checkpoint(args.checkpoint, tokenizer.contract)
    model.config = replace(model.config, checkpoint_blocks=False)
    model.eval()
    if args.context_length > model.config.max_context:
        p.error("Requested context exceeds model limit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    nll, tokens, size, documents = 0.0, 0, 0, 0
    samples = args.output.with_suffix(".samples.jsonl")
    with args.data.open() as source, samples.open("x") as dest:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            if args.max_documents and documents >= args.max_documents:
                break
            row = json.loads(line)
            result = score_text(
                model,
                tokenizer,
                row["text"],
                context_length=args.context_length,
                stride=args.stride,
                loss_chunk_size=args.loss_chunk_size,
            )
            record = {"line": line_number, "id": row.get("id", line_number), **result}
            dest.write(json.dumps(record) + "\n")
            dest.flush()
            nll += result["nll_sum"]
            tokens += result["tokens"]
            size += result["bytes"]
            documents += 1
            print(
                json.dumps(
                    {
                        "documents": documents,
                        "tokens": tokens,
                        "bpb": nll / (math.log(2) * size),
                    }
                ),
                flush=True,
            )
    if not documents:
        raise ValueError("No evaluation documents")
    mx.synchronize()
    elapsed = time.monotonic() - start
    report = {
        "format": "nanochat-base-eval-v1",
        "checkpoint_step": meta["step"],
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "weights_sha256": file_sha256(args.checkpoint.with_suffix(".safetensors")),
        "tokenizer": tokenizer.contract,
        "dataset_path": str(args.data.resolve()),
        "dataset_sha256": file_sha256(args.data),
        "model": meta["model"],
        "runtime": {
            "mlx_version": version("mlx"),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "checkpoint_blocks": False,
        },
        "protocol": {
            "chat_template": False,
            "initial_context": "eos",
            "scored_eos": False,
            "context_length": args.context_length,
            "stride": args.stride,
            "loss_chunk_size": args.loss_chunk_size,
            "byte_encoding": "utf-8",
            "document_state_reset": True,
        },
        "documents": documents,
        "tokens": tokens,
        "bytes": size,
        "nll_sum": nll,
        "mean_token_nll": nll / tokens,
        "token_perplexity": math.exp(nll / tokens),
        "bpb": nll / (math.log(2) * size),
        "elapsed_seconds_including_first_call": elapsed,
        "tokens_per_second_including_first_call": tokens / elapsed,
        "mlx_peak_memory_mib": get_peak_memory_mb(),
        "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024**2 if sys.platform == "darwin" else 1024),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Evaluate prepared synthetic memory records by task/distance/key count."""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np


def main(argv=None, *, backend="mlx"):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--tokenizer-dir", required=True, type=Path)
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--max-problems", type=int, default=100)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--output", required=True, type=Path)
    if backend == "mlx":
        p.add_argument("--memory-limit-gb", type=float, default=8)
    elif backend == "cuda":
        p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        p.add_argument("--gdn-backend", choices=("fla", "reference"), default="fla")
        p.add_argument("--attention-backend", choices=("sdpa", "flash"), default="sdpa")
    else:
        raise ValueError("Unknown evaluation backend")
    args = p.parse_args(argv)
    from nanochat_mlx.hybrid.tokenizer import DeepSeekTokenizer
    from nanochat_mlx.hybrid.data import TokenDataset

    tokenizer = DeepSeekTokenizer(args.tokenizer_dir)
    if backend == "mlx":
        from nanochat_mlx.hybrid.checkpoint import load_checkpoint
        from nanochat_mlx.hybrid.engine import HybridEngine
        from nanochat_mlx.common import set_memory_limit

        set_memory_limit(args.memory_limit_gb)
        model, _, _ = load_checkpoint(args.checkpoint, tokenizer.contract)
    else:
        from nanochat_cuda.checkpoint import load_checkpoint
        from nanochat_cuda.engine import HybridEngine

        model, _, _ = load_checkpoint(
            args.checkpoint,
            tokenizer.contract,
            args.device,
            args.gdn_backend,
            args.attention_backend,
        )
    engine = HybridEngine(model, tokenizer)
    meta = json.loads((args.data_dir / "meta.json").read_text())
    record_length = meta["record_length"]
    if not record_length:
        raise ValueError("Synthetic evaluation requires fixed-length records")
    dataset = TokenDataset(
        args.data_dir, "val", record_length - 1, tokenizer_contract=tokenizer.contract
    )
    buckets = defaultdict(lambda: {"correct": 0, "total": 0})
    results = []
    with (args.data_dir / "val.tasks.jsonl").open() as ledger:
        for i, line in enumerate(ledger):
            if i >= args.max_problems:
                break
            task = json.loads(line)
            start = i * record_length
            prompt = (
                dataset.tokens[start : start + task["answer_start"]]
                .astype(np.int32)
                .tolist()
            )
            budget = min(args.max_tokens, model.config.max_context - len(prompt))
            if budget <= 0:
                raise ValueError(
                    "Checkpoint context limit leaves no room for the answer"
                )
            generated, _ = engine.generate_batch(
                prompt, max_tokens=budget, temperature=0
            )
            answer = tokenizer.decode(generated[0][len(prompt) :]).strip()
            correct = answer == task["answer"]
            key = f"{task['task']}/distance={task['distance']}/keys={task['key_count']}"
            buckets[key]["correct"] += int(correct)
            buckets[key]["total"] += 1
            results.append({**task, "prediction": answer, "correct": correct})
    args.output.write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "buckets": dict(buckets),
                "examples": results,
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

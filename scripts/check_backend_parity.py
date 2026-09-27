"""Check an imported MLX checkpoint against saved short CUDA numerical references."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokenizer-dir", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if args.output.exists():
        p.error("Output exists; choose a new path")
    import mlx.core as mx
    import numpy as np

    from nanochat_mlx.common import set_memory_limit
    from nanochat_mlx.hybrid.checkpoint import file_sha256, load_checkpoint
    from nanochat_mlx.hybrid.tokenizer import DeepSeekTokenizer

    set_memory_limit(8, cache_gb=1)
    tokenizer = DeepSeekTokenizer(args.tokenizer_dir)
    model, meta, _ = load_checkpoint(args.checkpoint, tokenizer.contract)
    model.eval()
    reference = json.loads(args.reference.read_text())
    source_hash = (meta.get("run") or {}).get("source_weights_sha256")
    if not source_hash or reference.get("checkpoint_weights_sha256") != source_hash:
        raise ValueError("Reference weights fingerprint does not match imported source")
    if (
        reference["tokenizer"] != tokenizer.contract
        or reference["checkpoint_step"] != meta["step"]
    ):
        raise ValueError("Reference step/tokenizer does not match checkpoint")
    arrays = mx.load(str(args.reference.with_suffix(".safetensors")))
    # BF16 numerical smoke bounds, not bitwise equivalence or long-context proof.
    limits = {
        "logits_mean_abs_error": 0.03,
        "logits_max_abs_error": 0.3,
        "target_logprob_mean_abs_error": 0.05,
        "mean_nll_abs_error": 0.03,
        "cache_final_logits_max_abs_error": 0.3,
    }
    results = []
    for i, sample in enumerate(reference["samples"]):
        x = mx.array([sample["ids"][:-1]], dtype=mx.int32)
        y = mx.array([sample["ids"][1:]], dtype=mx.int32)
        logits = model(x)
        lp = mx.take_along_axis(
            logits - mx.logsumexp(logits, axis=-1, keepdims=True), y[..., None], axis=-1
        )[0, :, 0]
        selected = logits[0, sample["positions"]]
        expected = arrays[f"logits_{i}"]
        delta = np.abs(np.array(selected - expected))
        lpd = np.abs(np.array(lp - arrays[f"target_logprobs_{i}"]))
        cached = model.prefill(x, model.make_cache(), chunk_size=7)
        cd = np.abs(np.array(cached[0, -1] - logits[0, -1]))
        actual_nll, expected_nll = (
            -lp.mean().item(),
            sample["nll_sum"] / sample["tokens"],
        )
        result = {
            "sample": i,
            "tokens": sample["tokens"],
            "mlx_mean_nll": actual_nll,
            "cuda_mean_nll": expected_nll,
            "mean_nll_abs_error": abs(actual_nll - expected_nll),
            "logits_mean_abs_error": float(delta.mean()),
            "logits_max_abs_error": float(delta.max()),
            "target_logprob_mean_abs_error": float(lpd.mean()),
            "target_logprob_max_abs_error": float(lpd.max()),
            "greedy_agreement": float(
                np.mean(
                    np.argmax(np.array(selected), -1)
                    == np.argmax(np.array(expected), -1)
                )
            ),
            "cache_final_logits_max_abs_error": float(cd.max()),
        }
        result["passed"] = all(
            np.isfinite(result[k]) and result[k] <= limit for k, limit in limits.items()
        )
        results.append(result)
    report = {
        "scope": "Short BF16 numerical smoke check; not bitwise or long-context equivalence",
        "checkpoint_step": meta["step"],
        "weights_sha256": file_sha256(args.checkpoint.with_suffix(".safetensors")),
        "reference_sha256": file_sha256(args.reference),
        "reference_weights_sha256": file_sha256(
            args.reference.with_suffix(".safetensors")
        ),
        "limits": limits,
        "samples": results,
        "passed": bool(results) and all(r["passed"] for r in results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

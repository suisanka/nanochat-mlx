"""Hybrid training entry point. Inspection is the default; execution is explicit."""

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time

from .config import HybridConfig, TrainingConfig, config_for_depth


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--architecture",
        choices=["gdn_swa"],
        default=None,
    )
    p.add_argument("--recipe", type=Path)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--max-seq-len", "--context-length", type=int, default=None)
    p.add_argument("--max-context", type=int, default=None)
    p.add_argument("--window-size", type=int, default=None)
    p.add_argument("--rope-scaling", choices=["none", "linear", "yarn"], default=None)
    p.add_argument("--rope-factor", type=float, default=None)
    p.add_argument("--data-dir", type=Path)
    p.add_argument(
        "--stream-dataset", help="Hugging Face owner/dataset ID; tokenize online"
    )
    p.add_argument("--stream-name", help="Optional dataset configuration name")
    p.add_argument(
        "--stream-revision", help="Hub revision; resolved to a pinned commit at startup"
    )
    p.add_argument("--stream-train-split", default="train")
    p.add_argument(
        "--stream-val-split",
        help="Separate validation split, otherwise reserve first N documents",
    )
    p.add_argument("--stream-val-documents", type=int, default=1024)
    p.add_argument("--stream-text-column", default="text")
    p.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path(os.path.expanduser("~/.cache/nanochat/deepseek_tokenizer")),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            os.environ.get("NANOCHAT_BASE_DIR", os.path.expanduser("~/.cache/nanochat"))
        ),
    )
    p.add_argument("--device-batch-size", type=int, default=1)
    p.add_argument(
        "--total-batch-size", type=int, default=None, help="Tokens per optimizer step"
    )
    p.add_argument("--num-iterations", type=int, default=None)
    p.add_argument("--training-tokens", type=int, default=None)
    p.add_argument("--optimizer", choices=["muon", "adamw"], default=None)
    p.add_argument("--resume", type=Path)
    p.add_argument(
        "--init-from",
        type=Path,
        help="Warm-start weights for a new phase; optimizer starts fresh",
    )
    p.add_argument("--source", choices=["base", "sft"], default="base")
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-steps", type=int, default=5)
    p.add_argument("--diagnostics-every", type=int, default=100)
    p.add_argument("--memory-limit-gb", type=float, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-blocks", action="store_true", default=None)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved configuration without loading data or MLX",
    )
    p.add_argument(
        "--start-training",
        action="store_true",
        help="Explicitly execute training; otherwise only print the plan",
    )
    return p


def resolve_plan(args):
    recipe = json.loads(args.recipe.read_text()) if args.recipe else {}
    depth = args.depth if args.depth is not None else recipe.get("depth", 12)
    overrides = dict(recipe.get("model", {}))
    for name, attr in (
        ("architecture", "architecture"),
        ("sequence_len", "max_seq_len"),
        ("max_context", "max_context"),
        ("window_size", "window_size"),
        ("rope_scaling", "rope_scaling"),
        ("rope_factor", "rope_factor"),
        ("checkpoint_blocks", "checkpoint_blocks"),
    ):
        value = getattr(args, attr)
        if value is not None:
            overrides[name] = value
    model = config_for_depth(depth, **overrides)
    training = dict(recipe.get("training", {}))
    if args.total_batch_size is not None:
        training["tokens_per_step"] = args.total_batch_size
    if args.optimizer is not None:
        training["optimizer"] = args.optimizer
    training = TrainingConfig(**training)
    micro_tokens = args.device_batch_size * model.sequence_len
    if micro_tokens <= 0 or training.tokens_per_step % micro_tokens:
        raise ValueError(
            "tokens_per_step must be an exact multiple of device_batch_size * context length"
        )
    token_budget = (
        args.training_tokens
        if args.training_tokens is not None
        else recipe.get("training_tokens", 100_000_000)
    )
    steps = (
        args.num_iterations
        if args.num_iterations is not None
        else math.ceil(token_budget / training.tokens_per_step)
    )
    if steps <= 0:
        raise ValueError("Training requires a positive step/token budget")
    if args.resume and args.init_from:
        raise ValueError("Select --resume or --init-from, not both")
    if args.source == "sft" and not (args.resume or args.init_from):
        raise ValueError("SFT requires --init-from or --resume")
    stream = None
    if args.stream_dataset:
        from .streaming import StreamConfig

        if args.data_dir or args.source != "base" or recipe.get("data_profile"):
            raise ValueError(
                "Streaming text requires base training without --data-dir or a synthetic recipe"
            )
        stream = asdict(
            StreamConfig(
                dataset=args.stream_dataset,
                name=args.stream_name,
                revision=args.stream_revision,
                train_split=args.stream_train_split,
                val_split=args.stream_val_split,
                val_documents=args.stream_val_documents,
                text_column=args.stream_text_column,
            )
        )
    return {
        "model": model.to_dict(),
        "training": asdict(training),
        "steps": steps,
        "effective_tokens": steps * training.tokens_per_step,
        "gradient_accumulation": training.tokens_per_step // micro_tokens,
        "micro_batch": args.device_batch_size,
        "parameter_counts": model.parameter_counts(),
        "data_profile": "sft" if args.source == "sft" else recipe.get("data_profile"),
        "source": args.source,
        "streaming": stream,
        "training_started": False,
    }


def summarize_diagnostics(stats):
    result = {}
    for layer, values in stats.items():
        for name, value in values.items():
            result[f"layer/{layer}/{name}"] = {
                key: array.tolist() for key, array in value.items()
            }
    return result


def train(args, plan):
    # Intentionally imported only after --start-training, so dry-run cannot
    # initialize Metal, download anything, load a model or update parameters.
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_map
    from .model import HybridLM
    from .optim import HybridOptimizer, lr_multiplier
    from .tokenizer import DeepSeekTokenizer
    from .checkpoint import load_checkpoint, save_checkpoint, checkpoint_directory
    from nanochat_mlx.common import set_memory_limit

    if args.data_dir is None and plan["streaming"] is None:
        raise ValueError("--data-dir or --stream-dataset is required to start training")
    config, training = HybridConfig(**plan["model"]), TrainingConfig(**plan["training"])
    directory = checkpoint_directory(args.output_dir, config, args.source)
    if not args.resume and directory.exists() and any(directory.glob("step_*.json")):
        raise ValueError(
            "Output already has checkpoints; choose another --output-dir or use --resume"
        )
    tokenizer = DeepSeekTokenizer(args.tokenizer_dir)
    if config.vocab_size != tokenizer.get_vocab_size():
        raise ValueError("Model/tokenizer vocabulary mismatch")
    set_memory_limit(args.memory_limit_gb)
    mx.random.seed(args.seed)
    loader_state = None
    if args.resume:
        model, metadata, optimizer = load_checkpoint(
            args.resume, tokenizer.contract, load_optimizer=True
        )
        if model.config != config or metadata["training"] != plan["training"]:
            raise ValueError("Resume requires identical model and training settings")
        loader_state = metadata["loader"]
        if loader_state is None:
            raise ValueError("Resume checkpoint is missing data position")
        if metadata.get("run", {}) != {
            "steps": plan["steps"],
            "micro_batch": args.device_batch_size,
        }:
            raise ValueError(
                "Resume requires identical total steps and micro batch size"
            )
    else:
        model = HybridLM(config)
        if args.init_from:
            previous, metadata, _ = load_checkpoint(args.init_from, tokenizer.contract)
            # Context extension may change position/cache settings, not tensor shapes.
            source = dict(tree_flatten(previous.parameters()))
            if {p: v.shape for p, v in source.items()} != {
                p: v.shape for p, v in tree_flatten(model.parameters())
            }:
                raise ValueError("Warm-start parameter shapes do not match")
            model.load_weights(list(source.items()), strict=True)
            del previous
        optimizer = HybridOptimizer(model, training)
    loader, val = build_datasets(args, plan, tokenizer, loader_state)
    if plan["streaming"] is not None:
        print(json.dumps({"data_stream": loader.contract["source"]}), flush=True)
    if (
        plan["data_profile"] is not None
        and loader.meta.get("scenario_profile") != plan["data_profile"]
    ):
        raise ValueError("Recipe data profile does not match the prepared dataset")
    directory.mkdir(parents=True, exist_ok=True)
    grad_fn = nn.value_and_grad(model, lambda m, x, y: m(x, targets=y))
    start_step = optimizer.step
    if start_step >= plan["steps"]:
        raise ValueError("Checkpoint has already reached this training budget")
    mx.eval(model.parameters())
    with ExitStack() as resources, (directory / "metrics.jsonl").open("a") as metrics:
        if plan["streaming"] is not None:
            resources.callback(loader.close)
            resources.callback(val.close)
        for step in range(start_step, plan["steps"]):
            start = time.monotonic()
            total_loss, total_weight, accum = 0.0, 0, None
            for _ in range(plan["gradient_accumulation"]):
                x, y = next(loader)
                valid = int(mx.sum(y != -1).item())
                if valid == 0:
                    raise ValueError("Training batch has no supervised targets")
                loss, grads = grad_fn(model, x, y)
                if not math.isfinite(loss.item()):
                    raise RuntimeError("Non-finite training loss")
                # Weight by target count (synthetic answers have different lengths).
                grads = tree_map(lambda g: g.astype(mx.float32) * valid, grads)
                accum = (
                    grads
                    if accum is None
                    else tree_map(lambda a, b: a + b, accum, grads)
                )
                total_loss += loss.item() * valid
                total_weight += valid
                mx.eval(accum)
            grads = tree_map(lambda g: g / total_weight, accum)
            optimizer.multiplier = lr_multiplier(
                step, plan["steps"], training.warmup_ratio, training.min_lr_ratio
            )
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.arrays())
            record = {
                "step": step + 1,
                "loss": total_loss / total_weight,
                "lr_multiplier": optimizer.multiplier,
                "tokens_per_second": training.tokens_per_step
                / (time.monotonic() - start),
            }
            if args.diagnostics_every > 0 and (step + 1) % args.diagnostics_every == 0:
                stats = {}
                # Per-layer reductions keep monitoring bounded; observe the
                # complete actual sample, including late recurrent state.
                mx.eval(model.hidden(x, diagnostics=stats))
                record["diagnostics"] = summarize_diagnostics(stats)
                record["optimizer"] = {
                    k: v.item() for k, v in optimizer.last_metrics.items()
                }
            if (
                args.eval_every > 0
                and (step + 1) % args.eval_every == 0
                and args.eval_steps > 0
            ):
                val.reset()
                nll, count = 0.0, 0
                for _ in range(args.eval_steps):
                    vx, vy = next(val)
                    weight = int(mx.sum(vy != -1).item())
                    nll += model(vx, targets=vy).item() * weight
                    count += weight
                record["val_loss"] = nll / max(count, 1)
            metrics.write(json.dumps(record) + "\n")
            metrics.flush()
            print(json.dumps(record), flush=True)
            if step + 1 == plan["steps"] or (
                args.save_every > 0 and (step + 1) % args.save_every == 0
            ):
                save_checkpoint(
                    directory,
                    model,
                    step + 1,
                    tokenizer.contract,
                    optimizer,
                    loader.state_dict(),
                    training,
                    run={"steps": plan["steps"], "micro_batch": args.device_batch_size},
                )


def build_datasets(args, plan, tokenizer, state=None):
    """Create loaders without constructing a model (also used by data checks)."""
    if plan["streaming"] is not None:
        from .streaming import StreamConfig, open_streaming_datasets

        return open_streaming_datasets(
            StreamConfig(**plan["streaming"]),
            tokenizer,
            plan["model"]["sequence_len"],
            args.device_batch_size,
            state,
        )
    from .data import TokenDataset

    if state is not None and state.get("format") == "nanochat-hf-stream-v1":
        raise ValueError(
            "Streaming checkpoint requires the original --stream-dataset options"
        )
    return tuple(
        TokenDataset(
            args.data_dir,
            split,
            plan["model"]["sequence_len"],
            args.device_batch_size,
            state if split == "train" else None,
            tokenizer.contract,
        )
        for split in ("train", "val")
    )


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        plan = resolve_plan(args)
        if args.dry_run or not args.start_training:
            print(json.dumps(plan, indent=2))
            print(
                "Configuration only. Training is not started; execution requires --start-training."
            )
            return 0
        train(args, plan)
        return 0
    except (ValueError, FileNotFoundError) as exc:
        print(f"Setup error: {exc}")
        return 2

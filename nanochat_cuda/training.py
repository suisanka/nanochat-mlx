"""Explicit single-GPU PyTorch/FLA training; dry runs import no tensor runtime."""

import argparse
from contextlib import ExitStack
import json
import math
import os
import time
from nanochat_mlx.hybrid.training import (
    build_parser as shared_parser,
    resolve_plan as shared_plan,
)
from .config import HybridConfig, TrainingConfig


def build_parser():
    p = shared_parser(mlx_options=False)
    p.description = __doc__
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--gdn-backend", choices=("fla", "reference"), default="fla")
    p.add_argument("--attention-backend", choices=("sdpa", "flash"), default="sdpa")
    p.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="torch.compile feed-forward blocks and Muon; FLA also uses Triton JIT",
    )
    p.add_argument(
        "--cuda-memory-fraction",
        type=float,
        default=None,
        help="Optional CUDA caching allocator limit as a fraction of device memory",
    )
    return p


def resolve_plan(args):
    plan = shared_plan(args)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError(
            "This trainer supports a single GPU; do not launch multiple torchrun workers"
        )
    if args.device == "cpu" and (
        args.gdn_backend != "reference" or args.attention_backend != "sdpa"
    ):
        raise ValueError(
            "CPU checks require --gdn-backend reference --attention-backend sdpa"
        )
    if args.gdn_backend == "fla" and plan["model"]["dtype"] != "bfloat16":
        raise ValueError("FLA training requires model dtype=bfloat16")
    if args.cuda_memory_fraction is not None and not 0 < args.cuda_memory_fraction <= 1:
        raise ValueError("cuda-memory-fraction must be in (0, 1]")
    if args.device == "cpu" and args.cuda_memory_fraction is not None:
        raise ValueError("cuda-memory-fraction requires --device cuda")
    if (
        min(args.save_every, args.eval_every, args.eval_steps, args.diagnostics_every)
        < 0
    ):
        raise ValueError("Save/eval/diagnostic intervals must be nonnegative")
    plan["backend"] = dict(
        name="pytorch",
        device=args.device,
        gdn=args.gdn_backend,
        attention=args.attention_backend,
        compile=args.compile,
        cuda_memory_fraction=args.cuda_memory_fraction,
    )
    return plan


def run_contract(plan):
    return {
        key: plan[key]
        for key in ("steps", "micro_batch", "source", "data_profile", "backend")
    }


def accumulate_gradients(model, batches, tokens_per_step):
    """Compute weighted microbatch gradients without updating any parameters."""
    import torch

    accumulated, total_loss, total_weight = {}, 0.0, 0
    model.zero_grad(set_to_none=True)
    for x, y in batches:
        valid = int((y != -1).sum().item())
        if valid == 0:
            raise ValueError("Training batch has no supervised targets")
        loss = model(x, targets=y)
        value = loss.item()
        if not math.isfinite(value):
            raise RuntimeError("Non-finite training loss")
        # Fixed scaling keeps BF16 micro-gradients in range. Return the valid
        # count so the optimizer can normalize uneven SFT masks exactly.
        (loss * (valid / tokens_per_step)).backward()
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.grad is None:
                    raise RuntimeError(f"Missing gradient: {name}")
                if name not in accumulated:
                    accumulated[name] = p.grad.float().clone()
                else:
                    accumulated[name].add_(p.grad.float())
                p.grad = None
        total_loss += value * valid
        total_weight += valid
    if total_weight == 0:
        raise ValueError("Gradient accumulation requires at least one batch")
    return accumulated, total_loss / total_weight, total_weight


def train(args, plan):
    import torch
    from .checkpoint import (
        checkpoint_directory,
        initialize_weights,
        load_checkpoint,
        save_checkpoint,
    )
    from .data import build_datasets, next_batch
    from .gdn import fla_kernels
    from .attention import flash_kernel
    from .model import HybridLM
    from .optim import HybridOptimizer, lr_multiplier
    from .tokenizer import DeepSeekTokenizer

    if args.data_dir is None and plan["streaming"] is None:
        raise ValueError("--data-dir or --stream-dataset is required to start training")
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; this command requires an NVIDIA GPU")
        if not torch.cuda.is_bf16_supported():
            raise ValueError("This backend requires CUDA BF16 support")
        if args.cuda_memory_fraction is not None:
            torch.cuda.set_per_process_memory_fraction(
                args.cuda_memory_fraction, device
            )
        torch.cuda.reset_peak_memory_stats(device)
    if args.gdn_backend == "fla":
        fla_kernels()
    if args.attention_backend == "flash":
        flash_kernel()
    config, training = HybridConfig(**plan["model"]), TrainingConfig(**plan["training"])
    directory = checkpoint_directory(args.output_dir, config, args.source)
    if not args.resume and directory.exists() and any(directory.glob("step_*.json")):
        raise ValueError(
            "Output already has checkpoints; choose another --output-dir or use --resume"
        )
    tokenizer = DeepSeekTokenizer(args.tokenizer_dir)
    if config.vocab_size != tokenizer.get_vocab_size():
        raise ValueError("Model/tokenizer vocabulary mismatch")
    torch.manual_seed(args.seed)
    loader_state = None
    if args.resume:
        model, metadata, optimizer = load_checkpoint(
            args.resume,
            tokenizer.contract,
            device,
            args.gdn_backend,
            args.attention_backend,
            load_optimizer=True,
            compile=args.compile,
        )
        if (
            model.config != config
            or metadata["training"] != plan["training"]
            or metadata["run"] != run_contract(plan)
        ):
            raise ValueError(
                "Resume requires identical model, training, run and backend settings"
            )
        if metadata["torch_version"] != torch.__version__:
            raise ValueError(
                "Resume requires the same PyTorch version; use --init-from for a new run"
            )
        loader_state = metadata["loader"]
        if loader_state is None:
            raise ValueError("Resume checkpoint is missing data position")
    else:
        model = HybridLM(config, args.gdn_backend, args.attention_backend).to(device)
        if args.init_from:
            initialize_weights(model, args.init_from, tokenizer.contract)
        optimizer = HybridOptimizer(model, training, args.compile)
    if optimizer.step >= plan["steps"]:
        raise ValueError("Checkpoint has already reached this training budget")
    if args.compile:
        model.compile_hotpaths()
    model.train()
    directory.mkdir(parents=True, exist_ok=True)
    with ExitStack() as resources:
        loader, val = build_datasets(args, plan, tokenizer, loader_state)
        for dataset in (loader, val):
            if hasattr(dataset, "close"):
                resources.callback(dataset.close)
        if (
            plan["data_profile"] is not None
            and loader.meta.get("scenario_profile") != plan["data_profile"]
        ):
            raise ValueError("Recipe data profile does not match prepared data")
        if plan["streaming"] is not None:
            print(json.dumps({"data_stream": loader.contract["source"]}), flush=True)
        metrics = resources.enter_context((directory / "metrics.jsonl").open("a"))
        first_step = optimizer.step
        for step in range(first_step, plan["steps"]):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.monotonic()
            batches = (
                next_batch(loader, device) for _ in range(plan["gradient_accumulation"])
            )
            accumulated, mean_loss, total_weight = accumulate_gradients(
                model, batches, training.tokens_per_step
            )
            # Optimizer consumes the FP32 accumulated gradients directly.
            optimizer.multiplier = lr_multiplier(
                step, plan["steps"], training.warmup_ratio, training.min_lr_ratio
            )
            optimizer.update(accumulated, scale=training.tokens_per_step / total_weight)
            del accumulated
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            record = dict(
                step=step + 1,
                loss=mean_loss,
                lr_multiplier=optimizer.multiplier,
                tokens_per_second=training.tokens_per_step / (time.monotonic() - start),
            )
            if device.type == "cuda":
                record.update(
                    peak_memory_gb=torch.cuda.max_memory_allocated(device) / 1024**3,
                    reserved_memory_gb=torch.cuda.memory_reserved(device) / 1024**3,
                )
            if args.diagnostics_every and (step + 1) % args.diagnostics_every == 0:
                record["optimizer"] = optimizer.last_metrics
            if (
                args.eval_every
                and (step + 1) % args.eval_every == 0
                and args.eval_steps
            ):
                model.eval()
                val.reset()
                nll, count = 0.0, 0
                with torch.no_grad():
                    for _ in range(args.eval_steps):
                        vx, vy = next_batch(val, device)
                        weight = int((vy != -1).sum().item())
                        nll += model(vx, targets=vy).item() * weight
                        count += weight
                if count == 0:
                    raise ValueError("Validation has no supervised targets")
                record["val_loss"] = nll / count
                model.train()
            metrics.write(json.dumps(record) + "\n")
            metrics.flush()
            print(json.dumps(record), flush=True)
            if (
                (args.save_first_step and step == first_step)
                or step + 1 == plan["steps"]
                or (args.save_every and (step + 1) % args.save_every == 0)
            ):
                save_checkpoint(
                    directory,
                    model,
                    step + 1,
                    tokenizer.contract,
                    optimizer,
                    loader.state_dict(),
                    training,
                    run_contract(plan),
                )


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        plan = resolve_plan(args)
        if args.dry_run or not args.start_training:
            print(json.dumps(plan, indent=2))
            print("Configuration only. Execution requires --start-training.")
            return 0
        train(args, plan)
        return 0
    except (ValueError, FileNotFoundError, ImportError, RuntimeError) as exc:
        print(f"Training error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

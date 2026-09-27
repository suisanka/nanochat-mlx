"""Atomic metadata-last safetensors checkpoints, including optimizer/data/RNG."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import uuid
import torch
from safetensors.torch import load_file, save_file
from .config import HybridConfig, TrainingConfig
from .model import HybridLM
from .optim import HybridOptimizer


FORMAT = "nanochat-cuda-hybrid-v1"


def checkpoint_directory(base, config, source="base"):
    if source not in ("base", "sft"):
        raise ValueError("source must be base or sft")
    return (
        Path(base)
        / "cuda_checkpoints"
        / config.architecture
        / f"d{config.n_layer}"
        / source
    )


def save_checkpoint(
    directory,
    model,
    step,
    tokenizer_contract,
    optimizer=None,
    loader_state=None,
    training_config=None,
    run=None,
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"step_{step:08d}"
    metadata = directory / f"{stem}.json"
    if any(directory.glob(f"{stem}.*")):
        raise FileExistsError(f"Checkpoint already exists: {metadata}")
    pending = directory / f".pending-{uuid.uuid4().hex}"
    pending.mkdir()

    def write(name, tensors):
        save_file(
            {key: value.detach().cpu().contiguous() for key, value in tensors.items()},
            str(pending / name),
        )

    write(f"{stem}.safetensors", model.state_dict())
    if optimizer is not None:
        if optimizer.step != step:
            raise ValueError("Optimizer step and checkpoint step disagree")
        write(f"{stem}.optim.safetensors", optimizer.state_dict())
    rng = {"cpu": torch.get_rng_state()}
    if next(model.parameters()).is_cuda:
        rng["cuda"] = torch.cuda.get_rng_state(next(model.parameters()).device)
    write(f"{stem}.rng.safetensors", rng)
    meta = dict(
        format=FORMAT,
        step=step,
        model=model.config.to_dict(),
        tokenizer=tokenizer_contract,
        loader=loader_state,
        training=asdict(training_config) if training_config else None,
        optimizer=optimizer is not None,
        run=run,
        kernels=dict(
            gdn=model.gdn_backend,
            attention=model.attention_backend,
            loss="fla" if model.fused_loss else "recompute",
        ),
        torch_version=torch.__version__,
    )
    (pending / f"{stem}.json").write_text(json.dumps(meta, indent=2) + "\n")
    for path in pending.glob("*.safetensors"):
        os.replace(path, directory / path.name)
    os.replace(pending / f"{stem}.json", metadata)
    pending.rmdir()  # This function's own empty transaction directory.
    return metadata


def load_checkpoint(
    metadata,
    tokenizer_contract=None,
    device="cpu",
    gdn_backend="fla",
    attention_backend="sdpa",
    load_optimizer=False,
    compile=False,
    fused_loss=False,
):
    metadata = Path(metadata)
    meta = json.loads(metadata.read_text())
    if meta.get("format") != FORMAT:
        raise ValueError(
            "Resume/chat requires a CUDA checkpoint; use --init-from for MLX weights"
        )
    if tokenizer_contract is not None and meta["tokenizer"] != tokenizer_contract:
        raise ValueError("Checkpoint tokenizer contract mismatch")
    model = HybridLM(
        HybridConfig(**meta["model"]), gdn_backend, attention_backend, fused_loss
    ).to(device)
    model.load_state_dict(
        load_file(str(metadata.with_suffix(".safetensors")), device=str(device)),
        strict=True,
    )
    optimizer = None
    if load_optimizer:
        if not meta["optimizer"] or meta["training"] is None:
            raise ValueError(
                "Resume requires optimizer state and training configuration"
            )
        optimizer = HybridOptimizer(model, TrainingConfig(**meta["training"]), compile)
        optimizer.load_state_dict(
            load_file(
                str(metadata.with_suffix(".optim.safetensors")), device=str(device)
            )
        )
        if optimizer.step != meta["step"]:
            raise ValueError("Optimizer step and checkpoint step disagree")
        rng = load_file(str(metadata.with_suffix(".rng.safetensors")))
        torch.set_rng_state(rng["cpu"])
        if torch.device(device).type == "cuda":
            if "cuda" not in rng:
                raise ValueError("CUDA resume requires a CUDA RNG state")
            torch.cuda.set_rng_state(rng["cuda"], device)
    return model, meta, optimizer


def initialize_weights(model, metadata, tokenizer_contract):
    """Explicit weight-only warm start from CUDA or current MLX hybrid files."""
    metadata = Path(metadata)
    meta = json.loads(metadata.read_text())
    if meta.get("format") not in (FORMAT, "nanochat-mlx-hybrid-v1"):
        raise ValueError("Unsupported warm-start checkpoint format")
    if meta["tokenizer"] != tokenizer_contract:
        raise ValueError("Warm-start tokenizer contract mismatch")
    source = HybridConfig(**meta["model"])
    target = model.config
    # Context, loss tiling and checkpointing may change, semantic tensor shapes may not.
    for key in (
        "architecture",
        "n_layer",
        "n_embd",
        "intermediate_size",
        "vocab_size",
        "gdn_heads",
        "key_head_dim",
        "value_head_dim",
        "conv_kernel",
        "n_head",
        "n_kv_head",
        "head_dim",
    ):
        if getattr(source, key) != getattr(target, key):
            raise ValueError(f"Warm-start architecture mismatch: {key}")
    model.load_state_dict(
        load_file(
            str(metadata.with_suffix(".safetensors")),
            device=str(next(model.parameters()).device),
        ),
        strict=True,
    )

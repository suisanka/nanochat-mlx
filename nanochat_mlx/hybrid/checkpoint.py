"""Versioned, complete checkpoints isolated from the legacy nanochat format."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import uuid

from .config import HybridConfig, TrainingConfig


def checkpoint_directory(base, config, source="base"):
    if source not in ("base", "sft"):
        raise ValueError("source must be base or sft")
    return (
        Path(base)
        / "hybrid_checkpoints"
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
    import mlx.core as mx

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"step_{step:08d}"
    metadata = directory / f"{stem}.json"
    if metadata.exists():
        raise FileExistsError(f"Checkpoint already exists: {metadata}")
    temporary = directory / f".pending-{uuid.uuid4().hex}"
    temporary.mkdir()
    model.save_weights(str(temporary / f"{stem}.safetensors"))
    if optimizer is not None:
        optimizer.save(temporary / f"{stem}.optim.safetensors")
    meta = {
        "format": "nanochat-mlx-hybrid-v1",
        "step": step,
        "model": model.config.to_dict(),
        "tokenizer": tokenizer_contract,
        "loader": loader_state,
        "training": asdict(training_config) if training_config else None,
        "optimizer": optimizer is not None,
        "run": run,
    }
    (temporary / f"{stem}.json").write_text(json.dumps(meta, indent=2) + "\n")
    # Metadata is the commit marker and is moved last. No old checkpoint deletion.
    for filename in temporary.glob("*.safetensors"):
        os.replace(filename, directory / filename.name)
    os.replace(temporary / f"{stem}.json", metadata)
    temporary.rmdir()  # This function's own empty transaction directory.
    return metadata


def load_checkpoint(metadata, tokenizer_contract=None, load_optimizer=False):
    import mlx.core as mx
    from .model import HybridLM
    from .optim import HybridOptimizer

    metadata = Path(metadata)
    meta = json.loads(metadata.read_text())
    if meta.get("format") != "nanochat-mlx-hybrid-v1":
        raise ValueError(
            "Expected a hybrid checkpoint; legacy weights cannot be used as hybrid weights"
        )
    if tokenizer_contract is not None and meta["tokenizer"] != tokenizer_contract:
        raise ValueError("Checkpoint tokenizer contract does not match local tokenizer")
    model = HybridLM(HybridConfig(**meta["model"]))
    model.load_weights(str(metadata.with_suffix(".safetensors")), strict=True)
    mx.eval(model.parameters())
    optimizer = None
    if load_optimizer:
        if not meta["optimizer"] or meta["training"] is None:
            raise ValueError(
                "Resume requires optimizer state and training configuration"
            )
        optimizer = HybridOptimizer(model, TrainingConfig(**meta["training"]))
        optimizer.load(metadata.with_suffix(".optim.safetensors"))
        if optimizer.step != meta["step"]:
            raise ValueError("Optimizer step and checkpoint step disagree")
    return model, meta, optimizer

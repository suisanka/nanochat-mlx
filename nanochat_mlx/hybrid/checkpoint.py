"""Versioned, complete checkpoints isolated from the legacy nanochat format."""

import hashlib
import json
import os
import uuid
from dataclasses import asdict
from pathlib import Path

from .config import HybridConfig, TrainingConfig


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_cuda_checkpoint(metadata, directory, tokenizer_contract):
    """Import exact hybrid weights for MLX inference; never convert optimizer state."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from .model import HybridLM

    metadata = Path(metadata)
    meta = json.loads(metadata.read_text())
    if meta.get("format") != "nanochat-cuda-hybrid-v1":
        raise ValueError("Expected a nanochat-cuda-hybrid-v1 checkpoint")
    if meta["tokenizer"] != tokenizer_contract:
        raise ValueError("Checkpoint tokenizer contract does not match local tokenizer")
    if type(meta["step"]) is not int or meta["step"] < 0:
        raise ValueError("Invalid checkpoint step")
    weights = metadata.with_suffix(".safetensors")
    model = HybridLM(HybridConfig(**meta["model"]))
    expected = dict(tree_flatten(model.parameters()))
    arrays = mx.load(str(weights))
    if set(arrays) != set(expected):
        raise ValueError("Checkpoint parameter names do not match model")
    for name, value in arrays.items():
        reference = expected[name]
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise ValueError(f"Checkpoint shape/dtype mismatch: {name}")
        if not mx.all(mx.isfinite(value)).item():
            raise ValueError(f"Non-finite checkpoint parameter: {name}")
    model.load_weights(list(arrays.items()), strict=True)
    mx.eval(model.parameters())
    return save_checkpoint(
        directory,
        model,
        meta["step"],
        tokenizer_contract,
        run={
            "purpose": "inference-only CUDA weight import",
            "source_format": meta["format"],
            "source_metadata": str(metadata.resolve()),
            "source_metadata_sha256": file_sha256(metadata),
            "source_weights_sha256": file_sha256(weights),
        },
    )


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

"""Reuse NumPy packing/cursors without ever invoking the MLX iterator."""

import torch
from nanochat_mlx.hybrid.training import build_datasets


def next_batch(loader, device):
    x, y = loader.next_numpy()
    tensors = (torch.from_numpy(a).long() for a in (x, y))
    if device.type == "cuda":
        return tuple(a.pin_memory().to(device, non_blocking=True) for a in tensors)
    return tuple(a.to(device) for a in tensors)


__all__ = ["build_datasets", "next_batch"]

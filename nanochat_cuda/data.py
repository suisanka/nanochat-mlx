"""Reuse NumPy packing/cursors without ever invoking the MLX iterator."""

from copy import deepcopy
import queue
import threading
import multiprocessing
import torch
from nanochat_mlx.hybrid.training import build_datasets


def _prefetch_worker(loader, output, stop):
    def put(item):
        while not stop.is_set():
            try:
                output.put(item, timeout=0.1)
                return
            except queue.Full:
                pass

    try:
        while not stop.is_set():
            batch = loader.next_numpy()
            put((batch, deepcopy(loader.state_dict())))
    except Exception as exc:
        put(RuntimeError(f"{type(exc).__name__}: {exc}"))
    finally:
        if hasattr(loader, "close"):
            loader.close()


class PrefetchedDataset:
    """Bounded CPU read-ahead; checkpoints describe the last consumed batch."""

    def __init__(self, loader, capacity=2, process=False):
        if capacity < 1:
            raise ValueError("Prefetch capacity must be positive")
        if process:
            from nanochat_mlx.hybrid.streaming import StreamingTokenDataset

            if not isinstance(loader, StreamingTokenDataset):
                raise ValueError("Process prefetch requires a streaming dataset")
        self.loader, self.meta = loader, loader.meta
        if hasattr(loader, "contract"):
            self.contract = loader.contract
        self.consumed_state = deepcopy(loader.state_dict())
        self.process = process
        context = multiprocessing.get_context("spawn") if process else None
        self.queue = context.Queue(capacity) if process else queue.Queue(capacity)
        self.stop = context.Event() if process else threading.Event()
        factory = context.Process if process else threading.Thread
        self.thread = factory(
            target=_prefetch_worker, args=(loader, self.queue, self.stop), daemon=True
        )
        self.thread.start()

    def next_numpy(self):
        while True:
            try:
                item = self.queue.get(timeout=1)
                break
            except queue.Empty:
                if not self.thread.is_alive():
                    raise RuntimeError(
                        "Data prefetch worker exited without a batch"
                    ) from None
        if isinstance(item, Exception):
            raise item
        batch, self.consumed_state = item
        return batch

    def state_dict(self):
        return deepcopy(self.consumed_state)

    def close(self):
        self.stop.set()
        # A pending network request may finish later; the daemon owns cleanup.
        self.thread.join(timeout=1)
        if self.process:
            if self.thread.is_alive():
                self.thread.terminate()
                self.thread.join()
            self.queue.close()


def next_batch(loader, device):
    x, y = loader.next_numpy()
    tensors = (torch.from_numpy(a).long() for a in (x, y))
    if device.type == "cuda":
        return tuple(a.pin_memory().to(device, non_blocking=True) for a in tensors)
    return tuple(a.to(device) for a in tensors)


__all__ = ["build_datasets", "next_batch"]

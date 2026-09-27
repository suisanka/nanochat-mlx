"""
Common utilities for nanochat MLX.
"""

import os
import sys
import math


class SetupError(RuntimeError):
    """Raised when required local artifacts are missing or incomplete."""


def print0(s="", **kwargs):
    """Print (no DDP, always rank 0)."""
    print(s, **kwargs)


def print_setup_error(exc):
    """Render setup problems without a Python traceback."""
    print(f"Setup error: {exc}", file=sys.stderr)


def get_base_dir():
    """Get the nanochat base directory for cached data."""
    if os.environ.get("NANOCHAT_BASE_DIR"):
        nanochat_dir = os.environ.get("NANOCHAT_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanochat_dir = os.path.join(cache_dir, "nanochat")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir


def set_memory_limit(gb=16, cache_gb=None):
    """Set MLX's working-set guideline and optionally bound its unused cache."""
    import mlx.core as mx

    if not math.isfinite(gb) or gb <= 0:
        raise ValueError("Memory limit must be finite and positive")
    if cache_gb is not None and (
        not math.isfinite(cache_gb) or not 0 <= cache_gb <= gb
    ):
        raise ValueError("Cache limit must be finite and between zero and memory limit")
    limit = int(gb * 1024**3)
    try:
        mx.set_memory_limit(limit)
    except AttributeError:
        mx.metal.set_memory_limit(limit)
    if cache_gb is not None:
        mx.set_cache_limit(int(cache_gb * 1024**3))
    print0(
        f"MLX memory guideline: {gb} GiB; cache limit: {cache_gb if cache_gb is not None else 'default'} GiB"
    )


def get_active_memory_mb():
    """Get current MLX Metal memory usage in MB."""
    import mlx.core as mx

    try:
        return mx.get_active_memory() / 1024**2
    except AttributeError:
        try:
            return mx.metal.get_active_memory() / 1024**2
        except Exception:
            return 0.0


def get_peak_memory_mb():
    """Get peak MLX Metal memory usage in MB."""
    import mlx.core as mx

    try:
        return mx.get_peak_memory() / 1024**2
    except AttributeError:
        try:
            return mx.metal.get_peak_memory() / 1024**2
        except Exception:
            return 0.0


def download_file_with_lock(url, filename, postprocess_fn=None):
    """
    Downloads a file from a URL to a local path in the base directory.
    Uses a simple check-before-download approach (no DDP, single-device).
    """
    import urllib.request

    base_dir = get_base_dir()
    file_path = os.path.join(base_dir, filename)

    if os.path.exists(file_path):
        return file_path

    print(f"Downloading {url}...")
    with urllib.request.urlopen(url) as response:
        content = response.read()

    with open(file_path, "wb") as f:
        f.write(content)
    print(f"Downloaded to {file_path}")

    if postprocess_fn is not None:
        postprocess_fn(file_path)

    return file_path

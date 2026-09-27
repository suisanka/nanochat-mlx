"""Import a CUDA hybrid checkpoint into the MLX inference format."""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--memory-limit-gb", type=float, default=8)
    args = parser.parse_args(argv)
    from nanochat_mlx.common import set_memory_limit
    from nanochat_mlx.hybrid.checkpoint import import_cuda_checkpoint
    from nanochat_mlx.hybrid.tokenizer import DeepSeekTokenizer

    set_memory_limit(args.memory_limit_gb, cache_gb=1)
    tokenizer = DeepSeekTokenizer(args.tokenizer_dir)
    path = import_cuda_checkpoint(args.checkpoint, args.output_dir, tokenizer.contract)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

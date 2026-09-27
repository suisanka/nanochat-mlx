"""Fine-tune hybrid checkpoints on prepared DeepSeek V4.1 conversations."""

import sys
from nanochat_mlx.hybrid.training import main as train_main


def main(argv=None):
    return train_main([*(sys.argv[1:] if argv is None else argv), "--source=sft"])


if __name__ == "__main__":
    raise SystemExit(main())

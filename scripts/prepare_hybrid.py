"""Install the pinned tokenizer and prepare uint32 data, without training."""

import argparse
from pathlib import Path
import os
import json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer-dir",
        default=os.path.expanduser("~/.cache/nanochat/deepseek_tokenizer"),
    )
    parser.add_argument("--install-tokenizer", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--train-text", type=Path, help="UTF-8 text, one document per line"
    )
    parser.add_argument("--val-text", type=Path)
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        help="FineWeb shards: last sorted file is validation",
    )
    parser.add_argument(
        "--train-chat", type=Path, help="JSONL official-format conversations"
    )
    parser.add_argument("--val-chat", type=Path)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--synthetic", choices=["4k", "32k"])
    parser.add_argument("--train-examples", type=int, default=5400)
    parser.add_argument("--val-examples", type=int, default=540)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    from nanochat_mlx.hybrid.tokenizer import DeepSeekTokenizer, install_tokenizer

    tokenizer = (
        install_tokenizer(args.tokenizer_dir)
        if args.install_tokenizer
        else DeepSeekTokenizer(args.tokenizer_dir)
    )
    print(f"Tokenizer: {tokenizer.contract}")
    if args.output is None:
        return 0
    if not 0 < args.context_length <= 32768:
        parser.error("context-length must be in [1,32768]")
    if args.synthetic:
        if (
            args.train_text
            or args.val_text
            or args.train_chat
            or args.val_chat
            or args.parquet_dir
        ):
            parser.error("Select synthetic data or text documents, not both")
        from nanochat_mlx.hybrid.synthetic import prepare_synthetic

        prepare_synthetic(
            args.output,
            tokenizer,
            args.synthetic,
            args.train_examples,
            args.val_examples,
            args.seed,
        )
    elif args.parquet_dir:
        if args.train_text or args.val_text or args.train_chat or args.val_chat:
            parser.error("Use one input data format")
        files = sorted(args.parquet_dir.glob("*.parquet"))
        if len(files) < 2:
            parser.error(
                "At least 2 parquet shards are required for train/validation splits"
            )
        import pyarrow.parquet as pq
        from nanochat_mlx.hybrid.data import prepare_documents

        def documents(paths):
            for filename in paths:
                for batch in pq.ParquetFile(filename).iter_batches(columns=["text"]):
                    yield from batch.column("text").to_pylist()

        prepare_documents(
            args.output, documents(files[:-1]), documents(files[-1:]), tokenizer
        )
    elif args.train_chat or args.val_chat:
        if not args.train_chat or not args.val_chat or args.train_text or args.val_text:
            parser.error("Provide both --train-chat and --val-chat without text inputs")
        from nanochat_mlx.hybrid.data import prepare_conversations

        with args.train_chat.open() as train, args.val_chat.open() as val:
            prepare_conversations(
                args.output,
                (json.loads(s) for s in train if s.strip()),
                (json.loads(s) for s in val if s.strip()),
                tokenizer,
                args.context_length,
            )
    else:
        if not args.train_text or not args.val_text:
            parser.error("Both --train-text and --val-text are required")
        from nanochat_mlx.hybrid.data import prepare_documents

        with args.train_text.open() as train, args.val_text.open() as val:
            prepare_documents(args.output, train, val, tokenizer)
    print(f"Prepared data at {args.output}; no training was started")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

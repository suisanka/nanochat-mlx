"""Chat using a hybrid checkpoint and the official DeepSeek V4.1 encoder."""

import argparse
import os
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint", required=True, type=Path, help="Hybrid checkpoint metadata JSON"
    )
    p.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path(os.path.expanduser("~/.cache/nanochat/deepseek_tokenizer")),
    )
    p.add_argument("--prompt", "-p")
    p.add_argument("--interactive", action="store_true")
    p.add_argument(
        "--raw",
        action="store_true",
        help="Base-model continuation without chat framing",
    )
    p.add_argument("--thinking-mode", choices=["chat", "thinking"], default="chat")
    p.add_argument("--reasoning-effort", type=int, default=75)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--prefill-chunk-size", type=int, default=256)
    p.add_argument("--memory-limit-gb", type=float, default=8)
    args = p.parse_args(argv)
    if args.raw and args.interactive:
        p.error("--raw is a single-prompt base-model continuation")
    from nanochat_mlx.common import set_memory_limit
    from nanochat_mlx.hybrid.checkpoint import load_checkpoint
    from nanochat_mlx.hybrid.engine import HybridEngine
    from nanochat_mlx.hybrid.tokenizer import DeepSeekTokenizer

    set_memory_limit(args.memory_limit_gb)
    tokenizer = DeepSeekTokenizer(args.tokenizer_dir)
    model, _, _ = load_checkpoint(args.checkpoint, tokenizer.contract)
    model.eval()
    engine = HybridEngine(model, tokenizer)
    messages = []
    while True:
        prompt = input("You: ") if args.interactive else args.prompt or "Hello."
        if prompt.strip().lower() in ("quit", "exit"):
            break
        messages.append({"role": "user", "content": prompt})
        tokens = (
            tokenizer.encode(prompt, prepend=tokenizer.contract["eos"])
            if args.raw
            else tokenizer.apply_chat_template(
                messages,
                thinking_mode=args.thinking_mode,
                reasoning_effort=args.reasoning_effort,
            )
        )
        output, last = [], ""
        for column, _ in engine.generate(
            tokens,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            prefill_chunk_size=args.prefill_chunk_size,
        ):
            if column[0] == tokenizer.contract["eos"]:
                break
            output.append(column[0])
            decoded = tokenizer.decode(output)
            if not decoded.endswith("\ufffd"):
                print(decoded[len(last) :], end="", flush=True)
                last = decoded
        print()
        decoded = tokenizer.decode(output)
        if args.thinking_mode == "thinking":
            reasoning, sep, answer = decoded.partition("</think>")
            messages.append(
                {
                    "role": "assistant",
                    "content": answer if sep else decoded,
                    "reasoning_content": reasoning if sep else "",
                }
            )
        else:
            messages.append({"role": "assistant", "content": decoded})
        if not args.interactive:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

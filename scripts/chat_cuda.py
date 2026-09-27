"""Chat with a CUDA hybrid checkpoint and the official DeepSeek V4.1 encoder."""

import argparse
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path.home() / ".cache/nanochat/deepseek_tokenizer",
    )
    p.add_argument("--prompt", "-p", default="Hello.")
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--thinking-mode", choices=("chat", "thinking"), default="chat")
    p.add_argument("--reasoning-effort", type=int, default=75)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--prefill-chunk-size", type=int, default=256)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--gdn-backend", choices=("fla", "reference"), default="fla")
    p.add_argument("--attention-backend", choices=("sdpa", "flash"), default="sdpa")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    args = p.parse_args(argv)
    from nanochat_cuda.tokenizer import DeepSeekTokenizer
    from nanochat_cuda.checkpoint import load_checkpoint
    from nanochat_cuda.engine import HybridEngine

    tokenizer = DeepSeekTokenizer(args.tokenizer_dir)
    model, _, _ = load_checkpoint(
        args.checkpoint,
        tokenizer.contract,
        args.device,
        args.gdn_backend,
        args.attention_backend,
    )
    if args.compile:
        model.compile_hotpaths()
    engine = HybridEngine(model, tokenizer)
    messages = []
    while True:
        try:
            prompt = input("You: ") if args.interactive else args.prompt
        except EOFError:
            break
        if prompt.strip().lower() in ("quit", "exit"):
            break
        messages.append({"role": "user", "content": prompt})
        tokens = tokenizer.apply_chat_template(
            messages,
            thinking_mode=args.thinking_mode,
            reasoning_effort=args.reasoning_effort,
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
                dict(
                    role="assistant",
                    content=answer if sep else decoded,
                    reasoning_content=reasoning if sep else "",
                )
            )
        else:
            messages.append(dict(role="assistant", content=decoded))
        if not args.interactive:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

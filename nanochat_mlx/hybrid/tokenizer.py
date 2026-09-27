"""Pinned official DeepSeek V4.1 text tokenizer and prompt encoder adapter."""

import hashlib
import json
from pathlib import Path
from .vendor import deepseek_v41 as encoding

REPO = "deepseek-ai/DeepSeek-V4.1-Flash"
REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
VOCAB_SIZE, BOS, EOS, PAD = 129280, 0, 1, 2
CHAT_TEMPLATE_SHA256 = hashlib.sha256(Path(encoding.__file__).read_bytes()).hexdigest()


class DeepSeekTokenizer:
    def __init__(self, directory):
        from tokenizers import Tokenizer

        directory = Path(directory)
        self.contract = json.loads((directory / "contract.json").read_text())
        filename = directory / "tokenizer.json"
        digest = hashlib.sha256(filename.read_bytes()).hexdigest()
        if (
            self.contract.get("repo") != REPO
            or self.contract.get("revision") != REVISION
        ):
            raise ValueError("Unexpected DeepSeek tokenizer revision")
        if self.contract.get("sha256") != digest:
            raise ValueError("Tokenizer fingerprint mismatch")
        if self.contract.get("chat_template_sha256") != CHAT_TEMPLATE_SHA256:
            raise ValueError(
                "Official V4.1 chat template fingerprint mismatch; re-run --install-tokenizer"
            )
        if any(
            self.contract.get(k) != v
            for k, v in (
                ("vocab_size", VOCAB_SIZE),
                ("bos", BOS),
                ("eos", EOS),
                ("pad", PAD),
            )
        ):
            raise ValueError("Tokenizer vocabulary/special-token contract mismatch")
        self.tokenizer = Tokenizer.from_file(str(filename))
        if max(self.tokenizer.get_vocab().values()) >= VOCAB_SIZE:
            raise ValueError(
                "Tokenizer contains token IDs outside the model vocabulary"
            )
        if (
            self.tokenizer.token_to_id("<｜begin▁of▁sentence｜>") != BOS
            or self.tokenizer.token_to_id("<｜end▁of▁sentence｜>") != EOS
        ):
            raise ValueError("Unexpected DeepSeek BOS/EOS IDs")

    def get_vocab_size(self):
        return VOCAB_SIZE

    def get_bos_token_id(self):
        return BOS

    def encode(self, text, prepend=None, append=None):
        if isinstance(text, list):
            return [self.encode(t, prepend, append) for t in text]
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        if prepend is not None:
            ids.insert(0, prepend)
        if append is not None:
            ids.append(append)
        return ids

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    @staticmethod
    def _text_messages(messages):
        processed, images = encoding.process_image_messages(messages)
        if images:
            raise ValueError(
                "This language model is text-only; image inputs require a vision encoder"
            )
        return processed

    def apply_chat_template(
        self,
        messages,
        thinking_mode="chat",
        reasoning_effort=None,
        drop_thinking=True,
        tokenize=True,
    ):
        processed = self._text_messages(messages)
        prompt = encoding.encode_messages(
            processed,
            thinking_mode=thinking_mode,
            reasoning_effort=reasoning_effort,
            drop_thinking=drop_thinking,
        )
        return self.encode(prompt) if tokenize else prompt

    def render_conversation(self, conversation, max_tokens=None):
        messages = self._text_messages(conversation["messages"])
        mode = conversation.get("thinking_mode", "chat")
        effort = conversation.get("reasoning_effort")
        drop = conversation.get("drop_thinking", True)
        prompt = encoding.encode_messages(
            messages, thinking_mode=mode, reasoning_effort=effort, drop_thinking=drop
        )
        processed = encoding.sort_tool_results_by_call_order(
            encoding.merge_tool_messages(messages)
        )
        effective_drop = drop and not any(m.get("tools") for m in processed)
        if mode == "thinking" and effective_drop:
            processed = encoding._drop_thinking_messages(processed)
        text, spans = encoding.bos_token, []
        for i, message in enumerate(processed):
            part = encoding.render_message(
                i,
                processed,
                thinking_mode=mode,
                drop_thinking=effective_drop,
                reasoning_effort=effort,
            )
            if message["role"] == "assistant":
                spans.append((len(text), len(text) + len(part)))
            text += part
        if text != prompt:
            raise ValueError(
                "Supervision spans differ from official V4.1 prompt encoding"
            )
        encoded = self.tokenizer.encode(prompt, add_special_tokens=False)
        # Offsets avoid guessing role boundaries by scanning user-supplied tokens.
        mask = [
            int(any(start <= a and b <= end and b > a for start, end in spans))
            for a, b in encoded.offsets
        ]
        ids = encoded.ids
        return (
            (ids, mask) if max_tokens is None else (ids[:max_tokens], mask[:max_tokens])
        )

    def render_for_completion(self, conversation):
        messages = conversation["messages"]
        if messages and messages[-1]["role"] == "assistant":
            messages = messages[:-1]
        return self.apply_chat_template(
            messages,
            thinking_mode=conversation.get("thinking_mode", "chat"),
            reasoning_effort=conversation.get("reasoning_effort"),
            drop_thinking=conversation.get("drop_thinking", True),
        )


def install_tokenizer(directory):
    """Fetch tokenizer JSON only; no model weights, remote code or training."""
    from huggingface_hub import hf_hub_download

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "contract.json").exists():
        existing = json.loads((directory / "contract.json").read_text())
        if existing.get("chat_template_sha256") == CHAT_TEMPLATE_SHA256:
            return DeepSeekTokenizer(directory)
    filename = hf_hub_download(
        REPO, "tokenizer.json", revision=REVISION, local_dir=str(directory)
    )
    contract = {
        "repo": REPO,
        "revision": REVISION,
        "vocab_size": VOCAB_SIZE,
        "bos": BOS,
        "eos": EOS,
        "pad": PAD,
        "sha256": hashlib.sha256(Path(filename).read_bytes()).hexdigest(),
        "chat_template_sha256": CHAT_TEMPLATE_SHA256,
    }
    (directory / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    return DeepSeekTokenizer(directory)

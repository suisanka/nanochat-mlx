"""uint32 mmap data with exact resume positions and optional supervised labels."""

import hashlib
import json
import os
from pathlib import Path
import numpy as np


def sha256_file(filename):
    result = hashlib.sha256()
    with open(filename, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def prepare_documents(directory, train_documents, val_documents, tokenizer):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(
        (directory / name).exists() for name in ("train.bin", "val.bin", "meta.json")
    ):
        raise FileExistsError(
            "Prepared dataset already exists; choose another output directory"
        )
    counts = {}
    for split, documents in (("train", train_documents), ("val", val_documents)):
        count = 0
        with open(directory / f"{split}.bin", "wb") as handle:
            for document in documents:
                ids = tokenizer.encode(document) + [tokenizer.contract["eos"]]
                tokens = np.asarray(ids, dtype="<u4")
                if len(tokens) and int(tokens.max()) >= tokenizer.get_vocab_size():
                    raise ValueError("Token outside vocabulary")
                tokens.tofile(handle)
                count += len(tokens)
        counts[split] = count
    write_metadata(directory, tokenizer.contract, counts)


def write_metadata(
    directory, contract, counts, record_length=None, scenario_profile=None
):
    directory = Path(directory)
    files = {f.name: sha256_file(f) for f in directory.glob("*.bin")}
    meta = {
        "format": "nanochat-uint32-v1",
        "dtype": "<u4",
        "tokenizer": contract,
        "counts": counts,
        "files": files,
        "record_length": record_length,
        "scenario_profile": scenario_profile,
    }
    (directory / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def prepare_conversations(
    directory, train_conversations, val_conversations, tokenizer, sequence_len
):
    """Pack complete official-template conversations; preserve assistant masks.

    Oversized records fail explicitly instead of silently cropping an answer or
    leaving an unconsumable item in a packing buffer.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError("Conversation output directory must be empty")
    capacity = sequence_len + 1
    counts = {}
    for split, conversations in (
        ("train", train_conversations),
        ("val", val_conversations),
    ):
        count = 0
        with (
            open(directory / f"{split}.bin", "wb") as tokens_file,
            open(directory / f"{split}.labels.bin", "wb") as labels_file,
        ):
            row, labels = [], []

            def flush():
                nonlocal count, row, labels
                if not row:
                    return
                if not any(x != -1 for x in labels[1:]):
                    raise ValueError("Conversation record has no assistant targets")
                padding = capacity - len(row)
                np.asarray(
                    row + [tokenizer.contract["pad"]] * padding, dtype="<u4"
                ).tofile(tokens_file)
                np.asarray(labels + [-1] * padding, dtype="<i4").tofile(labels_file)
                count += capacity
                row, labels = [], []

            for conversation in conversations:
                if isinstance(conversation, list):
                    conversation = {"messages": conversation}
                ids, mask = tokenizer.render_conversation(conversation)
                if len(ids) > capacity:
                    raise ValueError(
                        f"Conversation has {len(ids)} tokens, exceeding context capacity {capacity}"
                    )
                if len(row) + len(ids) > capacity:
                    flush()
                row.extend(ids)
                labels.extend(
                    token if supervise else -1 for token, supervise in zip(ids, mask)
                )
            flush()
        counts[split] = count
    return write_metadata(directory, tokenizer.contract, counts, capacity, "sft")


class TokenDataset:
    def __init__(
        self,
        directory,
        split,
        sequence_len,
        batch_size=1,
        state=None,
        tokenizer_contract=None,
    ):
        self.directory = Path(directory)
        if split not in ("train", "val"):
            raise ValueError("split must be train or val")
        self.meta = json.loads((self.directory / "meta.json").read_text())
        if (
            self.meta.get("format") != "nanochat-uint32-v1"
            or self.meta.get("dtype") != "<u4"
        ):
            raise ValueError("Expected little-endian uint32 data")
        if (
            tokenizer_contract is not None
            and self.meta["tokenizer"] != tokenizer_contract
        ):
            raise ValueError("Dataset tokenizer contract mismatch")
        filename = self.directory / f"{split}.bin"
        if sha256_file(filename) != self.meta["files"][filename.name]:
            raise ValueError("Dataset fingerprint mismatch")
        if filename.stat().st_size % 4:
            raise ValueError("Truncated uint32 token file")
        self.tokens = np.memmap(filename, dtype="<u4", mode="r")
        if len(self.tokens) != self.meta["counts"][split]:
            raise ValueError("Dataset token count mismatch")
        self.T, self.B = sequence_len, batch_size
        self.record_length = self.meta.get("record_length")
        if self.T <= 0 or self.B <= 0 or len(self.tokens) < self.T + 1:
            raise ValueError(
                "Dataset requires at least sequence_len+1 tokens and positive batch size"
            )
        if self.record_length and (
            self.record_length != self.T + 1 or len(self.tokens) % self.record_length
        ):
            raise ValueError("Synthetic record length must equal sequence_len+1")
        self.labels = None
        labels_path = self.directory / f"{split}.labels.bin"
        if labels_path.exists():
            if self.meta["files"].get(labels_path.name) != sha256_file(labels_path):
                raise ValueError("Labels fingerprint mismatch")
            self.labels = np.memmap(labels_path, dtype="<i4", mode="r")
            if len(self.labels) != len(self.tokens):
                raise ValueError("Labels and token lengths differ")
        self.fingerprint = hashlib.sha256(
            (self.directory / "meta.json").read_bytes()
        ).hexdigest()
        self.split = split
        self.cursor, self.epoch = 0, 0
        if state is not None:
            if (
                state["fingerprint"] != self.fingerprint
                or state["split"] != split
                or state["sequence_len"] != self.T
            ):
                raise ValueError("Resume dataset/split/context mismatch")
            self.cursor, self.epoch = state["cursor"], state["epoch"]
            if not 0 <= self.cursor <= len(self.tokens):
                raise ValueError("Invalid resume cursor")

    def state_dict(self):
        return dict(
            cursor=self.cursor,
            epoch=self.epoch,
            fingerprint=self.fingerprint,
            split=self.split,
            sequence_len=self.T,
        )

    def next_numpy(self):
        rows, targets = [], []
        for _ in range(self.B):
            if self.cursor + self.T + 1 > len(self.tokens):
                self.cursor = 0
                self.epoch += 1
            row = np.array(
                self.tokens[self.cursor : self.cursor + self.T + 1], dtype=np.int32
            )
            if row.min() < 0 or row.max() >= self.meta["tokenizer"]["vocab_size"]:
                raise ValueError("Token ID outside vocabulary")
            target = (
                row[1:]
                if self.labels is None
                else np.array(
                    self.labels[self.cursor + 1 : self.cursor + self.T + 1],
                    dtype=np.int32,
                )
            )
            if np.any((target < -1) | (target >= self.meta["tokenizer"]["vocab_size"])):
                raise ValueError("Target ID outside vocabulary")
            rows.append(row[:-1])
            targets.append(target)
            self.cursor += self.record_length or self.T
        return np.stack(rows), np.stack(targets)

    def __iter__(self):
        return self

    def __next__(self):
        import mlx.core as mx

        x, y = self.next_numpy()
        return mx.array(x), mx.array(y)

"""Online Hugging Face text loading, EOS packing and resumable token buffers.

No dataset, tokenizer or MLX is loaded during configuration inspection. Streams
are deterministic: HF shuffle buffers cannot be restored exactly, so this loader
preserves source order. Memory scales with a batch and the largest document, not
the corpus. Hugging Face may also buffer remote file blocks / parquet row groups.
"""

from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import islice
import re

import numpy as np


@dataclass(frozen=True)
class StreamConfig:
    dataset: str
    name: str | None = None
    revision: str | None = None
    train_split: str = "train"
    val_split: str | None = None
    val_documents: int = 1024
    text_column: str = "text"
    cache_parquet: bool = False

    def __post_init__(self):
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", self.dataset):
            raise ValueError("stream-dataset must be a Hugging Face owner/dataset ID")
        if not self.text_column or self.val_documents <= 0:
            raise ValueError(
                "Stream text column and positive val-documents are required"
            )
        for split in (self.train_split, self.val_split):
            if split is not None and not re.fullmatch(r"[\w.-]+", split):
                raise ValueError(
                    "Stream splits must be plain split names, without slicing"
                )
        if self.train_split == self.val_split:
            raise ValueError("Stream train and validation splits must be distinct")
        if self.cache_parquet and (
            self.name is not None
            or self.train_split != "train"
            or self.val_split is not None
        ):
            raise ValueError(
                "Cached Parquet requires a single train split without a named configuration"
            )


def cached_parquet_documents(files, dataset, revision, text_column):
    """Download only the current pinned shard; iterate bounded Arrow batches.

    Explicit filenames avoid Hub directory pagination, which some network
    proxies incorrectly cache without the pagination cursor. HF's iterable
    dataset still owns shard/example cursors and exact resume semantics.
    """
    from huggingface_hub import hf_hub_download, try_to_load_from_cache
    import pyarrow.parquet as pq

    for filename in files:
        path = try_to_load_from_cache(
            dataset, filename, repo_type="dataset", revision=revision
        )
        if not isinstance(path, str):
            path = hf_hub_download(
                dataset, filename, repo_type="dataset", revision=revision
            )
        with pq.ParquetFile(path) as parquet:
            for batch in parquet.iter_batches(batch_size=256, columns=[text_column]):
                yield from batch.to_pylist()


class StreamingTokenDataset:
    """Pack documents into the same overlapping T+1 rows as TokenDataset."""

    def __init__(
        self, stream, tokenizer, source, split, sequence_len, batch_size=1, state=None
    ):
        if sequence_len <= 0 or batch_size <= 0 or split not in ("train", "val"):
            raise ValueError("Invalid stream batch/context/split")
        self.stream, self.tokenizer = stream, tokenizer
        self.T, self.B = sequence_len, batch_size
        self.contract = {
            "source": deepcopy(source),
            "split": split,
            "tokenizer": deepcopy(tokenizer.contract),
            "sequence_len": sequence_len,
            "batch_size": batch_size,
        }
        self.meta = {"scenario_profile": None}
        self.initial_state = deepcopy(stream.state_dict())
        self.reset()
        if state is not None:
            if (
                state.get("format") != "nanochat-hf-stream-v1"
                or state.get("contract") != self.contract
            ):
                raise ValueError(
                    "Resume streaming dataset/tokenizer/configuration mismatch"
                )
            self.epoch = state["epoch"]
            self.tokens_in_epoch = state["tokens_in_epoch"]
            self.buffer = deque(state["pending_tokens"])
            if self.epoch < 0 or self.tokens_in_epoch < len(self.buffer):
                raise ValueError("Invalid stream resume position")
            self._validate_tokens(self.buffer)
            stream.load_state_dict(deepcopy(state["stream_state"]))
            self.iterator = iter(stream)
        # Snapshot is valid even before the first next() on a restored iterator.
        self.stream_state = deepcopy(
            state["stream_state"] if state else self.initial_state
        )

    def _validate_tokens(self, tokens):
        if any(
            type(t) is not int or not 0 <= t < self.tokenizer.get_vocab_size()
            for t in tokens
        ):
            raise ValueError("Stream token ID outside vocabulary")

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("iterator", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.stream.load_state_dict(deepcopy(self.stream_state))
        self.iterator = iter(self.stream)

    def reset(self):
        """Return validation to its fixed initial sample; no network read yet."""
        self.close()
        self.buffer = deque()
        self.epoch, self.tokens_in_epoch = 0, 0
        self.stream.load_state_dict(deepcopy(self.initial_state))
        self.stream_state = deepcopy(self.initial_state)
        self.iterator = iter(self.stream)

    def close(self):
        iterator = getattr(self, "iterator", None)
        if iterator is not None:
            iterator.close()

    def state_dict(self):
        return {
            "format": "nanochat-hf-stream-v1",
            "contract": deepcopy(self.contract),
            "epoch": self.epoch,
            "tokens_in_epoch": self.tokens_in_epoch,
            "pending_tokens": list(self.buffer),
            "stream_state": deepcopy(self.stream_state),
        }

    def next_numpy(self):
        rows = []
        for _ in range(self.B):
            while len(self.buffer) < self.T + 1:
                try:
                    document = next(self.iterator)
                except StopIteration:
                    if self.tokens_in_epoch < self.T + 1:
                        raise ValueError(
                            "Streaming split needs at least sequence_len+1 tokens"
                        ) from None
                    # Match mmap packing: drop the incomplete epoch tail.
                    self.buffer.clear()
                    self.epoch += 1
                    self.tokens_in_epoch = 0
                    self.stream.load_state_dict(deepcopy(self.initial_state))
                    self.stream_state = deepcopy(self.initial_state)
                    self.iterator = iter(self.stream)
                    continue
                column = self.contract["source"]["request"]["text_column"]
                text = document.get(column)
                if not isinstance(text, str):
                    raise ValueError(
                        f"Streaming document requires string column {column!r}"
                    )
                tokens = self.tokenizer.encode(text) + [self.tokenizer.contract["eos"]]
                self._validate_tokens(tokens)
                self.buffer.extend(tokens)
                self.tokens_in_epoch += len(tokens)
                self.stream_state = deepcopy(self.stream.state_dict())
            rows.append(list(islice(self.buffer, self.T + 1)))
            for _ in range(self.T):
                self.buffer.popleft()
        rows = np.asarray(rows, dtype=np.int32)
        return rows[:, :-1].copy(), rows[:, 1:].copy()

    def __iter__(self):
        return self

    def __next__(self):
        import mlx.core as mx

        x, y = self.next_numpy()
        return mx.array(x), mx.array(y)


def open_streaming_datasets(config, tokenizer, sequence_len, batch_size=1, state=None):
    """Pin Hub revision once; resume keeps that SHA even if main moves."""
    import datasets
    from huggingface_hub import HfApi

    request = asdict(config)
    if not config.cache_parquet:
        request.pop("cache_parquet")  # Preserve existing stream checkpoint contracts.
    if state is not None:
        if state.get("format") != "nanochat-hf-stream-v1":
            raise ValueError("Resume checkpoint does not contain a streaming loader")
        source = state["contract"]["source"]
        if (
            source["request"] != request
            or source["datasets_version"] != datasets.__version__
        ):
            raise ValueError("Resume streaming source or datasets version mismatch")
        revision = source["revision"]
    else:
        info = HfApi().dataset_info(config.dataset, revision=config.revision)
        revision = info.sha
        source = {
            "request": request,
            "revision": revision,
            "datasets_version": datasets.__version__,
        }
        if config.cache_parquet:
            files = sorted(
                s.rfilename for s in info.siblings if s.rfilename.endswith(".parquet")
            )
            if not files or any("/" in f for f in files):
                raise ValueError(
                    "Cached Parquet requires flat, single-split Parquet shards"
                )
            source["files"] = files
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(
            "Streaming dataset must resolve to an immutable Hub commit SHA"
        )

    def load(split):
        if config.cache_parquet:
            return datasets.IterableDataset.from_generator(
                cached_parquet_documents,
                gen_kwargs=dict(
                    files=source["files"],
                    dataset=config.dataset,
                    revision=revision,
                    text_column=config.text_column,
                ),
            )
        return datasets.load_dataset(
            config.dataset,
            name=config.name,
            revision=revision,
            split=split,
            streaming=True,
        )

    # Separate instances prevent validation iteration from changing train state.
    train, val = load(config.train_split), load(config.val_split or config.train_split)
    if config.val_split is None:
        train = train.skip(config.val_documents)
        val = val.take(config.val_documents)
    return (
        StreamingTokenDataset(
            train, tokenizer, source, "train", sequence_len, batch_size, state
        ),
        StreamingTokenDataset(val, tokenizer, source, "val", sequence_len, batch_size),
    )

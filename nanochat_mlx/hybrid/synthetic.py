"""Reproducible memory-task records and answer-only labels for 4K / 32K.

This module prepares data; it never creates a model or starts training.
Distances are measured in tokenizer tokens, from the end of the last relevant
fact to the beginning of the query, and recorded for distance-binned evaluation.
"""

import json
import random
import string
from pathlib import Path
import numpy as np
from .data import write_metadata
from .vendor import deepseek_v41 as encoding

PROFILES = {
    "4k": {
        "sequence_len": 4096,
        "distances": [128, 256, 512, 768, 1024, 1152, 1280, 1536, 2048, 3072],
    },
    "32k": {
        "sequence_len": 32768,
        "distances": [128, 512, 1024, 1152, 1536, 4096, 8192, 16384, 24576, 30720],
    },
}
TASKS = (
    "recall",
    "overwrite",
    "multi_key",
    "parity",
    "state_transition",
    "variable_update",
    "counter",
    "stack",
    "exact_string",
)
KEY_COUNTS = (4, 8, 16, 32, 64, 128)


class RecordDoesNotFit(ValueError):
    """An exact-distance/key-count combination cannot fit this context."""


def make_task(kind, rng, key_count=4):
    value = lambda n=6: "".join(
        rng.choices(string.ascii_uppercase + string.digits, k=n)
    )
    if kind in ("recall", "overwrite", "multi_key", "exact_string"):
        count = key_count if kind == "multi_key" else 1
        keys = [f"key_{value(5)}" for _ in range(count)]
        values = [value(32 if kind == "exact_string" else 6) for _ in keys]
        target = rng.randrange(count)
        pairs = list(zip(keys, values))
        chosen = pairs.pop(target)
        pairs.append(chosen)  # The relevant fact ends exactly at the recorded gap.
        facts = "Remember the most recent value for each key.\n"
        if kind == "overwrite":
            facts += f"{keys[0]} = {value()}.\n"
        facts += "".join(f"{k} = {v}.\n" for k, v in pairs)
        return (
            facts,
            f"\nQuestion: What is the latest value of {keys[target]}?\nAnswer: ",
            values[target],
        )
    if kind == "parity":
        bits = [rng.randrange(2) for _ in range(32)]
        return (
            "Bits: " + " ".join(map(str, bits)) + ".\n",
            "\nQuestion: What is their parity (sum modulo 2)?\nAnswer: ",
            str(sum(bits) % 2),
        )
    if kind == "state_transition":
        actions = [rng.choice((-1, 1)) for _ in range(12)]
        return (
            "A three-state machine starts at 0.\n"
            + "".join(f"Move {x:+d} modulo 3.\n" for x in actions),
            "\nQuestion: What is the final state?\nAnswer: ",
            str(sum(actions) % 3),
        )
    if kind in ("variable_update", "counter"):
        initial = rng.randint(0, 9)
        moves = [rng.randint(-3, 3) for _ in range(8)]
        return (
            f"Counter starts at {initial}.\n"
            + "".join(f"Add {x:+d} to the counter.\n" for x in moves),
            "\nQuestion: What is the final counter value?\nAnswer: ",
            str(initial + sum(moves)),
        )
    if kind == "stack":
        values = [value(4) for _ in range(5)]
        return (
            "Start with an empty stack.\n"
            + "".join(f"Push {v}.\n" for v in values)
            + "Pop twice.\n",
            "\nQuestion: What is now on top of the stack?\nAnswer: ",
            values[-3],
        )
    raise ValueError(f"Unknown task {kind}")


def make_record(tokenizer, sequence_len, distance, kind, seed, key_count=4):
    rng = random.Random(seed)
    facts, question, answer = make_task(kind, rng, key_count)
    # Use the pinned official chat template for answer-supervised tasks.
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=False
    )
    prefix_text = encoding.bos_token + encoding.USER_SP_TOKEN
    if not prompt.startswith(prefix_text):
        raise ValueError("Unexpected official single-user chat prefix")
    prefix = tokenizer.encode(prefix_text)
    query_ids = tokenizer.encode(prompt[len(prefix_text) :])
    answer_ids = tokenizer.encode(answer)
    fragments = [tokenizer.encode(line + "\n") for line in facts.splitlines()]
    ending = answer_ids + [tokenizer.contract["eos"]]
    length = sequence_len + 1
    query_start = length - len(query_ids) - len(ending)
    fact_end = query_start - distance
    fact_start = fact_end - sum(map(len, fragments))
    if fact_start < len(prefix):
        raise RecordDoesNotFit(
            f"Task {kind} with {key_count} keys and gap {distance} does not fit context {sequence_len}"
        )
    filler = tokenizer.encode(
        "An unrelated passage describes quiet weather, ordinary roads, and distant trees.\n"
    )
    if not filler:
        raise ValueError("Tokenizer produced empty distractor text")
    offset = rng.randrange(len(filler))
    ids = np.asarray(
        [filler[(offset + i) % len(filler)] for i in range(length)], dtype="<u4"
    )
    ids[: len(prefix)] = prefix
    # Spread updates across the available history, so overwrite/state-tracking
    # require retaining state across distractor segments rather than one paragraph.
    first = len(prefix)
    gap = (fact_end - first - sum(map(len, fragments))) // max(len(fragments) - 1, 1)
    cursor = first if len(fragments) > 1 else fact_start
    spans = []
    for i, fragment in enumerate(fragments):
        if i == len(fragments) - 1:
            cursor = fact_end - len(fragment)
        ids[cursor : cursor + len(fragment)] = fragment
        spans.append([cursor, cursor + len(fragment)])
        cursor += len(fragment) + gap
    ids[query_start : query_start + len(query_ids)] = query_ids
    ids[-len(ending) :] = ending
    labels = np.full(length, -1, dtype="<i4")
    labels[-len(ending) :] = ending
    info = dict(
        task=kind,
        seed=seed,
        distance=distance,
        key_count=key_count,
        fact_start=spans[0][0],
        fact_end=fact_end,
        fact_spans=spans,
        query_start=query_start,
        answer_start=length - len(ending),
        answer=answer,
    )
    return ids, labels, info


def prepare_synthetic(
    directory, tokenizer, profile="4k", train_examples=900, val_examples=180, seed=42
):
    if train_examples < 1 or val_examples < 1:
        raise ValueError("Both splits must contain examples")
    settings = PROFILES[profile]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError("Synthetic output directory must be empty")
    sequence_len = settings["sequence_len"]
    counts = {}
    skipped = set()
    combinations = [
        (kind, distance, keys if kind == "multi_key" else 1)
        for keys in KEY_COUNTS
        for distance in settings["distances"]
        for kind in TASKS
    ]
    for split, count, seed_offset in (
        ("train", train_examples, 0),
        ("val", val_examples, 10**9),
    ):
        with (
            open(directory / f"{split}.bin", "wb") as tokens,
            open(directory / f"{split}.labels.bin", "wb") as labels,
            open(directory / f"{split}.tasks.jsonl", "w") as ledger,
        ):
            candidate = 0
            for i in range(count):
                for attempt in range(len(combinations)):
                    kind, distance, key_count = combinations[
                        candidate % len(combinations)
                    ]
                    candidate += 1
                    try:
                        record, target, info = make_record(
                            tokenizer,
                            sequence_len,
                            distance,
                            kind,
                            seed + seed_offset + i,
                            key_count,
                        )
                        break
                    except RecordDoesNotFit:
                        skipped.add((kind, distance, key_count))
                else:
                    raise ValueError("No requested scenario fits the context")
                record.tofile(tokens)
                target.tofile(labels)
                ledger.write(json.dumps(info) + "\n")
        counts[split] = count * (sequence_len + 1)
    result = write_metadata(
        directory, tokenizer.contract, counts, sequence_len + 1, profile
    )
    (directory / "coverage.json").write_text(
        json.dumps(
            {
                "profile": profile,
                "skipped_infeasible_combinations": sorted(skipped),
                "note": "See split tasks.jsonl for actual token distances and coverage; no distance is silently shortened.",
            },
            indent=2,
        )
        + "\n"
    )
    return result

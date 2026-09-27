# 32K memory-training scenarios

These are **prepared data generators, recipes and evaluation tooling**. They
have not been used to train a model, and no actual 32K model sequence is run as
part of the current verification.

## Data contract

`--synthetic 32k` produces 32,769-token records (`x = record[:-1]`,
`y = record[1:]`), official DeepSeek V4.1 non-thinking chat framing, answer-only
int32 labels (`-1` elsewhere), and uint32 input storage. The prompt consumes the
remaining context before its expected answer. The EOS target is included.

`train.tasks.jsonl` / `val.tasks.jsonl` record exact fact spans, query position,
answer start, expected answer, task type, key count, seed and measured gap.
Training and validation have disjoint seed ranges. The token gap runs from the
end of the most recent relevant fact to the start of the query.

| Scenario | Construction | Intended question |
|---|---|---|
| Recall | Key/value assignment, distractors, query | Retention vs distance |
| Boundary recall | Gaps 1024 and 1152 | Direct SWA boundary behavior |
| Long recall | Gaps 4096,8192,16384,24576,30720 | Retention outside short local paths |
| Overwrite | Old and new assignments distributed across history | Latest value wins |
| Multi-key | 4,8,16,32,64,128 concurrent associations | Capacity and interference |
| Parity | A bit sequence separated from its query | State tracking |
| State transition | Distributed signed moves in a 3-state machine | Persistent state updates |
| Variable/counter updates | Initial value and distributed additions | Update composition |
| Stack-like dependencies | Distributed push and pop instructions | Structured memory limits |
| Exact string | Random 32-character value | Lossy-memory exact-copy limits |

Not every high-key-count/high-distance combination fits a fixed context.
Infeasible combinations are skipped and listed in `coverage.json`; their distance
is never silently shortened. Actual coverage is authoritative in each ledger.
The default CLI dataset sizes (5400 train / 540 validation) traverse the scenario
matrix; smaller user-selected sizes may cover only part of it.

## Preparation and inspection

```bash
uv run python -m scripts.prepare_hybrid --install-tokenizer
uv run python -m scripts.prepare_hybrid --synthetic 32k \
  --train-examples 5400 --val-examples 540 --output /absolute/path/memory-32k

uv run python -m scripts.train --recipe configs/gdn_swa_32k_memory.json \
  --depth 12 --data-dir /absolute/path/memory-32k \
  --init-from /absolute/path/4k-checkpoint.json \
  --output-dir /absolute/path/32k-run --dry-run
```

The last command is inspection only. Add `--start-training` only when the user
requests actual execution. A new phase needs a separate output root to avoid
mixing checkpoint series. A complete exact continuation uses `--resume` with the
same configuration, token budget, micro batch and dataset fingerprint.

## Recipe

The 32K memory recipe keeps the v0 width, FFN, 1024 SWA window, GDN dimensions,
optimizer split and 131,072-token global batch. With micro-batch 1, accumulation
is 4 steps. It enables block checkpointing and uses a 50M-token target, rounded
up to complete optimizer steps. Depth overrides remain available. Native local
RoPE is the default; fixed linear/YaRN options are separate controlled changes.

The 4K synthetic recipe is available for mechanism validation before this phase.
There is no automatic curriculum or automatic launch of the next phase. SFT on
user-provided long conversations is also supported via `prepare_hybrid
--train-chat ... --val-chat ... --context-length 32768`, preserving official
assistant-only masks.

## Evaluation protocol (not executed)

Use `scripts.hybrid_eval` to emit exact-match accuracy by task, actual distance
and key count. Increase `--max-problems` to cover the full ledger. Select
`--max-tokens` large enough for the expected answer; generation is bounded by the
checkpoint's context budget. Report per-bin counts so missing bins cannot be
mistaken for zero accuracy.

Use identical data/token budgets and depth for the AdamW-only hybrid optimizer
control. Pure SWA, pure GDN and full-attention comparisons from the research plan
are deferred; only the hybrid architecture is implemented. Report parameter
counts and compute/memory separately.
Do not infer architecture ranking from random-weight outputs, tokenizer fixtures,
or one successful forward pass. Multi-layer SWA can propagate information farther
than its per-layer window, so evaluate against the actual baseline receptive field.

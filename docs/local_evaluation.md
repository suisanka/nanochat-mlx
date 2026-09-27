# Evaluate CUDA-trained hybrid models on Apple Silicon

Use MLX for the hybrid model on Apple Silicon. CUDA/FLA execution remains on
NVIDIA GPUs. Evaluation is forward-only and does not resume training.

## Preserve and import weights

Download the CUDA checkpoint JSON and matching `.safetensors`, tokenizer JSON
and `contract.json`, and metrics. Keep `.optim.safetensors` and
`.rng.safetensors` separately if CUDA training must be resumable. Verify the
downloads against source SHA-256 values before importing.

```bash
uv run python -m scripts.import_cuda \
  --checkpoint artifacts/h800-d4-4k/cuda/step_00003815.json \
  --tokenizer-dir artifacts/h800-d4-4k/tokenizer \
  --output-dir artifacts/h800-d4-4k/mlx
```

This explicit import accepts only `nanochat-cuda-hybrid-v1`. It checks the exact
tokenizer contract, parameter names, shapes, dtypes and finite values, then saves
an MLX checkpoint with source hashes. It does not quantize weights or import
optimizer, RNG or loader state. The imported checkpoint is for inference or a
new weight-only training phase, not continuation of the CUDA optimizer. Existing
checkpoints are never overwritten. This does not restore support for legacy GPT
weights.

## Check numerical interchange

A short CUDA reference contains `checkpoint_weights_sha256`, checkpoint step, tokenizer contract, input
token IDs, selected positions, NLL sums and token counts in JSON. Its matching
safetensors file contains `logits_0`, `target_logprobs_0`, etc. These arrays are
captured with the CUDA model in evaluation mode, using ordinary forward logits.
Keep references alongside the verified original checkpoint and record which
checkpoint generated them.

```bash
uv run python -m scripts.check_backend_parity \
  --checkpoint artifacts/h800-d4-4k/mlx/step_00003815.json \
  --tokenizer-dir artifacts/h800-d4-4k/tokenizer \
  --reference artifacts/h800-d4-4k/cuda/cuda-reference.json \
  --output artifacts/h800-d4-4k/eval/parity.json
```

The check reports numerical errors and greedy agreement. It also compares MLX
whole-sequence forward with chunked cached prefill. Bounds are BF16 smoke-test
tolerances, not proof of bitwise equivalence or long-context correctness. Keep
the per-sample errors; a passing status alone is not a quality benchmark.

## Score raw text

Prepare a local UTF-8 JSONL file with one complete document per line:

```json
{"id":"example-1","text":"A complete held-out document."}
```

Use complete documents from a documented held-out split, preserve dataset
revision and selection indices, and keep training/monitoring overlap explicit.
Scoring a later slice of the same validation partition is additional validation,
not an independent external benchmark. Do not truncate documents merely to hit
an exact token budget without recording that change.

```bash
uv run python -m scripts.base_eval \
  --checkpoint artifacts/h800-d4-4k/mlx/step_00003815.json \
  --tokenizer-dir artifacts/h800-d4-4k/tokenizer \
  --data artifacts/h800-d4-4k/eval/heldout.jsonl \
  --output artifacts/h800-d4-4k/eval/base-1024.json \
  --context-length 1024 --stride 512 --loss-chunk-size 128 \
  --memory-limit-gb 8
```

Each document starts from zero model state, with one EOS token as its initial
context. There is no chat template and no scored trailing EOS. Rolling windows
score every original text token exactly once, masking repeated context. The
context-length flag caps each window; a stride greater than one gives target
tokens varying amounts of preceding context. Report both parameters.

The scorer computes hidden states first and projects only bounded token tiles
into the vocabulary. It avoids materializing full `[batch, sequence, vocab]`
logits. MLX's memory setting is a working-set guideline, not a hard process RSS
cap. The report includes MLX peak allocation and process peak RSS separately.

Reports include aggregate NLL, token count, original UTF-8 byte count, token
perplexity, BPB, file hashes, tokenizer contract, scoring protocol and timing.
Per-document results are flushed to a sibling `.samples.jsonl`. Existing output
files are rejected. Timing includes first-call compilation but excludes model
loading. Warmed inference and cold start should be measured separately when
comparing performance.

Token perplexity is only comparable with the same tokenizer/protocol. A GPT-2
comparison must use identical raw documents and scoring rules, with a common
context policy that accounts for different tokenizers. Report BPB and separately
run fixed standard tasks. Matching numerical context lengths alone does not
guarantee identical textual context across tokenizers.

This command does not implement HellaSwag, PIQA, LAMBADA or a GPT-2 baseline. The
existing chat evaluation script uses chat framing and letter-token scoring; it
is not a substitute for standard candidate-continuation likelihood evaluation.

## Inspect a raw continuation

```bash
uv run python -m scripts.hybrid_chat \
  --checkpoint artifacts/h800-d4-4k/mlx/step_00003815.json \
  --tokenizer-dir artifacts/h800-d4-4k/tokenizer \
  --raw --prompt 'Water evaporates when' --temperature 0 --max-tokens 128
```

`--raw` is single-prompt continuation with an initial EOS context token and no
chat framing. Chat mode remains the default. Save prompts, decoding parameters
and outputs for comparison after SFT. No 32K execution is required by this local
evaluation workflow.

# nanochat-mlx — Gated DeltaNet + SWA

A single-device MLX research language model for Apple Silicon. The default backbone is **GDN → GDN → sliding-window attention**, with dense SwiGLU, tied embeddings, the official **DeepSeek V4.1 tokenizer and prompt encoder**, and Muon plus auxiliary AdamW.

Only the GDN/SWA hybrid architecture is supported. The old GPT architecture, BPE training, and checkpoint converter have been removed. Existing old-format checkpoints are not hybrid checkpoints. The AdamW-only recipe provides an optimizer control for the same hybrid model.

**Current scope:** architecture implementation, short numerical correctness checks, 4K and 32K training recipes, data preparation, and inference plumbing. No model has been trained as part of this migration. 32K is an engineering/configuration limit; actual 32K execution, memory/performance, and learned long-context ability are not validated.

## Setup

```bash
uv sync --python 3.13
```

Python >=3.10 and MLX >=0.32.2 are declared; the checked environment uses Python 3.13 and MLX 0.32.2. Metal access requires a usable Apple Silicon GPU session. `uv.lock` pins the dependency resolution. PyTorch, RustBPE, tiktoken and Transformers are not required.

Inspect the frozen v0 configuration without loading a model, downloading data, or training:

```bash
uv run python -m scripts.train --depth 12 --dry-run
uv run python -m scripts.train --depth 4 --recipe configs/gdn_swa_32k_memory.json --dry-run
bash runs/quickstart.sh
```

Training only executes with **`--start-training`**. Omitting that switch prints a plan.

## Depth presets

The original depth choices remain available. Width now follows the new architecture: `256 × ceil(depth / 6)`, FFN width is `3 × hidden`, and GDN Q/K width is `0.75 × hidden`. The 12-layer preset exactly matches the referenced 110M design. The GDN/GDN/SWA pattern is tiled and truncated for depths not divisible by three.

| Depth | Hidden | GDN heads | SWA Q/KV heads | SWA head dimension | GDN/SWA layers | Total parameters |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 256 | 3 | 4 / 1 | 64 | 3 / 1 | 36,814,994 |
| 12 | 512 | 6 | 4 / 1 | 128 | 8 / 4 | 109,819,488 |
| 20 | 1024 | 12 | 8 / 2 | 128 | 14 / 6 | 425,495,632 |
| 26 | 1280 | 15 | 20 / 5 | 64 | 18 / 8 | 759,617,564 |

These are exact counts from parameter shapes, not speed or memory benchmarks. The 64-dimensional attention heads at d4/d26 preserve 4:1 GQA while keeping projected attention width equal to hidden width.

## Official tokenizer and chat format

```bash
uv run python -m scripts.prepare_hybrid --install-tokenizer
```

Only tokenizer artifacts are downloaded, not DeepSeek model weights. The pinned upstream revision is `dba1be0a40aa45a94ad051997016db3960a90277` in `deepseek-ai/DeepSeek-V4.1-Flash`. Vocabulary is 129,280, BOS/EOS/PAD IDs are 0/1/2. The model's padding ID is used; upstream `tokenizer_config.json` aliases its pad token to EOS, while model config and tokenizer ID 2 identify the dedicated pad token.

V4.1 publishes a Python prompt encoder rather than a Jinja template. Its unmodified official implementation is vendored with its MIT license in `nanochat_mlx/hybrid/vendor/`. SFT, CLI chat and Web chat share that encoder: system messages, non-thinking/thinking mode, numeric reasoning effort, and V4.1 DSML tool messages retain the upstream format. This model has no vision encoder and rejects image inputs. Encoding tool calls does not execute tools.

Tokenizer and prompt-encoder SHA-256 fingerprints are recorded in prepared data and checkpoints. Mismatches fail before loading/resuming. Ordinary pretraining encodes documents with EOS separators and no chat template.

## Prepare data without training

Local prepared datasets use memory mapping during training, so they are not
loaded entirely into RAM. Preparation processes documents incrementally, but
completes the token files before training starts. To begin reading directly from
the Hub without that preparation step, use the streaming option below.

Use a new output directory for every prepared dataset. Files are little-endian **uint32**, with separate train/validation splits and content fingerprints.

```bash
# Plain text: one document per line; explicitly separated train and validation files.
uv run python -m scripts.prepare_hybrid \
  --train-text /absolute/path/train.txt --val-text /absolute/path/val.txt \
  --output /absolute/path/text-data

# Optional FineWeb download. The last sorted shard becomes validation.
uv run python -m nanochat_mlx.dataset -n 2
uv run python -m scripts.prepare_hybrid \
  --parquet-dir "$HOME/.cache/nanochat/base_data" --output /absolute/path/fineweb-data

# Official-template conversations: JSONL objects containing messages, or message arrays.
uv run python -m scripts.prepare_hybrid \
  --train-chat /absolute/path/train.jsonl --val-chat /absolute/path/val.jsonl \
  --context-length 4096 --output /absolute/path/sft-data

# Synthetic memory data; this command prepares records, not a trained model.
uv run python -m scripts.prepare_hybrid --synthetic 32k \
  --output /absolute/path/memory-32k
```

Conversation training preserves assistant-only masks, including exclusion of external tool results. Complete conversations are packed; oversized conversations fail explicitly. Samples start with zero GDN/conv/cache state. EOS inside a packed sample does not reset state.

The 32K memory profile covers local/boundary recall; gaps of 4K, 8K, 16K, 24K and 30K; distributed overwrite updates; 4–128 keys; parity, state transitions, counters, stack-like dependencies and exact strings. Inputs use the official non-thinking chat framing, with answer-only targets. Every record logs its actual token gap, answer boundary, key count and expected answer. Train/validation use disjoint seeds. See [32K training scenarios](docs/long_context_training.md).

## Stream text directly from Hugging Face

Install the tokenizer once, then select `--stream-dataset` instead of
`--data-dir`. No full parquet download or prepared `.bin` files are required:

```bash
uv run python -m scripts.prepare_hybrid --install-tokenizer

# Inspection is offline: no Hub resolution, dataset loading or training.
uv run python -m scripts.train --recipe configs/gdn_swa_4k.json \
  --depth 4 --stream-dataset karpathy/fineweb-edu-100b-shuffle \
  --stream-val-documents 1024 \
  --output-dir "$HOME/.cache/nanochat/runs/stream-d4" --dry-run
```

To execute, replace `--dry-run` with `--start-training`. Documents are tokenized
on demand with the pinned DeepSeek tokenizer, followed by EOS, and packed into
the same overlapping input/target sequences as the mmap loader. Working memory
depends on the batch, largest document and HF file/row-group buffers, not total
corpus size. Network access and online tokenization affect throughput. Streaming
still reads remote bytes and may cache metadata/file blocks locally.

- `--stream-name`: dataset configuration, if required by the repository.
- `--stream-text-column`: text field (default `text`).
- `--stream-train-split`: training split (default `train`).
- `--stream-val-split`: distinct validation split, if available. Otherwise the
  first `--stream-val-documents` documents (default 1024) are reserved for
  validation and skipped by training. These are document counts, not token counts.
- `--stream-revision`: optional Hub revision. Startup resolves it to an immutable
  commit and prints the pinned source; checkpoints retain that commit.

Use `--resume` with the same streaming flags and training settings. Resume uses
the saved commit even if the Hub branch has moved, checks the tokenizer/source/
context/batch and `datasets` version, and restores both HF iteration state and
unconsumed tokens. Reading part of the current shard again may be necessary.
Source order is preserved: HF shuffle buffers are deliberately not used because
their contents are lost on resume. Prefer a source already shuffled when needed.
Empty/too-short splits and missing/non-string text fields fail explicitly.

This path is for ordinary text pretraining at the configured context length,
including 32K. Synthetic memory recipes and chat SFT still use prepared datasets
to preserve record boundaries and assistant-only masks. The Web workbench offers
the same entry under **Training data → Hugging Face streaming text**. The loader
uses the official [HF streaming and checkpoint APIs](https://huggingface.co/docs/datasets/stream).

## Training recipes (prepared, not executed)

- `configs/gdn_swa_4k.json`: frozen hybrid backbone, natural-language validation recipe.
- `configs/gdn_swa_4k_memory.json`: 4K synthetic memory recipe.
- `configs/gdn_swa_32k_memory.json`: 32K memory training, 131,072 tokens/step, block checkpointing.
- `configs/gdn_swa_4k_adamw_control.json`: AdamW-only optimizer control.

For a future authorized run, add `--start-training` to an inspected command:

```bash
uv run python -m scripts.train --recipe configs/gdn_swa_4k.json \
  --depth 12 --data-dir /absolute/path/text-data --dry-run

# Weight warm-start for the 32K phase; optimizer starts fresh.
uv run python -m scripts.train --recipe configs/gdn_swa_32k_memory.json \
  --data-dir /absolute/path/memory-32k \
  --init-from /absolute/path/base-checkpoint.json \
  --output-dir /absolute/path/32k-run --dry-run

# SFT requires an initial checkpoint or a complete resume checkpoint.
uv run python -m scripts.sft --depth 12 --data-dir /absolute/path/sft-data \
  --init-from /absolute/path/base-checkpoint.json --dry-run
```

`--resume` restores model, FP32 optimizer/master state and the data-loader position (mmap cursor or streaming iterator plus pending tokens); configuration, dataset identity, total steps and micro-batch size must match. `--init-from` is a new training phase, permits compatible context changes, and uses a fresh optimizer. Checkpoints are written under `hybrid_checkpoints/<architecture>/d<depth>/<base|sft>/`. Metadata is published last so incomplete saves are not discoverable. Previous checkpoints are retained.

For throughput tuning, `--device-batch-size` changes the micro-batch and the
trainer adjusts accumulation to preserve `--total-batch-size` tokens per update.
The latter must remain exactly divisible by micro-batch × context. A larger
`--loss-chunk-size` reduces CE tile launches and FP32 gradient copies at the cost
of larger temporary logits. `--no-checkpoint-blocks` disables recomputation when
memory permits. Measure candidate settings on the target machine before a run.

`--memory-limit-gb` is MLX's working-set guideline, **not a hard process memory
cap**. `--cache-limit-gb` (default 1) separately limits unused cached allocations.
Leave headroom for transient buffers, host copies and system use. Training metrics
report peak MLX allocations in GiB. Use `--save-first-step` to save a complete
checkpoint after the first update as well as the normal `--save-every` interval.

## 32K context engineering

The baseline training context remains 4K and the maximum supported configuration is 32,768 tokens. SWA keeps a 1,024-token window and absolute positions after eviction. GDN has no positional embedding. Prefill is chunked; intermediate prefill chunks do not project a full vocabulary tensor. GDN uses FP32 recurrent accumulation, so its cache is larger than the design document's hypothetical BF16-state budget but remains independent of sequence length.

Native RoPE (`theta=10000`) remains the default: SWA's local distance range is unchanged at 32K. Optional fixed linear or YaRN scaling is serialized in checkpoints, e.g. `--rope-scaling linear --rope-factor 8`. Frequencies do not change mid-generation, preventing stale cached-key bases. These options are capabilities, not measured quality recommendations.

## Chat and evaluation

```bash
uv run python -m scripts.chat --checkpoint /absolute/path/checkpoint.json --interactive
uv run python -m scripts.chat --checkpoint /absolute/path/checkpoint.json \
  --thinking-mode thinking --reasoning-effort 75 -p "Explain the result."

uv run python -m scripts.chat_eval --checkpoint /absolute/path/checkpoint.json \
  --tokenizer-dir "$HOME/.cache/nanochat/deepseek_tokenizer" -a 'GSM8K|MMLU'

uv run python -m scripts.hybrid_eval --checkpoint /absolute/path/checkpoint.json \
  --tokenizer-dir "$HOME/.cache/nanochat/deepseek_tokenizer" \
  --data-dir /absolute/path/memory-data --output /absolute/path/results.json

uv run python -m scripts.quickstart
```

The local Web workbench defaults to `http://127.0.0.1:8000`: inspect depth/context, install tokenizer, prepare data, inspect training plans, load hybrid checkpoints and stream chat. Its training checkbox is off by default. No old-model import mode remains.

## Verification

```bash
# Install tokenizer first, then enable real-tokenizer adapter/data/API checks.
NANOCHAT_TEST_TOKENIZER="$HOME/.cache/nanochat/deepseek_tokenizer" \
  uv run python -m pytest tests -v
```

Without `NANOCHAT_TEST_TOKENIZER`, tokenizer-dependent checks skip explicitly; tests never download it automatically. Tests include official DeepSeek golden cases, GDN output/state/gradient parity at short lengths, local attention parity, bounded loss and both gradients, BF16 behavior, decode/cache parity, serialization, optimizer grouping and synthetic update equations, mmap resume, conversation masks and API chat with a tiny untrained model. They do **not** run training loops or actual 32K sequences.

See [architecture and validation boundaries](docs/architecture.md) and [release checklist](RELEASE_CHECKLIST.md). The project is an implementation for research; learned retrieval, architecture ranking, throughput and training stability require future experiments.

## Attribution

Originally based on Karpathy's nanochat MLX port. The new design follows the referenced GDN/SWA architecture, the [FLA Gated DeltaNet implementation](https://github.com/fla-org/flash-linear-attention), [Keller Jordan's Muon](https://github.com/KellerJordan/Muon), [MLX](https://github.com/ml-explore/mlx), and [DeepSeek V4.1's official prompt encoding](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/dba1be0a40aa45a94ad051997016db3960a90277/encoding). Third-party license notices are retained under `nanochat_mlx/hybrid/vendor/`.

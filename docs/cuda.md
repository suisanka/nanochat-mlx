# nanochat_cuda: PyTorch and FLA

`nanochat_cuda/` implements the same GDN/GDN/SWA hybrid architecture as the MLX
backend on a **single NVIDIA GPU**. The architecture, depth presets, DeepSeek
V4.1 tokenizer/prompt encoder, prepared data, streaming cursors, and 4K/32K recipes
are shared. Importing the shared contracts does not import MLX. Existing MLX CLI
commands and Web UI continue to select MLX; CUDA uses separate CLI entry points.

## Install

From the repository root on Linux, with an NVIDIA driver compatible with the
selected PyTorch CUDA runtime:

```bash
uv sync --python 3.13 --extra cuda --locked
uv run --extra cuda python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

The lock pins **PyTorch 2.14.0**, **fla-core 0.5.2**, and **Triton 3.8.0** on
Linux. PyPI's pinned PyTorch Linux wheel uses CUDA 13.0. CUDA BF16 support is
required by this training backend. Consult the [official PyTorch installation
selector](https://pytorch.org/get-started/locally/) when selecting a runtime for
your driver/hardware; changing the pinned PyTorch stack requires relocking and
rerunning the CUDA checks below.

FLA's [official backend installation guidance](https://github.com/fla-org/flash-linear-attention/blob/main/INSTALL.md)
separates the kernels (`fla-core`) from optional model/Transformers integration.
This project uses the kernels directly and does not require Transformers or
download pretrained DeepSeek model weights.

The default SWA implementation is PyTorch SDPA over bounded local tiles. To use
FlashAttention-2's fused sliding-window kernel, install it into the same environment
after PyTorch, following [upstream requirements](https://github.com/Dao-AILab/flash-attention#installation-and-features):

```bash
uv pip install packaging psutil ninja
MAX_JOBS=4 uv pip install --no-build-isolation 'flash-attn==2.8.3.post1'
# Preserve the optional extension when invoking commands:
uv run --no-sync python -m scripts.train_cuda --attention-backend flash --dry-run
```

FlashAttention is an optional native extension, installed separately because its
build depends on the host CUDA toolchain. It is not part of `uv.lock`; another
exact `uv sync` can remove it. Selecting `--attention-backend flash` requires it
and fails explicitly if unavailable. There is no silent dense-attention fallback.

For CPU numerical checks on macOS/Linux, use `uv sync --extra torch --locked`.
CPU reference GDN is deliberately slow and intended only for short checks;
the normal CUDA path always uses FLA. CUDA training cannot run on Apple Silicon.

## Inspect and train

Inspection does not import PyTorch/FLA/MLX, initialize a GPU, resolve the Hub,
download a tokenizer, or start training:

```bash
uv run --extra cuda python -m scripts.train_cuda \
  --recipe configs/gdn_swa_4k.json --depth 4 --dry-run
uv run --extra cuda python -m scripts.train_cuda \
  --recipe configs/gdn_swa_32k_memory.json --depth 12 --dry-run
```

Install the exact same official tokenizer once, then explicitly start training:

```bash
uv run --extra cuda python -m scripts.prepare_hybrid --install-tokenizer

uv run --extra cuda python -m scripts.train_cuda \
  --recipe configs/gdn_swa_4k.json --depth 4 \
  --context-length 4096 --device-batch-size 4 \
  --loss-chunk-size 1024 --checkpoint-blocks \
  --stream-dataset karpathy/fineweb-edu-100b-shuffle \
  --stream-val-documents 1024 \
  --output-dir "$HOME/.cache/nanochat-cuda" \
  --save-first-step --save-every 100 --eval-every 100 \
  --start-training
```

Batch 4 is an example, not a measured fit or optimum for your GPU. The effective
batch is 131,072 tokens by default: at 4K and batch 4, accumulate 8 microbatches.
Use `--device-batch-size`, `--loss-chunk-size`, and `--checkpoint-blocks` /
`--no-checkpoint-blocks` to fit the target GPU. `--cuda-memory-fraction 0.8`
optionally limits the PyTorch caching allocator to a fraction of device memory;
it is not a total process/driver memory cap. MLX's memory/cache flags do not apply.

`--compile` is enabled by default. It compiles SwiGLU blocks and Muon's
Newton–Schulz kernel with `torch.compile`; FLA uses its own Triton JIT kernels.
The training loop, cache mutation, and recomputed vocabulary loss are eager.
There is no claim of full-graph training compilation. Use `--no-compile` to
isolate compiler problems. First-call compilation costs are included in step time.

The trainer supports one process/GPU. `WORLD_SIZE > 1` is rejected; DDP, FSDP,
context parallelism, and multi-node training are not implemented.

## Data, phases, and checkpoints

Use `--data-dir /path/prepared-data` instead of `--stream-dataset` for the existing
uint32 mmap format. All preparation commands in the main README apply on Linux
without importing MLX. Base streaming preserves the pinned Hub SHA, exact pending
token buffer and dataset iterator state. Online SFT/synthetic streaming is not
supported. SFT uses the same official assistant-only masks as MLX.

CUDA checkpoints are isolated at:

```text
<output-dir>/cuda_checkpoints/gdn_swa/d<depth>/<base|sft>/
  step_00000100.json
  step_00000100.safetensors
  step_00000100.optim.safetensors
  step_00000100.rng.safetensors
  metrics.jsonl
```

Metadata is published last. Checkpoints preserve BF16 weights, FP32 optimizer
masters/moments, step, data cursor, tokenizer contract, configuration, and PyTorch
CPU/CUDA RNG. Resume with the original command plus `--resume /path/step_*.json`;
model, batch, total steps, backend settings, PyTorch version and data contract
must match. A normal interrupt does not save a new checkpoint; resume uses the
last completed checkpoint. Existing checkpoint files are not overwritten.

Use `--init-from /path/step_*.json` for a **new phase with a fresh optimizer and
data cursor**. It accepts CUDA checkpoints and the current MLX hybrid format.
Parameter names/layouts are shared and loaded with safetensors without importing
MLX. Tensor architecture and tokenizer must match. Context/RoPE settings can
change; kernel numerics differ between backends, so interchange is not a claim
of bit-for-bit training equivalence. MLX optimizer/RNG state is not resumed on CUDA.

Examples of additional phases (explicit training only when the final switch is
provided):

```bash
# Prepare 32K memory scenarios; this does not execute a model.
uv run --extra cuda python -m scripts.prepare_hybrid \
  --synthetic 32k --output /data/memory-32k

# Inspect the 32K continuation phase; use a distinct output directory.
uv run --extra cuda python -m scripts.train_cuda \
  --recipe configs/gdn_swa_32k_memory.json --depth 4 \
  --init-from /checkpoints/step_00001000.json \
  --data-dir /data/memory-32k --output-dir /checkpoints/memory-32k --dry-run

# Inspect SFT on data prepared with --train-chat / --val-chat.
uv run --extra cuda python -m scripts.train_cuda \
  --recipe configs/gdn_swa_4k.json --depth 4 --source sft \
  --init-from /checkpoints/step_00001000.json \
  --data-dir /data/sft --output-dir /checkpoints/sft --dry-run
```

32K is an engineering limit, not a measured GPU memory/performance result or
evidence of learned long-context ability. No actual 32K test is required by this
implementation.

## Inference

```bash
uv run --extra cuda python -m scripts.chat_cuda \
  --checkpoint /checkpoints/step_00001000.json \
  --prompt 'Hello' --max-tokens 128
```

`--interactive`, `--thinking-mode thinking`, `--reasoning-effort`, `--temperature`,
`--top-k` and `--prefill-chunk-size` are supported. Prefill emits only final-token
logits; GDN uses FP32 recurrent state and bounded convolution history. SWA evicts
KV only after all prefill queries, using absolute RoPE positions. Both KV and conv
tails own bounded storage. Caches are inference-only and each training sample
starts with zero state. Use an appropriately trained/SFT model for useful chat.

The memory-scenario evaluator uses the same ledger and exact-match buckets as MLX:

```bash
uv run --extra cuda python -m scripts.eval_cuda \
  --checkpoint /checkpoints/step_00001000.json \
  --tokenizer-dir "$HOME/.cache/nanochat/deepseek_tokenizer" \
  --data-dir /data/memory-4k --max-problems 100 --output /tmp/cuda-eval.json
```

## Verification

```bash
# CPU reference, short sequences, no corpus training.
uv run --extra torch python -m pytest tests/cuda -m 'not cuda' -q

# NVIDIA host: real FLA forward/backward/recurrent state and optional FlashAttention.
uv run --extra cuda python -m pytest tests/cuda -m cuda -q
# With separately installed FlashAttention:
uv run --no-sync python -m pytest tests/cuda -m cuda -q
```

CPU tests check loss/gradient equivalence to dense cross entropy, absence of saved
full vocabulary logits, local attention against a short dense oracle, causal/cache
equivalence including eviction, RoPE variants, activation checkpointing, compile
tracing, optimizer equations, safetensors/RNG recovery, and configuration guards.
GPU tests require NVIDIA hardware and are skipped on macOS. CPU checks do not
validate FLA/Triton kernels, CUDA compilation, GPU throughput or GPU peak memory.

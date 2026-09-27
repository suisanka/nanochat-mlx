# GDN/SWA implementation contract

Source design: the final v0 architecture in [解读Qwen架构](chatgpt-conversation://6ab9014c-2528-83ee-92b2-e4dd17951fae). Subsequent requirements preserve the depth dial, cap context engineering at 32K, add long-sequence training scenarios, use the official DeepSeek V4.1 tokenizer/chat format, remove old-model modes, and do not start training or actual long-context tests.

## Frozen d12 anchor

- Decoder-only, 12 pre-norm blocks, hidden 512, dense SwiGLU intermediate 1536.
- `[GDN, GDN, SWA] × 4`; no global attention in the default architecture.
- Vocabulary 129,280; one tied input/output matrix; no dropout, linear biases,
  value embeddings, residual lambdas or logit softcap.
- RMSNorm epsilon 1e-5. Normal weight initialization std 0.02; mixer output and
  FFN down-projection std `0.02 / sqrt(2*layers)`.
- GDN: six heads, key dimension 64, value dimension 128; depthwise causal Q/K/V
  convolution of width 4 and SiLU; Q/K L2 normalization; per-head update sigmoid
  and decay `-exp(A_log)*softplus(a+dt_bias)`; output RMSNorm and SiLU gate.
- GDN initialization follows the FLA reference: A uniform in (0,16), dt
  log-uniform in [0.001,0.1], inverse softplus bias. GDN has no RoPE.
- SWA: four query heads, one KV head, head dimension 128; causal local window
  1024 including the current token. Split-half full-dimension RoPE, theta 10000.
- 109,819,488 parameters, including 66,191,360 in the tied embedding.

## MLX backend adaptation

The linked design was written around nanoGPT/PyTorch/FLA CUDA. This repository
remains single-device MLX; there is no CUDA, PyTorch or DDP dependency. Backend
choices preserve the specified equations and expose numerical correctness paths:

| Design requirement | Implementation | Validation |
|---|---|---|
| GDN chunk size 64 | `gdn.chunk_gated_delta_rule`, parallel lower-triangular delta solve within each chunk | Short reference output/state and Q/K/V/gate/state gradients |
| Independent recurrent reference | `gdn.naive_recurrent_gdn` | T=16,32,64 and uneven T=79 |
| Recurrent decode | Compiled one-token MLX recurrence, same FP32 state contract | Chunked prefill and incremental decode vs full forward |
| Local-aware attention | Query tiles only submit their relevant local K/V slice to SDPA | Dense local-mask reference output and gradients |
| Fused linear CE | MLX custom operation accepting hidden/weight/targets, bounded token tiles and explicit recomputation VJP | Vanilla CE and hidden/weight gradients, float32/BF16 |
| Large-vocabulary inference | Last-position projection and bounded prefill chunks | Short cache/full-forward parity |
| Training precision | BF16 model weights, FP32 gate math/recurrent accumulation/optimizer master state | BF16 loss/gradient and checkpoint checks |
| Activation checkpointing | MLX parameter-aware block checkpointing | Short loss/backward test |

The linear CE implementation fuses the operation boundary and controls logits
lifetime; it is not FLA's CUDA kernel or a throughput-equivalent replacement.
Likewise, the GDN chunk implementation uses MLX matrix operations; fused Metal
kernel tuning is a future performance experiment. No full `[B,T,V]` tensor is
retained by training loss, and SWA never constructs a full-sequence `T*T` mask.

At the d12 BF16 anchor, persistent state is approximately 1.5 MiB GDN FP32 state,
2 MiB SWA KV, plus 72 KiB short-convolution state per sequence. This excludes
transient activations, logits, model weights and allocator overhead. It is a
shape-derived budget, not a measured peak-memory claim.

## Optimizer and schedule

Muon only owns Q/K/V/G/O representation projections and SwiGLU matrices. Tied
embedding, a/b controller projections, convolution weights, norms, A_log and
dt_bias use AdamW. No decay on norms/A_log/dt_bias; weight decay 0.01 elsewhere.
Muon lr 0.02, momentum 0.95, Nesterov, five Newton–Schulz iterations with the
Jordan aspect-ratio update scaling. AdamW lr 3e-4, betas (0.9,0.95), epsilon 1e-10.
Gradient clip 1.0; 2% linear warmup and cosine to 10% of peak learning rate.
The AdamW-only control retains architecture/data/other settings.

The batch contract is tokens per optimizer step, default 131,072. Exact
micro-batch divisibility is required rather than silently rounding. Gradient
accumulation weights micro-batches by supervised target count, which matters for
variable-length answers. Optimizer moments and master weights remain FP32.

## Data, prompt format and persistence

The official tokenizer and unmodified official Python prompt encoder are pinned
at the same DeepSeek V4.1 revision. Ordinary documents use EOS separators and
uint32 storage; conversation and memory-task data use the official chat format.
SFT masks come from official rendered assistant character spans and tokenizer
offsets; user/system/tool-result content is not supervised. Conversation packing
rejects oversized records, preventing non-progressing padding loops.

No state is carried between training samples. There is no EOS reset inside a
packed sample. Resumption tracks mmap cursor/epoch and data fingerprint exactly.
The optional online text loader pins a Hugging Face dataset commit and preserves
source order. It stores HF iteration state, epoch, unconsumed token buffer,
tokenizer/source/context/batch contract and datasets version. Validation either
uses a distinct split or holds out a fixed document prefix from training. Local
parquet tests check cross-shard, mid-document and epoch-boundary resume against
the mmap token sequence; online text uses the same EOS packing. SFT and synthetic
tasks retain prepared record/mask data. A remote stream may buffer file blocks or
row groups; this is not a zero-cache or latency-free promise.
Checkpoints serialize all architecture, context/RoPE, optimizer, tokenizer and
loader settings. Model tensors load strictly. Optimizer state includes master
weights, moments and step; missing or mismatched state fails closed. Metadata is
the save commit marker; incomplete transactions are not listed by the Web UI.
Old-format model weights have no import or compatibility mode.

## Context engineering

The baseline training length is 4096. `max_context` is capped at 32768 and applies
to prompt plus generated tokens. Larger values are rejected. SWA retains absolute
positions after eviction; it evicts only after all queries in a multi-token
prefill have attended. GDN state and convolution history are cached separately.
Cache reset and batch expansion are explicit operations.

Native RoPE is the default even for the 32K recipe: the local attention distance
range stays 1024. Fixed linear and YaRN frequency scaling are optional experiments,
serialized before cache construction. No dynamic frequency changes occur inside a
stream. A changed context/RoPE config requires a new cache. Engineering support
and recipe construction do not establish trained 32K capability.

The reference discussion's claim that all dependencies beyond 1024 tokens must
use GDN is too strong: multiple SWA layers can propagate information indirectly
across a wider receptive field. Only *direct* attention is limited to 1024.
The 16K–30K scenarios are useful controls beyond the d12 pure-SWA receptive field;
all architecture claims still require actual matched-budget experiments.

## Research controls and deferred phases

The current implementation supports only GDN/SWA hybrid, as requested. The
AdamW-only recipe retains the same architecture and provides an optimizer
control. Pure GDN, pure SWA and full-attention modes from the original research
plan are not available. Comparing those architectures would require a separately
authorized extension and matched data/token budgets, with parameter counts and
compute reported explicitly.

Short correctness tests are the current delivery gate. Synthetic training,
natural-language smoke training, 500M–1B architecture-ranking runs and KDA
replacement remain separate, unexecuted research phases. KDA/1B scale-up, MoE,
MTP, sparse global retrieval and context beyond 32K are not implemented or claimed.

Training code can record loss, validation loss, alpha/beta histograms, recurrent
state norm, Q/K/V norms, output gates, parameter gradient norms, optimizer update
RMS and SWA entropy. Diagnostics are reduced per layer so their raw activations
are not retained across layers. No monitoring results exist until a run is
explicitly started.

## Primary implementation references

- FLA GatedDeltaNet and naive recurrence: https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/gated_delta_rule
- GDN layer/init: https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/gated_deltanet.py
- Jordan Muon: https://github.com/KellerJordan/Muon/blob/master/muon.py
- MLX custom transforms: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.custom_function.html
- MLX-LM RoPE scaling: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/rope_utils.py
- DeepSeek V4.1 encoding: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/dba1be0a40aa45a94ad051997016db3960a90277/encoding

---
language:
  - en
pipeline_tag: text-generation
tags:
  - neko
  - causal-lm
  - gated-deltanet
  - sliding-window-attention
  - pytorch
  - mlx
  - pretrained
  - research
datasets:
  - karpathy/fineweb-edu-100b-shuffle
---

<div align="center">
  <h1>Neko V0 Preview</h1>
  <p><strong>A small hybrid language model for reproducible research.</strong></p>
  <p>36.8M parameters · 500M training tokens · 4K training context</p>
  <p>Gated DeltaNet + Sliding-Window Attention · PyTorch / CUDA · Apple Silicon / MLX</p>
  <p>
    <a href="#model-introduction">Introduction</a> ·
    <a href="#model-overview">Model Overview</a> ·
    <a href="#evaluation">Evaluation</a> ·
    <a href="#quickstart">Quickstart</a> ·
    <a href="#limitations">Limitations</a>
  </p>
</div>

> **Research preview.** Neko V0 Preview is a pretrained base model. It has completed a 500M-token training run and local evaluation, but has not undergone instruction tuning, preference optimization, or reinforcement learning. Generated text remains unreliable.

<a id="model-introduction"></a>

## Model Introduction

**Neko V0 Preview** is a 36.8-million-parameter, text-only language model trained from scratch on English FineWeb-Edu text. It combines Gated DeltaNet (GDN) with sliding-window attention (SWA) in a four-layer hybrid backbone.

This first preview establishes a reproducible path from single-GPU CUDA pretraining to local inference on Apple Silicon. Training completed on one NVIDIA H800 80GB GPU. The final checkpoint was imported into MLX without quantization and evaluated on an Apple M4.

The model is intended for architecture experiments, training-system development, local inference studies, and small-scale fine-tuning research. Its primary contribution at this stage is a working, inspectable training and evaluation pipeline.

**Highlights**

- **Hybrid sequence modeling.** Three GDN blocks and one local-attention block, with dense SwiGLU feed-forward networks and tied token embeddings.
- **Training from scratch.** 500,039,680 tokens processed over 3,815 optimizer steps, using Muon with auxiliary AdamW parameter groups.
- **Two execution backends.** PyTorch / FLA for NVIDIA GPUs and MLX for Apple Silicon, with an explicit checkpoint import path.
- **Pinned tokenization.** The DeepSeek V4.1 tokenizer and official Python prompt encoder are bound to a fixed upstream revision and recorded fingerprints.
- **Recorded evaluation.** Raw-text validation, short cross-backend numerical checks, and fixed generation diagnostics accompany the experiment.

<a id="model-overview"></a>

## Model Overview

| Property | Neko V0 Preview |
|---|---|
| Model type | Dense, autoregressive, text-only base language model |
| Training stage | Pretraining only |
| Total parameters | **36,814,994** |
| Tied embedding parameters | 33,095,680 |
| Non-embedding parameters | 3,719,314 |
| Number of layers | 4 |
| Layer layout | GDN → GDN → SWA → GDN |
| Hidden dimension | 256 |
| Feed-forward network | SwiGLU, intermediate dimension 768 |
| GDN heads | 3 |
| GDN head dimensions | Key: 64; value: 128 |
| GDN convolution / chunk size | 4 / 64 |
| SWA heads | 4 query heads / 1 key-value head |
| SWA head dimension | 64 |
| SWA window | 1,024 tokens |
| Position handling | GDN: no positional embedding; SWA: RoPE, θ = 10,000 |
| RoPE scaling in this checkpoint | None |
| Vocabulary size | 129,280 |
| Output projection | Tied to input token embeddings |
| Main weight precision | BF16 |
| Training context length | **4,096 tokens** |
| Configured maximum context | 32,768 tokens; engineering limit only |
| Training language | Primarily English |
| Training hardware | 1 × NVIDIA H800 80GB |
| Local evaluation hardware | Apple M4 |

Token embeddings account for approximately **89.9%** of the total parameter count. Only about **3.72M parameters** lie outside the shared embedding matrix. Comparisons with other small models should account for this allocation rather than relying on total parameter count alone.

The 32K configuration limit has not been validated by a 32K training or evaluation run. This checkpoint was trained at 4K; reliable long-context retrieval and reasoning have not been established.

## Training

Neko V0 Preview was initialized from scratch and trained with a causal next-token prediction objective. Documents were tokenized as ordinary text, separated by EOS, and packed into training sequences. Pretraining did not use a conversation template.

| Setting | Value |
|---|---|
| Dataset | [`karpathy/fineweb-edu-100b-shuffle`](https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle) |
| Dataset revision | `4c8f30d6756da75362432a4d5569e1b229263b71` |
| Data loading | Streaming with Parquet caching |
| Validation holdout | First 1,024 source documents excluded from training |
| Processed training tokens | **500,039,680** |
| Optimizer steps | **3,815** |
| Sequence length | 4,096 |
| Effective batch | 131,072 tokens per optimizer step |
| Microbatch | 16 sequences |
| Gradient accumulation | 2 microbatches per optimizer step |
| Optimizers | Muon and auxiliary AdamW |
| Initial learning rates | Muon: 0.02; AdamW: 0.0003 |
| Warmup fraction | 0.02 |
| Minimum learning-rate ratio | 0.1 |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |
| CUDA implementation | PyTorch 2.14.0+cu130; FLA GDN and fused loss; SDPA attention; compilation enabled |
| Final training-batch loss | 4.232251 |
| Last periodic validation loss | 4.206085 at step 3,800 |

Training-batch loss and periodic validation loss describe different samples and aggregation scopes. Neither is a standardized external benchmark score. The token count above is the number of processed training tokens, including the training stream's separators.

## Tokenizer and Prompt Format

The tokenizer comes from [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), pinned to revision `dba1be0a40aa45a94ad051997016db3960a90277`. Its vocabulary contains 129,280 entries, with BOS / EOS / PAD IDs of 0 / 1 / 2 in the project's tokenizer contract.

The runtime also includes the corresponding official Python prompt encoder for structured conversations. Neko V0 Preview uses no DeepSeek model weights. Sharing tokenization and prompt syntax does not transfer the upstream model's knowledge, reasoning, or tool-use abilities.

**Raw continuation is the appropriate starting point for this checkpoint.** Chat formatting is available in the runtime, but the model has not been trained to follow that format. Image inputs are unsupported.

<a id="evaluation"></a>

## Evaluation

### Raw-text language modeling

The final checkpoint was evaluated with MLX on **114 documents, 105,884 tokens, and 520,011 UTF-8 bytes**.

| Maximum scoring context | Mean token NLL ↓ | Token perplexity ↓ | Bits per byte ↓ |
|---:|---:|---:|---:|
| 1,024 | 4.274308 | 71.830 | 1.255620 |
| 4,096 | **4.268213** | **71.394** | **1.253829** |

**Protocol.** Documents 768–881, using zero-based indices, were selected from the first source shard. They belong to the 1,024-document partition excluded from training. Their prefix begins after 633,462 tokens, beyond the 131,072-token prefix used for periodic monitoring. This is an additional slice of the same holdout partition, not an independent external benchmark.

Each document starts from a reset model state with one EOS context token. Scoring uses no chat template and no scored trailing EOS. Rolling windows use stride 512, with every original text token scored once. Vocabulary projection is tiled in groups of 128 positions. BPB uses the original UTF-8 byte count.

Token perplexity is specific to this tokenizer and protocol. The small difference between the two context settings has not been tested for statistical significance and does not establish long-context capability.

### CUDA / MLX numerical checks

Two short samples were checked against saved CUDA reference outputs from the final checkpoint:

| Sample length | Absolute mean NLL difference | Greedy agreement at sampled positions |
|---:|---:|---:|
| 16 tokens | 0.002069 | 100% |
| 101 tokens | 0.006382 | 100% |

Both samples passed the recorded BF16 numerical tolerances. Agreement covers three selected positions per sample. These are short numerical checks, not bitwise-equivalence or full-context correctness guarantees.

### Generation diagnostics

Fixed diagnostics include eight greedy prompts and four sampled prompts. The sampled runs use temperature 0.8, top-k 50, and seed 42.

| Prompt | Recorded observation |
|---|---|
| `2 + 2 =` | Greedy continuation starts with `1`; sampled continuation starts with `50`. Both are incorrect. |
| `The capital of France is` | Greedy output repeats; sampled output does not provide a reliable answer. |
| `人工智能是` | Sampled output switches to incoherent English. |

The model produces some recognizable English phrasing, but repetition, factual errors, and arithmetic failures remain prominent. Chinese generation is not reliable in the recorded diagnostics.

No HellaSwag, PIQA, LAMBADA, or same-protocol GPT-2 baseline results are available. This preview makes no claim of matching GPT-2 quality.

<a id="quickstart"></a>

## Quickstart

Neko V0 Preview uses the project's custom hybrid checkpoint format and runtime. A Transformers `AutoModelForCausalLM` adapter and vLLM / SGLang integrations have not been provided.

The examples below run from a `nanochat-mlx` project checkout on Apple Silicon with its dependencies installed. They assume the experiment's local artifact bundle is available at `artifacts/h800-d4-4k/`. These artifacts are separate from the source checkout; this card does not specify a public model download endpoint.

The MLX path was evaluated with Python 3.13 and MLX 0.32.2. Use the project's `uv.lock` for the dependency environment.

### Interactive continuation on Apple Silicon

```bash
uv run python -m scripts.hybrid_chat \
  --checkpoint artifacts/h800-d4-4k/mlx/step_00003815.json \
  --tokenizer-dir artifacts/h800-d4-4k/tokenizer \
  --interactive --raw \
  --temperature 0.8 --top-k 50 \
  --max-tokens 128 --memory-limit-gb 8
```

Enter a text prefix at `Text:`. Each prompt is independent. Use `exit`, `quit`, or Ctrl-D to finish. The memory flag is an MLX working-set guideline, not a hard process memory cap.

### Single-prompt greedy continuation

```bash
uv run python -m scripts.hybrid_chat \
  --checkpoint artifacts/h800-d4-4k/mlx/step_00003815.json \
  --tokenizer-dir artifacts/h800-d4-4k/tokenizer \
  --raw --prompt 'Water evaporates when' \
  --temperature 0 --max-tokens 128 --memory-limit-gb 8
```

These settings are starting points for inspecting behavior, not validated optimal decoding parameters. Removing `--raw` enables the conversation encoder; it does not make this checkpoint instruction-tuned.

### Repeat the 4K text evaluation

With the preserved evaluation documents available, choose a new output filename:

```bash
uv run python -m scripts.base_eval \
  --checkpoint artifacts/h800-d4-4k/mlx/step_00003815.json \
  --tokenizer-dir artifacts/h800-d4-4k/tokenizer \
  --data artifacts/h800-d4-4k/eval/heldout.jsonl \
  --output artifacts/h800-d4-4k/eval/neko-v0-preview-4096-rerun.json \
  --context-length 4096 --stride 512 --loss-chunk-size 128 \
  --memory-limit-gb 8
```

The evaluator refuses to overwrite an existing report. It also writes per-document results to a sibling `.samples.jsonl` file.

<a id="limitations"></a>

## Intended Use and Limitations

Neko V0 Preview supports research into hybrid language-model architectures, small-scale pretraining, checkpoint interoperability, and local inference. It can also serve as an experimental starting checkpoint for further training.

- **Base-model behavior.** SFT, DPO, and RL have not been performed. Instruction following and multi-turn dialogue have not been established.
- **Limited generation quality.** Existing diagnostics show repetition, unsupported claims, and basic arithmetic errors. Outputs require independent checking.
- **English-focused training.** Tokenizer coverage does not establish multilingual proficiency. Chinese ability is unverified beyond the unsuccessful recorded examples.
- **Unvalidated long context.** Training used 4K sequences. Actual 32K execution and learned long-context ability have not been evaluated.
- **Narrow evaluation coverage.** Reported language-modeling scores cover one small holdout slice. There are no standard task results, statistical confidence intervals, or broad robustness assessments.
- **Safety evaluation pending.** This experiment has not characterized harmful-content generation, bias, privacy leakage, or adversarial robustness.

## Reproducibility

The experiment preserves the final CUDA and MLX checkpoints, tokenizer contract, optimizer and RNG state, training metrics, evaluation inputs, numerical references, and generation diagnostics. CUDA-to-MLX import preserves the stored weight precision without quantization; optimizer, RNG, and data-loader state are not imported into MLX.

| Record | Value |
|---|---|
| Experiment date | 2026-09-28 |
| Final checkpoint step | 3,815 |
| Training code commit | `dc9e301a310827335c1389672e9e9511f35c51ad` |
| Evaluation code commit | `b9d5370b5b1bee3cc60973a29a580240b30c5f88` |
| Interactive raw CLI commit | `84ac516c7223b69875bf1fce41491f711780e2af` |

<details>
<summary>Artifact fingerprints and local evidence</summary>

| Artifact | SHA-256 |
|---|---|
| CUDA weights | `62b1d1e10c4a91c4de79010791d3968e0aa7cc99046a26af0773f910b38df289` |
| Imported MLX weights | `21935f280d38984f23aa326006bba56b23049b63b12fb950305b52c9bebde704` |
| Tokenizer JSON | `c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b` |
| Prompt encoder fingerprint | `502bdaec8a3fd88ebc24c4721a7038fbe42f2063c664638127056107920035c1` |
| Evaluation JSONL | `8af026458d08b588174ae7834abd2eaadc4c473f26f5115646017523adc072f8` |

The local artifact root is `artifacts/h800-d4-4k/`. Relevant evidence files are:

- `experiment-summary.json` and `EVALUATION.md`
- `cuda/download-manifest.json` and `cuda/metrics.jsonl`
- `cuda/step_00003815.json` and `mlx/step_00003815.json`
- `tokenizer/contract.json`
- `eval/heldout.manifest.json`
- `eval/base-1024.json` and `eval/base-4096.json`
- `eval/parity-verified.json`
- `eval/generations.json` and `eval/generations-sampling.json`

</details>

## License and Acknowledgments

The project's source code carries an MIT license. A separate license for distributing the Neko V0 Preview weights has not yet been specified; no weight-license claim is made in this card. Dataset and tokenizer provenance are recorded above.

This experiment uses the nanochat-mlx hybrid implementation, PyTorch, Flash Linear Attention, MLX, FineWeb-Edu, and DeepSeek's tokenizer and prompt encoder.

The presentation follows the model-introduction, architecture-summary, evaluation, and usage structure of the [Kimi K3](https://huggingface.co/moonshotai/Kimi-K3) and [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) model cards. Their weights, benchmark results, and capabilities are not part of this experiment.

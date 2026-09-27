# Release notes

## Training memory and JIT kernels

- Release custom-loss tile graphs explicitly: MLX's custom VJP retains graphs
  even across `eval`, which otherwise causes memory growth at real vocabulary
  and context sizes. Preserve first-order gradients with bounded tile storage.
- JIT compile GDN chunk steps, CE forward/backward tiles and Muon Newton–Schulz;
  recurrent decode already uses JIT. Report MLX peak memory in training metrics.
- Add a peak-memory regression alongside loss/gradient numerical parity tests.

## Online dataset streaming

- Added CLI and Web Hugging Face text streaming with online DeepSeek tokenization
  and EOS packing; full corpus download / `.bin` preparation is optional.
- Pin dataset commits, isolate validation documents, and preserve iterator state
  plus unconsumed tokens for deterministic resume without HF shuffle buffers.
- Keep offline dry-run and prepared mmap/SFT/synthetic data paths.
- Validate with tiny local parquet corpora and a bounded remote data read;
  no training loop or actual 32K sequence is part of verification.

## Hybrid architecture migration

- Replaced the previous GPT implementation with the GDN/SWA research backbone.
- Preserved depth presets; d12 matches the 109,819,488-parameter v0 design.
- Pinned the official DeepSeek V4.1 tokenizer and Python prompt encoder.
- Added local-aware tiled attention, chunked GDN and recurrent decode, bounded
  linear cross entropy, tied embedding, SwiGLU and FP32 optimizer/master state.
- Added uint32 datasets, assistant supervision masks, complete versioned
  checkpoints, 4K/32K memory preparation, an AdamW optimizer control and Web/CLI tooling.
- Restricted all model configuration and execution to the GDN/SWA hybrid architecture.
- Removed old GPT/BPE training/checkpoint import modes and their dependencies.
- Training commands default to inspection and require --start-training to run.

The old March 2026 release's `29 passed, 1 skipped` and imported base-d20 evidence
apply to the removed architecture. They do not validate this implementation.
Current verification is short-sequence and untrained; no training run or actual
32K long-sequence validation was performed for this migration.

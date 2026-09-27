# Release notes

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

# Project instructions

This repository has two backends for one architecture: `nanochat_mlx/hybrid/`
(Apple Silicon) and `nanochat_cuda/` (single NVIDIA GPU, PyTorch/FLA).
Only the GDN/GDN/SWA hybrid architecture is supported. The AdamW-only recipe
changes the optimizer, not the model architecture. Do not reintroduce the removed GPT,
RustBPE tokenizer training or old Hugging Face checkpoint conversion path.

Use `uv` and `uv.lock`. MLX 0.32.2 or later is required. The official DeepSeek
V4.1 tokenizer and Python prompt encoder are pinned to the same upstream revision;
never silently substitute a V4/V3/Qwen/Jinja template. Preserve upstream vendor
code byte-for-byte and keep its MIT license/provenance. Text-only model inputs
must reject image content.

Preserve all depth presets (4, 12, 20, 26) and the canonical d12/512/1536 v0.
Baseline training context is 4K; configured maximum is 32K. GDN is NoPE; SWA
has local attention and cache-safe RoPE. Do not build dense T*T masks for SWA or
full B*T*V training logits. Keep the optimizer memory controllers out of Muon.

Training is explicit (`--start-training`). Do not start new training jobs or
actual long-context execution as part of implementation verification. Preserve
any training the user separately authorized. Short forward/gradient checks and synthetic
optimizer equation checks are allowed. Do not claim learned long-context ability
from those tests or from a dry-run.

Commands:

```bash
uv sync --python 3.13
uv run python -m scripts.train --depth 12 --dry-run
uv run python -m scripts.train --recipe configs/gdn_swa_32k_memory.json --dry-run
NANOCHAT_TEST_TOKENIZER=/absolute/path/tokenizer uv run python -m pytest tests -v
```

Source modules own separate responsibilities: config, model, GDN, local attention,
RoPE, loss, optimizer, tokenizer, mmap data, scenarios, checkpoint, engine and
training. CLI adapters stay under `scripts/`. CUDA/PyTorch dependencies are optional
extras; preserve MLX-only installations. CUDA reuses backend-independent config,
tokenizer, NumPy data and plan contracts without importing MLX. CUDA checkpoints
have a separate directory/format; MLX imports are weight-only warm starts.
Use `docs/cuda.md` for CUDA setup and short CPU/GPU checks. Do not start new CUDA
training or actual 32K tests as part of implementation verification. Preserve the
user-authorized existing MLX training process while working on the CUDA backend.

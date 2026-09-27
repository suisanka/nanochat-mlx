# Hybrid architecture verification gates

## Code-delivery gate (no training)

- Resolve the committed uv.lock and run the short pytest suite on Apple Silicon.
- Set NANOCHAT_TEST_TOKENIZER to a verified local official V4.1 tokenizer so
  real tokenizer/data/API cases run rather than skip.
- Check GDN chunk/reference output, state and gradient parity; local SDPA parity;
  bounded linear CE/vanilla CE output and both gradients; BF16 behavior;
  cached decode/full forward; strict checkpoint roundtrip and optimizer grouping.
- Check all original depths and 4K/32K recipe metadata with dry-run.
- Check the default CLI and Web plan flow cannot accidentally start training.
- Confirm there are no old GPT/tokenizer/conversion runtime branches.
- Do not run training or actual 32K sequences for the current task.

## Future research gates (require separate authorization)

- Measure actual 32K memory, throughput, gradients and long-prefill execution.
- Run 4K synthetic validation and then the prepared 32K memory scenarios.
- Compare the hybrid optimizer recipes with matched data and token budgets.
  Pure GDN, pure SWA and full-attention comparisons would require a separately
  authorized extension; those model modes are not available.
- Record alpha/beta distributions, recurrent state norms, Q/K/V norms, output
  gates, layer/parameter gradient norms, optimizer update RMS and SWA entropy.
- Run natural-language smoke training before the 500M-token architecture ranking.
- Only advance to KDA after stability, LM loss, local recall, long recall,
  overwrite and hybrid trade-off requirements are demonstrated empirically.

Passing code checks does not close research gates. No trained checkpoint or
32K quality/performance claim is part of this migration.

## Optional CUDA backend gate

- Resolve `uv.lock` with `--extra cuda` for Linux without installing MLX.
- Run `tests/cuda` with the `torch` extra for short CPU reference/loss/cache,
  weighted accumulation, optimizer and checkpoint/RNG checks.
- On Apple Silicon with both extras available, run `tests/test_cuda_interchange.py`
  to compare actual MLX safetensors, logits, loss and parameter gradients.
- Confirm dry-run works without importing either tensor runtime and that existing
  MLX-only installations do not require PyTorch.
- On an NVIDIA host, run `tests/cuda -m cuda`: actual FLA forward/backward/state,
  recurrent decode, compiled model gradients and optional FlashAttention parity.
  CPU passes or skipped CUDA tests do not close this GPU gate.
- Keep CUDA throughput/peak-memory and actual long-context claims separate from
  code delivery; see `docs/cuda.md` for installation and validation boundaries.

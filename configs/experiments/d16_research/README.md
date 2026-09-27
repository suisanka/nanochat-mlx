# d16 research experiment

These are proposed recipes for the [training plan](../../../docs/training_plan_d16.md), not records of completed training.

| File | Purpose |
|---|---|
| `pretrain_5b.json` | d16 / 4K English pretraining, 5B-token scheduler horizon |
| `cpt_bilingual_1b.json` | Weight-initialized 1B-token continued pretraining on a separately prepared mixture |
| `sft_pilot.json` | SFT pilot ceiling of 512 optimizer steps; reduce to the prepared dataset's one-epoch budget |
| `plan.json` | Experiment manifest and hypothetical cost scenarios; **not a trainer recipe** |

Only the first three files may be passed to `scripts.train_cuda --recipe`.
Inspect with `--dry-run`; see the plan for exact commands and required data,
checkpoint, backend and budget contracts. Recipes do not prepare datasets or
implement a mixture, budget stop, DPO, or RL trainer. Paid execution requires a
separately authorized run after the engineering and quality gates are met.

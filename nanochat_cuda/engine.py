"""Bounded prefill and recurrent generation using the official tokenizer."""

import math
import torch


class HybridEngine:
    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer

    @torch.no_grad()
    def generate(
        self,
        tokens,
        num_samples=1,
        max_tokens=256,
        temperature=0.8,
        top_k=50,
        seed=42,
        repetition_penalty=1.0,
        prefill_chunk_size=256,
    ):
        if not tokens or num_samples < 1 or max_tokens < 0:
            raise ValueError(
                "Require non-empty prompt, positive samples and nonnegative max_tokens"
            )
        if len(tokens) + max_tokens > self.model.config.max_context:
            raise ValueError("Prompt plus generation exceeds configured context limit")
        if (
            not math.isfinite(temperature)
            or temperature < 0
            or not math.isfinite(repetition_penalty)
            or repetition_penalty <= 0
            or top_k < 0
        ):
            raise ValueError("Invalid sampling settings")
        if max_tokens == 0:
            return
        self.model.eval()
        device = next(self.model.parameters()).device
        generator = torch.Generator(device=device).manual_seed(seed)
        cache = self.model.make_cache()
        logits = self.model.prefill(
            torch.tensor([tokens], device=device), cache, prefill_chunk_size
        )[:, -1]
        if num_samples > 1:
            logits = logits.repeat_interleave(num_samples, 0)
            cache = cache.repeat(num_samples)
        histories = [list(tokens) for _ in range(num_samples)]
        done = [False] * num_samples
        eos = self.tokenizer.contract["eos"]
        for step in range(max_tokens):
            sampled = []
            for i in range(num_samples):
                row = logits[i].clone()
                if repetition_penalty != 1:
                    seen = torch.tensor(list(set(histories[i][-256:])), device=device)
                    values = row[seen]
                    row[seen] = torch.where(
                        values > 0,
                        values / repetition_penalty,
                        values * repetition_penalty,
                    )
                if temperature == 0:
                    token = int(row.argmax().item())
                else:
                    if top_k:
                        threshold = row.topk(min(top_k, row.numel())).values[-1]
                        row = row.masked_fill(row < threshold, -torch.inf)
                    token = int(
                        torch.multinomial(
                            (row / temperature).softmax(-1), 1, generator=generator
                        ).item()
                    )
                token = eos if done[i] else token
                sampled.append(token)
                histories[i].append(token)
                done[i] |= token == eos
            yield sampled, [1] * num_samples
            if all(done) or step + 1 == max_tokens:
                break
            logits = self.model(
                torch.tensor(sampled, device=device)[:, None],
                kv_cache=cache,
                last_only=True,
            )[:, -1]

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        results = [list(tokens) for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        done = [False] * num_samples
        for column, _ in self.generate(tokens, num_samples=num_samples, **kwargs):
            for i, token in enumerate(column):
                done[i] |= token == self.tokenizer.contract["eos"]
                if not done[i]:
                    results[i].append(token)
                    masks[i].append(1)
        return results, masks

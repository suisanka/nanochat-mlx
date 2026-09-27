"""Bounded-prefill generation with the DeepSeek V4.1 tokenizer contract."""

import mlx.core as mx


class HybridEngine:
    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer

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
            raise ValueError(
                f"Prompt plus requested generation exceeds context limit {self.model.config.max_context}"
            )
        if temperature < 0 or repetition_penalty <= 0:
            raise ValueError("Invalid sampling temperature or repetition penalty")
        if max_tokens == 0:
            return
        mx.random.seed(seed)
        cache = self.model.make_cache()
        logits = self.model.prefill(
            mx.array([tokens], dtype=mx.int32), cache, prefill_chunk_size
        )[:, -1]
        if num_samples > 1:
            logits = mx.repeat(logits, num_samples, axis=0)
            cache = cache.repeat(num_samples)
        histories = [list(tokens) for _ in range(num_samples)]
        done = [False] * num_samples
        eos = self.tokenizer.contract["eos"]
        for step in range(max_tokens):
            sampled = []
            for i in range(num_samples):
                row = logits[i]
                if repetition_penalty != 1:
                    seen = mx.array(list(set(histories[i][-256:])), dtype=mx.int32)
                    values = row[seen]
                    row = row.at[seen].add(
                        mx.where(
                            values > 0,
                            values / repetition_penalty,
                            values * repetition_penalty,
                        )
                        - values
                    )
                if temperature == 0:
                    token = int(mx.argmax(row).item())
                else:
                    if top_k and top_k > 0:
                        threshold = mx.min(mx.topk(row, k=min(top_k, row.size)))
                        row = mx.where(row >= threshold, row, -mx.inf)
                    token = int(mx.random.categorical(row / temperature).item())
                token = eos if done[i] else token
                sampled.append(token)
                histories[i].append(token)
                done[i] = done[i] or token == eos
            yield sampled, [1] * num_samples
            if all(done) or step + 1 == max_tokens:
                break
            logits = self.model(
                mx.array(sampled, dtype=mx.int32)[:, None],
                kv_cache=cache,
                last_only=True,
            )[:, -1]
            mx.eval(logits, cache.arrays())

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        results = [list(tokens) for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        done = [False] * num_samples
        for column, _ in self.generate(tokens, num_samples=num_samples, **kwargs):
            for i, token in enumerate(column):
                done[i] = done[i] or token == self.tokenizer.contract["eos"]
                if not done[i]:
                    results[i].append(token)
                    masks[i].append(1)
        return results, masks

"""Raw-text likelihood evaluation with bounded vocabulary projection.

Each document starts with a single EOS context token, with no chat template or
scored trailing EOS. Rolling windows score every text token exactly once. BPB
uses the UTF-8 byte count of the original document, never decoded fragments.
"""

import math


def scoring_windows(ids, context_length, stride):
    if not 1 <= stride <= context_length:
        raise ValueError("Require 1 <= stride <= context_length")
    if len(ids) < 2:
        raise ValueError("Require an initial context token and at least one target")
    for start in range(1, len(ids), stride):
        stop = min(start + stride, len(ids))
        offset = max(0, stop - 1 - context_length)
        x = ids[offset : stop - 1]
        y = ids[offset + 1 : stop]
        masked = start - offset - 1
        yield x, [-1] * masked + y[masked:]


def score_ids(model, ids, context_length=1024, stride=512, loss_chunk_size=128):
    import mlx.core as mx

    from .loss import linear_cross_entropy

    if not 1 <= context_length <= model.config.max_context:
        raise ValueError("Context length exceeds model configuration")
    if loss_chunk_size <= 0:
        raise ValueError("loss_chunk_size must be positive")
    nll, count = 0.0, 0
    for x, y in scoring_windows(ids, context_length, stride):
        hidden = model.hidden(mx.array([x], dtype=mx.int32))
        loss = linear_cross_entropy(
            hidden, model.wte.weight, mx.array([y], dtype=mx.int32), loss_chunk_size
        ).item()
        if not math.isfinite(loss):
            raise RuntimeError("Non-finite evaluation loss")
        valid = sum(t != -1 for t in y)
        nll += loss * valid
        count += valid
    return {"nll_sum": nll, "tokens": count}


def score_text(model, tokenizer, text, **kwargs):
    if not isinstance(text, str) or not text:
        raise ValueError("Evaluation text must be a non-empty string")
    tokens = tokenizer.encode(text)
    if not tokens or tokenizer.decode(tokens) != text:
        raise ValueError("BPB requires exact text/tokenizer round-trip")
    result = score_ids(model, [tokenizer.contract["eos"], *tokens], **kwargs)
    size = len(text.encode("utf-8"))
    return {**result, "bytes": size, "bpb": result["nll_sum"] / (math.log(2) * size)}

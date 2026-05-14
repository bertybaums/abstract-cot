"""Self-distillation — Phase 1b trainer pieces.

Given a model trained via bottlenecked SFT, sample abstract sequences
on-policy from the prompt alone (no verbal CoT in context), pair with the
gold answer, and SFT on [x; z̃; y] under standard causal masking.

Per paper §3.2 Eq. 4:
    L_Distill(θ; x, ẑ, y) = -Σ_{j ∈ Z_abs ∪ Y} log π_θ(s_j | s_{<j})

The structure is identical to bottlenecked SFT with an empty verbal-CoT
segment — the block mask degenerates to standard causal when c_len=0, and
the loss mask covers exactly the same positions. So we reuse
BottleneckCollator with `verbal_cot=""`.

What's new here is the on-policy abstract-sequence generation step.

See PLAN.md §6.5.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import LogitsProcessorList, PreTrainedTokenizerBase

from .bottlenecked_sft import BottleneckExample, bottleneck_loss
from .constrained_decoding import AbstractCotLogitsProcessor
from .tokenizer import AbstractVocab

# Re-export for symmetric naming
self_distill_loss = bottleneck_loss


@dataclass
class _GenSpec:
    prompt: str
    answer: str
    cot_for_conditioning: str = ""  # empty for self-distill; populated for PI step 1


def generate_abstract_sequences(
    model,
    tokenizer: PreTrainedTokenizerBase,
    vocab: AbstractVocab,
    prompts: list[str],
    cots: list[str] | None = None,
    m_max: int = 128,
    batch_size: int = 4,
    temperature: float = 1.0,
    top_p: float = 0.95,
    device: torch.device | str | None = None,
) -> list[list[int]]:
    """Sample abstract token sequences on-policy via constrained decoding.

    Args:
        prompts: per-example prompt text.
        cots: optional per-example verbal-CoT text. If provided, the model
            sees (x, c) before the <beginabstract> token. Used at PI step 1
            (bottlenecked-SFT sampling at t>=2). If None or empty strings,
            the model sees only x — the self-distillation regime.

    Returns:
        For each prompt, a list of codebook token IDs (delimiters stripped).
    """
    if cots is not None and len(cots) != len(prompts):
        raise ValueError("len(cots) must equal len(prompts) when provided")
    cots = cots or ["" for _ in prompts]

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()

    out: list[list[int]] = []
    try:
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            batch_cots = cots[start : start + batch_size]
            # Construct per-example prefix as token IDs to preserve length variation.
            prefixes = []
            for p, c in zip(batch_prompts, batch_cots):
                ids = tokenizer.encode(p, add_special_tokens=False)
                if c:
                    ids = ids + tokenizer.encode(c, add_special_tokens=False)
                ids = ids + [vocab.begin_id]
                prefixes.append(ids)
            # Left-pad so that the begin delimiter is the last token in each row.
            max_pre = max(len(p) for p in prefixes)
            input_ids = torch.full(
                (len(prefixes), max_pre), tokenizer.pad_token_id, dtype=torch.long
            )
            attention_mask = torch.zeros_like(input_ids)
            for b, p in enumerate(prefixes):
                input_ids[b, max_pre - len(p) :] = torch.tensor(p, dtype=torch.long)
                attention_mask[b, max_pre - len(p) :] = 1
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            proc = AbstractCotLogitsProcessor(vocab=vocab, m_max=m_max, start_inside=True)
            with torch.no_grad():
                gen = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=m_max + 1,  # +1 for the <endabstract>
                    do_sample=True,
                    top_p=top_p,
                    temperature=temperature,
                    logits_processor=LogitsProcessorList([proc]),
                    pad_token_id=tokenizer.pad_token_id,
                )
            # Strip prefix + delimiters from each generated row.
            for b, prefix_ids in enumerate(prefixes):
                generated = gen[b, max_pre:].tolist()
                # Trim at first <endabstract> if present
                if vocab.end_id in generated:
                    generated = generated[: generated.index(vocab.end_id)]
                # Keep only codebook tokens (drop any stragglers — defensive)
                codebook = set(vocab.abstract_ids)
                cleaned = [t for t in generated if t in codebook]
                out.append(cleaned)
    finally:
        if was_training:
            model.train()
    return out


def make_distill_examples(
    prompts: list[str],
    abstract_id_lists: list[list[int]],
    answers: list[str],
) -> list[BottleneckExample]:
    """Build BottleneckExamples with empty verbal_cot for self-distill SFT."""
    if not (len(prompts) == len(abstract_id_lists) == len(answers)):
        raise ValueError("inputs must have equal length")
    return [
        BottleneckExample(prompt=p, verbal_cot="", abstract_ids=z, answer=y)
        for p, z, y in zip(prompts, abstract_id_lists, answers)
    ]

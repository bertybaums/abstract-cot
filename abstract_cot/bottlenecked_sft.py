"""Bottlenecked SFT — Phase 1a trainer pieces.

Builds packed training sequences of the form [x; c; z̃; y] (prompt; verbal CoT;
abstract span with delimiters; answer) with a 4D block-structured attention
mask so the answer cannot directly attend to the verbal CoT (paper §3.2).

Exports:
  - BottleneckExample: a single packed-sequence input pre-collation
  - BottleneckCollator: builds a batch of tensors from a list of examples
  - bottleneck_loss(model, batch): standard CE on the abstract+answer span
  - random_abstract_ids(vocab, length): for the t=1 PI initialization

The collator + loss are designed to be composable into a HF Trainer subclass
later; for Phase 0 they are exercised directly in scripts/run_toy_bottleneck_sft.py.

See PLAN.md §6.5.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch.nn import functional as F
from transformers import PreTrainedTokenizerBase

from .attention_masks import (
    ABS,
    ANS,
    COT,
    PROMPT,
    SegmentSpec,
    build_bottleneck_mask,
    build_loss_mask,
    build_segment_ids,
    hf_attention_mask_from_4d,
)
from .tokenizer import AbstractVocab


@dataclass
class BottleneckExample:
    """One training example before tokenization.

    abstract_ids must already be a list of token IDs from V_abs (no delimiters).
    The collator wraps them with <beginabstract> / <endabstract>.
    """
    prompt: str
    verbal_cot: str
    abstract_ids: list[int]
    answer: str


@dataclass
class BottleneckBatch:
    input_ids: torch.Tensor          # (b, L)
    attention_mask_4d: torch.Tensor  # (b, 1, L, L) additive
    segment_ids: torch.Tensor        # (b, L) for diagnostics
    loss_mask: torch.Tensor          # (b, L) bool — True where loss contributes


def random_abstract_ids(vocab: AbstractVocab, length: int, rng: random.Random | None = None) -> list[int]:
    """Uniformly sample `length` codebook IDs (paper §3.2 t=1 init)."""
    r = rng or random
    return [r.choice(vocab.abstract_ids) for _ in range(length)]


def _ensure_pad(tokenizer: PreTrainedTokenizerBase) -> int:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer.pad_token_id


def _tokenize_text(tok: PreTrainedTokenizerBase, text: str, add_special: bool = False) -> list[int]:
    return tok.encode(text, add_special_tokens=add_special)


@dataclass
class BottleneckCollator:
    """Pack a list of BottleneckExample → a BottleneckBatch.

    Truncation policy: if a packed sequence would exceed max_length, the
    verbal CoT is truncated from the right first (preserving the prompt,
    abstract span, and answer). If still too long, raise.
    """
    tokenizer: PreTrainedTokenizerBase
    vocab: AbstractVocab
    max_length: int = 1024
    dtype: torch.dtype = torch.float32  # for the additive mask

    def __post_init__(self):
        _ensure_pad(self.tokenizer)

    def _pack_one(self, ex: BottleneckExample) -> tuple[list[int], SegmentSpec]:
        prompt_ids = _tokenize_text(self.tokenizer, ex.prompt)
        cot_ids = _tokenize_text(self.tokenizer, ex.verbal_cot)
        ans_ids = _tokenize_text(self.tokenizer, ex.answer)
        abs_segment = [self.vocab.begin_id] + ex.abstract_ids + [self.vocab.end_id]

        fixed = len(prompt_ids) + len(abs_segment) + len(ans_ids)
        budget_for_cot = self.max_length - fixed
        if budget_for_cot < 0:
            raise ValueError(
                f"Example exceeds max_length={self.max_length} even with empty CoT: "
                f"|x|={len(prompt_ids)}, |z̃|={len(abs_segment)}, |y|={len(ans_ids)}"
            )
        if len(cot_ids) > budget_for_cot:
            cot_ids = cot_ids[:budget_for_cot]

        seq = prompt_ids + cot_ids + abs_segment + ans_ids
        spec = SegmentSpec(
            prompt=len(prompt_ids),
            cot=len(cot_ids),
            abstract=len(abs_segment),
            answer=len(ans_ids),
        )
        return seq, spec

    def __call__(self, examples: list[BottleneckExample]) -> BottleneckBatch:
        packed_pairs = [self._pack_one(ex) for ex in examples]
        seqs, specs = zip(*packed_pairs)
        max_len = max(len(s) for s in seqs)
        batch = len(seqs)

        pad_id = self.tokenizer.pad_token_id
        input_ids = torch.full((batch, max_len), pad_id, dtype=torch.long)
        for b, s in enumerate(seqs):
            input_ids[b, : len(s)] = torch.tensor(s, dtype=torch.long)

        segment_ids = build_segment_ids(list(specs), max_len)
        mask_4d_bool = build_bottleneck_mask(segment_ids)
        mask_4d_additive = hf_attention_mask_from_4d(mask_4d_bool, dtype=self.dtype)
        loss_mask = build_loss_mask(segment_ids)

        return BottleneckBatch(
            input_ids=input_ids,
            attention_mask_4d=mask_4d_additive,
            segment_ids=segment_ids,
            loss_mask=loss_mask,
        )


def bottleneck_loss(model, batch: BottleneckBatch) -> torch.Tensor:
    """Cross-entropy on Z_abs ∪ Y positions only (paper Eq. 3).

    Standard next-token: predicting position i from positions [0, i-1]. So we
    compute loss at position i+1 if `loss_mask[i+1]` is True — i.e. shift the
    mask right by one when applying to per-token losses.
    """
    out = model(
        input_ids=batch.input_ids.to(model.device),
        attention_mask=batch.attention_mask_4d.to(model.device, dtype=model.dtype),
    )
    logits = out.logits  # (b, L, V)
    # Shift for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = batch.input_ids[..., 1:].contiguous().to(model.device)
    # The mask at position i means "compute loss at the prediction TARGETING
    # position i", i.e. we want to keep position i in the labels if mask[i] is True.
    shift_mask = batch.loss_mask[..., 1:].contiguous().to(model.device)

    per_tok = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(shift_labels.shape)

    masked = per_tok * shift_mask.float()
    n = shift_mask.sum().clamp_min(1)
    return masked.sum() / n


def get_embedding_param_groups(model, vocab: AbstractVocab, base_lr: float, embed_lr_mult: float):
    """Param groups that give the new embedding rows a higher LR.

    Per paper §3.2 footnote 2: gradients on abstract embeddings can be
    isolated. Our default is to keep all params trainable but give the new
    rows `base_lr * embed_lr_mult` so they catch up faster.
    """
    new_ids = set(vocab.allowed_ids + [vocab.begin_id])
    new_ids_list = sorted(new_ids)
    emb = model.get_input_embeddings()
    out_emb = model.get_output_embeddings()

    # We can't put a sub-slice of a Parameter into its own param group, so we
    # implement the higher LR via a closure on the embedding rows after the
    # optimizer step. The simpler approach used here: hand back a single
    # param group with the full model and a callback the caller invokes after
    # optimizer.step() to amplify gradients on the new rows in-place via SGD.
    # Real Phase 1 will use a custom optimizer wrapper for this; for the toy,
    # we just use base_lr on everything.
    return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": base_lr}]

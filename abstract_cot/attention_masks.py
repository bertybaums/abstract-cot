"""Block-structured attention mask for bottlenecked SFT.

Implements paper §3.2 Eq. 3: within a packed sequence [x; c; z; y]
(prompt; verbal CoT; abstract span; answer):

    Default rule:  causal mask (q attends to kv iff kv <= q and kv is not pad).
    NEW rule:      if q ∈ y and kv ∈ c, attention is blocked.

This creates a conditional Markov bottleneck c -> H_{Z_abs} -> y: the answer
sees the verbal CoT only through the hidden states at abstract-token positions.

We use SEGMENT IDS to mark positions:
    0 = prompt (x)
    1 = verbal CoT (c)
    2 = abstract span (z)  — includes <beginabstract> / <endabstract> delimiters
    3 = answer (y)
    -1 = padding

The mask is a 4D bool tensor (batch, 1, q_len, kv_len) suitable for passing
to HuggingFace models with attn_implementation="eager". A flex_attention
variant can be added later for production speed (see PLAN.md §6.2).

See PLAN.md section 6.2.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

PROMPT = 0
COT = 1
ABS = 2
ANS = 3
PAD = -1


@dataclass
class SegmentSpec:
    """Per-example segment lengths. Sum must equal sequence length (before padding)."""
    prompt: int
    cot: int
    abstract: int
    answer: int

    @property
    def total(self) -> int:
        return self.prompt + self.cot + self.abstract + self.answer


def build_segment_ids(
    specs: list[SegmentSpec], max_len: int, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Build a (batch, max_len) segment-id tensor, right-padded with PAD=-1.

    Args:
        specs: per-example segment lengths.
        max_len: pad length. Must be >= max(spec.total for spec in specs).
    """
    batch = len(specs)
    out = torch.full((batch, max_len), PAD, dtype=torch.long, device=device)
    for b, s in enumerate(specs):
        if s.total > max_len:
            raise ValueError(
                f"example {b}: total {s.total} > max_len {max_len}"
            )
        i = 0
        for seg_id, length in [
            (PROMPT, s.prompt),
            (COT, s.cot),
            (ABS, s.abstract),
            (ANS, s.answer),
        ]:
            out[b, i : i + length] = seg_id
            i += length
    return out


def build_bottleneck_mask(segment_ids: torch.Tensor) -> torch.Tensor:
    """Build the 4D attention mask for bottlenecked SFT.

    Args:
        segment_ids: (batch, seq_len) tensor with values in {PROMPT, COT, ABS, ANS, PAD}.

    Returns:
        (batch, 1, seq_len, seq_len) bool tensor: True = attend, False = block.
        Compatible with HuggingFace eager attention: pass as `attention_mask`
        when the underlying model accepts 4D masks (most decoder-only architectures
        in transformers >=4.40 do via the additive-mask path).
    """
    if segment_ids.dim() != 2:
        raise ValueError(f"segment_ids must be 2D, got shape {tuple(segment_ids.shape)}")
    batch, seq = segment_ids.shape
    device = segment_ids.device

    # Causal: kv_idx <= q_idx
    idx = torch.arange(seq, device=device)
    causal = idx.unsqueeze(0) <= idx.unsqueeze(1)  # (q, kv) bool

    # Bottleneck: NOT (q in ANS and kv in COT)
    q_is_ans = (segment_ids == ANS).unsqueeze(2)  # (b, q, 1)
    kv_is_cot = (segment_ids == COT).unsqueeze(1)  # (b, 1, kv)
    bottleneck_block = q_is_ans & kv_is_cot  # (b, q, kv)

    # kv must not be padding
    kv_valid = (segment_ids != PAD).unsqueeze(1)  # (b, 1, kv)

    allow = causal.unsqueeze(0) & (~bottleneck_block) & kv_valid  # (b, q, kv)
    return allow.unsqueeze(1)  # (b, 1, q, kv)


def build_loss_mask(segment_ids: torch.Tensor) -> torch.Tensor:
    """Loss is computed only at abstract and answer positions (paper Eq. 3).

    Returns (batch, seq_len) bool: True where the token contributes to the loss.
    """
    return (segment_ids == ABS) | (segment_ids == ANS)


def hf_attention_mask_from_4d(
    mask_4d: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Convert a (b, 1, q, kv) bool mask to the additive form HF eager expects.

    Some HF attention paths want an additive mask where blocked positions are
    a large negative number and allowed positions are 0. This helper does
    the conversion for callers that need it.
    """
    additive = torch.zeros_like(mask_4d, dtype=dtype)
    additive.masked_fill_(~mask_4d, torch.finfo(dtype).min)
    return additive

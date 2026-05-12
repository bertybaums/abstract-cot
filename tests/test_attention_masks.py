"""Unit tests for the block-structured attention mask.

Two layers:
  1. Pure mask-tensor tests: verify the boolean structure matches the paper.
  2. End-to-end invariance test: y's hidden states must not directly depend on
     c, given the mask. We probe this by perturbing c and verifying that the
     direct attention path is gone — concretely, by inspecting attention
     weights at y positions over c columns.
"""
from __future__ import annotations

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from abstract_cot.attention_masks import (
    ABS,
    ANS,
    COT,
    PAD,
    PROMPT,
    SegmentSpec,
    build_bottleneck_mask,
    build_loss_mask,
    build_segment_ids,
    hf_attention_mask_from_4d,
)

TINY_MODEL = "sshleifer/tiny-gpt2"


def test_build_segment_ids_simple():
    spec = SegmentSpec(prompt=2, cot=3, abstract=2, answer=2)
    ids = build_segment_ids([spec], max_len=10)
    expected = torch.tensor(
        [[PROMPT, PROMPT, COT, COT, COT, ABS, ABS, ANS, ANS, PAD]]
    )
    assert torch.equal(ids, expected)


def test_build_segment_ids_two_examples_different_lengths():
    s1 = SegmentSpec(prompt=1, cot=1, abstract=1, answer=1)  # total 4
    s2 = SegmentSpec(prompt=2, cot=2, abstract=2, answer=2)  # total 8
    ids = build_segment_ids([s1, s2], max_len=8)
    assert ids.shape == (2, 8)
    assert (ids[0, 4:] == PAD).all()
    assert (ids[1] != PAD).all()


def test_build_segment_ids_rejects_overflow():
    with pytest.raises(ValueError, match="> max_len"):
        build_segment_ids([SegmentSpec(2, 2, 2, 2)], max_len=4)


def test_bottleneck_mask_basic_structure():
    """y -> c is blocked; everything else follows causal."""
    spec = SegmentSpec(prompt=2, cot=3, abstract=2, answer=2)
    ids = build_segment_ids([spec], max_len=spec.total)
    mask = build_bottleneck_mask(ids)[0, 0]  # (q, kv)

    # Indices: [0,1]=x, [2,3,4]=c, [5,6]=z, [7,8]=y

    # Causal: q=2 (first c position) attends to {0,1,2} only
    assert mask[2].tolist() == [True, True, True, False, False, False, False, False, False]

    # z (q=5,6) attends to x and c and itself (causal)
    assert mask[5].tolist() == [True, True, True, True, True, True, False, False, False]
    assert mask[6].tolist() == [True, True, True, True, True, True, True, False, False]

    # y (q=7) attends to x, z, and itself — NOT c
    assert mask[7].tolist() == [True, True, False, False, False, True, True, True, False]
    # y (q=8) likewise
    assert mask[8].tolist() == [True, True, False, False, False, True, True, True, True]


def test_bottleneck_mask_z_can_see_c():
    """Sanity: abstract tokens DO attend to verbal CoT (that's the whole point)."""
    spec = SegmentSpec(prompt=1, cot=2, abstract=2, answer=1)
    ids = build_segment_ids([spec], max_len=spec.total)
    mask = build_bottleneck_mask(ids)[0, 0]
    # Indices: [0]=x, [1,2]=c, [3,4]=z, [5]=y
    # z at q=3 should attend to c at kv=1,2
    assert mask[3, 1].item() is True
    assert mask[3, 2].item() is True


def test_bottleneck_mask_respects_padding():
    spec = SegmentSpec(prompt=1, cot=1, abstract=1, answer=1)  # total 4
    ids = build_segment_ids([spec], max_len=8)
    mask = build_bottleneck_mask(ids)[0, 0]
    # kv positions 4..7 are PAD and should be blocked from any q
    assert mask[:, 4:].any().item() is False


def test_loss_mask():
    spec = SegmentSpec(prompt=2, cot=2, abstract=3, answer=2)
    ids = build_segment_ids([spec], max_len=spec.total)
    lm = build_loss_mask(ids)[0]
    # Indices: [0,1]=x (False), [2,3]=c (False), [4,5,6]=z (True), [7,8]=y (True)
    assert lm.tolist() == [False, False, False, False, True, True, True, True, True]


def test_additive_mask_conversion():
    spec = SegmentSpec(prompt=1, cot=1, abstract=1, answer=1)
    ids = build_segment_ids([spec], max_len=spec.total)
    bool_mask = build_bottleneck_mask(ids)
    add = hf_attention_mask_from_4d(bool_mask, dtype=torch.float32)
    # Blocked → very negative; allowed → 0
    assert add[bool_mask].abs().max().item() == 0.0
    assert add[~bool_mask].max().item() < -1e30


def test_end_to_end_y_does_not_attend_to_c():
    """Forward pass with output_attentions=True; check y rows have ~0 attention
    on c columns at every layer."""
    config = AutoConfig.from_pretrained(TINY_MODEL)
    config.attn_implementation = "eager"  # required for 4D masks + attn output
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL, config=config)
    model.eval()

    spec = SegmentSpec(prompt=2, cot=3, abstract=2, answer=2)
    ids = build_segment_ids([spec], max_len=spec.total)
    mask_4d = build_bottleneck_mask(ids)
    add_mask = hf_attention_mask_from_4d(mask_4d, dtype=torch.float32)

    vocab_size = model.get_input_embeddings().weight.shape[0]
    torch.manual_seed(0)
    input_ids = torch.randint(0, vocab_size, (1, spec.total))

    with torch.no_grad():
        out = model(
            input_ids,
            attention_mask=add_mask,  # additive 4D
            output_attentions=True,
        )

    # y indices = [7,8], c indices = [2,3,4]
    y_idx = [7, 8]
    c_idx = [2, 3, 4]
    for layer_idx, attn in enumerate(out.attentions):
        # attn shape: (batch=1, heads, q, kv)
        slice_ = attn[0, :, y_idx][:, :, c_idx]
        assert slice_.abs().max().item() < 1e-6, (
            f"Layer {layer_idx}: y attended to c with weight up to "
            f"{slice_.abs().max().item():.3e} (expected ~0)"
        )


def test_end_to_end_loss_mask_zeros_x_and_c_positions():
    """Compute a fake loss and verify gradient only flows through z and y."""
    spec = SegmentSpec(prompt=2, cot=2, abstract=2, answer=2)
    ids = build_segment_ids([spec], max_len=spec.total)
    lm = build_loss_mask(ids)[0]

    # Fake per-token loss tensor; aggregate using the loss mask
    per_tok = torch.arange(spec.total, dtype=torch.float32)
    aggregated = (per_tok * lm.float()).sum()
    # Only positions [4,5,6,7] contribute: 4 + 5 + 6 + 7 = 22
    assert aggregated.item() == 22.0

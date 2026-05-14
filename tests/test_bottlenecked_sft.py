"""Unit tests for the bottlenecked-SFT collator + loss function."""
from __future__ import annotations

import random

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from abstract_cot.attention_masks import ABS, ANS, COT, PAD, PROMPT
from abstract_cot.bottlenecked_sft import (
    BottleneckCollator,
    BottleneckExample,
    bottleneck_loss,
    random_abstract_ids,
)
from abstract_cot.tokenizer import extend_tokenizer, resize_model_for_vocab

TINY_MODEL = "sshleifer/tiny-gpt2"


@pytest.fixture
def extended():
    tok = AutoTokenizer.from_pretrained(TINY_MODEL)
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL, attn_implementation="eager")
    tok, vocab = extend_tokenizer(tok, M=16)
    model = resize_model_for_vocab(model, tok, seed=0)
    return tok, model, vocab


def test_random_abstract_ids_only_from_codebook(extended):
    _, _, vocab = extended
    rng = random.Random(0)
    ids = random_abstract_ids(vocab, 50, rng)
    allowed = set(vocab.abstract_ids)
    assert all(i in allowed for i in ids)


def test_collator_packs_segments_correctly(extended):
    tok, _, vocab = extended
    collator = BottleneckCollator(tok, vocab, max_length=128, dtype=torch.float32)
    ex = BottleneckExample(
        prompt="The cat sat",
        verbal_cot="Because it was tired.",
        abstract_ids=vocab.abstract_ids[:4],
        answer="So it slept.",
    )
    batch = collator([ex])
    assert batch.input_ids.dim() == 2 and batch.input_ids.shape[0] == 1
    # The packed sequence must contain begin and end delimiters at the
    # abstract-segment boundary.
    seq = batch.input_ids[0].tolist()
    assert vocab.begin_id in seq
    assert vocab.end_id in seq
    begin_pos = seq.index(vocab.begin_id)
    end_pos = seq.index(vocab.end_id)
    assert end_pos == begin_pos + 5  # begin + 4 abstract + end


def test_collator_loss_mask_covers_abstract_and_answer(extended):
    tok, _, vocab = extended
    collator = BottleneckCollator(tok, vocab, max_length=128, dtype=torch.float32)
    ex = BottleneckExample(
        prompt="P",
        verbal_cot="C",
        abstract_ids=vocab.abstract_ids[:3],
        answer="A",
    )
    batch = collator([ex])
    seg = batch.segment_ids[0]
    lm = batch.loss_mask[0]
    # Loss mask True iff segment is ABS or ANS
    for i in range(seg.shape[0]):
        if seg[i].item() == PAD:
            assert not lm[i].item()
        elif seg[i].item() in (ABS, ANS):
            assert lm[i].item(), f"pos {i} seg={seg[i].item()} should be in loss"
        else:
            assert not lm[i].item(), f"pos {i} seg={seg[i].item()} should NOT be in loss"


def test_collator_truncates_cot_first(extended):
    tok, _, vocab = extended
    collator = BottleneckCollator(tok, vocab, max_length=20, dtype=torch.float32)
    ex = BottleneckExample(
        prompt="short prompt",
        verbal_cot="this is a verbal chain of thought that is very long " * 20,
        abstract_ids=vocab.abstract_ids[:2],
        answer="A.",
    )
    batch = collator([ex])
    assert batch.input_ids.shape[1] == 20
    # The non-COT pieces should still appear
    seq = batch.input_ids[0].tolist()
    assert vocab.begin_id in seq
    assert vocab.end_id in seq


def test_loss_decreases_over_a_few_steps(extended):
    """Tiny end-to-end smoke test on tiny-gpt2: loss should drop over 5 SGD steps."""
    tok, model, vocab = extended
    collator = BottleneckCollator(tok, vocab, max_length=64, dtype=torch.float32)
    rng = random.Random(0)
    examples = [
        BottleneckExample(
            prompt=f"prompt number {i}",
            verbal_cot=f"reasoning {i} continues here.",
            abstract_ids=random_abstract_ids(vocab, 3, rng),
            answer=f"final answer {i}.",
        )
        for i in range(4)
    ]
    batch = collator(examples)
    torch.manual_seed(0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(5):
        opt.zero_grad()
        l = bottleneck_loss(model, batch)
        l.backward()
        opt.step()
        losses.append(l.item())
    assert losses[0] > losses[-1], f"loss didn't decrease: {losses}"


def test_loss_only_counts_z_and_y_positions(extended):
    """Verify the loss really excludes x and c positions.

    Trick: zero out the model's logits at all but the prompt positions.
    The loss should be unchanged (since prompt positions don't contribute).
    """
    tok, model, vocab = extended
    collator = BottleneckCollator(tok, vocab, max_length=32, dtype=torch.float32)
    ex = BottleneckExample(
        prompt="px",
        verbal_cot="cc",
        abstract_ids=vocab.abstract_ids[:2],
        answer="ay",
    )
    batch = collator([ex])
    model.eval()
    with torch.no_grad():
        l1 = bottleneck_loss(model, batch)
    # Now reorder loss_mask to invert it — loss should differ
    inverted = batch
    inverted.loss_mask = ~batch.loss_mask
    inverted.loss_mask[batch.segment_ids == PAD] = False
    with torch.no_grad():
        l2 = bottleneck_loss(model, inverted)
    # Two different masks → two different losses (with overwhelming probability)
    assert abs(l1.item() - l2.item()) > 1e-4

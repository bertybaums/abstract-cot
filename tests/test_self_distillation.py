"""Unit tests for self-distillation generation + collation."""
from __future__ import annotations

import random

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from abstract_cot.bottlenecked_sft import BottleneckCollator
from abstract_cot.self_distillation import (
    generate_abstract_sequences,
    make_distill_examples,
)
from abstract_cot.tokenizer import extend_tokenizer, resize_model_for_vocab

TINY = "sshleifer/tiny-gpt2"


@pytest.fixture
def extended():
    tok = AutoTokenizer.from_pretrained(TINY)
    model = AutoModelForCausalLM.from_pretrained(TINY, attn_implementation="eager")
    tok, vocab = extend_tokenizer(tok, M=16)
    model = resize_model_for_vocab(model, tok, seed=0)
    return tok, model, vocab


def test_generate_returns_codebook_ids_only(extended):
    tok, model, vocab = extended
    prompts = ["What is 1+1?", "What is 2+2?", "What is 3+3?"]
    seqs = generate_abstract_sequences(
        model, tok, vocab, prompts=prompts, m_max=8, batch_size=2,
    )
    allowed = set(vocab.abstract_ids)
    assert len(seqs) == len(prompts)
    for s in seqs:
        assert all(t in allowed for t in s), f"non-codebook token in {s}"
        assert len(s) <= 8


def test_generate_with_cots_conditioning_runs(extended):
    """PI iteration t>=2 sampling step: model conditions on (x, c)."""
    tok, model, vocab = extended
    prompts = ["What is 1+1?", "What is 2+2?"]
    cots = ["Add 1 and 1.", "Add 2 and 2."]
    seqs = generate_abstract_sequences(
        model, tok, vocab, prompts=prompts, cots=cots, m_max=8, batch_size=2,
    )
    allowed = set(vocab.abstract_ids)
    for s in seqs:
        assert all(t in allowed for t in s)


def test_generate_cot_length_mismatch_raises(extended):
    tok, model, vocab = extended
    with pytest.raises(ValueError):
        generate_abstract_sequences(
            model, tok, vocab, prompts=["a", "b"], cots=["only one"], m_max=4,
        )


def test_make_distill_examples_strips_cot(extended):
    _, _, vocab = extended
    examples = make_distill_examples(
        prompts=["p1", "p2"],
        abstract_id_lists=[vocab.abstract_ids[:3], vocab.abstract_ids[3:5]],
        answers=["a1", "a2"],
    )
    assert len(examples) == 2
    assert all(ex.verbal_cot == "" for ex in examples)
    assert examples[0].abstract_ids == vocab.abstract_ids[:3]


def test_distill_collator_with_empty_cot_yields_causal_mask(extended):
    """If verbal_cot is empty, the bottleneck mask reduces to standard causal."""
    tok, _, vocab = extended
    collator = BottleneckCollator(tok, vocab, max_length=64, dtype=torch.float32)
    examples = make_distill_examples(
        prompts=["short prompt"],
        abstract_id_lists=[vocab.abstract_ids[:2]],
        answers=["answer"],
    )
    batch = collator(examples)
    # No COT segment in segment_ids
    from abstract_cot.attention_masks import COT
    assert (batch.segment_ids == COT).sum().item() == 0
    # And the mask should look standard-causal: y attends to all positions <= itself
    # except padding. We test this by checking that no allowed positions are blocked
    # due to the bottleneck rule (which is vacuous here).
    bool_mask = (batch.attention_mask_4d == 0)  # additive 0 = allowed
    seq_len = batch.input_ids.shape[1]
    seg = batch.segment_ids[0]
    real_len = (seg != -1).sum().item()
    for q in range(real_len):
        for kv in range(real_len):
            should_attend = kv <= q
            assert bool_mask[0, 0, q, kv].item() == should_attend, (
                f"causal violation at q={q} kv={kv}"
            )

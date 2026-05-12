"""Unit tests for the constrained-decoding LogitsProcessor."""
from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

from abstract_cot.constrained_decoding import (
    AbstractCotLogitsProcessor,
    assert_abstract_span_legal,
)
from abstract_cot.tokenizer import extend_tokenizer, resize_model_for_vocab

TINY_MODEL = "sshleifer/tiny-gpt2"


@pytest.fixture
def extended_model():
    tok = AutoTokenizer.from_pretrained(TINY_MODEL)
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
    tok, vocab = extend_tokenizer(tok, M=8)
    model = resize_model_for_vocab(model, tok, seed=0)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model.eval()
    return tok, model, vocab


def test_processor_masks_after_begin_delim(extended_model):
    tok, model, vocab = extended_model
    proc = AbstractCotLogitsProcessor(vocab=vocab, m_max=16)

    # Sequence ending with <beginabstract> — next step should be constrained.
    input_ids = torch.tensor([[vocab.begin_id]])
    scores = torch.randn(1, len(tok))
    new_scores = proc(input_ids, scores.clone())

    finite_mask = torch.isfinite(new_scores[0])
    finite_ids = finite_mask.nonzero(as_tuple=True)[0].tolist()
    assert set(finite_ids) == set(vocab.allowed_ids), (
        f"Allowed set mismatch: expected {sorted(vocab.allowed_ids)}, "
        f"got {sorted(finite_ids)}"
    )


def test_processor_does_not_mask_before_begin(extended_model):
    tok, model, vocab = extended_model
    proc = AbstractCotLogitsProcessor(vocab=vocab, m_max=16)

    # First step, sequence has only a prompt token, no <beginabstract> yet.
    input_ids = torch.tensor([[42]])  # arbitrary prompt token
    scores = torch.randn(1, len(tok))
    new_scores = proc(input_ids, scores.clone())
    assert torch.isfinite(new_scores).all(), "BEFORE state should not mask"


def test_processor_releases_after_end_delim(extended_model):
    tok, model, vocab = extended_model
    proc = AbstractCotLogitsProcessor(vocab=vocab, m_max=16)

    # Simulate: <beginabstract>, then a few abstract tokens, then <endabstract>.
    # After <endabstract>, we should be unconstrained.
    seq = [vocab.begin_id, vocab.abstract_ids[0], vocab.abstract_ids[3], vocab.end_id]
    for i in range(1, len(seq) + 1):
        input_ids = torch.tensor([seq[:i]])
        scores = torch.randn(1, len(tok))
        proc(input_ids, scores.clone())
    # Now query one more step — proc should be in AFTER state, unconstrained.
    input_ids = torch.tensor([seq + [99]])
    scores = torch.randn(1, len(tok))
    new_scores = proc(input_ids, scores.clone())
    assert torch.isfinite(new_scores).all(), "AFTER state should not mask"


def test_processor_forces_end_at_m_max(extended_model):
    tok, model, vocab = extended_model
    m_max = 3
    proc = AbstractCotLogitsProcessor(vocab=vocab, m_max=m_max)

    # Feed begin + 3 abstract tokens — that hits m_max, next step must force end.
    seq = [vocab.begin_id] + vocab.abstract_ids[:m_max]
    for i in range(1, len(seq) + 1):
        input_ids = torch.tensor([seq[:i]])
        scores = torch.randn(1, len(tok))
        new_scores = proc(input_ids, scores.clone())
    # Last call corresponds to "we just emitted the m_max-th abstract token";
    # the returned scores should put +inf weight (= 0.0, others = -inf) on end_id.
    assert new_scores[0, vocab.end_id].item() == 0.0
    others = new_scores[0].clone()
    others[vocab.end_id] = float("-inf")
    assert torch.isinf(others).all() and (others < 0).all()


def test_generate_produces_only_legal_abstract_tokens(extended_model):
    """End-to-end: tiny-gpt2 with extended vocab + processor → check legality."""
    tok, model, vocab = extended_model
    m_max = 12

    prompt_ids = torch.tensor([[vocab.begin_id]])
    proc = AbstractCotLogitsProcessor(vocab=vocab, m_max=m_max, start_inside=True)
    # start_inside=True because the prompt's LAST token is <beginabstract> AND
    # we want the FIRST generation step to already be constrained — without it,
    # the processor only transitions to INSIDE after reading the begin token at
    # input_ids[:, -1] on step 1, which works in practice but is cleaner with
    # start_inside set explicitly.
    out = model.generate(
        prompt_ids,
        max_new_tokens=m_max + 5,  # room for end delim + a few response tokens
        do_sample=True,
        top_k=50,
        temperature=1.0,
        logits_processor=LogitsProcessorList([proc]),
        pad_token_id=tok.pad_token_id,
    )
    generated = out[0].tolist()
    # Inject a synthetic begin at the start since prompt was just the delim:
    assert generated[0] == vocab.begin_id
    assert_abstract_span_legal(generated, vocab, m_max=m_max)


def test_assert_helper_catches_illegal_token(extended_model):
    tok, model, vocab = extended_model
    bad = [vocab.begin_id, vocab.abstract_ids[0], 99999, vocab.end_id]
    with pytest.raises(AssertionError, match="not in V_abs"):
        assert_abstract_span_legal(bad, vocab, m_max=16)


def test_assert_helper_catches_unclosed_span(extended_model):
    tok, model, vocab = extended_model
    unclosed = [vocab.begin_id, vocab.abstract_ids[0], vocab.abstract_ids[1]]
    with pytest.raises(AssertionError, match="never closed"):
        assert_abstract_span_legal(unclosed, vocab, m_max=16)


def test_assert_helper_catches_overlong_span(extended_model):
    tok, model, vocab = extended_model
    overlong = [vocab.begin_id] + vocab.abstract_ids * 5 + [vocab.end_id]
    with pytest.raises(AssertionError, match="exceeded m_max"):
        assert_abstract_span_legal(overlong, vocab, m_max=5)

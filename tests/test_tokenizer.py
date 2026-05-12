"""Unit tests for tokenizer extension.

Uses sshleifer/tiny-gpt2 (~2 MB) so tests run instantly without HPC.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from abstract_cot.tokenizer import (
    BEGIN_DELIM,
    END_DELIM,
    abstract_token_names,
    extend_tokenizer,
    resize_model_for_vocab,
)


TINY_MODEL = "sshleifer/tiny-gpt2"


def test_abstract_token_names_single_letter():
    names = abstract_token_names(5)
    assert names == [
        "<TOKEN_A>", "<TOKEN_B>", "<TOKEN_C>", "<TOKEN_D>", "<TOKEN_E>"
    ]


def test_abstract_token_names_two_letter_overflow():
    names = abstract_token_names(30)
    assert len(names) == 30
    assert names[25] == "<TOKEN_Z>"
    assert names[26] == "<TOKEN_AA>"
    assert names[27] == "<TOKEN_AB>"
    assert names[29] == "<TOKEN_AD>"


def test_abstract_token_names_paper_default():
    # M = 64 → 26 single-letter + 38 two-letter (AA..BL)
    names = abstract_token_names(64)
    assert len(names) == 64
    assert names[0] == "<TOKEN_A>"
    assert names[25] == "<TOKEN_Z>"
    assert names[26] == "<TOKEN_AA>"
    assert names[-1] == "<TOKEN_BL>"


def test_abstract_token_names_invalid():
    with pytest.raises(ValueError):
        abstract_token_names(0)


def test_extend_tokenizer_grows_vocab():
    tok = AutoTokenizer.from_pretrained(TINY_MODEL)
    base_size = len(tok)
    tok, vocab = extend_tokenizer(tok, M=64)

    # 64 abstract + 2 delimiters = 66 new tokens
    assert len(tok) == base_size + 66
    assert vocab.M == 64
    assert len(vocab.abstract_ids) == 64
    assert vocab.begin_id == tok.convert_tokens_to_ids(BEGIN_DELIM)
    assert vocab.end_id == tok.convert_tokens_to_ids(END_DELIM)
    # All IDs distinct
    assert len(set(vocab.abstract_ids + [vocab.begin_id, vocab.end_id])) == 66
    # Allowed-set helper has 65 entries (64 abstract + end delim)
    assert len(vocab.allowed_ids) == 65


def test_extend_tokenizer_roundtrip_through_disk():
    tok = AutoTokenizer.from_pretrained(TINY_MODEL)
    tok, vocab = extend_tokenizer(tok, M=8)
    with tempfile.TemporaryDirectory() as tmp:
        tok.save_pretrained(tmp)
        tok2 = AutoTokenizer.from_pretrained(tmp)
    assert len(tok2) == len(tok)
    assert tok2.convert_tokens_to_ids(BEGIN_DELIM) == vocab.begin_id
    assert tok2.convert_tokens_to_ids("<TOKEN_C>") == vocab.abstract_ids[2]


def test_extend_tokenizer_refuses_double_extension():
    tok = AutoTokenizer.from_pretrained(TINY_MODEL)
    tok, _ = extend_tokenizer(tok, M=8)
    with pytest.raises(RuntimeError, match="already contains"):
        extend_tokenizer(tok, M=8)


def test_resize_model_inits_new_rows_with_norm_matched_std():
    tok = AutoTokenizer.from_pretrained(TINY_MODEL)
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
    old_size = model.get_input_embeddings().weight.shape[0]

    tok, vocab = extend_tokenizer(tok, M=16)
    model = resize_model_for_vocab(model, tok, init_std_scale=1.0, seed=0)
    new_size = model.get_input_embeddings().weight.shape[0]

    assert new_size == old_size + 18  # 16 abstract + 2 delimiters
    assert new_size == len(tok)

    # New rows should be non-zero and roughly N(0, sigma^2) — sigma matched to
    # existing rows' RMS norm. Just check the new-row std is in a plausible band.
    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        existing_rms = emb[:old_size].float().pow(2).mean(dim=-1).sqrt().mean().item()
        new_rms = emb[old_size:].float().pow(2).mean(dim=-1).sqrt().mean().item()
    # New rows are N(0, sigma^2) where sigma = existing_rms, so new_rms ≈ existing_rms.
    assert 0.3 * existing_rms < new_rms < 3.0 * existing_rms, (
        f"new_rms={new_rms:.4f} far from existing_rms={existing_rms:.4f}"
    )


def test_resize_model_forward_pass_works():
    """Smoke test: extended model can forward a batch containing new tokens."""
    tok = AutoTokenizer.from_pretrained(TINY_MODEL)
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
    tok, vocab = extend_tokenizer(tok, M=8)
    model = resize_model_for_vocab(model, tok, seed=0)
    model.eval()

    input_ids = torch.tensor(
        [[vocab.begin_id] + vocab.abstract_ids[:3] + [vocab.end_id]]
    )
    with torch.no_grad():
        out = model(input_ids)
    assert out.logits.shape == (1, 5, len(tok))
    assert torch.isfinite(out.logits).all()

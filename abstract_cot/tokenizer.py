"""Tokenizer + embedding extension for Abstract-CoT.

Adds M reserved abstract tokens (<TOKEN_A>, <TOKEN_B>, ..., <TOKEN_BL>, ...)
and two delimiters (<beginabstract>, <endabstract>) to a HuggingFace
tokenizer + resizes the model's embedding and LM-head matrices.

New embedding rows are initialized from N(0, sigma^2) where sigma matches the
RMS norm of existing rows, scaled by `init_std_scale`. This is less cold than
HF's default (which inits from the model's std but not row-norm-matched).

See PLAN.md section 6.1.
"""
from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

BEGIN_DELIM = "<beginabstract>"
END_DELIM = "<endabstract>"
TOKEN_PREFIX = "TOKEN_"


def abstract_token_names(M: int) -> list[str]:
    """Generate M abstract token strings: <TOKEN_A>..<TOKEN_Z>, then <TOKEN_AA>..

    Matches paper §3.1 footnote 1.
    """
    if M < 1:
        raise ValueError(f"M must be >= 1, got {M}")
    letters = string.ascii_uppercase
    names = [f"<{TOKEN_PREFIX}{ch}>" for ch in letters[: min(M, 26)]]
    if M > 26:
        # Two-letter overflow: AA, AB, ..., AZ, BA, ..., ZZ (676 max).
        # Paper notes this extends to longer suffixes if needed.
        extra = M - 26
        idx = 0
        for a in letters:
            for b in letters:
                if idx >= extra:
                    break
                names.append(f"<{TOKEN_PREFIX}{a}{b}>")
                idx += 1
            if idx >= extra:
                break
        if idx < extra:
            raise NotImplementedError(
                f"M={M} > 26+676=702; three-letter overflow not implemented"
            )
    return names


@dataclass
class AbstractVocab:
    """Resolved token IDs for the extended tokenizer."""
    M: int
    begin_id: int
    end_id: int
    abstract_ids: list[int]  # length M, in order TOKEN_A, TOKEN_B, ...

    @property
    def allowed_ids(self) -> list[int]:
        """The constrained-decoding allowed set: V_abs ∪ {<endabstract>}."""
        return self.abstract_ids + [self.end_id]


def extend_tokenizer(
    tokenizer: PreTrainedTokenizerBase, M: int = 64
) -> tuple[PreTrainedTokenizerBase, AbstractVocab]:
    """Add abstract tokens + delimiters to the tokenizer.

    Each added token is registered as a `special_token` so the BPE/SP merges
    treat it atomically. Returns (tokenizer, AbstractVocab).
    """
    new_tokens = [BEGIN_DELIM, END_DELIM] + abstract_token_names(M)
    added = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    if added != len(new_tokens):
        # Already-present tokens are not re-added. Caller probably loaded an
        # already-extended tokenizer — surface that instead of silently passing.
        existing = [t for t in new_tokens if tokenizer.convert_tokens_to_ids(t) != tokenizer.unk_token_id]
        raise RuntimeError(
            f"Tokenizer already contains {len(existing)} of the {len(new_tokens)} "
            f"requested tokens. Use the existing tokenizer instead of re-extending."
        )

    begin_id = tokenizer.convert_tokens_to_ids(BEGIN_DELIM)
    end_id = tokenizer.convert_tokens_to_ids(END_DELIM)
    abstract_ids = [
        tokenizer.convert_tokens_to_ids(name) for name in abstract_token_names(M)
    ]
    return tokenizer, AbstractVocab(
        M=M, begin_id=begin_id, end_id=end_id, abstract_ids=abstract_ids
    )


def resize_model_for_vocab(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    init_std_scale: float = 1.0,
    seed: Optional[int] = 0,
) -> PreTrainedModel:
    """Resize input embeddings + LM head to match tokenizer; init new rows.

    New rows are drawn from N(0, sigma^2) where sigma = scale * mean RMS of
    existing rows. This is row-norm-matched, unlike HF's default `mean_resizing`
    which uses the model's overall init std.
    """
    old_size = model.get_input_embeddings().weight.shape[0]
    new_size = len(tokenizer)
    if new_size <= old_size:
        return model

    # Capture sigma BEFORE resize so the calculation only sees real rows.
    with torch.no_grad():
        existing = model.get_input_embeddings().weight[:old_size]
        sigma = (existing.float().pow(2).mean(dim=-1).sqrt().mean() * init_std_scale).item()

    model.resize_token_embeddings(new_size)

    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    with torch.no_grad():
        in_emb = model.get_input_embeddings().weight
        new_rows = torch.empty(new_size - old_size, in_emb.shape[1])
        torch.nn.init.normal_(new_rows, mean=0.0, std=sigma, generator=gen)
        in_emb[old_size:].copy_(new_rows.to(dtype=in_emb.dtype, device=in_emb.device))

        # If untied: re-init LM head new rows too. (If tied, they share storage.)
        out_emb = model.get_output_embeddings()
        if out_emb is not None and out_emb.weight.data_ptr() != in_emb.data_ptr():
            new_out = torch.empty(new_size - old_size, out_emb.weight.shape[1])
            torch.nn.init.normal_(new_out, mean=0.0, std=sigma, generator=gen)
            out_emb.weight[old_size:].copy_(
                new_out.to(dtype=out_emb.weight.dtype, device=out_emb.weight.device)
            )

    return model

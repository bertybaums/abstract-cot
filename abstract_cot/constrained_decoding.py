"""Constrained decoding for Abstract-CoT.

A LogitsProcessor that restricts generation inside the abstract span to
V_abs ∪ {<endabstract>}, then releases the constraint once <endabstract>
fires (or m_max tokens have been emitted, in which case <endabstract> is
forced). The response after <endabstract> is generated unconstrained.

State machine (per sequence):
    BEFORE  → INSIDE  on emit(<beginabstract>)
    INSIDE  → AFTER   on emit(<endabstract>)
                       OR on reaching m_max abstract tokens (force <endabstract>)

The processor assumes <beginabstract> is provided in the prompt prefix; if
not, generation starts in BEFORE state and never constrains. Callers that
expect constrained-from-start (e.g. GRPO rollouts that prepend the begin
delimiter) must include it in the prompt.

See PLAN.md section 6.3. Pattern adapted from
_RCDS/compression/translator/model.py::StateMachineLogitsProcessor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

import torch
from transformers import LogitsProcessor

from .tokenizer import AbstractVocab


class _State(Enum):
    BEFORE = 0
    INSIDE = 1
    AFTER = 2


@dataclass
class _SeqState:
    state: _State = _State.BEFORE
    abstract_count: int = 0
    forced_end: bool = False  # next step must emit <endabstract>


@dataclass
class AbstractCotLogitsProcessor(LogitsProcessor):
    """Constrain generation to V_abs ∪ {<endabstract>} inside the abstract span.

    Args:
        vocab: AbstractVocab with begin_id, end_id, abstract_ids.
        m_max: hard cap on abstract sequence length (default 128 per paper).
        start_inside: if True, all sequences start in INSIDE state — useful when
            <beginabstract> is the last prompt token and we want the FIRST
            generated token to be constrained. Defaults to False (standard).
    """

    vocab: AbstractVocab
    m_max: int = 128
    start_inside: bool = False

    _states: list[_SeqState] = field(default_factory=list)
    _allowed_ids_tensor: torch.Tensor | None = None

    def _ensure_states(self, batch_size: int):
        if len(self._states) != batch_size:
            initial = _State.INSIDE if self.start_inside else _State.BEFORE
            self._states = [_SeqState(state=initial) for _ in range(batch_size)]

    def _allowed_mask(self, vocab_size: int, device: torch.device) -> torch.Tensor:
        if self._allowed_ids_tensor is None:
            self._allowed_ids_tensor = torch.tensor(
                self.vocab.allowed_ids, dtype=torch.long
            )
        mask = torch.full((vocab_size,), False, device=device)
        mask[self._allowed_ids_tensor.to(device)] = True
        return mask

    def reset(self):
        """Clear per-sequence state. Call before each new generation batch."""
        self._states = []

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        batch_size, vocab_size = scores.shape
        self._ensure_states(batch_size)

        # First, update state machine based on the last token in each sequence.
        # input_ids reflects the tokens already produced (including any prompt);
        # the LAST token tells us what was just emitted at the previous step.
        if input_ids.shape[1] > 0:
            last_tokens = input_ids[:, -1].tolist()
            for b, tok in enumerate(last_tokens):
                st = self._states[b]
                if st.state == _State.BEFORE:
                    if tok == self.vocab.begin_id:
                        st.state = _State.INSIDE
                        st.abstract_count = 0
                elif st.state == _State.INSIDE:
                    if tok == self.vocab.end_id:
                        st.state = _State.AFTER
                    elif tok in self.vocab.abstract_ids:
                        st.abstract_count += 1
                        if st.abstract_count >= self.m_max:
                            st.forced_end = True
                # AFTER: terminal, no transition.

        # Then, mask logits for sequences in INSIDE state.
        allowed = self._allowed_mask(vocab_size, scores.device)
        for b, st in enumerate(self._states):
            if st.state != _State.INSIDE:
                continue
            if st.forced_end:
                # Force <endabstract> by giving it +inf and everything else -inf.
                scores[b] = torch.full_like(scores[b], float("-inf"))
                scores[b, self.vocab.end_id] = 0.0
                st.forced_end = False  # one-shot; next step will be AFTER
            else:
                scores[b] = torch.where(
                    allowed, scores[b], torch.full_like(scores[b], float("-inf"))
                )
        return scores


def assert_abstract_span_legal(
    sequence: Iterable[int], vocab: AbstractVocab, m_max: int = 128
) -> None:
    """Validate a fully generated sequence's abstract span. Used by tests."""
    state = _State.BEFORE
    count = 0
    for tok in sequence:
        if state == _State.BEFORE:
            if tok == vocab.begin_id:
                state = _State.INSIDE
                count = 0
        elif state == _State.INSIDE:
            if tok == vocab.end_id:
                state = _State.AFTER
            elif tok in vocab.abstract_ids:
                count += 1
                if count > m_max:
                    raise AssertionError(
                        f"Abstract span exceeded m_max={m_max} (got {count} tokens)"
                    )
            else:
                raise AssertionError(
                    f"Illegal token id {tok} in abstract span — not in V_abs "
                    f"∪ {{<endabstract>}}"
                )
        # AFTER: anything goes.
    if state == _State.INSIDE:
        raise AssertionError("Abstract span never closed with <endabstract>")

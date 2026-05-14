"""Policy iteration warm-up loop — paper Algorithm 1.

Top-level driver for Phase 1: alternates between bottlenecked SFT (with
verbal-CoT-guided abstract sequence generation) and self-distillation (with
prompt-only on-policy abstract generation), for T iterations.

Pseudo-code (paper Algorithm 1):
    for t = 1 .. T:
        # Bottlenecked SFT step
        D_{t,1} ⊂ D
        if t == 1:  ẑ^(t) ~ uniform from V_abs
        else:       ẑ^(t) ~ π_θ^abs(· | x, c)  (constrained decode, sees verbal CoT)
        θ_bar ← arg min L_SFT(θ; x, c, ẑ^(t), y) over D_{t,1}

        # Self-distillation step
        D_{t,2} ⊂ D
        ẑ' ~ π_θ_bar^abs(· | x)  (constrained decode, prompt only)
        θ^(t) ← arg min L_Distill(θ; x, ẑ', y) over D_{t,2}

See PLAN.md §6.6.
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from .bottlenecked_sft import (
    BottleneckCollator,
    BottleneckExample,
    bottleneck_loss,
    random_abstract_ids,
)
from .self_distillation import generate_abstract_sequences, make_distill_examples
from .tokenizer import AbstractVocab

logger = logging.getLogger(__name__)


@dataclass
class TrainingRecord:
    phase: str   # "bottlenecked_sft" or "self_distill"
    iteration: int
    epoch: int
    step: int
    loss: float
    wall_sec: float


@dataclass
class PolicyIterationConfig:
    T: int = 3                       # PI iterations
    epochs_per_phase: int = 3
    batch_size: int = 4
    grad_accum: int = 1
    learning_rate: float = 2e-5
    max_grad_norm: float = 1.0
    max_length: int = 1024
    m_max: int = 128
    on_policy_gen_batch: int = 8
    on_policy_temperature: float = 1.0
    on_policy_top_p: float = 0.95
    log_every: int = 10
    seed: int = 0
    save_dir: Optional[str] = None  # checkpoint output (per iter)


def _split_dataset(
    examples: list[dict], T: int, seed: int
) -> list[tuple[list[dict], list[dict]]]:
    """Stage the full dataset across iterations and phases.

    Per paper §3.2: D = ∪_t (D_{t,1} ∪ D_{t,2}). We split evenly across
    2*T buckets, so each iteration sees a disjoint slice for each phase.
    """
    rng = random.Random(seed)
    shuffled = examples.copy()
    rng.shuffle(shuffled)
    buckets = [shuffled[i :: 2 * T] for i in range(2 * T)]
    return [(buckets[2 * t], buckets[2 * t + 1]) for t in range(T)]


def _train_one_phase(
    model,
    tokenizer,
    vocab: AbstractVocab,
    examples: list[BottleneckExample],
    config: PolicyIterationConfig,
    phase: str,
    iteration: int,
    optimizer,
) -> list[TrainingRecord]:
    """Standard SFT loop using bottleneck_loss (works for both phases)."""
    collator = BottleneckCollator(
        tokenizer=tokenizer, vocab=vocab, max_length=config.max_length,
        dtype=torch.bfloat16 if model.dtype == torch.bfloat16 else torch.float32,
    )
    model.train()
    records: list[TrainingRecord] = []
    rng = random.Random(config.seed + iteration)

    for epoch in range(config.epochs_per_phase):
        rng.shuffle(examples)
        running_steps = 0
        for start in range(0, len(examples), config.batch_size):
            batch_examples = examples[start : start + config.batch_size]
            if len(batch_examples) < config.batch_size:
                break
            batch = collator(batch_examples)

            t0 = time.monotonic()
            optimizer.zero_grad()
            loss = bottleneck_loss(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            wall = time.monotonic() - t0

            loss_val = float(loss.detach().item())
            records.append(TrainingRecord(
                phase=phase, iteration=iteration, epoch=epoch,
                step=running_steps, loss=loss_val, wall_sec=wall,
            ))
            if running_steps % config.log_every == 0:
                logger.info(
                    "[iter=%d %s ep=%d step=%d] loss=%.4f  wall=%.2fs",
                    iteration, phase, epoch, running_steps, loss_val, wall,
                )
            running_steps += 1
    return records


def run_policy_iteration(
    model,
    tokenizer,
    vocab: AbstractVocab,
    dataset: list[dict],
    config: PolicyIterationConfig,
    optimizer_factory: Callable | None = None,
) -> list[TrainingRecord]:
    """Run T iterations of (bottlenecked SFT + self-distillation).

    dataset entries must contain keys: 'prompt', 'verbal_cot', 'answer'.
    """
    optimizer_factory = optimizer_factory or (
        lambda params: torch.optim.AdamW(params, lr=config.learning_rate)
    )
    optimizer = optimizer_factory(model.parameters())
    rng = random.Random(config.seed)

    splits = _split_dataset(dataset, T=config.T, seed=config.seed)
    all_records: list[TrainingRecord] = []

    for t in range(1, config.T + 1):
        D_t1, D_t2 = splits[t - 1]
        logger.info(
            "=== PI iteration %d / %d ===  |D_{t,1}|=%d  |D_{t,2}|=%d",
            t, config.T, len(D_t1), len(D_t2),
        )

        # --- Phase 1a: bottlenecked SFT ---
        # Generate ẑ^(t) for D_{t,1}: random at t=1, else on-policy conditioned on (x, c)
        prompts_1 = [d["prompt"] for d in D_t1]
        cots_1 = [d["verbal_cot"] for d in D_t1]
        answers_1 = [d["answer"] for d in D_t1]
        if t == 1:
            z_seqs_1 = [
                random_abstract_ids(vocab, length=rng.randint(1, max(2, config.m_max // 4)), rng=rng)
                for _ in D_t1
            ]
        else:
            t_gen0 = time.monotonic()
            z_seqs_1 = generate_abstract_sequences(
                model, tokenizer, vocab,
                prompts=prompts_1, cots=cots_1,
                m_max=config.m_max,
                batch_size=config.on_policy_gen_batch,
                temperature=config.on_policy_temperature,
                top_p=config.on_policy_top_p,
            )
            logger.info("Generated %d abstract seqs in %.1fs", len(z_seqs_1),
                        time.monotonic() - t_gen0)

        bsft_examples = [
            BottleneckExample(prompt=p, verbal_cot=c, abstract_ids=z, answer=y)
            for p, c, z, y in zip(prompts_1, cots_1, z_seqs_1, answers_1)
        ]
        records_a = _train_one_phase(
            model, tokenizer, vocab, bsft_examples, config,
            phase="bottlenecked_sft", iteration=t, optimizer=optimizer,
        )
        all_records.extend(records_a)

        # --- Phase 1b: self-distillation ---
        # Generate ẑ' for D_{t,2}: on-policy, conditioned on x ONLY (no c)
        prompts_2 = [d["prompt"] for d in D_t2]
        answers_2 = [d["answer"] for d in D_t2]
        t_gen0 = time.monotonic()
        z_seqs_2 = generate_abstract_sequences(
            model, tokenizer, vocab,
            prompts=prompts_2, cots=None,
            m_max=config.m_max,
            batch_size=config.on_policy_gen_batch,
            temperature=config.on_policy_temperature,
            top_p=config.on_policy_top_p,
        )
        logger.info("Generated %d distill abstract seqs in %.1fs",
                    len(z_seqs_2), time.monotonic() - t_gen0)

        distill_examples = make_distill_examples(prompts_2, z_seqs_2, answers_2)
        records_b = _train_one_phase(
            model, tokenizer, vocab, distill_examples, config,
            phase="self_distill", iteration=t, optimizer=optimizer,
        )
        all_records.extend(records_b)

        if config.save_dir:
            ckpt_path = f"{config.save_dir}/pi_{t}"
            logger.info("Saving checkpoint to %s", ckpt_path)
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)

    return all_records

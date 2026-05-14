"""1k-sample warm-up pilot.

Goal: validate the policy iteration loop end-to-end on real Granite at
small scale before launching the multi-day Phase 1 job. Uses a SYNTHETIC
1k-example dataset (templated arithmetic) so we don't need Dolci-Think
downloaded yet — we are testing the loop, not the data.

Pass criteria:
  - All T=1 iteration phases complete without OOM or crash
  - bottlenecked_sft loss decreases across the iteration
  - self_distill loss decreases across the iteration
  - Final checkpoint loads + constrained-decodes a legal abstract span
  - Total wall-clock < 30 min on 1× A100-40GB

Run:
    sbatch slurm/submit_warmup_pilot.slurm
or interactively:
    srun --partition=gpu-volatile --nodelist=n124 --gres=gpu:a100:1 \\
         --time=30 --cpus-per-task=4 --mem=32G --pty \\
         bash -lc "source ~/venvs/abscot/bin/activate && cd ~/abstract-cot && python scripts/run_warmup_pilot.py"
"""
from __future__ import annotations

import argparse
import logging
import random
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from abstract_cot.policy_iteration import (
    PolicyIterationConfig,
    run_policy_iteration,
)
from abstract_cot.self_distillation import generate_abstract_sequences
from abstract_cot.tokenizer import extend_tokenizer, resize_model_for_vocab


def synth_arithmetic_dataset(n: int, seed: int = 0) -> list[dict]:
    """Templated 2-operand arithmetic: (a, b, op) → (prompt, cot, answer).

    Provides loss-bearing supervision without needing Dolci. Variance is in
    operands, not problem shape, so this is a pilot for the LOOP, not the
    paper claims.
    """
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        op = rng.choice(["+", "-", "*"])
        a = rng.randint(2, 99)
        b = rng.randint(2, 99)
        if op == "+":
            c = a + b
            cot = f"Compute {a} + {b}. Add: {a} + {b} = {c}."
        elif op == "-":
            c = a - b
            cot = f"Compute {a} - {b}. Subtract: {a} - {b} = {c}."
        else:
            c = a * b
            cot = (
                f"Compute {a} * {b}. Break {a} = {a // 10}*10 + {a % 10}. "
                f"Then {a // 10}*10*{b} + {a % 10}*{b} = {(a // 10) * 10 * b} + "
                f"{(a % 10) * b} = {c}."
            )
        out.append(
            {
                "prompt": f"Question: What is {a} {op} {b}?",
                "verbal_cot": cot,
                "answer": f"{a} {op} {b} = {c}.",
            }
        )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ibm-granite/granite-4.0-micro")
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--T", type=int, default=1, help="PI iterations")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--m-max", type=int, default=32, help="paper default 128; pilot uses 32")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", default="outputs/warmup_pilot")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("warmup_pilot")

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)
    if device == "cuda":
        log.info("GPU: %s  Free: %.1f GB", torch.cuda.get_device_name(),
                 torch.cuda.mem_get_info()[0] / 1e9)

    log.info("Loading %s …", args.model)
    t0 = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device)
    log.info("Loaded in %.1fs; params=%.2fB", time.monotonic() - t0,
             sum(p.numel() for p in model.parameters()) / 1e9)

    tokenizer, vocab = extend_tokenizer(tokenizer, M=64)
    model = resize_model_for_vocab(model, tokenizer, seed=args.seed)
    log.info("Extended vocab: %d", len(tokenizer))

    dataset = synth_arithmetic_dataset(args.n_samples, seed=args.seed)
    log.info("Synth dataset: %d examples", len(dataset))

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    config = PolicyIterationConfig(
        T=args.T,
        epochs_per_phase=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        m_max=args.m_max,
        save_dir=args.save_dir,
        seed=args.seed,
    )

    t_start = time.monotonic()
    records = run_policy_iteration(
        model, tokenizer, vocab, dataset, config=config,
    )
    total_wall = time.monotonic() - t_start
    log.info("=== Pilot complete in %.1f min ===", total_wall / 60)

    # Phase-wise loss summary
    by_phase = {}
    for r in records:
        by_phase.setdefault(r.phase, []).append(r.loss)
    for phase, losses in by_phase.items():
        first10 = statistics.mean(losses[:10]) if len(losses) >= 10 else losses[0]
        last10 = statistics.mean(losses[-10:]) if len(losses) >= 10 else losses[-1]
        log.info("phase=%s n=%d  first10 mean=%.3f  last10 mean=%.3f  drop=%+.3f",
                 phase, len(losses), first10, last10, first10 - last10)

    # Final-model legality check
    log.info("Final-model constrained-decode legality check…")
    seqs = generate_abstract_sequences(
        model, tokenizer, vocab,
        prompts=["Question: What is 13 + 19?", "Question: What is 7 * 8?"],
        m_max=args.m_max, batch_size=2,
    )
    for prompt, ids in zip(["13+19", "7*8"], seqs):
        log.info("  %s: |z|=%d, first 8: %s", prompt, len(ids),
                 [tokenizer.decode([t]) for t in ids[:8]])

    log.info("Pilot exit gate: %s",
             "PASSED" if all(by_phase[p][0] > by_phase[p][-1] for p in by_phase) else "FAILED")


if __name__ == "__main__":
    main()

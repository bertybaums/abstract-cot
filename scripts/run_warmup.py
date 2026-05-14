"""Full Phase 1 warm-up on Dolci-Think-SFT data.

Loads the JSONL produced by scripts/prepare_dolci.py and runs the T-iteration
policy iteration loop (bottlenecked SFT + self-distillation) on Granite
4.0 Micro.

This is the multi-day Phase 1 job. Designed to run under SLURM as
slurm/submit_warmup.slurm; can also be launched interactively via srun.

Resumption: if --resume-from points to a directory containing a saved
checkpoint named pi_<k> for some k < T, we reload it and restart at
iteration k+1. (For now, manual — automatic mid-iteration resume requires
tracking the optimizer state, which we don't yet checkpoint.)
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from abstract_cot.policy_iteration import (
    PolicyIterationConfig,
    run_policy_iteration,
)
from abstract_cot.tokenizer import extend_tokenizer, resize_model_for_vocab


def load_dolci_jsonl(path: str | Path) -> list[dict]:
    """Load the JSONL produced by prepare_dolci.py."""
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def find_resume_checkpoint(resume_dir: str | Path) -> tuple[str, int] | None:
    """Find the highest pi_<k> directory in resume_dir."""
    if not resume_dir:
        return None
    p = Path(resume_dir)
    if not p.exists():
        return None
    best = -1
    best_path = None
    for child in p.iterdir():
        if child.is_dir() and child.name.startswith("pi_"):
            try:
                k = int(child.name.split("_")[1])
                if k > best:
                    best = k
                    best_path = str(child)
            except ValueError:
                continue
    return (best_path, best) if best_path else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/granite_3b.yaml")
    p.add_argument("--data", default="data/dolci_warmup_100k.jsonl")
    p.add_argument("--output-dir", default="outputs/warmup")
    p.add_argument("--resume-from", default=None,
                   help="output-dir of a prior run; auto-detects latest pi_<k>")
    p.add_argument("--T-override", type=int, default=None,
                   help="override config T (e.g. for a 1-iter smoke test)")
    p.add_argument("--n-override", type=int, default=None,
                   help="override n_samples (cap dataset size for a smoke test)")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("warmup")

    # Load config
    cfg_dict = yaml.safe_load(open(args.config))
    model_cfg = cfg_dict["model"]
    codebook_cfg = cfg_dict["codebook"]
    warmup_cfg = cfg_dict["warmup"]
    data_cfg = cfg_dict["data"]

    T = args.T_override if args.T_override is not None else warmup_cfg["policy_iterations"]
    n = args.n_override if args.n_override is not None else data_cfg["warmup_subsample"]

    log.info("Phase 1 warm-up | model=%s | M=%d | T=%d | n=%d",
             model_cfg["hf_id"], codebook_cfg["M"], T, n)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load + filter data
    log.info("Loading data from %s …", args.data)
    t0 = time.monotonic()
    dataset = load_dolci_jsonl(args.data)
    log.info("Loaded %d examples in %.1fs", len(dataset), time.monotonic() - t0)
    if len(dataset) > n:
        dataset = dataset[:n]
        log.info("Truncated dataset to %d", len(dataset))

    # Resume?
    resume = find_resume_checkpoint(args.resume_from)
    if resume:
        ckpt_path, start_t = resume
        log.info("RESUMING from %s (iter %d completed; restarting at %d)",
                 ckpt_path, start_t, start_t + 1)
        model_load_path = ckpt_path
        tokenizer_load_path = ckpt_path
        resume_t = start_t
    else:
        model_load_path = model_cfg["hf_id"]
        tokenizer_load_path = model_cfg["hf_id"]
        resume_t = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)
    if device == "cuda":
        log.info("GPU: %s  Free: %.1f GB", torch.cuda.get_device_name(),
                 torch.cuda.mem_get_info()[0] / 1e9)

    log.info("Loading model %s…", model_load_path)
    t0 = time.monotonic()
    tok = AutoTokenizer.from_pretrained(tokenizer_load_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_load_path,
        dtype=torch.bfloat16,
        attn_implementation=model_cfg["attn_implementation"],
    ).to(device)
    log.info("Loaded in %.1fs; params=%.2fB", time.monotonic() - t0,
             sum(p.numel() for p in model.parameters()) / 1e9)

    # Extend tokenizer only on a fresh load (resume keeps the extended tokenizer)
    if not resume:
        tok, vocab = extend_tokenizer(tok, M=codebook_cfg["M"])
        model = resize_model_for_vocab(model, tok, seed=0)
        log.info("Extended vocab: %d (M=%d)", len(tok), codebook_cfg["M"])
    else:
        # Reconstruct the vocab handle from the loaded extended tokenizer
        from abstract_cot.tokenizer import AbstractVocab, abstract_token_names, BEGIN_DELIM, END_DELIM
        M = codebook_cfg["M"]
        begin_id = tok.convert_tokens_to_ids(BEGIN_DELIM)
        end_id = tok.convert_tokens_to_ids(END_DELIM)
        abstract_ids = [tok.convert_tokens_to_ids(name) for name in abstract_token_names(M)]
        vocab = AbstractVocab(M=M, begin_id=begin_id, end_id=end_id, abstract_ids=abstract_ids)
        if any(i == tok.unk_token_id for i in abstract_ids):
            raise RuntimeError("Loaded tokenizer is missing abstract tokens — resume mismatched.")

    # Build training config
    bsft_cfg = warmup_cfg["bottlenecked_sft"]
    config = PolicyIterationConfig(
        T=T,
        epochs_per_phase=bsft_cfg["epochs"],
        batch_size=bsft_cfg["batch_size"],
        learning_rate=bsft_cfg["learning_rate"],
        max_length=data_cfg["max_seq_length"],
        m_max=codebook_cfg["m_max"],
        save_dir=str(out_dir),
        seed=0,
    )
    # If resuming, slice the iteration range; for simplicity we restart from
    # iteration resume_t+1 by reducing T and re-using the same dataset splits
    # (acceptable because data is sharded deterministically given seed).
    if resume_t > 0:
        # Run iterations resume_t+1 .. T by patching T to (T - resume_t) and
        # offsetting bucket indices via a new seed; simpler approach: still
        # run T total iterations but call with T_remaining and let the user
        # know they should adjust by hand for now.
        config.T = T - resume_t
        log.info("Will run %d remaining iterations (resume_t=%d, T=%d)",
                 config.T, resume_t, T)

    log.info("Starting policy iteration: T=%d, epochs=%d, batch=%d, "
             "lr=%g, seq_len=%d, m_max=%d",
             config.T, config.epochs_per_phase, config.batch_size,
             config.learning_rate, config.max_length, config.m_max)
    t_start = time.monotonic()
    records = run_policy_iteration(
        model, tok, vocab, dataset, config=config,
    )
    total = time.monotonic() - t_start
    log.info("=== warm-up complete in %.1f h ===", total / 3600)

    # Dump training log
    log_path = out_dir / "training_log.jsonl"
    with log_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({
                "phase": r.phase, "iteration": r.iteration,
                "epoch": r.epoch, "step": r.step,
                "loss": r.loss, "wall_sec": r.wall_sec,
            }) + "\n")
    log.info("Wrote training log: %s", log_path)

    # Final summary
    by_phase: dict[str, list[float]] = {}
    for r in records:
        by_phase.setdefault(r.phase, []).append(r.loss)
    for phase, losses in by_phase.items():
        first50 = sum(losses[:50]) / max(1, min(50, len(losses)))
        last50 = sum(losses[-50:]) / max(1, min(50, len(losses)))
        log.info("phase=%s n=%d  first50 mean=%.3f  last50 mean=%.3f",
                 phase, len(losses), first50, last50)


if __name__ == "__main__":
    main()

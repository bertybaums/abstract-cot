# abstract-cot

**Date kicked off:** May 12, 2026

Reproduction of *Thinking Without Words: Efficient Latent Reasoning with Abstract Chain-of-Thought* (Ramji, Naseem, Astudillo, IBM Research AI, arXiv:2604.22709) on fortyfive HPC.

Start with **`PLAN.md`** — that's the comprehensive plan. This README is just orientation.

## TL;DR

Train a small instruction-tuned LM (Granite 4.0 Micro 3B → Qwen3-4B → Qwen3-8B) to replace its verbal chain-of-thought with a short sequence of *reserved* tokens (`<TOKEN_A>` … `<TOKEN_BL>`), getting ~10× token efficiency at comparable accuracy on MATH-500, AlpacaEval, and HotpotQA.

Three training phases:
1. **Bottlenecked SFT** with a block-structured attention mask (answer attends to abstract tokens, not the verbal CoT).
2. **Self-distillation** from on-policy abstract sequences.
3. **GRPO** with constrained decoding and an external generative reward (gpt-oss-20b via MindRouter).

Phases 1+2 iterate T=3 times (policy iteration warm-up); then Phase 3 (RL).

## Status

Pre-Phase-0. Folder is seeded with `PLAN.md` and skeleton directories. No code written, no jobs run.

## Quickstart (eventually — once Phase 0 is done)

```bash
# On fortyfive login node:
git pull
bash scripts/setup_env.sh           # builds ~/venvs/abscot
bash scripts/prepare_data.sh        # downloads HF models + datasets
sbatch slurm/submit_warmup.slurm    # ~3 days on cmci-gpu-8
sbatch slurm/submit_rl.slurm        # ~3-5 days on cmci-gpu-8
sbatch slurm/submit_eval.slurm
```

## Key references
- Paper: `notes/paper-v2.pdf` (arXiv:2604.22709v2)
- Datasets: `allenai/Dolci-Think-SFT-7B`, `allenai/Dolci-Think-RL-7B`
- Reward model: `openai/gpt-oss-20b` via MindRouter
- Base models: `ibm-granite/granite-4.0-micro`, `Qwen/Qwen3-4B`, `Qwen/Qwen3-8B`

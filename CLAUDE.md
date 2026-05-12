# CLAUDE.md — abstract-cot

## What This Is

Reproduction of *Thinking Without Words: Efficient Latent Reasoning with Abstract Chain-of-Thought* (Ramji et al., arXiv:2604.22709v2, IBM Research AI, April 2026). Train small instruction-tuned LMs to emit a short sequence of reserved "abstract" tokens in place of a verbalized chain-of-thought, getting ~10× compression at comparable accuracy.

Start with `PLAN.md`. The paper PDF lives at `notes/paper-v2.pdf`.

## Project conventions

- **Reports** must include the date in the filename: `report-YYYY-MM-DD.md` (per global `~/.claude/CLAUDE.md`).
- **Folder structure** follows the cx-bot / marc-interp pattern: `abstract_cot/` (Python package), `scripts/` (entry points), `configs/` (YAML), `slurm/` (sbatch scripts), `eval/`, `data/`, `notes/`, `outputs/`, `logs/`.
- **HPC venv:** `~/venvs/abscot/` on fortyfive — Python 3.11.11 + PyTorch 2.6 cu124 + transformers + trl + peft + bitsandbytes + flash-attn + openai (for MindRouter).
- **SLURM:** `source /etc/profile` **before** `set -e`; `--exclude=n113` always; for bf16+compile on gpu-8 also exclude `n118,n121,n122`. Primary target partition is `cmci-gpu-8` (A100s).
- **Reward model:** `openai/gpt-oss-20b` via MindRouter, 4 concurrent requests max (use a semaphore). Endpoint: `https://mindrouter.uidaho.edu/v1`. API key in `MINDROUTER_API_KEY` env var.

## Key design notes

- **Codebook size M = 64** (paper's best). Tokens are named alphabetically: `<TOKEN_A>` … `<TOKEN_Z>`, then two-letter `<TOKEN_AA>` …. Two delimiters: `<beginabstract>`, `<endabstract>`.
- **Max abstract length m_max = 128**.
- **Policy iteration T = 3** rounds of (bottlenecked SFT → self-distillation).
- **GRPO** for RL, with KL regularization vs. warm-started reference policy.
- **Constrained decoding** restricts the abstract span to `V_abs ∪ {<endabstract>}`; response span is unconstrained.
- **Disable thinking mode** on Qwen3 models when comparing — paper does this; baselines must match.

## Engineering risks (see PLAN.md §10 for full table)

- Block-structured attention mask in HuggingFace is non-trivial. Use `attn_implementation="flex_attention"` (PyTorch 2.5+) with a custom mask function. Parity-test against an eager-mode reference on small inputs.
- MindRouter reward-call latency will dominate RL wall-clock. Cache by `hash(prompt, completion)`; consider falling back to gpt-oss-120b (8 concurrent) if 20b is too slow.
- Compute nodes have no internet — MindRouter unreachable from compute. Either run RL on the login node (CPU-only orchestration with GPU rollouts via slurm), or set up an SSH tunnel through the login node. Test in Phase 0.

## Related projects

- [[cx-bot]] — Tier-3 finding (logical coherence inconsistent) is the *exact* gap Abstract-CoT's RL stage targets. Natural follow-on: A-CoT-style RL on cx-bot data.
- [[gettier-repair-space]] — abstract tokens give a clean instrument for "does a small LM have a latent reasoning vocabulary?"
- [[marc-interp]] — the power-law codebook is a mech-interp target; post-RL checkpoint embeddings can be SAE'd.

# Abstract-CoT Reproduction — Plan

**Date:** May 12, 2026
**Author:** Bert Baumgaertner (RCDS, U of Idaho)
**Paper:** Ramji, Naseem & Astudillo, *Thinking Without Words: Efficient Latent Reasoning with Abstract Chain-of-Thought*, arXiv:2604.22709v2 (April 27 2026), IBM Research AI.
**Status:** pre-Phase-0 (folder seed, no code written, no jobs run)

---

## 1. What we are reproducing

The paper introduces **Abstract Chain-of-Thought (Abstract-CoT, "A-CoT")**: a post-training mechanism that replaces a verbalized chain-of-thought with a short sequence of *reserved* abstract tokens drawn from a learned codebook. The model emits something like

```
<beginabstract> E C AE F A BB D G BA H AC B AD F <endabstract>
Answer: d = 120 km
```

instead of an 8-step natural-language rationale, and gets **10.4–11.6×** fewer reasoning tokens on MATH-500 while matching SFT+RL with verbal CoT. Comparable findings on AlpacaEval (1.9–2.2× compression) and HotpotQA (4.0–4.3×). The effect generalizes across Qwen3-8B, Qwen3-4B, and Granite 4.0 Micro (3B).

The recipe has three phases (see paper Figure 2 / Algorithm 1):

1. **Bottlenecked SFT** — train on `[prompt; verbal_CoT; abstract_seq; answer]` with a *block-structured attention mask* that forces the answer to attend to the abstract tokens but **not** the verbal CoT. This makes the abstract segment carry the information.
2. **Self-distillation** — sample abstract sequences on-policy from the prompt alone (under constrained decoding) and SFT on `[prompt; abstract_seq; answer]`.
3. **Warm-started RL (GRPO)** — sample K trajectories per prompt, score with a generative reward model (paper uses `gpt-oss-20b`), update with GRPO. The action space during the abstract segment is restricted to the codebook ∪ `<endabstract>` via constrained decoding; the response segment decodes unconstrained.

Phases 1+2 run as a **policy iteration loop** for *T* = 3 iterations. RL runs for 1M episodes after warm-up.

Headline configuration (paper):
- Codebook size **M = 64**, max abstract length **m_max = 128**.
- Tokens named `<TOKEN_A>` … `<TOKEN_Z>`, then `<TOKEN_AA>` … (alphabetical).
- Initialized as new embedding rows (random); two delimiters `<beginabstract>`, `<endabstract>` are also new tokens.

Two empirical findings of interest:
- A **power-law / Zipfian** frequency distribution emerges over the codebook during RL.
- **Permutation ablation:** shuffling the abstract tokens drops MATH-500 by 7.8 pts → the *order* carries signal, not just the bag of tokens.

---

## 2. Why this is a fit for fortyfive

The paper trains on **8× H100 (SFT) and up to 32× H100 (RL)**. We do not have H100s. But:

- The recipe is **post-training**, not pre-training. Base model already has its full capability; we are only learning embeddings for ~66 new rows + adapting the policy.
- Granite 4.0 Micro is **3 B parameters** and was one of the paper's three models. At 3 B, bf16 full fine-tune fits on a single A100 (40 GB) with conservative batch size, or on 2× A6000 (48 GB each) with FSDP/ZeRO-2. QLoRA + trainable embeddings fits on a single A6000.
- The **reward model is `gpt-oss-20b`**, and that is already available to us via **MindRouter** (`openai/gpt-oss-20b`, 4 concurrent requests). We do not host it — we hit MindRouter during RL rollouts. This removes the heaviest single piece of paper-scale infrastructure.
- The training corpora (`allenai/Dolci-Think-SFT-7B`, `allenai/Dolci-Think-RL-7B`) are public on HuggingFace. The paper used 600k subsampled examples for warm-up. We can subsample further if needed.

What we sacrifice vs. the paper:
- **Wall-clock per run.** A 3 B model at 600k samples × 3 epochs × 3 PI iterations is non-trivial on a single A100; expect days, not hours.
- **Headroom to ablate.** The paper runs `M ∈ {1, 2, 4, …, 512}`. We will run *one* full sweep at the chosen scale and pick **M = 64** (paper's best) for the main results; only ablate if cycles allow.
- **Largest model.** Reproducing on Qwen3-8B is feasible (QLoRA on 1× A100 80 GB or full FT on 2× A100), but expensive. We treat 8B as a stretch goal, not a milestone.

---

## 3. Goals and scope (three tiers)

### Tier 1 — Must-replicate (the headline claim)
Reproduce Abstract-CoT on **Granite 4.0 Micro (3 B)** at **M = 64, m_max = 128**, full pipeline (Warm-up via PI ×3 → GRPO). Compare against the SFT(CoT) and SFT+RL baselines from the paper on the same model. Success criteria:
- Abstract-CoT (Warm-up + RL) achieves **≥ 70 % on MATH-500** (paper: 74.4).
- Token compression **≥ 5×** on MATH-500 vs. SFT(CoT) tokens (paper: ~9×, 1412 → 153).
- AlpacaEval win-rate **≥ 30 %** (paper: 33.5).
- HotpotQA F1 **≥ 38** (paper: 42.6).
We do not need to *exceed* the paper's numbers — we need to land in the same regime to confirm the recipe works under our compute budget.

### Tier 2 — Target
Add **Qwen3-4B** as a second base model. Run an **M ablation** at `{16, 64, 256}` (the paper's saturation curve) on the cheaper model.

### Tier 3 — Stretch
Add **Qwen3-8B** (QLoRA, single A100 80 GB). Generate the **token-frequency power-law plot** (paper Figure 4) — needs logging of per-step token frequencies during RL.

Out of scope (for now): Qwen3-32B, AIME'25/GPQA-Diamond evals beyond a sanity-check sample, mechanistic interpretability of the learned codebook (interesting downstream — links to [[mech-interp-big-picture]]).

---

## 4. Hardware mapping (paper → fortyfive)

| Paper phase | Paper HW | Our HW (Granite 3B) | Our HW (Qwen3-8B) |
|---|---|---|---|
| Bottlenecked SFT | 8× H100 | 1× A100-40GB (n123/126/127) **or** 2× A6000 (n112/n130) | 1× A100-80GB w/ QLoRA |
| Self-distillation | 8× H100 | same | same |
| GRPO | up to 32× H100 | 2× A100 preferred (rollouts) | 2× A100-80GB |
| Reward model serving | gpt-oss-20b co-located | **MindRouter** (`openai/gpt-oss-20b`) | same |

**Partition strategy** (per `~/.claude/projects/-Users-bbaum-Documents--RCDS/memory/fortyfive.md`):
- Primary target: `cmci-gpu-8` for A100 access (n123 has 4× A100; n126–127 each have 2× A100).
- Fallback: `gpu-volatile` (n124 = 2× A100, preemptible — must use `--requeue` + checkpoint-resume).
- Backup: `gpu-8` on `--exclude=n113,n118,n121,n122` for bf16+compile.
- Avoid `gpu-9` (Python 3.9 on compute nodes; would require a separate venv).

**Reward-model rate limit:** MindRouter caps `gpt-oss-20b` at ~4 concurrent. GRPO with K=4 rollouts × batch_size=4 prompts = 16 outstanding scoring calls. We will queue reward calls with a semaphore of 4 and batch K rollouts asynchronously. Expected reward latency budget: ~3–8 s per scoring call → ~30–60 s per GRPO step. This will dominate wall-clock, not GPU compute. Mitigation discussed in §10.

---

## 5. Models and datasets

### Base models
| Model | HF ID | Params | Plan |
|---|---|---|---|
| Granite 4.0 Micro | `ibm-granite/granite-4.0-micro` | 3 B | Tier-1 primary |
| Qwen3-4B (instruct) | `Qwen/Qwen3-4B` | 4 B | Tier-2 |
| Qwen3-8B (instruct) | `Qwen/Qwen3-8B` | 8 B | Tier-3 (QLoRA) |

**Important:** the paper evaluates models *without* their built-in "thinking mode" (Qwen3 has one). We must explicitly disable it for both baselines and Abstract-CoT to make the comparison apples-to-apples. Confirm by inspecting the chat template and stripping any `<think>...</think>` scaffolding in the tokenizer.

### Datasets
| Dataset | HF ID | Use |
|---|---|---|
| Dolci-Think-SFT-7B | `allenai/Dolci-Think-SFT-7B` | Warm-up. Subsample 600 k (paper) → 100 k–300 k for our budget |
| Dolci-Think-RL-7B | `allenai/Dolci-Think-RL-7B` | RL prompts (prompts + gold answers only; no verbal CoT used at RL) |
| MATH-500 | `HuggingFaceH4/MATH-500` | Eval (math) |
| AlpacaEval-LC-2.0 | `tatsu-lab/alpaca_eval` (LC v2) | Eval (instruction following, judge=gpt-oss-120b via MindRouter) |
| HotpotQA | `hotpot_qa` | Eval (multi-hop QA, 500-sample subset) |
| AIME'25 | art-of-problem-solving scrape | Eval (hard math) — sanity-check sample only |
| GPQA-Diamond | `Idavidrein/gpqa` | Eval — sanity-check sample only |

**Internet caveat:** compute nodes have no internet. We pre-download all five datasets + the three base models to `~/abstract-cot/data/` and `~/hf_cache/` on the login node before any sbatch.

---

## 6. Engineering plan — what we have to build

This is where the project becomes non-trivial. There are five engineering pieces that are not turn-key in HuggingFace + TRL today:

### 6.1 Tokenizer extension
- Add 64 reserved tokens (`<TOKEN_A>` … `<TOKEN_BL>` with two-letter overflow) + two delimiters (`<beginabstract>`, `<endabstract>`).
- Resize embedding + LM head. Initialize new rows from `N(0, σ²)` where σ matches the mean L2 norm of existing rows (a less-cold start than the default).
- Persist tokenizer + initial embedding rows so all three phases share the same indices.
- File: `abstract_cot/tokenizer.py`.

### 6.2 Block-structured attention mask (Bottlenecked SFT)
This is the recipe-defining piece. Within a single sequence `[x; c; z̃; y]`:
- `x` (prompt), `c` (verbal CoT), `z̃` (abstract sequence), `y` (answer).
- Default causal mask everywhere, with one exception: **`y` cannot attend to `c`**. Concretely, for token positions `i ∈ y` and `j ∈ c`, set mask entry to 0.
- All other attention follows standard causal.

HuggingFace's default `attention_mask` is a 1D padding mask, not a 2D per-position mask. Options:
- **Use `attn_implementation="flex_attention"` (PyTorch 2.5+)** with a custom mask function that takes `(b, h, q_idx, kv_idx)` and reads a per-example block-partition tensor. This is the cleanest path.
- Fallback: manually construct a `(seq_len, seq_len)` boolean mask per sample and pass via `_attn_implementation="eager"`. Slow but correct — useful for unit tests.

The model needs to emit a per-example segment-index tensor (which positions are in `x`, `c`, `z̃`, `y`). We pack these into the dataset during tokenization.
- File: `abstract_cot/attention_masks.py`.
- Unit test: construct a 32-token toy sequence, build the mask both ways, assert equality with the manual mask.

### 6.3 Constrained decoding
*Borrow the wrapping pattern from `_RCDS/compression/translator/model.py::StateMachineLogitsProcessor` — same shape (per-sequence state, mask logits, force end-token at cap), strictly simpler constraint (closed 65-token allowed set, no BPE trie).*
At decode time the abstract span must come only from the allowed set 𝒜 = 𝒱_abs ∪ {`<endabstract>`}. After `<endabstract>` is emitted (or the m_max cap is hit), generation switches to unconstrained.
- Implementation: a HF `LogitsProcessor` that:
  - Tracks a per-sequence state machine: `awaiting_begin → inside_abstract → after_end`.
  - In `inside_abstract`, sets logits outside 𝒜 to `-inf`.
  - On `<endabstract>` (or when generated-count hits `m_max`, force `<endabstract>`), transitions to `after_end` and stops masking.
  - At `m_max`, forcibly emits `<endabstract>` regardless of the argmax.
- File: `abstract_cot/constrained_decoding.py`.
- Used identically in (a) self-distillation on-policy sampling, (b) GRPO rollouts, (c) eval-time generation.

### 6.4 GRPO with constrained decoding and external reward
TRL has `GRPOTrainer` (≥ 0.10). We need to subclass it to:
- Inject our `LogitsProcessor` during the generation step.
- Compute reward via an async MindRouter client instead of the built-in reward model.
- Apply KL regularization against the warm-started reference policy (paper Eq. 5).
- Update both the abstract span *and* the response span (paper does both; can be ablated to abstract-only).

Reference policy: snapshot of the warm-up model. Frozen, on CPU between batches.

Reward function: per paper §3.3, a *generative* reward model. Appendix B is now extracted to `configs/reward_prompt.txt`. Build a `score_completion(prompt, completion) -> float` function that:
- Calls MindRouter with `openai/gpt-oss-20b` and **`reasoning_effort: "medium"`** (paper specifies "medium" thinking mode — this is non-trivial extra latency per call; budget accordingly).
- Substitutes `{CONVERSATION_HISTORY}` (= the prompt + any prior turns) and `{RESPONSE_TO_SCORE}` (= the completion) into the template.
- **Parses JSON output** of the form `{"score": <0-10>, "reasoning": "..."}`. Normalize: `r = score / 10.0` → scalar in [0, 1].
- Logs both `score` and `reasoning` to a JSONL trace file (per-call) so we can audit reward signals later.
- Has retry with exponential backoff for 429/500 and a graceful fallback (e.g. return `r = 0.0` and log) on persistent JSON-parse failure — these will happen.
- File: `abstract_cot/reward_model.py`, with the MindRouter prompt in `configs/reward_prompt.txt`.

**Reuse, do not reinvent:** the rate-limiter pattern at `_RCDS/compression/corpus/generation/generate_reasoning.py::AsyncTokenBucket` is already battle-tested against MindRouter. The cap is **200 req/min per account** (not per model — shared across all our calls); 429 retries amplify outgoing rate 2–5× if the bucket isn't process-global, which caused full-service cascades on April 17, 2026. Read `compression/CLAUDE.md` MindRouter section before writing this layer. Net effect: combine the 200 rpm global cap with the gpt-oss-20b per-model 4-concurrent cap as nested limits.

**Latency note:** with `reasoning_effort: "medium"`, expect ~5–15 s per reward call. At GRPO batch_size=4, K=4 rollouts = 16 reward calls per step. Even with concurrency=4, that's ~4 sequential rounds → ~20–60 s per step on the reward path alone. **Reward latency, not GPU compute, will dominate RL wall-clock.** Cache aggressively by `hash(prompt, completion)` — Dolci-Think-RL-7B has identical prompts across the sample stream.

**Reuse, do not reinvent:** the rate-limiter pattern at `_RCDS/compression/corpus/generation/generate_reasoning.py::AsyncTokenBucket` is already battle-tested against MindRouter. The cap is **200 req/min per account** (not per model — shared across all our calls); 429 retries amplify outgoing rate 2–5× if the bucket isn't process-global, which caused full-service cascades on April 17, 2026. Read `compression/CLAUDE.md` MindRouter section before writing this layer. Net effect: combine the 200 rpm global cap with the gpt-oss-20b per-model 4-concurrent cap as nested limits.

### 6.5 Bottlenecked-SFT trainer + self-distillation trainer
Both are causal-LM SFT under the standard loss, but:
- Bottlenecked SFT uses the custom attention mask **and** restricts the loss to `j ∈ z̃ ∪ y` positions (mask out prompt and verbal CoT from the loss).
- Self-distillation uses standard causal mask but the data is `[x; z̃; y]` (no `c`), with abstract sequences sampled on-policy from the previous iteration.
- The simplest path: subclass `transformers.Trainer` with a custom `compute_loss` and a custom collator. TRL's `SFTTrainer` does not give us the mask hook we need cleanly.
- File: `abstract_cot/bottlenecked_sft.py`, `abstract_cot/self_distillation.py`.

### 6.6 Policy iteration loop
A top-level driver that:
- Initializes from base instruction-tuned model + extends tokenizer (once).
- For t = 1 … T:
  - **(t.1)** Generate abstract traces for batch `D_{t,1}`:
    - t=1: random sample from 𝒱_abs (paper found uniform-random best vs. alphabetical or power-law init).
    - t≥2: on-policy via constrained decoding from current model.
  - **(t.2)** Bottlenecked SFT for 3 epochs on `(x, c, z̃, y)`.
  - **(t.3)** On-policy sample `z'` from current model (constrained decode, no verbal CoT in context).
  - **(t.4)** Self-distillation SFT for 3 epochs on `(x, z', y)`.
- Save checkpoint at end of each iteration.
- File: `abstract_cot/policy_iteration.py`, driven by `scripts/run_warmup.py`.

---

## 7. Evaluation harness

Build once, run across every checkpoint (base, post-PI-1, post-PI-2, post-PI-3, post-RL):

- `eval/math500.py` — MATH-500 with exact-answer extraction (paper uses Hendrycks-style boxed answer matching).
- `eval/alpaca_eval.py` — wraps `tatsu-lab/alpaca_eval` v2 LC; judge = `openai/gpt-oss-120b` via MindRouter (paper uses GPT-4-based judge but we substitute the campus-hosted 120 B for cost — this should be a defensible substitution and we will note it in the writeup).
- `eval/hotpotqa.py` — 500-sample subset, F1.
- `eval/aime25.py` — sanity-check only (30 problems).
- `eval/gpqa_diamond.py` — sanity-check only (sample of 50).
- `eval/run_all.py` — one driver, one report.

**Token-count instrumentation.** The compression claim *is* the result. Log:
- Per-example abstract-segment length (m).
- Per-example response length.
- Per-example total reasoning + response tokens.
- Aggregate ratio `E[c_verbal] / E[m]` vs. the SFT(CoT) baseline.

Reports land at `outputs/report-YYYY-MM-DD.md` per the global convention.

---

## 8. SLURM submission scripts

One per phase + one for eval:

- `slurm/submit_pretrain.slurm` — none needed (pretraining not in scope).
- `slurm/submit_warmup.slurm` — runs the PI loop end-to-end, 3 iterations, ~3 days, partition `cmci-gpu-8` (n123 or n126/127) with `--gres=gpu:a100:1`.
- `slurm/submit_rl.slurm` — runs GRPO for N episodes (start with 100 k to validate, then scale to 1 M), partition `cmci-gpu-8`, 2 GPUs (1 for policy + ref, 1 for rollout cache; configurable).
- `slurm/submit_eval.slurm` — 1 GPU, 30 min.

All scripts must:
- `source /etc/profile` *before* `set -e` (per cluster gotcha).
- `module load python/3.11.11 cuda/12.8 || true`.
- Activate `~/venvs/abscot/` (we will build this — based on `~/ARC/venv/`).
- `--exclude=n113` always.
- Have a `--requeue` + checkpoint-resume block for `gpu-volatile`.

---

## 9. Phase plan and milestones

### Phase 0 — Environment, paper extraction, smoke test (1 week)
- [ ] Clone paper PDF v2 to `notes/paper-v2.pdf`. Extract Appendix B (reward prompt) and Appendix A (ablation tables). The reward prompt is the single biggest missing detail right now.
- [ ] Build `~/venvs/abscot/` on fortyfive: PyTorch 2.6 (cu124), transformers ≥ 4.46, trl ≥ 0.13, peft, bitsandbytes, datasets, accelerate, flash-attn (if A100), `openai` client (for MindRouter). Snapshot the requirements lock.
- [ ] Pre-download all three base models + all five datasets to `/mnt/ceph/bbaum/hf_cache/` on the login node.
- [ ] Confirm MindRouter `openai/gpt-oss-20b` round-trip with a dummy reward prompt. Measure latency × 20 samples.
- [ ] Write `abstract_cot/tokenizer.py` and unit-test the extension (load → resize → re-save → reload).
- [ ] Write `abstract_cot/attention_masks.py` with flex_attention + eager parity test.
- [ ] Smoke test: load Granite 4.0 Micro, extend tokenizer, run one forward pass with the custom mask on a fake batch.

**Exit criterion:** all five engineering pieces (§6.1–6.5) have at least a skeleton, the attention mask passes its parity test, and a single bottlenecked-SFT step runs end-to-end on a 4-example toy batch.

### Phase 1 — PI warm-up on Granite 4.0 Micro (1.5 weeks)
- [ ] Subsample 100 k examples from Dolci-Think-SFT-7B; filter for short prompts to control sequence length.
- [ ] Run PI iteration 1 (bottlenecked SFT, 3 epochs) on `cmci-gpu-8`. Checkpoint per epoch.
- [ ] Eval checkpoint on MATH-500 (small sample, 50 items) — sanity that loss is dropping in the right direction.
- [ ] Run PI iteration 1 self-distillation. Eval.
- [ ] Repeat for iterations 2 and 3. Full eval (MATH-500 + AlpacaEval + HotpotQA) after iteration 3.

**Exit criterion:** Granite + Warm-up post-PI-3 reaches ≥ 65 % on MATH-500 (paper: 71.8) and produces well-formed abstract sequences under constrained decoding ≥ 95 % of the time.

### Phase 2 — GRPO RL on Granite (2 weeks)
- [ ] Build `abstract_cot/reward_model.py` with the extracted reward prompt + MindRouter async client + semaphore=4.
- [ ] Subclass `GRPOTrainer`; integrate `LogitsProcessor`; freeze reference policy.
- [ ] Start with 5 k-episode dry run on a 1 k-prompt subset to verify the trainer runs end-to-end.
- [ ] Scale to 100 k episodes. Re-eval.
- [ ] Scale to 1 M episodes if Phase 1 timing allows. Re-eval.

**Exit criterion:** Granite + Warm-up + RL hits the Tier-1 numbers in §3.

### Phase 3 — Add Qwen3-4B + ablations (1.5 weeks)
- [ ] Rerun the full pipeline on Qwen3-4B with the same hyperparameters.
- [ ] M ablation at {16, 64, 256} on whichever model is cheaper to retrain (likely Qwen3-4B).

### Phase 4 — Stretch (Qwen3-8B + power-law plot)
- [ ] QLoRA + trainable embeddings on Qwen3-8B (1× A100 80 GB).
- [ ] Log per-step token frequencies during RL; reproduce paper Figure 4.

---

## 10. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| **Block-structured attention mask is wrong** | Medium | Parity test against an eager-mode reference implementation on tiny inputs; verify gradient flow with a known-toy task before scaling |
| **MindRouter rate-limit dominates wall-clock during RL** | High | (a) Batch K rollouts under a semaphore of 4; (b) cache reward calls keyed on (prompt, completion) hash — Dolci-Think-RL-7B has repeats; (c) fall back to `openai/gpt-oss-120b` (8 concurrent) if latency is the bottleneck — note in writeup; (d) consider a *local* lightweight reward model (a small classifier) as a sanity baseline |
| **m_max = 128 wastes capacity for easy prompts** | Low | Paper's truncation ablation (Table 3b) shows degradation is graceful; if needed, add a length-penalty term to reward |
| **Embedding rows for new tokens collapse / stay at init** | Medium | Monitor mean ‖embed[abs]‖₂ over training; if no growth, increase LR on new embedding rows only via param groups |
| **GRPO instability (paper note: cold-start RL underperforms base)** | Confirmed in paper | Always run *warm-started* RL — never cold-start. Use β (KL) = 0.04 as a starting point; sweep {0.01, 0.04, 0.1} if collapse observed |
| **Generative reward model is noisy / hackable** | Medium | Run a permutation check during RL (shuffle abstract tokens, score should drop) as a hack-detector. If reward saturates without performance gain, suspect hacking |
| **Compute nodes have no internet → MindRouter unreachable** | Confirmed | Submit RL job to login-node-adjacent partition, OR: run a tiny SSH tunnel from the compute node back through the login node. Test this in Phase 0 |
| **A100 nodes always full** | Medium | Use `gpu-volatile` with `--requeue` + 30-minute checkpoint interval; warm-up SFT can run on 2× A6000 with FSDP |

---

## 11. Open questions for Bert

(Recording, not blocking — making the reasonable call per session instructions.)

1. **Reward prompt.** Paper Appendix B has the *exact* generative-reward prompt template; we need to verify it once we extract the PDF section. If unavailable, draft our own and note the substitution.
2. **Eval-time judge.** Paper uses GPT-4-class judge for AlpacaEval. We substitute `openai/gpt-oss-120b` via MindRouter — defensible but a non-trivial change to flag in the writeup.
3. **Scaling-down ratios.** Paper used 600 k warm-up samples and 1 M RL episodes. We will start at 100 k / 100 k and scale up only if metrics underperform. We should agree on a budget cap before launching.
4. **Will we open-source this?** If yes, push to `bertybaums/abstract-cot` on GitHub from day one; otherwise keep local.

---

## 12. Connections to other RCDS projects

### Compression / UGF (`_RCDS/compression/`) — sister project on vocabulary compression
The closest sibling. Compression asks whether Up Goer Five (~3,643 inflected forms of the 1,000 most common English words) is *expressively adequate* for philosophical reasoning — a 1.5B Llama-style Reasoner trained from scratch on a pure-UGF corpus, with a T5-small Translator bridging full English ↔ UGF. **Both projects ask: what's the minimum vocabulary for reasoning?** They sit at adjacent points on the same axis:

```
fully verbal CoT   →   UGF (compression)   →   Abstract-CoT (this)   →   continuous latents (Coconut)
fully readable     human-readable, smaller  opaque, learned, ~10× shorter   fully opaque
~3,643 tokens used as English    M=64 learned tokens    n-dim vectors
```

There is a **Sheffer-stroke trade-off lurking here**: NAND gives a smaller alphabet at the cost of longer formulas. Compression's hypothesis is that UGF preserves expressive power at the cost of longer sentences (the Sheffer trade). A-CoT's headline claim is the *opposite* trade-off — a smaller alphabet with *shorter* sequences. Putting the two side-by-side is a compelling expository move for a future writeup. (Compression already has a `docs/followups/sheffer.md` — worth aligning with.)

**Two pieces of engineering to borrow directly, do not reinvent:**
1. **`StateMachineLogitsProcessor`** at `_RCDS/compression/translator/model.py` is the same shape as our constrained-decoder (§6.3). A-CoT's case is strictly simpler (closed 65-token allowed set, no BPE trie) but the wrapping pattern — track per-sequence state, mask logits, force end-token at cap — is identical.
2. **`AsyncTokenBucket`** at `_RCDS/compression/corpus/generation/generate_reasoning.py` is *exactly* the MindRouter rate-limiter we need for §6.4. Compression's documented gotchas (200 rpm cap, 429 cascades amplified by retries, login-node-only access) all apply to us. Read `compression/CLAUDE.md` MindRouter section before writing the reward-call layer.

**Stretch experimental connection:** train A-CoT *on top of* the UGF Reasoner once both exist. A model that can only express itself in UGF, but reasons internally in a learned 64-token abstract vocabulary, is a double-compression setup. That's a paper of its own.

### Numerals (`_RCDS/Numerals/`) — direct RL precedent
Numerals already wrote the RL lessons that A-CoT's GRPO phase will need to follow:
- **Use `{0, +1}` rewards, not `{-1, +1}`** (avoids zero-gradient traps).
- **SFT data mixing is essential for RL stability** — pure RL drifts. We should consider mixing a small SFT batch into each GRPO step (`λ_sft * CE(sft_example)` term) even though the paper doesn't explicitly do so.
- **KL penalty alone is insufficient** — one-sample KL estimates miss unseen tokens. Paper Eq. 5 uses a one-sample-style KL; we should monitor whether format collapse begins after some episodes and be ready to add an SFT-mixing term.
- **Scaffold design matters more than reward algorithm** — for A-CoT, the analog is *codebook construction* (M, m_max, init scheme). Treat these as first-order knobs.

### cx-bot (`_RCDS/cx-bot/`)
Tier 3 finding: SFT nails format but logical coherence is inconsistent (~37 % weak). Abstract-CoT's RL stage with a generative-quality reward is *exactly* the recipe cx-bot's open question was asking about. **Follow-on after Tier 1 lands: A-CoT-style RL on cx-bot data, with the reward measuring logical coherence rather than answer correctness.**

### marc-interp (`_RCDS/marc-interp/`) — interpretability followup
The power-law / Zipfian finding over the abstract codebook (paper Figure 4) is a perfect mech-interp target. After Phase 2 we will have a Granite checkpoint whose 64 abstract embeddings can be SAE'd. **Pre-registered followup: do abstract tokens cluster into interpretable concept families?** This plugs straight into [[mech-interp-big-picture]].

### gettier (`_RCDS/gettier/`)
The repair-space-in-the-air reframe is about whether *latent moves* exist in a small LM. Abstract-CoT gives a clean instrument: train an abstract codebook on Gettier-adjacent data and look at which abstract tokens fire on Gettier prompts. Not Tier-1, but a natural Phase-4 probe.

### arc3 (`_RCDS/arc3/`) — adjacent reframe
arc3 reformulates reasoning as intentional-stance narration; A-CoT reformulates it as token-budget-constrained latent. Both are also pre-Phase-0, both bet that the right reformulation buys efficiency. Worth checking arc3's narrator-format conventions before designing our abstract-token nomenclature.

---

## 13. File-level deliverables (initial scaffold — empty until Phase 0)

```
abstract-cot/
├── PLAN.md                           ← this file
├── README.md                         ← quickstart
├── CLAUDE.md                         ← project-level Claude instructions
├── abstract_cot/                     ← Python package
│   ├── __init__.py
│   ├── tokenizer.py                  ← §6.1
│   ├── attention_masks.py            ← §6.2
│   ├── constrained_decoding.py       ← §6.3
│   ├── reward_model.py               ← §6.4
│   ├── bottlenecked_sft.py           ← §6.5
│   ├── self_distillation.py          ← §6.5
│   ├── grpo_trainer.py               ← §6.4
│   └── policy_iteration.py           ← §6.6
├── scripts/
│   ├── prepare_data.py
│   ├── run_warmup.py
│   ├── run_rl.py
│   └── run_eval.py
├── configs/
│   ├── granite_3b.yaml
│   ├── qwen3_4b.yaml
│   ├── qwen3_8b_qlora.yaml
│   └── reward_prompt.txt
├── slurm/
│   ├── submit_warmup.slurm
│   ├── submit_rl.slurm
│   └── submit_eval.slurm
├── eval/
│   ├── math500.py
│   ├── alpaca_eval.py
│   ├── hotpotqa.py
│   ├── aime25.py
│   ├── gpqa_diamond.py
│   └── run_all.py
├── data/                             ← .gitignored (datasets pre-downloaded)
├── notes/
│   ├── paper-v2.pdf
│   └── reward-prompt-extracted.md    ← TODO Phase 0
├── logs/                             ← .gitignored
└── outputs/                          ← .gitignored (checkpoints, reports)
```

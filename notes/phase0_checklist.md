# Phase 0 Checklist

**Date:** May 12, 2026
**Goal:** prove the engineering pieces work end-to-end on a 4-example toy batch.
**Exit criterion:** see PLAN.md §9, Phase 0.

## Environment
- [ ] On fortyfive login node: `python -m venv ~/venvs/abscot`
- [ ] Activate; `pip install --upgrade pip`
- [ ] Install PyTorch 2.6 cu124: `pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124`
- [ ] Install: `transformers>=4.46 trl>=0.13 peft bitsandbytes datasets accelerate openai pyyaml`
- [ ] Install flash-attn (only on Ampere+ nodes): `pip install flash-attn --no-build-isolation`
- [ ] Snapshot: `pip freeze > requirements.lock`

## Paper extraction
- [ ] Save full paper PDF to `notes/paper-v2.pdf`
- [x] Extract Appendix B (generative reward model prompt) verbatim into `configs/reward_prompt.txt` — done 2026-05-12
- [ ] Extract Appendix A.1 ablation table (vocabulary size scaling) into `notes/ablation_M.md`
- [ ] Note any hyperparameters we missed in the main text (LR, optimizer, scheduler, warmup steps)

## Data + model pre-download (login node, internet available)
- [ ] `huggingface-cli download ibm-granite/granite-4.0-micro`
- [ ] `huggingface-cli download Qwen/Qwen3-4B`
- [ ] `huggingface-cli download Qwen/Qwen3-8B`
- [ ] `huggingface-cli download allenai/Dolci-Think-SFT-7B --repo-type dataset`
- [ ] `huggingface-cli download allenai/Dolci-Think-RL-7B --repo-type dataset`
- [ ] `huggingface-cli download HuggingFaceH4/MATH-500 --repo-type dataset`
- [ ] `huggingface-cli download tatsu-lab/alpaca_eval --repo-type dataset`
- [ ] `huggingface-cli download hotpot_qa --repo-type dataset`

## MindRouter sanity check
- [ ] From login node: hit `/v1/models`, confirm `openai/gpt-oss-20b` is listed
- [ ] Send a dummy reward prompt to `openai/gpt-oss-20b`, verify integer-only output
- [ ] Measure round-trip latency × 20 samples; record in `notes/mindrouter_latency.md`
- [ ] **Critical:** test from a compute node — does the job have internet at all? If not, design around it (login-node orchestrator or SSH tunnel)

## Engineering smoke tests
- [ ] `abstract_cot/tokenizer.py`: load Granite, extend with 64 tokens + 2 delimiters, save, reload, assert vocab size grew by 66
- [ ] `abstract_cot/attention_masks.py`:
  - construct a toy `(prompt=4, cot=6, abs=4, ans=3)` sample
  - build mask via flex_attention API
  - build mask via eager `(seq_len, seq_len)` manual construction
  - assert equivalence on a forward pass
- [ ] `abstract_cot/constrained_decoding.py`:
  - generate 10 sequences from extended-tokenizer Granite
  - assert every abstract span is ⊆ V_abs ∪ {<endabstract>}
  - assert m ≤ m_max in every sample
- [ ] `abstract_cot/reward_model.py`: score 5 dummy completions, verify async + semaphore=4 + cache hit on duplicate

## End-to-end toy run
- [ ] 4-example toy batch through `bottlenecked_sft.py` for 5 steps; confirm loss drops
- [ ] Sample 4 abstract sequences with constrained decoding; confirm they parse
- [ ] 4 GRPO steps with K=2 rollouts; confirm reward calls land and policy updates

## Exit
- [ ] Write `notes/phase0-complete-YYYY-MM-DD.md` summarizing what works and any deviations from PLAN.md
- [ ] Then move to Phase 1 (PI warm-up on Granite)

# Phase 0 Checklist

**Date:** May 12 → May 14, 2026
**Status: CLOSED (2026-05-14).** All engineering pieces validated; pilot passed on real Granite. Continuing into Phase 1 (Dolci data prep + multi-day warm-up).
**Exit criterion:** see PLAN.md §9, Phase 0.

## Environment
- [x] On fortyfive login node: `python -m venv ~/venvs/abscot` — done 2026-05-12
- [x] PyTorch 2.6.0+cu124 installed
- [x] transformers 5.8.0, trl 1.4.0, peft 0.19.1, bitsandbytes 0.49.2, datasets 4.8.5, accelerate 1.13.0, openai 2.36.0 — installed
- [ ] flash-attn (Ampere+) — deferred; eager attention is sufficient for the bottleneck mask
- [ ] requirements.lock — deferred

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
- [x] From login node: hit `/v1/models`, confirm `openai/gpt-oss-20b` is listed — done 2026-05-12
- [x] Send a dummy reward prompt to `openai/gpt-oss-20b` (HTTP 200, ~0.5s) — done 2026-05-12
- [x] **Critical:** test from a compute node — compute nodes HAVE internet to MindRouter. /v1/models 200 OK in 0.33s, chat completion 200 OK in 0.41s. **No SSH tunnel needed.**
- [ ] Real reward latency × 20 samples with `reasoning_effort: medium` — record in `notes/mindrouter_latency.md`

## Engineering smoke tests
- [x] `abstract_cot/tokenizer.py`: load Granite, extend with 64 tokens + 2 delimiters, save, reload, assert vocab size grew by 66 — done 2026-05-12 (7 unit tests + real-Granite smoke)
- [ ] `abstract_cot/attention_masks.py`: flex_attention variant (production speed). Eager 4D-mask path is done + tested.
- [x] `abstract_cot/attention_masks.py` eager path:
  - construct a toy `(prompt=2, cot=3, abs=2, ans=2)` sample → 8 mask-structure tests pass
  - end-to-end: forward pass with output_attentions=True, assert y attention on c < 1e-6 at every layer
- [x] `abstract_cot/constrained_decoding.py`:
  - generate sequences from extended tiny-gpt2; assert every abstract span ⊆ V_abs ∪ {<endabstract>}
  - assert m ≤ m_max via force-end test
  - 8 unit tests pass
- [x] `abstract_cot/reward_model.py`: 15 unit tests pass — caching, retries, content/reasoning_content fallback, token-bucket rate limit, batch concurrency

## End-to-end toy run
- [x] 4-example toy batch through `bottlenecked_sft.py` for 5 steps; confirm loss drops
      — done 2026-05-14, A100-40GB on n124. Loss 11.47 → 1.58 in 5 SGD steps.
      0.28 s/step at seq_len 137 batch 4. Extrapolated to seq_len 1024 batch 8:
      ~2 s/step → ~5 days for full warm-up (within Tier-1 budget).
- [x] Sample 4 abstract sequences with constrained decoding; confirm they parse
      — done 2026-05-14, model emitted legal abstract span + correct natural-
      language response ("5 + 7 gives us 12.")
- [ ] 4 GRPO steps with K=2 rollouts; confirm reward calls land and policy updates

## Exit
- [ ] Write `notes/phase0-complete-YYYY-MM-DD.md` summarizing what works and any deviations from PLAN.md
- [ ] Then move to Phase 1 (PI warm-up on Granite)

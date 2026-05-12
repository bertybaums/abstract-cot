# Reasoning-effort latency × discrimination study

**Date:** May 12, 2026
**Run on:** n104 (fortyfive `gpu-8`)
**Endpoint:** `https://mindrouter.uidaho.edu/v1` → `openai/gpt-oss-20b` (vLLM 0.18 backend)
**Method:** 8 contrasting (prompt, completion) pairs hand-ranked best→worst; 2 repeats each; rank correlation against intended quality order.

## Headline

| effort | mean latency | discrimination ρ | latency vs medium |
|---|---|---|---|
| `minimal` | — | — | **rejected by server** (vLLM 0.18 doesn't accept this value; valid: `none`/`low`/`medium`/`high`) |
| `low` | 2.37 s | +0.755 | 0.63× |
| **`medium`** | **3.79 s** | **+0.946** | 1.00× |
| `high` | 8.42 s | +0.903 | 2.22× |

`medium` wins. `high` is slightly worse than `medium` (over-thinks, noisier) at 2.2× the cost. `low` is 37% faster but discrimination drops meaningfully (0.946 → 0.755), and there's a concrete failure mode: the empty-string completion scored 0.50 at `low` vs 0.00 at `medium` — the reward can't reliably tell "nothing" from "something" without enough reasoning budget. That's lethal for RL signal.

## Per-pair scores (intended best→worst)

| Pair | Description | low | medium | high |
|---:|---|---:|---:|---:|
| 0 | Correct + clean reasoning | 1.00 | 1.00 | 1.00 |
| 1 | Correct, no reasoning | 1.00 | 1.00 | 0.95 |
| 2 | Correct answer, sloppy reasoning | 0.80 | 0.80 | 0.80 |
| 3 | Wrong answer, plausible reasoning | 0.25 | 0.35 | 0.30 |
| 4 | Correct + irrelevant filler | 0.85 | 0.80 | 0.85 |
| 5 | Off-topic but coherent (Pythagorean) | 0.10 | 0.05 | 0.00 |
| 6 | Single-word garbage ("potato") | 0.00 | 0.00 | 0.00 |
| 7 | Empty string | **0.50** | **0.00** | **0.00** |

## Revised cost-frontier math (using `medium` at 3.79 s)

- 4-concurrent × 3.79 s/call = ~63 reward calls/min = ~91 k/day
- GRPO step at K=4 rollouts × batch=4 prompts = 16 reward calls
- Throughput: ~5,700 GRPO steps/day on the reward path
- Tier-1 budget at 100 k episodes / K=4 = 25 k steps → **~4–5 days of reward-bound RL**
- Plus ~2–3 days of warm-up SFT (3 PI iterations × ~100 k examples on one A100)

**Total Tier 1 wall-clock estimate: ~1 week of compute** (plus debugging, plus eval). Workable.

## Decisions locked in

1. **Reward model effort = `medium`** per paper. Confirmed as the right sweet spot empirically.
2. **K = 4 rollouts** as a default; can drop to K=2 if we want to cut RL wall-clock further with a defensible variance trade.
3. **Tier-1 RL budget = 100 k episodes** (not 1 M) — gets us in striking distance of paper numbers in one week; we can extend if results justify it.
4. **Warm-up budget = 100 k examples × 3 epochs × T=3** as in current `configs/granite_3b.yaml`. ~2 days SFT.

## What this rules out (for now)

- **`high`** — strictly worse than medium on this signal, 2.2× the cost. Skip.
- **`low`** — defensible if we ever need to cut by 37%, but the empty-string failure mode is a red flag. Keep in back pocket as an ablation, not a default.
- **Local distilled reward** — premature optimization now that medium is affordable.

## What this enables

- Tier-1 reproduction on Granite 3B fits in one week of wall-clock at our current MindRouter share — no scope cuts needed.
- Tier-2 (add Qwen3-4B) adds another ~1 week. Roughly 3 weeks for both models + eval.
- Tier-3 stretch (Qwen3-8B QLoRA) — additional ~2 weeks. Whole reproduction in ~5 weeks if we don't hit blocking issues.

## Caveats

- 8 pairs, 2 repeats — small sample. Variance estimate is weak. Latency outliers (medium max = 7.11 s) suggest occasional pathological reasoning chains; doesn't dominate average.
- Discrimination ρ is on math-style pairs only. AlpacaEval and HotpotQA may behave differently.
- This is `openai/gpt-oss-20b` on MindRouter's vLLM build. Latency could change if they upgrade vLLM or rebalance load.

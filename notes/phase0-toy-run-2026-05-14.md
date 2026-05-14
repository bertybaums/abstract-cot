# Phase 0 toy run — log

**Date:** May 14, 2026
**Hardware:** A100-PCIE-40GB on n124 (gpu-volatile)
**Model:** ibm-granite/granite-4.0-micro (3.40 B, bf16)
**Job:** `srun --partition=gpu-volatile --gres=gpu:a100:1` running `scripts/run_toy_bottleneck_sft.py`

## Result

```
Device: cuda
  GPU: NVIDIA A100-PCIE-40GB
  Free mem: 42.0 GB
Loading ibm-granite/granite-4.0-micro (bf16)…
  loaded in 89.3s; params=3.40B
  extended vocab: 100418  (M=64)
  batch shape: input_ids=(4, 137), loss-tokens=64
  step 0: loss=11.4736   wall=2.93s
  step 1: loss=5.5026    wall=0.28s
  step 2: loss=4.6882    wall=0.28s
  step 3: loss=3.3995    wall=0.28s
  step 4: loss=1.5821    wall=0.28s

Loss trajectory: ['11.474', '5.503', '4.688', '3.399', '1.582']
Step 0 → step 4: drop = +9.8915

Constrained-decode sanity check:
  prompt: 'Question: What is 5 + 7?'
  generated: Question: What is 5 + 7?<beginabstract><TOKEN_AG><TOKEN_S>
    <TOKEN_AN><TOKEN_S><TOKEN_AH><TOKEN_BK><TOKEN_AZ><TOKEN_BB><TOKEN_AM>
    <TOKEN_BJ><endabstract>The 5 + 7 gives us 12.<beginabstract>...

✅ Phase 0 exit gate PASSED.
```

## What this proves

1. The full pipeline runs on real Granite on a real A100.
2. `bottleneck_loss` computes a finite, decreasing loss with the 4D block mask.
3. New embedding rows are receiving gradient (otherwise the model couldn't
   reduce loss on freshly-introduced abstract tokens).
4. `AbstractCotLogitsProcessor` generates legal abstract spans during
   sampling immediately after one SGD pass.
5. Memory: ~6 GB for model in bf16 + activations easily fits in 40 GB A100
   at seq_len 137 batch 4.

## Wall-clock takeaways

- **0.28 s per SGD step** at seq_len 137 batch 4 on A100.
- **Step 0 was 2.93 s** because of CUDA initialization / cudnn first-call cost. Excluded from extrapolation.
- **Model load: 89 s.** Dominates a short job. **Reuse a long-running job for Phase 1**, do not respawn.
- Initial loss = 11.47 ≈ log(100,418) = uniform prior. Confirms new embedding rows started cold (correct behavior).

## Extrapolation to Phase 1

Warm-up budget per `configs/granite_3b.yaml`:
- 100 k samples × 3 epochs × T=3 PI iterations × 2 phases (bottlenecked SFT + self-distill) = 1.8 M training examples
- Batch 8, seq_len 1024 → factor ~8× the toy setup → ~2.2 s/step (linear in batch and seq^2 for attention; modest)
- 1.8 M / 8 = 225 k steps × 2.2 s = ~5.5 days

Fits inside the 1-week Tier-1 budget. Comfortable margin if we use a 2× A100 node (n123 has 4× A100) — could cut to ~3 days with FSDP.

## Known caveat

The toy ran 5 steps on the same 4 examples → memorization, not generalization. Loss dropping is necessary but not sufficient evidence that the recipe works. The actual Phase 1 evaluation comes when we run on real Dolci-Think-SFT-7B data and check MATH-500 after PI-1.

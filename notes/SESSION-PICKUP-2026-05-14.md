# Session pickup — written 2026-05-14, intended for next session

You stopped mid-Phase-1 prep. Here's exactly where things are and what to do next.

## Status

**Phase 0 done.** 53/53 unit tests, all engineering pieces validated on real Granite on A100. Full audit trail in:
- `notes/phase0-toy-run-2026-05-14.md`
- `notes/phase1-pilot-2026-05-14.md`
- `notes/latency-study-2026-05-12.md`

**Phase 1 prep was in flight at session close.** Two things were running on fortyfive in the background:

### 1. Dolci data prep
- Process: `python scripts/prepare_dolci.py --n 100000 --max-seq-length 2048 --out data/dolci_warmup_100k.jsonl --shuffle-buffer 50000 --seed 0`
- Output: `~/abstract-cot/data/dolci_warmup_100k.jsonl`
- Last seen progress: 55k scanned, 19.5k kept (~33% retention), 74/s. Was on track to reach 100k in ~30 more min.
- Log: `/tmp/dolci_prep2.log` (may have rotated/gone)
- **Verify on pickup:** `wc -l data/dolci_warmup_100k.jsonl` — should be 100000 if it finished cleanly, less if it died

### 2. Smoke v4 (SLURM job 5144718)
- Real-data 2k-example smoke on n124 (A100-40GB, gpu-volatile)
- T=1, 3 epochs/phase, batch=2, seq_len=2048, SDPA + gradient_checkpointing
- Last seen at step 190 with loss=1.27 (descending from 2.23 → 0.91 → 0.67 → 1.27 oscillating but trending down)
- **Verify on pickup:**
  - `sacct -j 5144718 -o JobID,State,Elapsed`  → should be `COMPLETED`
  - `ls outputs/warmup_smoke_2k/` → should have a `pi_1/` checkpoint
  - `tail -50 logs/5144718_warmup_smoke_2k.err | tr '\r' '\n' | tail -20` → look for "warm-up complete in X.X h" line and the phase=... first50/last50 summary

## Bugs fixed in this session

Three real bugs caught by running real Dolci data through the pipeline. All committed:

1. **PyYAML parses `2e-5` as string.** Fixed to `2.0e-5` + `float()` cast in run_warmup.py.
2. **`attn_implementation: flex_attention` silently bypasses our 4D additive mask.** Switched to `sdpa` (supports 4D masks and is memory-efficient). flex_attention path still TODO — would need a custom `mask_mod` rather than a tensor.
3. **OOM at seq_len=2048 batch=4 with eager attention.** Fixed by `sdpa` + `batch_size: 2` + `gradient_checkpointing_enable()` + `use_cache=False`.

GitHub repo `bertybaums/abstract-cot` is fully up to date.

## Next step (the actual next push)

**If smoke v4 passed and Dolci prep completed (100k lines):**
```bash
ssh fortyfive.hpc.uidaho.edu
cd ~/abstract-cot
git pull
wc -l data/dolci_warmup_100k.jsonl   # must be 100000
sbatch slurm/submit_warmup.slurm     # launches full Tier-1 warmup, ~6 days
squeue -u bbaum
```

The full warmup will save checkpoints to `outputs/warmup/pi_1/`, `pi_2/`, `pi_3/`.

**If smoke v4 had issues (OOM still, loss explodes, etc.):**
Inspect `logs/5144718_warmup_smoke_2k.err` and decide. Common fixes:
- Lower seq_len in config from 2048 → 1024 (most examples fit, faster steps, fewer rejected)
- Switch to bnb 8-bit AdamW to free ~12 GB of optimizer state
- Reduce dataset size / epochs

**If Dolci prep died early:**
Just rerun: `python scripts/prepare_dolci.py --n 100000 --max-seq-length 2048 --out data/dolci_warmup_100k.jsonl --shuffle-buffer 50000` — it overwrites. The process is reproducible at seed=0.

## After warmup completes

Phase 2 = GRPO RL. Engineering pieces remaining:
1. `abstract_cot/grpo_trainer.py` — TRL `GRPOTrainer` subclass that:
   - Plugs `AbstractCotLogitsProcessor` into the rollout generation
   - Calls `AbstractCotRewardModel.score_batch(...)` for rewards (already built, tested, validated live against MindRouter)
   - Holds a frozen reference policy for KL
2. `scripts/run_rl.py` driver
3. `slurm/submit_rl.slurm` already exists, just needs the entry point wired

Then Phase 3 = eval harness (MATH-500, AlpacaEval, HotpotQA).

## Wall-clock budget (for the rest of Tier-1)

- Warm-up: ~6 days at current config (per smoke 0.6 s/step, 900k steps)
- RL: ~4-5 days reward-bound
- Eval: ~12 hours
- **Total remaining: ~10-12 days of compute**

## Files added this session (all on github)

```
abstract_cot/self_distillation.py          # Phase 1b
abstract_cot/policy_iteration.py           # Algorithm 1 driver
scripts/run_warmup_pilot.py                # 1k synthetic pilot (passed)
scripts/run_warmup.py                      # full warmup entry
scripts/prepare_dolci.py                   # streaming + length-filter
slurm/submit_warmup_smoke_2k.slurm         # smoke
slurm/submit_warmup_smoke.slurm            # 10k smoke template
tests/test_self_distillation.py            # 5 tests
notes/phase1-pilot-2026-05-14.md
notes/latency-study-2026-05-12.md
notes/SESSION-PICKUP-2026-05-14.md         # this file
```

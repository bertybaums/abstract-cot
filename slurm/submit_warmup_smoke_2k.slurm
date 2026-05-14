#!/usr/bin/env bash
#SBATCH --job-name=abscot-smk2k
#SBATCH --partition=gpu-volatile
#SBATCH --nodelist=n124
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%j_warmup_smoke_2k.out
#SBATCH --error=logs/%j_warmup_smoke_2k.err

# 2k-sample real-Dolci smoke. Validates the pipeline on real reasoning
# data (large CoTs, varied prompts) before launching the full warm-up.
# Expected wall-clock: ~10-15 min on A100.

source /etc/profile
set -e
module load python/3.11.11 || true
module load cuda/12.8 || true
source "$HOME/venvs/abscot/bin/activate"
cd "$SLURM_SUBMIT_DIR"

python3 scripts/run_warmup.py \
    --config configs/granite_3b.yaml \
    --data data/dolci_smoke_partial_2k.jsonl \
    --output-dir outputs/warmup_smoke_2k \
    --T-override 1 \
    --n-override 2000

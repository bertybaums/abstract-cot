#!/usr/bin/env bash
#SBATCH --job-name=abscot-smoke
#SBATCH --partition=gpu-volatile
#SBATCH --exclude=n113
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=logs/%j_warmup_smoke.out
#SBATCH --error=logs/%j_warmup_smoke.err
#SBATCH --requeue

# Smoke test: T=1 PI, 10k Dolci examples, 1 epoch. Validates real-data
# pipeline before the multi-day Tier-1 launch. Expected wall-clock: ~2 hours.

source /etc/profile
set -e

module load python/3.11.11 || true
module load cuda/12.8 || true

source "$HOME/venvs/abscot/bin/activate"
cd "$SLURM_SUBMIT_DIR"

python3 scripts/run_warmup.py \
    --config configs/granite_3b.yaml \
    --data data/dolci_warmup_100k.jsonl \
    --output-dir outputs/warmup_smoke_10k \
    --T-override 1 \
    --n-override 10000

echo "Smoke complete. If loss curves are sensible, sbatch submit_warmup.slurm for the full T=3, 100k run."

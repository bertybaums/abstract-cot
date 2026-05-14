#!/usr/bin/env bash
#SBATCH --job-name=abscot-warmup
#SBATCH --partition=cmci-gpu-8
#SBATCH --exclude=n113
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/%j_warmup.out
#SBATCH --error=logs/%j_warmup.err
#SBATCH --requeue

# IMPORTANT: source /etc/profile must come BEFORE set -e
# (sourcing the profile can trigger early exit otherwise)
source /etc/profile
set -e

module load python/3.11.11 || true
module load cuda/12.8 || true

source "$HOME/venvs/abscot/bin/activate"

cd "$SLURM_SUBMIT_DIR"

# Resume from latest checkpoint if present (for --requeue on preemption)
RESUME_FLAG=""
if [ -d "outputs/warmup/latest" ]; then
    RESUME_FLAG="--resume-from outputs/warmup/latest"
fi

python3 scripts/run_warmup.py \
    --config configs/granite_3b.yaml \
    --data data/dolci_warmup_100k.jsonl \
    --output-dir outputs/warmup \
    $RESUME_FLAG

echo "Warmup complete. Submit submit_rl.slurm next."

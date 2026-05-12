#!/usr/bin/env bash
#SBATCH --job-name=abscot-rl
#SBATCH --partition=cmci-gpu-8
#SBATCH --exclude=n113
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/%j_rl.out
#SBATCH --error=logs/%j_rl.err
#SBATCH --requeue

source /etc/profile
set -e

module load python/3.11.11 || true
module load cuda/12.8 || true

source "$HOME/venvs/abscot/bin/activate"

cd "$SLURM_SUBMIT_DIR"

# MindRouter reward model needs internet — compute nodes don't have it directly.
# Phase 0 TODO: verify SSH-tunnel-through-login-node works, OR submit a small
# orchestrator on the login node that calls into this job. Until that's
# resolved, this script may need to be split into a CPU orchestrator + GPU rollout.
export MINDROUTER_API_KEY="${MINDROUTER_API_KEY:?MINDROUTER_API_KEY must be set in your shell env}"

# Reference policy = final warm-up checkpoint
REF_POLICY="outputs/warmup/pi_3_final"
if [ ! -d "$REF_POLICY" ]; then
    echo "ERROR: warm-up checkpoint $REF_POLICY not found. Run submit_warmup.slurm first."
    exit 1
fi

RESUME_FLAG=""
if [ -d "outputs/rl/latest" ]; then
    RESUME_FLAG="--resume-from outputs/rl/latest"
fi

python3 scripts/run_rl.py \
    --config configs/granite_3b.yaml \
    --reference-policy "$REF_POLICY" \
    --output-dir outputs/rl \
    $RESUME_FLAG

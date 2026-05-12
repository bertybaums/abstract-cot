#!/usr/bin/env bash
#SBATCH --job-name=abscot-eval
#SBATCH --partition=gpu-8
#SBATCH --exclude=n113,n118,n121,n122
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%j_eval.out
#SBATCH --error=logs/%j_eval.err

source /etc/profile
set -e

module load python/3.11.11 || true
module load cuda/12.8 || true

source "$HOME/venvs/abscot/bin/activate"

cd "$SLURM_SUBMIT_DIR"

# AlpacaEval judge calls MindRouter — same internet caveat as RL.
export MINDROUTER_API_KEY="${MINDROUTER_API_KEY:?MINDROUTER_API_KEY must be set}"

CHECKPOINT="${CHECKPOINT:-outputs/rl/latest}"

python3 scripts/run_eval.py \
    --checkpoint "$CHECKPOINT" \
    --config configs/granite_3b.yaml \
    --output-dir "outputs/eval/$(basename $CHECKPOINT)"

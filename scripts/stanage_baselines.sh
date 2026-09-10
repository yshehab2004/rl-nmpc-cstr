#!/bin/bash
# The scale: random / constant / perfect-information ceiling. Minutes, not hours.
#   mkdir -p rl/logs && sbatch scripts/stanage_baselines.sh
#SBATCH --job-name=rl-scale
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=rl/logs/%x_%j.out
#SBATCH --error=rl/logs/%x_%j.out
set -uo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
module load Python/3.11.3-GCCcore-12.3.0
# Absolute path, no srun: see scripts/stanage_train.sh for why.
"${HOME}/venvs/rlcstr/bin/python" -m rl.run_baselines

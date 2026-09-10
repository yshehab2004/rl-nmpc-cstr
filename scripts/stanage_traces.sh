#!/bin/bash
# Full campaign traces on S3, all four tiers plus three learned seeds.
#   mkdir -p rl/logs && sbatch scripts/stanage_traces.sh
#
# Six campaigns run SEQUENTIALLY inside run_traces.py, roughly 460 s each,
# so budget ~50 min. One core would do; 8 are requested because casadi and
# the NMPC solve pick up some thread parallelism and the node is not
# contended at this size.
#SBATCH --job-name=traces
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=rl/logs/%x_%j.out
#SBATCH --error=rl/logs/%x_%j.out
set -uo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
module load Python/3.11.3-GCCcore-12.3.0
# Absolute path, no srun: see scripts/stanage_train.sh for why.
"${HOME}/venvs/rlcstr/bin/python" -u -m supervisors.run_traces

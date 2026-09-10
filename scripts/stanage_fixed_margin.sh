#!/bin/bash
# A2: the prudent fixed target at margin 2.0 and 2.5, ten episodes each.
#   mkdir -p rl/logs && sbatch scripts/stanage_fixed_margin.sh
#
# 20 campaigns, one worker each, so all 20 run concurrently on one node and
# the job costs one campaign's wall time rather than twenty.
#
# OMP_NUM_THREADS=1 is not boilerplate here. Run locally without it, each
# worker starts its own BLAS pool, and 10 workers on 10 cores contend badly
# enough to halve throughput (measured: 67 CPU-minutes returned in 14 minutes
# of wall on a 10-core laptop, i.e. 4.7 cores of 10 doing useful work).
#SBATCH --job-name=rl-fixmargin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=rl/logs/%x_%j.out
#SBATCH --error=rl/logs/%x_%j.out
set -uo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
module load Python/3.11.3-GCCcore-12.3.0
# Absolute path, no srun: see scripts/stanage_train.sh for why.
"${HOME}/venvs/rlcstr/bin/python" -u -m rl.run_fixed_margin_baseline

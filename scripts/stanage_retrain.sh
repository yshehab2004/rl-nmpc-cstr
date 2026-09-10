#!/bin/bash
# D-028 retrain: 4 arms x 10 seeds = 40 runs.
#   mkdir -p rl/logs rl/runs && sbatch scripts/stanage_retrain.sh
#
# arm  beta sampling  cooling margin  gamma   answers
#  A1  new (uniform)  observed        0.995   T3 vs T2 -- the thesis
#  A2  new            withheld        0.995   the ablation, first fair run
#  A3  OLD (legacy)   observed        0.995   did coverage cause the change?
#  A4  new            observed        0.99    gamma, controlled
#
# A3 is the point of the design. Without it, "we fixed coverage and it
# improved" cannot be separated from "we retrained and drew luckier seeds".
# gamma changes in A1/A2/A3 alike, so A1-vs-A3 isolates coverage and nothing
# else; A4 is what makes gamma controlled rather than merely disclosed.
#
# Held constant in every arm: weight 330000, 150k steps, 48 workers, SAC,
# learning rate, network, buffer, batch, tau, and the margin ceiling at 5.0
# (D-028 explains why it was NOT raised despite the observed clipping).
#
# MEMORY: 80G, MEASURED, not copied. Every completed 150k run peaks at
# 48-50 GB (sacct MaxRSS: 50358252K, 50520300K, 50195368K, 49041652K...).
# An earlier submission asked for 180G -- inherited from the retired
# branch's job_long.sh without checking -- and all 40 tasks sat PENDING
# with reason (Resources) despite 2777 idle CPUs. The binding constraint
# was never CPUs: a node with 48 free cores but 150 GB free cannot host a
# 180 GB request. An over-request is not free caution; it is a job that
# does not run.
#
# 40 jobs sits under the QOS `normal` cap of 50 running jobs per user.
# Measured: 2.5 h per run, so ~2.5 h wall if they run concurrently.
# --time is 4 h, not 6: measured runs are 2.5 h and a shorter request
# backfills far more easily when the partition is busy.

#SBATCH --job-name=rl-retrain
#SBATCH --array=1-40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=rl/logs/%x_%A_%a.out
#SBATCH --error=rl/logs/%x_%A_%a.out

set -uo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

module load Python/3.11.3-GCCcore-12.3.0

SEEDS=(1 2 3 4 5 6 7 8 9 10)
IDX=$(( SLURM_ARRAY_TASK_ID - 1 ))
ARM=$(( IDX / 10 ))
SEED=${SEEDS[$(( IDX % 10 ))]}

case ${ARM} in
  0) NAME="A1_new_withmargin";   COV="--uniform-in-beta"; MARG="";                     GAMMA=0.995 ;;
  1) NAME="A2_new_nomargin";     COV="--uniform-in-beta"; MARG="--no-cooling-margin";  GAMMA=0.995 ;;
  2) NAME="A3_old_withmargin";   COV="";                  MARG="";                     GAMMA=0.995 ;;
  3) NAME="A4_new_gamma099";     COV="--uniform-in-beta"; MARG="";                     GAMMA=0.99  ;;
esac

echo "=== task ${SLURM_ARRAY_TASK_ID}: arm ${NAME}, seed ${SEED}, gamma ${GAMMA} ==="
echo "=== node $(hostname), ${SLURM_CPUS_PER_TASK} cpus ==="

# Absolute path, no srun -- Sheffield's srun does not inherit the venv.
"${HOME}/venvs/rlcstr/bin/python" -m rl.train \
    --timesteps 150000 \
    --n-envs 48 \
    --seed "${SEED}" \
    --campaign-h 75.0 \
    --n-horizon 20 \
    --algo sac \
    --learner-threads 8 \
    --safety-weight 330000 \
    --gamma "${GAMMA}" \
    --out "rl/runs_d028/${NAME}" \
    ${COV} ${MARG}

echo "=== task ${SLURM_ARRAY_TASK_ID} (${NAME} seed ${SEED}) complete ==="

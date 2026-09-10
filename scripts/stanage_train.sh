#!/bin/bash
# Phase 2 training on Stanage (Sheffield HPC).
#
#   sbatch scripts/stanage_train.sh
#
# Submits a 6-task array: 3 seeds x 2 ablation arms (cooling margin observed
# vs withheld, D-017). Each task gets its own node and trains one policy.
#
# ---------------------------------------------------------------------------
# THE ONE SETTING THAT WILL SILENTLY RUIN THIS RUN IF LEFT OUT
#
# IPOPT (via CasADi) and NumPy both spawn threads on their own initiative. A
# node running 48 worker processes, each helpfully starting 64 BLAS threads,
# oversubscribes the machine ~64x: the cores spend their time context
# switching and the job runs SLOWER than one core, while looking perfectly
# healthy in squeue. The OMP/MKL/OPENBLAS exports below pin every worker to a
# single thread, which is correct here because the parallelism we want is
# across ENVIRONMENTS, not inside one linear solve. Do not remove them.
# ---------------------------------------------------------------------------
#
# BUDGET (measured 2026-09-06, see rl/train.py):
# BUDGET, from the PILOT (job 11471382) rather than from extrapolation:
#
#   measured throughput        14 fps (not the ~48 first projected; the
#                              learner was single-threaded, now fixed via
#                              --learner-threads)
#   reward plateau             ~43k steps. The pilot reached ep_rew_mean
#                              740 at 43k and sat at 744 / 741 through 72k.
#   150k steps                 ~3 h at 14 fps, less once the learner has
#                              threads. 3.5x past the observed plateau.
#
# VIOLATION PRICE: 330000, DERIVED BY SWEEP, not chosen. Four weights were
# trained and evaluated on identical episodes (rl/policy_eval.json):
#
#   weight    profit/h   violation   episodes violating   margin requested
#      250      2472.7      0.1751          7/10                0.54
#    10000      2433.2      0.0810          7/10                0.75
#   100000      2416.2      0.0455          6/10                0.90
#   330000      2375.9      0.0003          1/10                1.27
#
# Violations fall 580x for 3.9% of profit, and the agent buys the safety by
# LEARNING TO REQUEST MARGIN (0.54 -> 1.27 degC), which is the mechanism the
# thesis predicts. The rule is D-011's, applied to the price rather than to
# the margin: take the smallest value that achieves the constraint.
#
# CAVEAT, and it is why a 1e6 arm runs alongside this one: the sweep
# brackets 330000 only from BELOW. If 1e6 also reaches ~zero violations at
# comparable profit then 330000 is not the smallest sufficient weight, it is
# merely the largest that was tried -- which is a different claim.
#
# MEMORY: 180G, not 32G. The pilot was OOM-killed at 32G after 1h51m with
# 72k steps done. 48 workers each hold a do-mpc/IPOPT instance. The retired
# branch's own job_long.sh requested 180G on this same machine -- that
# number is empirical, from runs that completed.

#SBATCH --job-name=rl-cstr
#SBATCH --array=1-6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=06:00:00
# NOTE: rl/logs must EXIST before sbatch is called. SLURM opens these files
# when the job starts, before this script's own mkdir runs, and a job whose
# output file cannot be created dies immediately with nowhere to say so.
#   mkdir -p rl/logs rl/runs && sbatch scripts/stanage_train.sh
#SBATCH --output=rl/logs/%x_%A_%a.out
#SBATCH --error=rl/logs/%x_%A_%a.out

set -euo pipefail

# Single-threaded numerics per worker -- see the block above.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

module load Python/3.11.3-GCCcore-12.3.0 || true
if [[ -d "${HOME}/venvs/rlcstr" ]]; then
    source "${HOME}/venvs/rlcstr/bin/activate"
else
    echo "Expected a virtualenv at ~/venvs/rlcstr. Create it once with:"
    echo "  python -m venv ~/venvs/rlcstr"
    echo "  source ~/venvs/rlcstr/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

mkdir -p rl/logs rl/runs

# Task index -> (seed, ablation arm). Three seeds because a single training
# run is one sample from a stochastic process, and a difference between two
# single runs is not evidence of anything.
SEEDS=(1 3 5)
IDX=$(( SLURM_ARRAY_TASK_ID - 1 ))
SEED=${SEEDS[$(( IDX % 3 ))]}
if (( IDX < 3 )); then
    ARM_FLAG=""
    ARM_NAME="with_margin"
else
    ARM_FLAG="--no-cooling-margin"
    ARM_NAME="no_margin"
fi

echo "=== task ${SLURM_ARRAY_TASK_ID}: seed ${SEED}, arm ${ARM_NAME} ==="
echo "=== node $(hostname), ${SLURM_CPUS_PER_TASK} cpus allocated ==="
# nproc alone would print 1 here: GNU nproc honours OMP_NUM_THREADS, which is
# pinned to 1 above. --all reports the machine; the allocation is what SLURM
# says it is.
echo "cores on node: $(nproc --all), allowed: $(grep -c . <<< "$(taskset -cp $$ 2>/dev/null)")"
"${HOME}/venvs/rlcstr/bin/python" -c "import sys; print('python:', sys.version.split()[0], sys.executable)"

# NO srun, and the interpreter by ABSOLUTE PATH. Sheffield configures SLURM so
# srun does not inherit the batch script's environment: the venv activation is
# discarded, `python` resolves to the system /usr/bin/python (2.7), and the job
# dies on an f-string four seconds in. With --ntasks=1 srun buys nothing anyway
# -- the batch step already holds the whole CPU allocation (verified: its
# Cpus_allowed_list matches the request).
#
# 48 workers on a 64-core node: leaves headroom for the learner process, the
# OS, and SubprocVecEnv's own IPC rather than contending with them.
"${HOME}/venvs/rlcstr/bin/python" -m rl.train \
    --timesteps 150000 \
    --n-envs 48 \
    --seed "${SEED}" \
    --campaign-h 75.0 \
    --n-horizon 20 \
    --algo sac \
    --learner-threads 8 \
    --safety-weight 330000 \
    ${ARM_FLAG}

echo "=== task ${SLURM_ARRAY_TASK_ID} complete ==="

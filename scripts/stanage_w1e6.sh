#!/bin/bash
# Weight sweep: three training runs at three violation prices, one seed each.
#
#   mkdir -p rl/logs rl/runs && sbatch scripts/stanage_weight_sweep.sh
#
# WHY A SWEEP AND NOT A NUMBER. The configured 250 is measurably too small
# -- the pilot reached 101.6% of the constrained ceiling (F-049), which no
# safe policy can do. A break-even of 109,984 was computed from the
# unconstrained-vs-constrained oracle pair, and 3x that was briefly written
# into the config. That was wrong twice over: the break-even comes from ONE
# pair of runs at ONE operating point while the weight applies to every
# episode, and the 3x had nothing behind it. D-013 already established that
# no single weight is principled -- any weight only moves the price at
# which violating becomes rational.
#
# So the weight is DERIVED BY A RULE, the same rule D-011 uses for margin:
# sweep it, and take the smallest value that achieves zero violations. The
# return-vs-violation curve is the evidence, and it is a figure.
#
# Together with the already-measured 250 (pilot 11471382, run 11472255)
# this gives four points across three orders of magnitude.
#
# The proper form is a Lagrangian / constrained-RL formulation: set a
# violation BUDGET and solve for the multiplier, so nobody picks a number
# at all. That is the follow-up; this sweep is tonight's version of it.

#SBATCH --job-name=rl-w1e6

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --output=rl/logs/%x_%j.out
#SBATCH --error=rl/logs/%x_%j.out

set -uo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

module load Python/3.11.3-GCCcore-12.3.0

WEIGHTS=(1000000)
W=${WEIGHTS[0]}

echo "=== BRACKET CHECK: safety weight ${W}, seed 1, with_margin ==="
echo "=== node $(hostname), ${SLURM_CPUS_PER_TASK} cpus ==="

# Absolute path, no srun -- see scripts/stanage_train.sh for why.
"${HOME}/venvs/rlcstr/bin/python" -m rl.train \
    --timesteps 150000 \
    --n-envs 48 \
    --seed 1 \
    --campaign-h 75.0 \
    --n-horizon 20 \
    --algo sac \
    --learner-threads 8 \
    --safety-weight "${W}"

echo "=== bracket check complete ==="

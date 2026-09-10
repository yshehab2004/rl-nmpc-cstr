#!/bin/bash
# Cheap pre-flight job: verify the environment on a COMPUTE node before
# spending a training node on it.
#
#   mkdir -p rl/logs && sbatch scripts/stanage_smoke.sh
#
# Runs on a compute node rather than the login node deliberately. Sheffield
# reaps CPU-using processes on Stanage's login nodes, silently and within
# seconds -- a pytest run there produces an empty log and no error, which
# looks exactly like a hang. Anything that actually computes goes through
# SLURM.
#
# What this checks, in the order that matters:
#   1. the compute node's glibc and CPU count (login and compute nodes are
#      not guaranteed identical, and glibc 2.17 is why numpy/scipy had to
#      be pinned to the last manylinux_2_17 wheels)
#   2. the imports
#   3. the RL test suite
#   4. the GOLDEN TRACE -- whether a campaign is bit-identical here and on
#      the laptop. If it is, Phase 1's numbers transfer and results from
#      both machines can be pooled. If it is not, they cannot, and that is
#      a finding worth having before training rather than after.

#SBATCH --job-name=rl-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=rl/logs/%x_%j.out
#SBATCH --error=rl/logs/%x_%j.out

set -uo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

module load Python/3.11.3-GCCcore-12.3.0
source "${HOME}/venvs/rlcstr/bin/activate"

echo "=== node: $(hostname) ==="
echo "=== cpus: $(nproc) ==="
ldd --version | head -1
python -V

echo "=== imports ==="
python - <<'PY'
import do_mpc, casadi, stable_baselines3, gymnasium, numpy, scipy
print("do-mpc", do_mpc.__version__, "casadi", casadi.__version__,
      "sb3", stable_baselines3.__version__, "gymnasium", gymnasium.__version__,
      "numpy", numpy.__version__, "scipy", scipy.__version__)
PY

echo "=== RL tests ==="
python -m pytest rl/ -q -p no:cacheprovider 2>&1 | tail -5

echo "=== golden trace (does the plant behave identically to the laptop?) ==="
python -m pytest supervisors/test_campaign_golden.py -q -p no:cacheprovider 2>&1 | tail -5

echo "=== single-worker throughput (sets the training budget) ==="
python - <<'PY'
import time
import numpy as np
from rl.env import SupervisoryEnv, RandomAgeing
from rl.train import _training_feed

env = SupervisoryEnv(campaign_h=75.0, decision_interval_h=0.25, n_horizon=20,
                     ageing=RandomAgeing(), feed=_training_feed())
env.reset(seed=1)
env.step(np.zeros(3, dtype=np.float32))          # discard: builds the NMPC
ts = []
for _ in range(10):
    t0 = time.time()
    env.step(np.array([0.1, 0.2, -0.4], dtype=np.float32))
    ts.append(time.time() - t0)
step = float(np.mean(ts))
print(f"step {step:.3f}s | episode(300) {step*300/60:.1f} min | "
      f"200k steps 1 core {step*200000/3600:.0f} h | on 48 workers "
      f"{step*200000/3600/48:.1f} h")
PY

echo "=== smoke complete ==="

# RL-Assisted MPC for Autonomous Economic Operation of a Nonlinear CSTR

Code for the MSc thesis "Reinforcement Learning Assisted Model Predictive
Control for Autonomous Economic Operation of a Nonlinear CSTR"
(Youssef Shehab, University of Sheffield).

## What this is

The plant is the Klatt-Engell jacketed CSTR (Klatt and Engell, 1998):
a continuous stirred tank reactor running the Van de Vusse cyclopentenol
kinetics, with a catalyst that deactivates over the course of a campaign
(the `beta` state in the code). As the catalyst ages, the reactor's
achievable operating point drifts, and something has to decide what to
aim for next.

That "what to aim for" question is answered by a three-layer control
architecture, run every simulated timestep:

```
supervisor   proposes (F_ref, T_R_ref, margin)   OWNS all economic judgement
    |
SSTO         projects that onto the nearest FEASIBLE steady state
    |        (geometric only, no economic opinion of its own)
NMPC         tracks the projected target          no economics, tracking only
    |
plant
```

The supervisor is the only layer that reasons about profit. The
steady-state target optimiser (SSTO) only ever projects a proposed target
onto something the plant can physically reach; the nonlinear MPC (NMPC)
only ever tracks whatever the SSTO hands it. Because every strategy below
writes to the same `(F_ref, T_R_ref, margin)` interface and everything
underneath it is identical, any difference in outcome is attributable to
the supervisory strategy and nothing else.

Four supervisory strategies are compared, interchangeable at that one
interface:

- **T0a**, naive fixed target: pick one setting at the start of a
  campaign and never revisit it.
- **T0b**, prudent fixed target: also fixed, but chosen with a margin
  that survives the whole campaign.
- **T2**, classical real-time optimisation (RTO): re-optimise on a slow
  cadence from a state estimate.
- **T3**, learned Soft Actor-Critic (SAC) policy: a reinforcement
  learning agent trained to choose the target itself.

### Headline result

Across all five evaluation scenarios, the classical layer (T2) beat the
learned layer (T3) by 8.7 to 31.7 per cent. But the spread across 10
independently trained SAC seeds is larger than that gap: on every one of
the five scenarios, at least one trained policy scores below the prudent
fixed target (T0b), a single operating point with no adaptation at all.
The conclusion is not that the learned layer is simply worse than the
classical one. It is that the learned layer is unreliable: which seed you
happen to train can matter more than which supervisory strategy you
chose.

## Directory layout

- `configs/`: `reactor_params.yaml`, the single source of truth for
  every physical, economic and control constant used anywhere in the
  project (see below).
- `models/`: the CSTR ODE model, the catalyst deactivation schedules,
  and the feed/temperature disturbance generators.
- `controllers/`: the nonlinear MPC and the studies run against it
  (horizon sweep, chatter sweep, margin derivation, stress checks).
- `estimators/`: the extended Kalman filter (EKF) used to estimate
  catalyst activity from noisy measurements, plus its validation studies.
- `simulation/`: the plant simulator and the measurement model that sits
  between the plant and the estimator.
- `supervisors/`: the four supervisory tiers (T0a, T0b, T2, T3's
  runtime side), the campaign engine that runs them all through an
  identical schedule, and the comparison/validation/phase-3 experiments.
- `rl/`: the SAC training environment, the training script, the policy
  supervisor that wraps a trained policy for use inside `supervisors/`,
  and the RL-specific evaluation and ablation studies.
- `open_loop_testing/`: open-loop characterisation of the plant
  independent of any supervisor, covering the feasible region, constraint
  margins, sensitivity sweeps, and related studies.
- `evaluation/`: the steady-state oracle and the Phase 0 reproduction of
  the Klatt-Engell paper's reported operating point, used to validate the
  model before anything else is built on it.
- `scripts/`: shell scripts for running experiments on the Stanage HPC
  cluster.
- `utils/`: shared plotting style and `provenance.py`, the module every
  result-producing script uses to write its provenance sidecar (see
  below).

## Install and run

Install dependencies from `requirements.txt`:

```
pip install -r requirements.txt
```

Everything below is run as a module from inside `rl-nmpc-cstr/`, since
the code imports across top-level packages (`from models.deactivation
import ...`, `from supervisors.tiers import ...`, and so on).

Train a SAC policy (T3):

```
python -m rl.train --timesteps 200000 --n-envs 48 --seed 1
```

The `--no-cooling-margin` flag runs the ablation arm that withholds
remaining cooling authority from the observation (see `rl/train.py`).

Run the head-to-head comparison of the three built classical tiers
(T0a, T0b, T2) over the E1-E5 scenario matrix:

```
python -m supervisors.run_comparison
```

Run the Phase 3 comparison, which is the thesis question itself: T3
against T2, gated on whether T3 generalised to held-out scenarios rather
than memorised its training distribution:

```
python -m supervisors.run_phase3
```

Run the tests:

```
pytest
```

## Configuration

`configs/reactor_params.yaml` is the single source of truth for every
constant in the project: kinetic parameters, physical bounds, economic
prices, controller tuning, RL hyperparameters, all of it. Every script
loads it through `models/params.py::load_params()`. Nothing is
hardcoded a second time elsewhere. The file carries its own provenance
comments recording which values were verified against Klatt and Engell
(1998) and which were chosen by this project; those comments are load
bearing and are not summarised here.

## Provenance

Every script that writes a result (a `.json`, `.npz` or `.png`) also
writes a sidecar `<stem>.provenance.json` next to it, via
`utils/provenance.py::save_with_provenance`. The sidecar records:

- the producing script (path relative to the repo root)
- the git commit the result was generated from
- whether the working tree was clean or dirty at the time (a dirty-tree
  result is provisional and must be regenerated from a clean commit
  before it is used in the dissertation)
- a UTC timestamp
- a hash of `configs/reactor_params.yaml` (`config_hash`), so a result
  can be checked against the exact configuration that produced it

This is what makes a figure or a number in the dissertation traceable
back to exactly what generated it, rather than merely dated.

## Trained model checkpoints

Trained SAC policy weights (`rl/runs*/**/model.zip`) are excluded from
this repository. The 51 checkpoints total 159 MB, several times the size
of everything else here, and each one is reproducible from its commit,
seed and `configs/reactor_params.yaml`, which is exactly what its
`.provenance.json` sidecar and `train_meta.json` record. The sidecars
stay in the repo; the weights do not.

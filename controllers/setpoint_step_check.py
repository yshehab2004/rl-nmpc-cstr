import json
import time

import numpy as np

from controllers.nmpc import build_nmpc, run_closed_loop
from models.params import load_params
from simulation.simulator import build_simulator, initial_state

PARAMS = load_params()
FEED = PARAMS["nominal_feed"]
T_STEP = PARAMS["nmpc"]["t_step"]
SUPERVISORY_STEPS = int(round(PARAMS["rl"]["decision_interval_h"] / T_STEP))

RTERM_GRID = [10.0, 100.0, 300.0]

SCENARIOS = [
    {"label": "nominal beta=1.0", "beta": 1.0, "F_from": 18.0, "F_to": 25.0, "T_R_ref": 134.0},
    {"label": "severe beta=0.2", "beta": 0.2, "F_from": 12.0, "F_to": 18.0, "T_R_ref": 140.0},
]

T_STEP_TIME = 0.1


def steps_to_reach(F, target, tol):
    within = np.abs(F - target) <= tol
    if not within[-1]:
        return None
    if within.all():
        return 0
    return int(np.where(~within)[0][-1] + 1)


def run_step_test(scenario, rterm_F, n_steps=140):
    beta = scenario["beta"]
    F_from, F_to, T_R_ref = scenario["F_from"], scenario["F_to"], scenario["T_R_ref"]

    def target(t_now):
        return (F_from if t_now < T_STEP_TIME else F_to, T_R_ref)

    mpc = build_nmpc(target=target, n_horizon=20, beta_belief=beta, rterm_F=rterm_F)
    simulator = build_simulator(ageing_fn=lambda t_now: (1.0, beta))
    run = run_closed_loop(mpc, simulator, initial_state(), n_steps=n_steps)

    F = run["u"][:, 0]
    step_idx = int(round(T_STEP_TIME / T_STEP))
    after = F[step_idx:]

    reached = steps_to_reach(after, F_to, tol=0.2)
    reached_5pct = steps_to_reach(after, F_to, tol=0.05 * F_to)

    return {
        "scenario": scenario["label"], "beta": beta, "rterm_F": rterm_F,
        "F_from": F_from, "F_to": F_to,
        "F_before_step": float(F[step_idx - 1]),
        "F_final": float(F[-1]),
        "steps_to_within_0.2": reached,
        "steps_to_within_5pct": reached_5pct,
        "hours_to_within_0.2": None if reached is None else reached * T_STEP,
        "fraction_of_supervisory_interval": None if reached is None else reached / SUPERVISORY_STEPS,
        "T_R_max": float(run["x"][:, 2].max()),
    }


if __name__ == "__main__":
    from utils.provenance import save_with_provenance

    start_time = time.time()
    print(f"One supervisory interval = {SUPERVISORY_STEPS} NMPC steps "
          f"({PARAMS['rl']['decision_interval_h']}h at t_step={T_STEP}h)\n")
    print(f"{'scenario':>18} {'rterm_F':>8} {'F_final':>8} {'steps->0.2':>11} "
          f"{'hours':>7} {'x interval':>11}")

    results = []
    for scenario in SCENARIOS:
        for rterm_F in RTERM_GRID:
            r = run_step_test(scenario, rterm_F)
            results.append(r)
            reached = r["steps_to_within_0.2"]
            frac = r["fraction_of_supervisory_interval"]
            print(f"{r['scenario']:>18} {rterm_F:8.0f} {r['F_final']:8.3f} "
                  f"{str(reached):>11} "
                  f"{'--' if reached is None else f'{reached * T_STEP:7.3f}':>7} "
                  f"{'--' if frac is None else f'{frac:11.2f}':>11}")

    out_path = "controllers/setpoint_step_check.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    sidecar = save_with_provenance(
        out_path, data=None,
        params={"scenarios": SCENARIOS, "rterm_grid": RTERM_GRID,
                "step_time_h": T_STEP_TIME, "supervisory_steps": SUPERVISORY_STEPS},
        script=__file__, start_time=start_time, extra={"results": results},
    )
    print(f"\nSaved {out_path}, {sidecar}")

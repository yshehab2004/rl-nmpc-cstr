import json
import time

import numpy as np

from controllers.nmpc import build_nmpc, run_closed_loop
from models.deactivation import make_deactivation_fn
from models.params import load_params
from open_loop_testing.target_optimizer import solve_target
from simulation.simulator import build_simulator, initial_state

PARAMS = load_params()
C = PARAMS["certain_params"]
FEED = PARAMS["nominal_feed"]
T_R_LIMIT = PARAMS["state_bounds"]["T_R"]["upper_soft"]

HORIZONS = [10, 20, 30, 40, 50, 60, 80]


def evaluate(n_horizon, beta, n_steps=100, beta_belief=None, disturbance_fn=None,
              rterm_F=None, penalty_term_cons=None):
    ageing_fn = (lambda t_now: (PARAMS["uncertain_params"]["alpha_nominal"], beta))

    projected = solve_target(35.0, 133.0, margin=0.0, beta_hat=beta,
                              C_A0=FEED["C_A0"], alpha=PARAMS["uncertain_params"]["alpha_nominal"],
                              T_in=FEED["T_in"], C=C)
    target = (projected["F_s"], projected["T_R_s"])

    mpc = build_nmpc(target=target, n_horizon=n_horizon,
                      beta_belief=beta if beta_belief is None else beta_belief,
                      rterm_F=rterm_F, penalty_term_cons=penalty_term_cons)
    simulator = build_simulator(ageing_fn=ageing_fn, disturbance_fn=disturbance_fn)

    t0 = time.time()
    run = run_closed_loop(mpc, simulator, initial_state(), n_steps=n_steps)
    elapsed = time.time() - t0

    T_R = run["x"][:, 2]
    F = run["u"][:, 0]
    settled = T_R[len(T_R) // 2:]

    within = np.abs(T_R - target[1]) <= 0.5
    if within.all():
        steps_to_settle = 0
    elif not within[-1]:
        steps_to_settle = None
    else:
        steps_to_settle = int(np.where(~within)[0][-1] + 1)

    return {
        "n_horizon": n_horizon,
        "beta": beta,
        "rterm_F": rterm_F,
        "penalty_term_cons": penalty_term_cons,
        "target_F": target[0],
        "target_T_R": target[1],
        "T_R_final": float(T_R[-1]),
        "T_R_offset": float(abs(T_R[-1] - target[1])),
        "T_R_rms_offset_settled": float(np.sqrt(np.mean((settled - target[1]) ** 2))),
        "T_R_max": float(T_R.max()),
        "T_R_violation": float(max(0.0, T_R.max() - T_R_LIMIT)),
        "F_total_variation": float(np.abs(np.diff(F)).sum()),
        "F_offset_final": float(abs(F[-1] - target[0])),
        "steps_to_settle": steps_to_settle,
        "solve_time_s": elapsed,
        "solve_time_per_step_ms": 1000.0 * elapsed / n_steps,
    }


if __name__ == "__main__":
    from utils.provenance import save_with_provenance

    start_time = time.time()
    severity = PARAMS["uncertain_params"]
    betas = {"mild": 1.0, "severe": 0.2}

    from models.disturbances import build_feed_disturbance, step_disturbance

    disturbed = build_feed_disturbance(
        C_A0_gen=step_disturbance(base=FEED["C_A0"], magnitude=0.6, t_start=0.1)
    )
    scenarios = [(label, beta, None) for label, beta in betas.items()]
    scenarios.append(("severe+C_A0 step", 0.2, disturbed))

    results = []
    print(f"{'scenario':>18} {'N':>4} {'T_R off':>9} {'RMS off':>9} "
          f"{'viol':>7} {'F TV':>9} {'ms/step':>9}")
    for label, beta, dist in scenarios:
        for n_horizon in HORIZONS:
            r = evaluate(n_horizon, beta, disturbance_fn=dist)
            r["severity"] = label
            r["disturbed"] = dist is not None
            results.append(r)
            print(f"{label:>18} {n_horizon:4d} "
                  f"{r['T_R_offset']:9.4f} {r['T_R_rms_offset_settled']:9.4f} "
                  f"{r['T_R_violation']:7.4f} {r['F_total_variation']:9.3f} "
                  f"{r['solve_time_per_step_ms']:9.1f}")

    out_path = "controllers/horizon_sweep.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    sidecar = save_with_provenance(
        out_path, data=None,
        params={"horizons": HORIZONS, "betas": betas, "n_steps": 100,
                "t_step": PARAMS["nmpc"]["t_step"]},
        script=__file__, start_time=start_time, extra={"results": results},
    )
    print(f"Saved {out_path}, {sidecar}")

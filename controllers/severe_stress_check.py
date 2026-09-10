import json
import time

import numpy as np

from controllers.nmpc import build_nmpc
from models.disturbances import build_feed_disturbance, step_disturbance
from models.params import load_params
from open_loop_testing.target_optimizer import solve_target
from simulation.simulator import build_simulator, initial_state

PARAMS = load_params()
C = PARAMS["certain_params"]
FEED = PARAMS["nominal_feed"]
BX = PARAMS["state_bounds"]
BETA = 0.2
N_STEPS = 100


def run_instrumented(collocation_deg=None):
    projected = solve_target(35.0, 133.0, margin=0.0, beta_hat=BETA,
                              C_A0=FEED["C_A0"], alpha=1.0, T_in=FEED["T_in"], C=C)
    disturbed = build_feed_disturbance(
        C_A0_gen=step_disturbance(base=FEED["C_A0"], magnitude=0.6, t_start=0.1)
    )
    mpc = build_nmpc(target=(projected["F_s"], projected["T_R_s"]),
                      n_horizon=20, beta_belief=BETA, collocation_deg=collocation_deg)
    simulator = build_simulator(ageing_fn=lambda t_now: (1.0, BETA),
                                 disturbance_fn=disturbed)

    x0 = initial_state()
    mpc.x0 = x0
    simulator.x0 = x0
    mpc.set_initial_guess()

    xs, us, ok = [x0.flatten()], [], []
    x = x0
    for _ in range(N_STEPS):
        u = mpc.make_step(x)
        stats = getattr(mpc, "solver_stats", {}) or {}
        ok.append(bool(stats.get("success", True)))
        x = simulator.make_step(u)
        us.append(np.asarray(u).flatten())
        xs.append(np.asarray(x).flatten())

    return np.array(xs), np.array(us), np.array(ok), projected


def chatter_stats(signal):
    d = np.diff(signal)
    post = d[22:]
    flips = int(np.sum(np.sign(post[1:]) * np.sign(post[:-1]) < 0))
    return {
        "total_variation": float(np.abs(d).sum()),
        "sign_changes_post_step": flips,
        "sign_change_fraction": float(flips / max(len(post) - 1, 1)),
        "max_abs_delta": float(np.abs(d).max()),
    }


if __name__ == "__main__":
    from utils.provenance import save_with_provenance

    start_time = time.time()
    x, u, ok, target = run_instrumented()
    C_a, C_b, T_R, T_K = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    F, Q_dot = u[:, 0], u[:, 1]

    report = {"target": {"F": target["F_s"], "T_R": target["T_R_s"]}}

    report["rterm_Q_dot"] = chatter_stats(Q_dot)
    report["rterm_F"] = chatter_stats(F)

    report["penalty_term_cons"] = {
        "T_R_max": float(T_R.max()),
        "violation": float(max(0.0, T_R.max() - BX["T_R"]["upper_soft"])),
        "steps_above_limit": int(np.sum(T_R > BX["T_R"]["upper_soft"])),
        "constraint_is_active": bool(T_R.max() > BX["T_R"]["upper_soft"] - 0.5),
    }

    report["state_guards"] = {
        "C_a_margin_to_lower": float(C_a.min() - BX["C_a"]["lower"]),
        "C_a_margin_to_upper": float(BX["C_a"]["upper"] - C_a.max()),
        "C_b_margin_to_lower": float(C_b.min() - BX["C_b"]["lower"]),
        "C_b_margin_to_upper": float(BX["C_b"]["upper"] - C_b.max()),
        "T_K_margin_to_lower": float(T_K.min() - BX["T_K"]["lower"]),
        "T_K_max": float(T_K.max()),
        "T_K_would_breach_unimposed_upper": bool(T_K.max() > BX["T_K"]["upper"]),
    }

    report["solver"] = {
        "steps": int(len(ok)),
        "successes": int(ok.sum()),
        "all_converged": bool(ok.all()),
    }

    x3, u3, _, _ = run_instrumented(collocation_deg=3)
    report["collocation_deg"] = {
        "max_abs_diff_T_R": float(np.abs(x3[:, 2] - T_R).max()),
        "max_abs_diff_F": float(np.abs(u3[:, 0] - F).max()),
        "max_abs_diff_C_b": float(np.abs(x3[:, 1] - C_b).max()),
    }

    print(f"target: F={target['F_s']:.3f}, T_R={target['T_R_s']:.1f}   (beta={BETA}, C_A0 step at 0.1h)\n")
    for section, vals in report.items():
        if section == "target":
            continue
        print(f"{section}:")
        for k, v in vals.items():
            print(f"    {k:34} {v}")
        print()

    out_path = "controllers/severe_stress_check.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    sidecar = save_with_provenance(
        out_path, data=None,
        params={"beta": BETA, "n_steps": N_STEPS, "n_horizon": 20,
                "C_A0_step": {"magnitude": 0.6, "t_start": 0.1},
                "config_under_test": PARAMS["nmpc"]},
        script=__file__, start_time=start_time, extra={"report": report},
    )
    print(f"Saved {out_path}, {sidecar}")

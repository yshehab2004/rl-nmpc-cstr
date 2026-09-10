import numpy as np
from scipy.integrate import solve_ivp

from open_loop_testing.linearization import ode_rhs


def time_to_settle(t, x, x_final, tol, initial_value):
    band = tol * abs(x_final - initial_value)
    outside = np.abs(x - x_final) > band
    if not np.any(outside):
        return 0.0
    last_outside_idx = np.where(outside)[0][-1]
    if last_outside_idx == len(t) - 1:
        return float(t[-1])
    return float(t[last_outside_idx + 1])


def settling_time_for_step(F0, T_R0, beta, du, alpha, C_A0, T_in, C, t_max, tol=0.02, n_eval=20000):
    from open_loop_testing.steady_state_map import solve_steady_state
    C_a0, C_b0, T_K0, Q_dot0 = solve_steady_state(F0, T_R0, beta, alpha, C_A0, T_in, C)
    x0 = np.array([C_a0, C_b0, T_R0, T_K0])
    u = np.array([F0, Q_dot0]) + du
    kw = dict(alpha=alpha, beta=beta, C_A0=C_A0, T_in=T_in, C=C)

    t_eval = np.linspace(0.0, t_max, n_eval)
    sol = solve_ivp(lambda t, x: ode_rhs(x, u, **kw), (0.0, t_max), x0,
                      t_eval=t_eval, method="RK45", rtol=1e-9, atol=1e-9)

    x_final = sol.y[:, -1]
    return {
        "C_b": time_to_settle(sol.t, sol.y[1], x_final[1], tol, x0[1]),
        "T_R": time_to_settle(sol.t, sol.y[2], x_final[2], tol, x0[2]),
    }


if __name__ == "__main__":
    import json
    import time as timemod

    from models.params import load_params
    from utils.provenance import save_with_provenance

    start_time = timemod.time()
    PARAMS = load_params()
    C = PARAMS["certain_params"]
    feed = PARAMS["nominal_feed"]
    kw = dict(alpha=PARAMS["uncertain_params"]["alpha_nominal"], C_A0=feed["C_A0"], T_in=feed["T_in"], C=C)

    with open("open_loop_testing/optimum_trajectory.json") as f:
        table = json.load(f)

    step_F = np.array([1.0, 0.0])
    step_Q = np.array([0.0, -500.0])
    t_max = 5.0

    print(f"{'beta':>6} {'F':>6} {'T_R':>7} {'t_settle(F step) Cb/TR':>24} {'t_settle(Qdot step) Cb/TR':>26}")
    results = []
    for row in table:
        F0, T_R0, beta = row["F_star"], row["T_R_star"], row["beta"]
        r_F = settling_time_for_step(F0, T_R0, beta, step_F, t_max=t_max, **kw)
        r_Q = settling_time_for_step(F0, T_R0, beta, step_Q, t_max=t_max, **kw)
        results.append({"beta": beta, "F": F0, "T_R": T_R0,
                         "step_F_settle": r_F, "step_Q_settle": r_Q})
        print(f"{beta:6.3f} {F0:6.2f} {T_R0:7.2f} "
              f"{r_F['C_b']:10.3f} / {r_F['T_R']:8.3f}     "
              f"{r_Q['C_b']:10.3f} / {r_Q['T_R']:8.3f}")

    out_path = "open_loop_testing/settling_time.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    sidecar = save_with_provenance(
        out_path, data=None,
        params={"step_F": step_F.tolist(), "step_Q": step_Q.tolist(), "t_max": t_max, "tol": 0.02},
        script=__file__, start_time=start_time, extra={"results": results},
    )
    print(f"Saved {out_path}")
    print(f"Saved {sidecar}")

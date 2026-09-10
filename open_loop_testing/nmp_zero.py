import numpy as np

from open_loop_testing.linearization import ode_rhs, linearize

_INF_THRESHOLD = 1e6


def transmission_zeros(A, B_col, C_row):
    n = A.shape[0]
    M0 = np.zeros((n + 1, n + 1))
    M0[:n, :n] = A
    M0[:n, n] = B_col
    M0[n, :n] = C_row

    M1 = np.zeros((n + 1, n + 1))
    M1[:n, :n] = np.eye(n)

    from scipy.linalg import eig
    eigvals = eig(M0, M1, right=False)
    return [z for z in eigvals if np.isfinite(z) and abs(z) < _INF_THRESHOLD]


def F_to_Cb_zeros(F, T_R, beta, alpha, C_A0, T_in, C):
    from open_loop_testing.steady_state_map import solve_steady_state
    C_a, C_b, T_K, Q_dot = solve_steady_state(F, T_R, beta, alpha, C_A0, T_in, C)
    A, B = linearize(np.array([C_a, C_b, T_R, T_K]), np.array([F, Q_dot]), alpha, beta, C_A0, T_in, C)
    C_row = np.array([0.0, 1.0, 0.0, 0.0])
    return transmission_zeros(A, B[:, 0], C_row)


def simulate_step_response(x0, u0, du, alpha, beta, C_A0, T_in, C, dt, n_steps):
    kw = dict(alpha=alpha, beta=beta, C_A0=C_A0, T_in=T_in, C=C)
    u = u0 + du
    x = x0.copy()
    traj = np.zeros((n_steps + 1, len(x0)))
    traj[0] = x
    for i in range(n_steps):
        k1 = ode_rhs(x, u, **kw)
        k2 = ode_rhs(x + 0.5 * dt * k1, u, **kw)
        k3 = ode_rhs(x + 0.5 * dt * k2, u, **kw)
        k4 = ode_rhs(x + dt * k3, u, **kw)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj[i + 1] = x
    t = np.arange(n_steps + 1) * dt
    return t, traj


if __name__ == "__main__":
    import json
    import time
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from models.params import load_params
    from open_loop_testing.steady_state_map import solve_steady_state
    from utils.provenance import save_with_provenance

    start_time = time.time()
    PARAMS = load_params()
    C = PARAMS["certain_params"]
    feed = PARAMS["nominal_feed"]
    kw = dict(alpha=PARAMS["uncertain_params"]["alpha_nominal"], C_A0=feed["C_A0"], T_in=feed["T_in"], C=C)

    d = np.load("open_loop_testing/steady_state_map.npz")
    F_grid, T_R_grid, betas = d["F"][::6], d["T_R"][::6], d["beta"]

    print("RHP zero (max real part among finite zeros) across a subsampled grid, by beta:")
    results = []
    for beta in betas:
        rhp_count, total = 0, 0
        max_re = -np.inf
        for F in F_grid:
            for T_R in T_R_grid:
                zeros = F_to_Cb_zeros(F, T_R, beta, **kw)
                total += 1
                if any(z.real > 0 for z in zeros):
                    rhp_count += 1
                if zeros:
                    max_re = max(max_re, max(z.real for z in zeros))
        results.append({"beta": float(beta), "rhp_fraction": rhp_count / total, "max_zero_real_part": max_re})
        print(f"  beta={beta:.2f}: RHP zero present at {rhp_count}/{total} nodes "
              f"({100*rhp_count/total:.1f}%), max Re(zero)={max_re:.3f}")

    F0, T_R0, beta0 = 18.83, 134.14, 1.0
    C_a0, C_b0, T_K0, Q_dot0 = solve_steady_state(F0, T_R0, beta0, **kw)
    x0 = np.array([C_a0, C_b0, T_R0, T_K0])
    u0 = np.array([F0, Q_dot0])
    t, traj = simulate_step_response(x0, u0, du=np.array([2.0, 0.0]), beta=beta0, dt=1e-4, n_steps=3000, **kw)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(t, traj[:, 1], label="C_b(t)")
    ax.axhline(C_b0, color="gray", ls=":", lw=1, label="initial C_b")
    _, C_b_new, _, _ = solve_steady_state(F0 + 2.0, T_R0, beta0, **kw)
    ax.axhline(C_b_new, color="green", ls="--", lw=1, label="new steady state")
    ax.set_xlabel("time (h)")
    ax.set_ylabel("C_b (mol/L)")
    ax.set_title(f"Step response: F {F0} -> {F0+2.0} h$^{{-1}}$ at t=0 (beta={beta0})")
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_path = "open_loop_testing/nmp_zero_step_response.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")

    json_path = "open_loop_testing/nmp_zero.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    sidecar = save_with_provenance(
        [json_path, out_path], data=None,
        params={"betas": betas.tolist(), "F_grid_subsample": F_grid.tolist(), "T_R_grid_subsample": T_R_grid.tolist()},
        script=__file__, start_time=start_time, extra={"results": results},
    )
    print(f"Saved {json_path}, {sidecar}")

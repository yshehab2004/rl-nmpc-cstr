import numpy as np

from open_loop_testing.steady_state_map import solve_steady_state
from open_loop_testing.linearization import linearize


def max_real_eigenvalue(F, T_R, beta, alpha, C_A0, T_in, C):
    C_a, C_b, T_K, Q_dot = solve_steady_state(F, T_R, beta, alpha, C_A0, T_in, C)
    A, _ = linearize(np.array([C_a, C_b, T_R, T_K]), np.array([F, Q_dot]), alpha, beta, C_A0, T_in, C)
    return float(np.max(np.linalg.eigvals(A).real))


if __name__ == "__main__":
    import time
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from models.params import load_params
    from open_loop_testing.feasible_region import classify, FEASIBLE
    from utils.provenance import save_with_provenance

    start_time = time.time()
    PARAMS = load_params()
    C = PARAMS["certain_params"]
    feed = PARAMS["nominal_feed"]
    kw = dict(alpha=PARAMS["uncertain_params"]["alpha_nominal"], C_A0=feed["C_A0"], T_in=feed["T_in"], C=C)

    d = np.load("open_loop_testing/steady_state_map.npz")
    F_grid, T_R_grid, betas, Q_dot_map = d["F"], d["T_R"], d["beta"], d["Q_dot"]

    fig, axes = plt.subplots(1, len(betas), figsize=(3.4 * len(betas), 4.0), sharey=True)
    n_unstable_feasible = {}
    for i, (ax, beta) in enumerate(zip(axes, betas)):
        max_eig = np.array([[max_real_eigenvalue(F, T_R, beta, **kw) for T_R in T_R_grid] for F in F_grid])
        T_R_mesh = np.broadcast_to(T_R_grid[None, :], max_eig.shape)
        feasible = classify(Q_dot_map[i], T_R_mesh) == FEASIBLE
        n_unstable_feasible[float(beta)] = int(np.sum((max_eig > 0) & feasible))

        vmax = np.max(np.abs(max_eig))
        cf = ax.pcolormesh(F_grid, T_R_grid, max_eig.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="nearest")
        ax.contour(F_grid, T_R_grid, (feasible.T).astype(float), levels=[0.5], colors="lime", linewidths=1.3)
        ax.contour(F_grid, T_R_grid, max_eig.T, levels=[0.0], colors="black", linewidths=1.5)
        ax.set_title(f"beta = {beta:.2f}")
        ax.set_xlabel("F (h$^{-1}$)")
        if i == 0:
            ax.set_ylabel("T_R (degC)")
    fig.colorbar(cf, ax=axes, label="max Re(eigenvalue) of state Jacobian (red = unstable)")
    fig.suptitle("Open-loop stability across the grid (lime = feasible region, black = stability boundary)")

    out_path = "open_loop_testing/open_loop_stability.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")
    print("Unstable-AND-feasible node counts by beta:", n_unstable_feasible)

    sidecar = save_with_provenance(
        out_path, data=None, params={"betas": betas.tolist()}, script=__file__, start_time=start_time,
        extra={"n_unstable_feasible_nodes_by_beta": n_unstable_feasible},
    )
    print(f"Saved {sidecar}")

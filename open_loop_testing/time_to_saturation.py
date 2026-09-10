import numpy as np

from open_loop_testing.steady_state_map import solve_steady_state


def critical_beta(F, T_R, alpha, C_A0, T_in, C, q_lower=-8500.0,
                    beta_lo=0.05, beta_hi=1.0, maxit=60):
    F = np.asarray(F, dtype=float)
    T_R = np.asarray(T_R, dtype=float)
    shape = np.broadcast_shapes(F.shape, T_R.shape)
    lo = np.full(shape, beta_lo)
    hi = np.full(shape, beta_hi)

    for _ in range(maxit):
        mid = 0.5 * (lo + hi)
        _, _, _, Q_dot = solve_steady_state(F, T_R, mid, alpha, C_A0, T_in, C)
        slack = Q_dot - q_lower
        lo = np.where(slack < 0, mid, lo)
        hi = np.where(slack < 0, hi, mid)

    result = 0.5 * (lo + hi)
    return float(result) if result.ndim == 0 else result


if __name__ == "__main__":
    import time
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from models.params import load_params
    from utils.provenance import save_with_provenance

    start_time = time.time()
    PARAMS = load_params()
    C = PARAMS["certain_params"]
    feed = PARAMS["nominal_feed"]
    kw = dict(alpha=PARAMS["uncertain_params"]["alpha_nominal"], C_A0=feed["C_A0"], T_in=feed["T_in"], C=C)

    d = np.load("open_loop_testing/steady_state_map.npz")
    F_grid, T_R_grid = d["F"], d["T_R"]
    F_mesh, T_R_mesh = np.meshgrid(F_grid, T_R_grid, indexing="ij")

    beta_crit = critical_beta(F_mesh, T_R_mesh, **kw)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    levels = np.linspace(0.05, 1.0, 20)
    cf = ax.contourf(F_grid, T_R_grid, beta_crit.T, levels=levels, cmap="viridis_r")
    ax.contour(F_grid, T_R_grid, beta_crit.T, levels=[0.385], colors="red", linewidths=1.5)
    fig.colorbar(cf, ax=ax, label="critical beta (saturates below this)")
    ax.axhline(140.0, color="black", ls="--", lw=1, label="T_R = 140 limit")
    ax.set_xlabel("F (h$^{-1}$)")
    ax.set_ylabel("T_R (degC)")
    ax.set_title("Beta-to-saturation: ageing survived before Q_dot hits -8500\n"
                 "(red contour: beta=0.385, the F=35 switch beta from F-003)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()

    out_path = "open_loop_testing/time_to_saturation.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")
    print(f"beta_crit range: {np.nanmin(beta_crit):.3f} - {np.nanmax(beta_crit):.3f}")

    sidecar = save_with_provenance(
        out_path, data=None,
        params={"q_lower": -8500.0, "beta_lo": 0.05, "beta_hi": 1.0,
                "source_map": "open_loop_testing/steady_state_map.npz"},
        script=__file__, start_time=start_time,
    )
    print(f"Saved {sidecar}")

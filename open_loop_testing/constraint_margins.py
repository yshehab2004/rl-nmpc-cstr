import numpy as np


def compute_margins(Q_dot, T_R, q_lower=-8500.0, T_R_max=140.0):
    Q_dot = np.asarray(Q_dot, dtype=float)
    T_R = np.asarray(T_R, dtype=float)
    return {
        "Q_dot_margin": Q_dot - q_lower,
        "T_R_margin": T_R_max - T_R,
    }


if __name__ == "__main__":
    import time
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from utils.provenance import save_with_provenance

    start_time = time.time()
    d = np.load("open_loop_testing/steady_state_map.npz")
    F, T_R_grid, betas, Q_dot = d["F"], d["T_R"], d["beta"], d["Q_dot"]
    T_R_mesh = np.broadcast_to(T_R_grid[None, :], (len(F), len(T_R_grid)))

    fig, axes = plt.subplots(1, len(betas), figsize=(3.2 * len(betas), 3.8), sharey=True)
    vmax = 20000.0
    for i, (ax, beta) in enumerate(zip(axes, betas)):
        margins = compute_margins(Q_dot[i], T_R_mesh)
        cf = ax.pcolormesh(F, T_R_grid, margins["Q_dot_margin"].T, cmap="RdBu",
                            vmin=-vmax, vmax=vmax, shading="nearest")
        ax.contour(F, T_R_grid, margins["Q_dot_margin"].T, levels=[0.0], colors="black", linewidths=1.5)
        ax.axhline(140.0, color="green", ls="--", lw=1)
        ax.set_title(f"beta = {beta:.2f}")
        ax.set_xlabel("F (h$^{-1}$)")
        if i == 0:
            ax.set_ylabel("T_R (degC)")
    fig.colorbar(cf, ax=axes, label="Q_dot margin (kJ/h); 0-line = cooling floor")
    fig.suptitle("Constraint-margin fields: Q_dot margin (colour) and T_R=140 limit (green dashed)")

    out_path = "open_loop_testing/constraint_margin_fields.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")

    sidecar = save_with_provenance(
        out_path, data=None,
        params={"q_lower": -8500.0, "T_R_max": 140.0, "source_map": "open_loop_testing/steady_state_map.npz"},
        script=__file__, start_time=start_time,
    )
    print(f"Saved {sidecar}")

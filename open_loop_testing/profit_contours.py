import numpy as np

from open_loop_testing.steady_state_map import solve_steady_state
from open_loop_testing.optimum_trajectory import compute_profit, find_optimum
from open_loop_testing.feasible_region import classify, FEASIBLE


def find_unconstrained_optimum(F_grid, T_R_grid, beta, alpha, C_A0, T_in, C,
                                 price_B, price_A0, price_energy):
    F_grid = np.asarray(F_grid, dtype=float)
    T_R_grid = np.asarray(T_R_grid, dtype=float)
    F_mesh, T_R_mesh = np.meshgrid(F_grid, T_R_grid, indexing="ij")
    _, C_b, _, Q_dot = solve_steady_state(F_mesh, T_R_mesh, beta, alpha, C_A0, T_in, C)
    profit = compute_profit(F_mesh, C_b, C_A0, Q_dot, price_B, price_A0, price_energy)
    i_star = np.unravel_index(np.argmax(profit), profit.shape)
    return {"F_star": float(F_mesh[i_star]), "T_R_star": float(T_R_mesh[i_star]),
            "profit_star": float(profit[i_star])}


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
    econ = PARAMS["economics"]
    kw = dict(alpha=PARAMS["uncertain_params"]["alpha_nominal"], C_A0=feed["C_A0"], T_in=feed["T_in"], C=C)
    prices = dict(price_B=econ["price_B"], price_A0=econ["price_A0"], price_energy=econ["price_energy"])

    d = np.load("open_loop_testing/steady_state_map.npz")
    F_grid, T_R_grid, betas = d["F"], d["T_R"], d["beta"]

    fig, axes = plt.subplots(1, len(betas), figsize=(3.4 * len(betas), 4.0), sharey=True)
    F_mesh, T_R_mesh = np.meshgrid(F_grid, T_R_grid, indexing="ij")
    summary = []
    for i, (ax, beta) in enumerate(zip(axes, betas)):
        _, C_b, _, Q_dot = solve_steady_state(F_mesh, T_R_mesh, beta, **kw)
        profit = compute_profit(F_mesh, C_b, kw["C_A0"], Q_dot, **prices)
        code = classify(Q_dot, T_R_mesh)

        cf = ax.contourf(F_grid, T_R_grid, profit.T, levels=25, cmap="plasma")
        ax.contour(F_grid, T_R_grid, (code.T == FEASIBLE).astype(float), levels=[0.5],
                   colors="cyan", linewidths=1.3)

        unc = find_unconstrained_optimum(F_grid, T_R_grid, beta, **kw, **prices)
        con = find_optimum(F_grid, T_R_grid, beta, **kw, **prices)
        ax.scatter([unc["F_star"]], [unc["T_R_star"]], marker="x", s=110, color="white",
                   linewidths=3, label="unconstrained argmax", clip_on=False, zorder=6)
        ax.scatter([con["F_star"]], [con["T_R_star"]], marker="o", s=70, color="lime",
                   edgecolors="black", label="constrained argmax", clip_on=False, zorder=6)
        summary.append({"beta": float(beta), "unconstrained": unc, "constrained": con})

        ax.set_title(f"beta = {beta:.2f}")
        ax.set_xlabel("F (h$^{-1}$)")
        if i == 0:
            ax.set_ylabel("T_R (degC)")
    axes[-1].legend(fontsize=7, loc="lower left")
    fig.colorbar(cf, ax=axes, label="profit")
    fig.suptitle("Profit contours (cyan = feasible boundary), unconstrained vs. constrained argmax")

    out_path = "open_loop_testing/profit_contours.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")
    for row in summary:
        print(f"beta={row['beta']:.2f}  unconstrained=({row['unconstrained']['F_star']:.2f},"
              f"{row['unconstrained']['T_R_star']:.2f},pi={row['unconstrained']['profit_star']:.1f})  "
              f"constrained=({row['constrained']['F_star']:.2f},{row['constrained']['T_R_star']:.2f},"
              f"pi={row['constrained']['profit_star']:.1f})")

    sidecar = save_with_provenance(
        out_path, data=None,
        params={"betas": betas.tolist(), **prices}, script=__file__, start_time=start_time,
        extra={"summary": summary},
    )
    print(f"Saved {sidecar}")

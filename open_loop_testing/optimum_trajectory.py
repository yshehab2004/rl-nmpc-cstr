import numpy as np

from open_loop_testing.steady_state_map import solve_steady_state
from open_loop_testing.feasible_region import classify, FEASIBLE


def compute_profit(F, C_b, C_A0, Q_dot, price_B, price_A0, price_energy):
    F = np.asarray(F, dtype=float)
    C_b = np.asarray(C_b, dtype=float)
    Q_dot = np.asarray(Q_dot, dtype=float)
    return price_B * F * C_b - price_A0 * F * C_A0 - price_energy * (-Q_dot)


def find_optimum(F_grid, T_R_grid, beta, alpha, C_A0, T_in, C,
                  price_B, price_A0, price_energy,
                  q_lower=-8500.0, q_upper=0.0, T_R_max=140.0):
    F_grid = np.asarray(F_grid, dtype=float)
    T_R_grid = np.asarray(T_R_grid, dtype=float)
    F_mesh, T_R_mesh = np.meshgrid(F_grid, T_R_grid, indexing="ij")

    C_a, C_b, T_K, Q_dot = solve_steady_state(F_mesh, T_R_mesh, beta, alpha, C_A0, T_in, C)
    code = classify(Q_dot, T_R_mesh, q_lower, q_upper, T_R_max)
    profit = compute_profit(F_mesh, C_b, C_A0, Q_dot, price_B, price_A0, price_energy)

    feasible = (code == FEASIBLE)
    if not np.any(feasible):
        raise ValueError(f"no feasible node at beta={beta} over this grid")

    masked_profit = np.where(feasible, profit, -np.inf)
    i_star = np.unravel_index(np.argmax(masked_profit), masked_profit.shape)

    return {
        "F_star": float(F_mesh[i_star]),
        "T_R_star": float(T_R_mesh[i_star]),
        "profit_star": float(profit[i_star]),
        "C_b_star": float(C_b[i_star]),
        "Q_dot_star": float(Q_dot[i_star]),
        "n_feasible": int(np.sum(feasible)),
    }


def evaluate_point(F, T_R, beta, alpha, C_A0, T_in, C,
                    price_B, price_A0, price_energy,
                    q_lower=-8500.0, q_upper=0.0, T_R_max=140.0):
    _, C_b, _, Q_dot = solve_steady_state(F, T_R, beta, alpha, C_A0, T_in, C)
    code = classify(np.array([Q_dot]), np.array([T_R]), q_lower, q_upper, T_R_max)[0]
    profit = compute_profit(F, C_b, C_A0, Q_dot, price_B, price_A0, price_energy)
    return {"profit": float(profit), "feasible": bool(code == FEASIBLE),
            "C_b": float(C_b), "Q_dot": float(Q_dot)}


def compute_gap_table(F_grid, T_R_grid, betas, alpha, C_A0, T_in, C,
                        price_B, price_A0, price_energy,
                        q_lower=-8500.0, q_upper=0.0, T_R_max=140.0):
    kw = dict(alpha=alpha, C_A0=C_A0, T_in=T_in, C=C)
    prices = dict(price_B=price_B, price_A0=price_A0, price_energy=price_energy)
    mask_kw = dict(q_lower=q_lower, q_upper=q_upper, T_R_max=T_R_max)

    fixed = find_optimum(F_grid, T_R_grid, beta=1.0, **kw, **prices, **mask_kw)
    F_fixed, T_R_fixed = fixed["F_star"], fixed["T_R_star"]

    rows = []
    for beta in betas:
        star = find_optimum(F_grid, T_R_grid, beta=beta, **kw, **prices, **mask_kw)
        fixed_eval = evaluate_point(F_fixed, T_R_fixed, beta, **kw, **prices, **mask_kw)

        gap_pct = None
        if fixed_eval["feasible"]:
            gap_pct = 100.0 * (star["profit_star"] - fixed_eval["profit"]) / star["profit_star"]

        rows.append({
            "beta": beta,
            "F_star": star["F_star"], "T_R_star": star["T_R_star"], "Pi_star": star["profit_star"],
            "F_fixed": F_fixed, "T_R_fixed": T_R_fixed,
            "Pi_fixed": fixed_eval["profit"], "fixed_feasible": fixed_eval["feasible"],
            "gap_pct": gap_pct,
        })
    return rows


def plot_trajectory(F_grid, T_R_grid, betas, table, alpha, C_A0, T_in, C,
                      q_lower=-8500.0, q_upper=0.0, T_R_max=140.0, out_path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    cmap = matplotlib.colormaps["viridis"]
    norm = matplotlib.colors.Normalize(vmin=min(betas), vmax=max(betas))

    for beta in betas:
        T_R_mesh = np.broadcast_to(T_R_grid[None, :], (len(F_grid), len(T_R_grid)))
        _, _, _, Q_dot = solve_steady_state(
            np.broadcast_to(F_grid[:, None], (len(F_grid), len(T_R_grid))), T_R_mesh,
            beta, alpha, C_A0, T_in, C,
        )
        code = classify(Q_dot, T_R_mesh, q_lower, q_upper, T_R_max)
        is_feasible = (code == FEASIBLE).astype(float)
        ax.contour(F_grid, T_R_grid, is_feasible.T, levels=[0.5],
                   colors=[cmap(norm(beta))], linewidths=1.6)

    F_star = [row["F_star"] for row in table]
    T_R_star = [row["T_R_star"] for row in table]
    ax.plot(F_star, T_R_star, "-", color="black", lw=1.2, zorder=4)
    sc = ax.scatter(F_star, T_R_star, c=betas, cmap=cmap, norm=norm,
                     s=90, edgecolors="black", zorder=5, label="optimum (F*, T_R*)")

    F_fixed, T_R_fixed = table[0]["F_fixed"], table[0]["T_R_fixed"]
    ax.scatter([F_fixed], [T_R_fixed], marker="*", s=350, color="red",
               edgecolors="black", zorder=6, label=f"frozen beta=1.0 point ({F_fixed:.1f}, {T_R_fixed:.1f})")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("beta")
    ax.set_xlabel("F (h$^{-1}$)")
    ax.set_ylabel("T_R (degC)")
    ax.set_title("Profit-optimum trajectory vs. feasible-region boundary, by beta\n"
                 "(boundary colour = beta; region left of/above each curve is feasible)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    import json
    import time

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
    F_grid, T_R_grid = d["F"], d["T_R"]

    betas = [1.0, 0.8, 0.6, 0.4, 0.385, 0.3, 0.2]
    table = compute_gap_table(F_grid, T_R_grid, betas, **kw, **prices)

    print(f"{'beta':>6} {'F*':>7} {'T_R*':>8} {'Pi*':>10} {'Pi_fixed':>12} {'fixed_feas':>11} {'gap%':>8}")
    for row in table:
        gap_str = f"{row['gap_pct']:7.2f}%" if row["gap_pct"] is not None else "N/A (infeasible)"
        print(f"{row['beta']:6.3f} {row['F_star']:7.2f} {row['T_R_star']:8.2f} {row['Pi_star']:10.1f} "
              f"{row['Pi_fixed']:12.1f} {str(row['fixed_feasible']):>11} {gap_str:>16}")

    fig_path = "open_loop_testing/optimum_trajectory.png"
    plot_trajectory(F_grid, T_R_grid, betas, table, **kw, out_path=fig_path)
    print(f"Saved {fig_path}")

    json_path = "open_loop_testing/optimum_trajectory.json"
    with open(json_path, "w") as f:
        json.dump(table, f, indent=2)

    sidecar = save_with_provenance(
        [json_path, fig_path], data=None,
        params={"betas": betas, "F_range": [float(F_grid[0]), float(F_grid[-1])],
                "T_R_range": [float(T_R_grid[0]), float(T_R_grid[-1])], **prices},
        script=__file__, start_time=start_time,
        extra={"gap_table": table},
    )
    print(f"Saved {json_path}")
    print(f"Saved {sidecar}")

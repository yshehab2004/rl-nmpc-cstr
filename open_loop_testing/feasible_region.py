import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

FEASIBLE = 0
COOLING_FLOOR = 1
HEATING_LIMIT = 2
TEMP_LIMIT = 3

_LABELS = {
    FEASIBLE: "feasible",
    COOLING_FLOOR: "cooling floor (Q̇ < -8500)",
    HEATING_LIMIT: "heating limit (Q̇ > 0)",
    TEMP_LIMIT: "temperature limit (T_R > 140)",
}


def classify(Q_dot, T_R, q_lower=-8500.0, q_upper=0.0, T_R_max=140.0):
    Q_dot = np.asarray(Q_dot, dtype=float)
    T_R = np.asarray(T_R, dtype=float)
    code = np.full(Q_dot.shape, FEASIBLE, dtype=int)
    code[Q_dot < q_lower] = COOLING_FLOOR
    code[Q_dot > q_upper] = HEATING_LIMIT
    code[T_R > T_R_max] = TEMP_LIMIT
    return code


def _node_weights(grid):
    d = np.diff(grid)
    w = np.zeros_like(grid)
    w[:-1] += d / 2
    w[1:] += d / 2
    return w


def feasible_area(F_grid, T_R_grid, code):
    F_grid = np.asarray(F_grid, dtype=float)
    T_R_grid = np.asarray(T_R_grid, dtype=float)
    w_F = _node_weights(F_grid)
    w_T = _node_weights(T_R_grid)
    feasible_mask = (code == FEASIBLE)
    return float(np.sum(feasible_mask * w_F[:, None] * w_T[None, :]))


def plot_feasible_regions(F_grid, T_R_grid, betas, Q_dot, q_lower=-8500.0,
                            q_upper=0.0, T_R_max=140.0, out_path=None):
    n = len(betas)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6), sharey=True)
    if n == 1:
        axes = [axes]

    cmap = ListedColormap(["#d9f0d3", "#fdae61", "#d73027", "#4575b4"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    T_R_mesh = np.broadcast_to(T_R_grid[None, :], (len(F_grid), len(T_R_grid)))

    areas = []
    for i, (ax, beta) in enumerate(zip(axes, betas)):
        code = classify(Q_dot[i], T_R_mesh, q_lower, q_upper, T_R_max)
        area = feasible_area(F_grid, T_R_grid, code)
        areas.append(area)

        ax.pcolormesh(F_grid, T_R_grid, code.T, cmap=cmap, norm=norm, shading="nearest")

        is_feasible = (code.T == FEASIBLE)
        for other_code, colour in ((COOLING_FLOOR, "#e66101"), (HEATING_LIMIT, "#b2182b"), (TEMP_LIMIT, "#08306b")):
            field = np.where(is_feasible, 1.0, np.where(code.T == other_code, 0.0, np.nan))
            if np.nanmax(field, initial=0.0) == 1.0 and np.nanmin(field, initial=1.0) == 0.0:
                ax.contour(F_grid, T_R_grid, field, levels=[0.5], colors=colour, linewidths=2.2)

        ax.set_title(f"beta = {beta:.2f}\narea = {area:.0f} h$^{{-1}}$degC")
        ax.set_xlabel("F (h$^{-1}$)")
        if i == 0:
            ax.set_ylabel("T_R (degC)")

    handles = [Patch(facecolor=cmap(k), label=_LABELS[k]) for k in (FEASIBLE, COOLING_FLOOR, HEATING_LIMIT, TEMP_LIMIT)]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Steady-state feasible region: Q̇ in [-8500, 0] kJ/h and T_R <= 140 degC")
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return fig, np.array(areas)


if __name__ == "__main__":
    import time

    from utils.provenance import save_with_provenance

    start_time = time.time()
    map_path = "open_loop_testing/steady_state_map.npz"
    out_path = "open_loop_testing/feasible_region_vs_beta.png"

    d = np.load(map_path)
    F, T_R_grid, betas, Q_dot = d["F"], d["T_R"], d["beta"], d["Q_dot"]

    fig, areas = plot_feasible_regions(F, T_R_grid, betas, Q_dot, out_path=out_path)

    total = (F[-1] - F[0]) * (T_R_grid[-1] - T_R_grid[0])
    area_pct = {f"{beta:.2f}": {"area": float(area), "pct_of_domain": 100 * float(area) / total}
                for beta, area in zip(betas, areas)}

    print("Feasible area (h^-1 x degC) per beta:")
    for beta_str, vals in area_pct.items():
        print(f"  beta = {beta_str}: area = {vals['area']:8.1f}  ({vals['pct_of_domain']:5.1f}% of domain)")

    sidecar = save_with_provenance(
        out_path, data=None,
        params={"source_map": map_path, "q_lower": -8500.0, "q_upper": 0.0, "T_R_max": 140.0},
        script=__file__, start_time=start_time,
        extra={"feasible_area_by_beta": area_pct},
    )
    print(f"Saved {out_path}")
    print(f"Saved {sidecar}")

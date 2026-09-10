import json
import time

import numpy as np

from models.params import load_params
from open_loop_testing.feasible_region import FEASIBLE, classify
from open_loop_testing.optimum_trajectory import compute_profit, find_optimum
from open_loop_testing.steady_state_map import solve_steady_state
from utils.provenance import save_with_provenance

BETAS = (1.0, 0.8, 0.6, 0.4, 0.2)

CEILINGS = (145.0, 150.0, None)

REFERENCE_CEILING = 140.0

F_GRID = np.linspace(5.0, 35.0, 601)
T_R_GRID_WIDE = np.linspace(100.0, 200.0, 1001)


def unconstrained_optimum(F_grid, T_R_grid, beta, kw, prices):
    F_mesh, T_R_mesh = np.meshgrid(F_grid, T_R_grid, indexing="ij")
    _, C_b, _, Q_dot = solve_steady_state(F_mesh, T_R_mesh, beta, **kw)
    profit = compute_profit(F_mesh, C_b, kw["C_A0"], Q_dot, **prices)
    i = np.unravel_index(np.argmax(profit), profit.shape)
    return {"F_star": float(F_mesh[i]), "T_R_star": float(T_R_mesh[i]),
            "profit_star": float(profit[i])}


def main():
    t_start = time.time()
    P = load_params()
    C = P["certain_params"]
    feed = P["nominal_feed"]
    econ = P["economics"]
    kw = dict(alpha=P["uncertain_params"]["alpha_nominal"],
              C_A0=feed["C_A0"], T_in=feed["T_in"], C=C)
    prices = dict(price_B=econ["price_B"], price_A0=econ["price_A0"],
                  price_energy=econ["price_energy"])

    cached = np.load("open_loop_testing/steady_state_map.npz")
    T_cached_max = float(cached["T_R"].max())

    rows = []
    for beta in BETAS:
        base = find_optimum(F_GRID, T_R_GRID_WIDE, beta, **kw, **prices,
                            T_R_max=REFERENCE_CEILING)

        unc_wide = unconstrained_optimum(F_GRID, T_R_GRID_WIDE, beta, kw, prices)
        unc_cached = unconstrained_optimum(F_GRID, cached["T_R"], beta, kw, prices)

        row = {
            "beta": beta,
            "constrained_140": base["profit_star"],
            "constrained_140_T_R": base["T_R_star"],
            "unconstrained_wide_profit": unc_wide["profit_star"],
            "unconstrained_wide_T_R": unc_wide["T_R_star"],
            "unconstrained_cached_profit": unc_cached["profit_star"],
            "unconstrained_cached_T_R": unc_cached["T_R_star"],
            "pct_cost_unbounded_wide":
                100.0 * (unc_wide["profit_star"] - base["profit_star"])
                / unc_wide["profit_star"],
            "pct_cost_unbounded_cached":
                100.0 * (unc_cached["profit_star"] - base["profit_star"])
                / unc_cached["profit_star"],
        }

        for ceil in CEILINGS:
            if ceil is None:
                continue
            hi = find_optimum(F_GRID, T_R_GRID_WIDE, beta, **kw, **prices,
                              T_R_max=ceil)
            row[f"opt_at_{ceil:g}"] = hi["profit_star"]
            row[f"T_R_at_{ceil:g}"] = hi["T_R_star"]
            row[f"pct_cost_vs_{ceil:g}"] = (
                100.0 * (hi["profit_star"] - base["profit_star"])
                / hi["profit_star"])
        rows.append(row)

    print(f"cached grid tops out at T_R = {T_cached_max:g} degC\n")
    print("Cost of the 140 degC ceiling, priced against several alternatives.")
    hdr = (f"{'beta':>5} {'P(140)':>9} {'vs 145':>8} {'vs 150':>8} "
           f"{'vs unbdd':>9} {'unbdd T_R':>10} {'cached T_R':>11} {'cached %':>9}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['beta']:5.2f} {r['constrained_140']:9.1f} "
              f"{r['pct_cost_vs_145']:7.1f}% {r['pct_cost_vs_150']:7.1f}% "
              f"{r['pct_cost_unbounded_wide']:8.1f}% "
              f"{r['unconstrained_wide_T_R']:10.1f} "
              f"{r['unconstrained_cached_T_R']:11.1f} "
              f"{r['pct_cost_unbounded_cached']:8.1f}%")

    censored = [r for r in rows
                if abs(r["unconstrained_cached_T_R"] - T_cached_max) < 1e-6]
    print(f"\nbeta values whose CACHED unconstrained argmax sits exactly on the "
          f"grid edge ({T_cached_max:g} degC): "
          f"{[r['beta'] for r in censored] or 'none'}")
    if censored:
        print("  -> for these the published figure is censored by the grid, not "
              "located by the optimiser.")

    out = {"rows": rows, "cached_T_R_max": T_cached_max,
           "reference_ceiling": REFERENCE_CEILING,
           "wide_grid": [100.0, 200.0, len(T_R_GRID_WIDE)]}
    path = "open_loop_testing/constraint_cost_vs_ceiling.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    save_with_provenance(
        path, None,
        params={"betas": list(BETAS), "ceilings": [c for c in CEILINGS if c],
                "reference_ceiling": REFERENCE_CEILING,
                "F_grid": [5.0, 35.0, len(F_GRID)],
                "T_R_grid": [100.0, 200.0, len(T_R_GRID_WIDE)]},
        script=__file__, start_time=t_start,
        extra={"cached_T_R_max": T_cached_max})
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

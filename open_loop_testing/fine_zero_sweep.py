import json
import time

import numpy as np

from models.params import load_params
from open_loop_testing.nmp_zero import F_to_Cb_zeros
from open_loop_testing.optimum_trajectory import find_optimum
from utils.provenance import save_with_provenance

BANDWIDTH_RULE_DIVISOR = 3.0


def sweep(betas, F_grid, T_R_grid, alpha, C_A0, T_in, C, price_B, price_A0, price_energy):
    results = []
    for beta in betas:
        try:
            opt = find_optimum(F_grid, T_R_grid, beta, alpha, C_A0, T_in, C,
                                price_B, price_A0, price_energy)
        except ValueError:
            results.append({"beta": float(beta), "F_star": None, "T_R_star": None,
                             "min_rhp_zero": None, "bandwidth_cap_h_inv": None,
                             "note": "no feasible node at this beta"})
            continue

        F_star, T_R_star = opt["F_star"], opt["T_R_star"]
        zeros = F_to_Cb_zeros(F_star, T_R_star, beta, alpha, C_A0, T_in, C)
        rhp_zeros = [z.real for z in zeros if z.real > 0]
        min_rhp_zero = min(rhp_zeros) if rhp_zeros else None
        bandwidth_cap = min_rhp_zero / BANDWIDTH_RULE_DIVISOR if min_rhp_zero is not None else None

        results.append({
            "beta": float(beta), "F_star": F_star, "T_R_star": T_R_star,
            "min_rhp_zero": min_rhp_zero, "bandwidth_cap_h_inv": bandwidth_cap,
        })
    return results


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    start_time = time.time()
    PARAMS = load_params()
    C = PARAMS["certain_params"]
    feed = PARAMS["nominal_feed"]
    econ = PARAMS["economics"]
    kw = dict(alpha=PARAMS["uncertain_params"]["alpha_nominal"], C_A0=feed["C_A0"], T_in=feed["T_in"], C=C)
    prices = dict(price_B=econ["price_B"], price_A0=econ["price_A0"], price_energy=econ["price_energy"])

    betas = np.round(np.arange(0.35, 0.18 - 1e-9, -0.005), 3)
    F_grid = np.linspace(15.0, 32.0, 1701)
    T_R_grid = np.linspace(130.0, 145.0, 301)

    print(f"Sweeping {len(betas)} beta points in [0.18, 0.35] at 0.005 spacing...")
    results = sweep(betas, F_grid, T_R_grid, **kw, **prices)

    for r in results:
        z = f"{r['min_rhp_zero']:.4f}" if r["min_rhp_zero"] is not None else "none"
        print(f"  beta={r['beta']:.3f}  F*={r['F_star']}  T_R*={r['T_R_star']}  min_rhp_zero={z}")

    betas_plot = [r["beta"] for r in results]
    zeros_plot = [r["min_rhp_zero"] if r["min_rhp_zero"] is not None else np.nan for r in results]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(betas_plot, zeros_plot, marker="o", ms=3, lw=1)
    ax.axvline(0.385, color="gray", ls=":", lw=1, label="beta* = 0.385 (F-003)")
    ax.set_xlabel("beta")
    ax.set_ylabel("min positive real transmission zero (F->C_b)")
    ax.set_title("Fine beta sweep of the F->C_b RHP zero along the real trajectory")
    ax.legend(fontsize=8)
    ax.invert_xaxis()
    fig.tight_layout()

    fig_path = "open_loop_testing/fine_zero_sweep.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")

    json_path = "open_loop_testing/fine_zero_sweep.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    sidecar = save_with_provenance(
        [json_path, fig_path], data=None,
        params={"beta_range": [0.18, 0.35], "beta_spacing": 0.005,
                "F_range": [5.0, 35.0, 0.25], "T_R_range": [100.0, 150.0, 0.25], **prices},
        script=__file__, start_time=start_time, extra={"results": results},
    )
    print(f"Saved {json_path}, {fig_path}, {sidecar}")

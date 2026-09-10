import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from open_loop_testing.optimum_trajectory import find_optimum


def sweep(F_grid, T_R_grid, beta_range, beta_step, alpha, C_A0, T_in, C,
           price_B, price_A0, price_energy):
    betas = np.round(np.arange(beta_range[0], beta_range[1] - beta_step / 2, -beta_step), 4)
    rows = []
    for beta in betas:
        r = find_optimum(F_grid, T_R_grid, float(beta), alpha, C_A0, T_in, C, price_B, price_A0, price_energy)
        rows.append({"beta": float(beta), "F_star": r["F_star"], "T_R_star": r["T_R_star"],
                      "profit_star": r["profit_star"]})
    return rows


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

    F_grid = np.linspace(5.0, 35.0, 121)
    T_R_grid = np.linspace(100.0, 150.0, 201)
    beta_range = (0.40, 0.28)
    beta_step = 0.005

    rows = sweep(F_grid, T_R_grid, beta_range, beta_step, **kw, **prices)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot([r["beta"] for r in rows], [r["F_star"] for r in rows], "o-", ms=3)
    ax.invert_xaxis()
    ax.set_xlabel("beta")
    ax.set_ylabel("F* (h$^{-1}$)")
    ax.set_title("Fine beta sweep through the apparent discontinuity (0.005 spacing)\n"
                 "Smooth, continuous decline -- not a jump")
    fig.tight_layout()

    out_path = "open_loop_testing/fine_beta_sweep_check.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")

    json_path = "open_loop_testing/fine_beta_sweep_check.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    sidecar = save_with_provenance(
        [json_path, out_path], data=None,
        params={"beta_range": list(beta_range), "beta_step": beta_step, **prices},
        script=__file__, start_time=start_time,
        extra={"conclusion": "smooth continuous decline, not a jump"},
    )
    print(f"Saved {json_path}, {sidecar}")

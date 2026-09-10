import numpy as np

from open_loop_testing.optimum_trajectory import find_optimum
from open_loop_testing.active_constraint_switch import peak_cooling_floor_T_R, find_switch_beta


def _find_bracket(T_lim, alpha, C_A0, T_in, C, n_scan=40):
    betas = np.linspace(0.05, 1.0, n_scan)
    gaps = []
    for beta in betas:
        peak, _ = peak_cooling_floor_T_R(beta, alpha, C_A0, T_in, C, F_range=(5.0, 35.0))
        gaps.append(peak - T_lim)
    gaps = np.array(gaps)
    sign_changes = np.where(np.diff(np.sign(gaps)) != 0)[0]
    if len(sign_changes) == 0:
        raise ValueError(f"no sign change scanning beta in [0.05,1.0] for T_lim={T_lim}")
    i = sign_changes[0]
    return float(betas[i]), float(betas[i + 1])


def switch_beta_for_T_lim(T_lim, alpha, C_A0, T_in, C):
    beta_lo, beta_hi = _find_bracket(T_lim, alpha, C_A0, T_in, C)
    return find_switch_beta(beta_lo, beta_hi, alpha, C_A0, T_in, C, T_R_max=T_lim, F_range=(5.0, 35.0))


def trajectory_for_T_lim(T_lim, betas, alpha, C_A0, T_in, C,
                           price_B=100.0, price_A0=5.0, price_energy=0.02,
                           F_range=(5.0, 35.0), n_F=121, T_R_range=(100.0, 150.0), n_T_R=201):
    F_grid = np.linspace(F_range[0], F_range[1], n_F)
    T_R_grid = np.linspace(T_R_range[0], T_R_range[1], n_T_R)
    rows = []
    for beta in betas:
        try:
            opt = find_optimum(F_grid, T_R_grid, beta, alpha, C_A0, T_in, C,
                                 price_B, price_A0, price_energy, T_R_max=T_lim)
            rows.append({"beta": beta, "F_star": opt["F_star"], "T_R_star": opt["T_R_star"],
                          "profit_star": opt["profit_star"], "feasible": True})
        except ValueError:
            rows.append({"beta": beta, "F_star": None, "T_R_star": None,
                          "profit_star": None, "feasible": False})
    return rows


if __name__ == "__main__":
    import json
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

    T_lims = [135.0, 140.0, 145.0]
    betas = [1.0, 0.8, 0.6, 0.4, 0.3, 0.2]

    all_results = {}
    switch_betas = {}
    for T_lim in T_lims:
        switch_betas[T_lim] = switch_beta_for_T_lim(T_lim, **kw)
        rows = trajectory_for_T_lim(T_lim, betas, **kw, **prices)
        all_results[T_lim] = rows
        print(f"\n--- T_lim = {T_lim} degC, switch beta* = {switch_betas[T_lim]:.5f} ---")
        print(f"{'beta':>6} {'F*':>7} {'T_R*':>8} {'profit*':>10} {'feasible':>9}")
        for row in rows:
            if row["feasible"]:
                print(f"{row['beta']:6.3f} {row['F_star']:7.2f} {row['T_R_star']:8.2f} "
                      f"{row['profit_star']:10.1f} {'yes':>9}")
            else:
                print(f"{row['beta']:6.3f} {'--':>7} {'--':>8} {'--':>10} {'no':>9}")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for T_lim in T_lims:
        rows = all_results[T_lim]
        bs = [r["beta"] for r in rows if r["feasible"]]
        Fs = [r["F_star"] for r in rows if r["feasible"]]
        ax.plot(bs, Fs, "o-", label=f"T_lim={T_lim:.0f} (beta*={switch_betas[T_lim]:.3f})")
    ax.invert_xaxis()
    ax.set_xlabel("beta")
    ax.set_ylabel("F* (h$^{-1}$)")
    ax.set_title("Optimum F* vs. beta, across temperature-limit choices")
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_path = "open_loop_testing/switch_beta_vs_temperature_limit.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved {out_path}")

    json_path = "open_loop_testing/temperature_limit_sensitivity.json"
    with open(json_path, "w") as f:
        json.dump({"switch_betas": switch_betas, "trajectories": all_results}, f, indent=2)

    sidecar = save_with_provenance(
        [json_path, out_path], data=None,
        params={"T_lims": T_lims, "betas": betas, **prices},
        script=__file__, start_time=start_time,
        extra={"switch_betas": switch_betas},
    )
    print(f"Saved {json_path}, {sidecar}")

from evaluation.phase0_validation import rate_constants


def F_peak(T_R, alpha, beta, C):
    k1, _, _ = rate_constants(T_R, alpha=alpha, beta=beta)
    return float(k1)


def crossover_beta_for_bound(F_bound, T_R, alpha, C):
    k1_at_beta_1, _, _ = rate_constants(T_R, alpha=alpha, beta=1.0)
    return F_bound / k1_at_beta_1


if __name__ == "__main__":
    import time
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from models.params import load_params
    from evaluation.phase0_validation import steady_state_concentrations
    from open_loop_testing.optimum_trajectory import find_optimum
    from utils.provenance import save_with_provenance

    start_time = time.time()
    PARAMS = load_params()
    C = PARAMS["certain_params"]
    feed = PARAMS["nominal_feed"]
    econ = PARAMS["economics"]
    THETA = 134.14

    beta_upper = crossover_beta_for_bound(35.0, THETA, 1.0, C)
    beta_lower = crossover_beta_for_bound(5.0, THETA, 1.0, C)
    print(f"At fixed T_R={THETA}: F_peak enters the operating range (F_peak=35) at "
          f"beta={beta_upper:.4f}, reaches the F lower bound (F_peak=5) at beta={beta_lower:.4f}")

    betas_to_plot = [1.0, 0.8, beta_upper, 0.6, 0.4, 0.2]
    F = np.linspace(0.5, 60.0, 2000)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for beta in sorted(set(betas_to_plot), reverse=True):
        k1, k2, k3 = rate_constants(THETA, alpha=1.0, beta=beta)
        _, C_b = steady_state_concentrations(F, feed["C_A0"], k1, k2, k3)
        axes[0].plot(F, C_b, label=f"beta={beta:.3f}")
        axes[0].axvline(k1, color=axes[0].lines[-1].get_color(), ls=":", lw=1)
    axes[0].axvspan(5.0, 35.0, color="gray", alpha=0.15, label="operating range")
    axes[0].set_xlabel("F (h$^{-1}$)")
    axes[0].set_ylabel("C_b (mol/L)")
    axes[0].set_title(f"C_b(F) at fixed T_R={THETA}, by beta\n(dotted = F_peak = k1)")
    axes[0].legend(fontsize=7)
    axes[0].set_xlim(0, 60)

    kw = dict(alpha=PARAMS["uncertain_params"]["alpha_nominal"], C_A0=feed["C_A0"], T_in=feed["T_in"], C=C)
    prices = dict(price_B=econ["price_B"], price_A0=econ["price_A0"], price_energy=econ["price_energy"])
    d = np.load("open_loop_testing/steady_state_map.npz")
    F_grid, T_R_grid = d["F"], d["T_R"]

    betas = [1.0, 0.8, 0.6, 0.4, 0.385, 0.3, 0.2]
    rows = []
    for beta in betas:
        opt = find_optimum(F_grid, T_R_grid, beta, **kw, **prices)
        peak = F_peak(opt["T_R_star"], kw["alpha"], beta, C)
        rows.append({"beta": beta, "F_star": opt["F_star"], "T_R_star": opt["T_R_star"],
                      "F_peak_at_T_R_star": peak, "past_peak": opt["F_star"] > peak})

    axes[1].plot([r["beta"] for r in rows], [r["F_star"] for r in rows], "o-", label="F* (profit optimum)")
    axes[1].plot([r["beta"] for r in rows], [r["F_peak_at_T_R_star"] for r in rows], "s--",
                 label="F_peak = k1 at that T_R*")
    axes[1].invert_xaxis()
    axes[1].set_xlabel("beta")
    axes[1].set_ylabel("F (h$^{-1}$)")
    axes[1].set_title("Real trajectory: F* vs. the C_b(F) peak at its own T_R*")
    axes[1].legend(fontsize=8)
    fig.tight_layout()

    out_path = "open_loop_testing/input_multiplicity.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")

    print(f"\n{'beta':>6} {'F_star':>8} {'T_R_star':>9} {'F_peak':>8} {'past peak?':>11}")
    for r in rows:
        print(f"{r['beta']:6.3f} {r['F_star']:8.2f} {r['T_R_star']:9.2f} "
              f"{r['F_peak_at_T_R_star']:8.2f} {str(r['past_peak']):>11}")
    print(f"Saved {out_path}")

    sidecar = save_with_provenance(
        out_path, data=None,
        params={"theta_reference": THETA, "betas": betas, **prices},
        script=__file__, start_time=start_time,
        extra={"beta_upper_crossover_at_reference_T_R": beta_upper,
               "beta_lower_crossover_at_reference_T_R": beta_lower,
               "real_trajectory_rows": rows},
    )
    print(f"Saved {sidecar}")

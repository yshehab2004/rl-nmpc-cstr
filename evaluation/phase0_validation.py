import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models.params import load_params

PARAMS = load_params()
C = PARAMS["certain_params"]
F_BOUNDS = PARAMS["input_bounds"]["F"]
THETA = 134.14


def rate_constants(T_R_degC, alpha=1.0, beta=1.0):
    T = T_R_degC + 273.15
    k1 = beta * C["K0_ab"] * np.exp(-C["E_A_ab"] / T)
    k2 = beta * C["K0_bc"] * np.exp(-C["E_A_bc"] / T)
    k3 = C["K0_ad"] * np.exp(-alpha * C["E_A_ad"] / T)
    return k1, k2, k3


def steady_state_concentrations(F, C_A0, k1, k2, k3):
    F = np.asarray(F, dtype=float)
    a = k3
    b = F + k1
    c = -F * C_A0
    C_a = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    C_b = k1 * C_a / (F + k2)
    return C_a, C_b


def main():
    import time
    start_time = time.time()

    k1, k2, k3 = rate_constants(THETA)
    F_peak = k1
    print(f"theta = {THETA} degC -> k1 = k2 = {k1:.4f} h^-1, k3 = {k3:.4f} L/mol.h")
    print(f"Predicted C_b peak at F* = k1 = {F_peak:.2f} h^-1 "
          f"({'beyond' if F_peak > F_BOUNDS['upper'] else 'within'} "
          f"the operating bound F <= {F_BOUNDS['upper']:.0f})")

    F = np.linspace(0.5, 100.0, 500)
    C_A0_values = [4.5, 5.1, 5.7]

    fig, ax = plt.subplots(figsize=(7, 5))
    for C_A0 in C_A0_values:
        C_a, C_b = steady_state_concentrations(F, C_A0, k1, k2, k3)
        style = "-" if C_A0 == 5.1 else "--"
        ax.plot(F, C_b, style, label=f"C_A0 = {C_A0} mol/L")

    ax.axvline(F_BOUNDS["upper"], color="k", ls=":", lw=1,
               label=f"F upper bound = {F_BOUNDS['upper']:.0f}")
    ax.axvline(F_peak, color="tab:red", ls=":", lw=1,
               label=f"F* = k1 = {F_peak:.1f} (peak, off-plot bound)")
    ax.set_xlabel("F (h$^{-1}$)")
    ax.set_ylabel("C_b (mol/L)")
    ax.set_title("Steady-state C_b vs F at theta = 134.14 degC\n"
                 "(reproduction of Klatt & Engell 1998, Fig. 3)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 100)
    plt.tight_layout()

    out_dir = "evaluation/phase0_validation"
    import os
    os.makedirs(out_dir, exist_ok=True)
    fig_path = f"{out_dir}/klatt_engell_reproduction.png"
    fig.savefig(fig_path, dpi=300)
    print(f"Saved {fig_path}")

    _, C_b_at_35 = steady_state_concentrations(35.0, 5.1, k1, k2, k3)
    print(f"C_b(F=35, C_A0=5.1) = {C_b_at_35:.4f} mol/L "
          f"(still rising -- curve has not turned over by the operating bound)")

    from utils.provenance import save_with_provenance
    sidecar = save_with_provenance(
        fig_path, data=None,
        params={"theta_degC": THETA, "C_A0_values": C_A0_values, "F_range": [0.5, 100.0]},
        script=__file__, start_time=start_time,
        extra={"k1": float(k1), "k2": float(k2), "k3": float(k3),
               "F_peak": float(F_peak), "C_b_at_F35_C_A0_5p1": float(C_b_at_35)},
    )
    print(f"Saved {sidecar}")


if __name__ == "__main__":
    main()

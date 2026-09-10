import json
import math
import time

import numpy as np

from models.params import load_params
from utils.provenance import save_with_provenance

CONFIG = "configs/reactor_params.yaml"

BETAS = (1.0, 0.6, 0.4, 0.2)

TEMPERATURES = (140.0, 120.0)


def rate_constants(T_R, beta, alpha, p):
    T_K = T_R + 273.15
    K_1 = beta * p["K0_ab"] * math.exp(-p["E_A_ab"] / T_K)
    K_2 = beta * p["K0_bc"] * math.exp(-p["E_A_bc"] / T_K)
    K_3 = p["K0_ad"] * math.exp(-alpha * p["E_A_ad"] / T_K)
    return K_1, K_2, K_3


def C_B_at(F, K_1, K_2, K_3, C_A0):
    if F <= 0.0:
        return 0.0
    a, b, c = K_3, F + K_1, -F * C_A0
    C_A = (-b + math.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)
    return K_1 * C_A / (F + K_2)


def C_B_no_dimerisation(F, K_1, K_2, C_A0):
    if F <= 0.0:
        return 0.0
    C_A = F * C_A0 / (F + K_1)
    return K_1 * C_A / (F + K_2)


def argmax_F(fn, lo, hi, n_scan=200_001, n_refine=200):
    grid = np.linspace(lo, hi, n_scan)
    vals = np.array([fn(F) for F in grid])
    i = int(np.argmax(vals))
    a = grid[max(i - 1, 0)]
    b = grid[min(i + 1, n_scan - 1)]

    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = b - invphi * (b - a), a + invphi * (b - a)
    for _ in range(n_refine):
        if fn(c) > fn(d):
            b, d = d, c
            c = b - invphi * (b - a)
        else:
            a, c = c, d
            d = a + invphi * (b - a)
    F_star = 0.5 * (a + b)
    return F_star, fn(F_star)


def main():
    t_start = time.time()
    cfg = load_params()
    p = cfg["certain_params"]
    C_A0 = cfg["nominal_feed"]["C_A0"]
    alpha = cfg["uncertain_params"]["alpha_nominal"]
    print(f"C_A0 = {C_A0}   alpha = {alpha}")

    rows = []
    for T_R in TEMPERATURES:
        for beta in BETAS:
            K_1, K_2, K_3 = rate_constants(T_R, beta, alpha, p)
            sqrt_k1k2 = math.sqrt(K_1 * K_2)

            F_true, CB_true = argmax_F(
                lambda F: C_B_at(F, K_1, K_2, K_3, C_A0), 1e-3, 400.0)
            F_nodim, _ = argmax_F(
                lambda F: C_B_no_dimerisation(F, K_1, K_2, C_A0), 1e-3, 400.0)

            CB_at_sqrt = C_B_at(sqrt_k1k2, K_1, K_2, K_3, C_A0)

            a, b, c = K_3, F_true + K_1, -F_true * C_A0
            C_A_star = (-b + math.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)
            dimer_share = K_3 * C_A_star / K_1

            row = {
                "T_R_degC": T_R, "beta": beta,
                "k1": K_1, "k2": K_2, "k3": K_3,
                "sqrt_k1k2": sqrt_k1k2,
                "F_star_true": F_true,
                "F_star_no_dimerisation": F_nodim,
                "pct_error_in_F": 100.0 * (sqrt_k1k2 - F_true) / F_true,
                "C_B_at_true_optimum": CB_true,
                "C_B_at_sqrt_k1k2": CB_at_sqrt,
                "pct_C_B_lost_using_sqrt": 100.0 * (CB_true - CB_at_sqrt) / CB_true,
                "dimerisation_share_k3CA_over_k1": dimer_share,
            }
            rows.append(row)

    print()
    hdr = (f"{'T_R':>6} {'beta':>5} {'sqrt(k1k2)':>11} {'true F*':>10} "
           f"{'err in F*':>10} {'C_B lost':>10} {'k3CA/k1':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['T_R_degC']:6.0f} {r['beta']:5.2f} {r['sqrt_k1k2']:11.4f} "
              f"{r['F_star_true']:10.4f} {r['pct_error_in_F']:9.2f}% "
              f"{r['pct_C_B_lost_using_sqrt']:9.3f}% "
              f"{r['dimerisation_share_k3CA_over_k1']:9.4f}")

    worst_nodim = max(abs(100.0 * (r["sqrt_k1k2"] - r["F_star_no_dimerisation"])
                          / r["F_star_no_dimerisation"]) for r in rows)
    print(f"\ncontrol: max |error| of sqrt(k1k2) against the k3=0 argmax "
          f"= {worst_nodim:.2e}%  (must be ~0, else the optimiser is at fault)")

    K_1, K_2, K_3 = rate_constants(140.0, 0.2, alpha, p)
    perturbation = []
    for ratio in (1.0, 1.2, 1.5, 2.0, 3.0, 0.5):
        k2 = K_1 * ratio
        F_t, _ = argmax_F(lambda F: C_B_at(F, K_1, k2, K_3, C_A0), 1e-3, 400.0)
        s = math.sqrt(K_1 * k2)
        a, b, c = K_3, F_t + K_1, -F_t * C_A0
        C_A = (-b + math.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)
        perturbation.append({
            "k2_over_k1": ratio,
            "sqrt_k1k2": s,
            "F_star_true": F_t,
            "pct_error_in_F": 100.0 * (s - F_t) / F_t,
            "stationarity_residual": F_t * F_t + K_3 * C_A * F_t
                                     - k2 * K_3 * C_A - K_1 * k2,
        })

    print(f"\ncontrol: is the exactness due to k1 = k2 rather than small k3?")
    print(f"  (beta=0.20, T_R=140 degC, k3 held fixed, k2 varied)")
    print(f"  {'k2/k1':>7} {'sqrt(k1k2)':>11} {'true F*':>10} {'err in F*':>10} "
          f"{'stat. resid':>12}")
    for q in perturbation:
        print(f"  {q['k2_over_k1']:7.2f} {q['sqrt_k1k2']:11.4f} "
              f"{q['F_star_true']:10.4f} {q['pct_error_in_F']:9.3f}% "
              f"{q['stationarity_residual']:12.2e}")

    out = {"rows": rows,
           "C_A0": C_A0, "alpha": alpha,
           "control_max_pct_error_k3_zero": worst_nodim,
           "k2_perturbation_control": perturbation}
    path = "open_loop_testing/fstar_dimerisation_check.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    save_with_provenance(
        path, None,
        params={"betas": list(BETAS), "temperatures_degC": list(TEMPERATURES),
                "C_A0": C_A0, "alpha": alpha,
                "F_scan": [1e-3, 400.0, 200001], "golden_iters": 200},
        script=__file__, start_time=t_start,
        extra={"control_max_pct_error_k3_zero": worst_nodim})
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

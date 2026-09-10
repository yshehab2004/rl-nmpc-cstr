import numpy as np

from open_loop_testing.steady_state_map import solve_steady_state
from open_loop_testing.optimum_trajectory import compute_profit


def _profit(F, T_R, beta, alpha, C_A0, T_in, C, price_B, price_A0, price_energy):
    _, C_b, _, Q_dot = solve_steady_state(F, T_R, beta, alpha, C_A0, T_in, C)
    return compute_profit(F, C_b, C_A0, Q_dot, price_B, price_A0, price_energy)


def profit_hessian(F, T_R, beta, alpha, C_A0, T_in, C,
                     price_B, price_A0, price_energy, h_F=1e-2, h_T=1e-2):
    kw = dict(alpha=alpha, C_A0=C_A0, T_in=T_in, C=C,
              price_B=price_B, price_A0=price_A0, price_energy=price_energy)
    f = lambda F_, T_: _profit(F_, T_, beta, **kw)

    f00 = f(F, T_R)
    fpp = f(F + h_F, T_R + h_T)
    fpm = f(F + h_F, T_R - h_T)
    fmp = f(F - h_F, T_R + h_T)
    fmm = f(F - h_F, T_R - h_T)
    fp0 = f(F + h_F, T_R)
    fm0 = f(F - h_F, T_R)
    f0p = f(F, T_R + h_T)
    f0m = f(F, T_R - h_T)

    d2F = (fp0 - 2 * f00 + fm0) / h_F ** 2
    d2T = (f0p - 2 * f00 + f0m) / h_T ** 2
    dFT = (fpp - fpm - fmp + fmm) / (4 * h_F * h_T)

    return np.array([[d2F, dFT], [dFT, d2T]])


def condition_number(H):
    eigvals = np.linalg.eigvalsh(H)
    return float(np.max(np.abs(eigvals)) / np.min(np.abs(eigvals)))


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

    with open("open_loop_testing/optimum_trajectory.json") as f:
        table = json.load(f)

    print(f"{'beta':>6} {'F*':>7} {'T_R*':>8} {'eig1':>12} {'eig2':>12} {'condition #':>12}")
    results = []
    for row in table:
        H = profit_hessian(row["F_star"], row["T_R_star"], row["beta"], **kw, **prices)
        eigvals = np.linalg.eigvalsh(H)
        kappa = condition_number(H)
        results.append({"beta": row["beta"], "F_star": row["F_star"], "T_R_star": row["T_R_star"],
                         "eigenvalues": eigvals.tolist(), "condition_number": kappa})
        print(f"{row['beta']:6.3f} {row['F_star']:7.2f} {row['T_R_star']:8.2f} "
              f"{eigvals[0]:12.3f} {eigvals[1]:12.3f} {kappa:12.1f}")

    out_path = "open_loop_testing/hessian_conditioning.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    sidecar = save_with_provenance(
        out_path, data=None,
        params={"h_F": 1e-2, "h_T": 1e-2, **prices},
        script=__file__, start_time=start_time, extra={"results": results},
    )
    print(f"Saved {out_path}")
    print(f"Saved {sidecar}")

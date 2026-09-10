import numpy as np

from open_loop_testing.steady_state_map import solve_steady_state
from open_loop_testing.optimum_trajectory import compute_profit


def gradient_at_F(F, T_R, beta, alpha, C_A0, T_in, C,
                    price_B, price_A0, price_energy, h=1e-3):
    _, C_b_plus, _, Q_dot_plus = solve_steady_state(F + h, T_R, beta, alpha, C_A0, T_in, C)
    _, C_b_minus, _, Q_dot_minus = solve_steady_state(F - h, T_R, beta, alpha, C_A0, T_in, C)

    profit_plus = compute_profit(F + h, C_b_plus, C_A0, Q_dot_plus, price_B, price_A0, price_energy)
    profit_minus = compute_profit(F - h, C_b_minus, C_A0, Q_dot_minus, price_B, price_A0, price_energy)

    return {
        "dC_b_dF": float((C_b_plus - C_b_minus) / (2 * h)),
        "dPi_dF": float((profit_plus - profit_minus) / (2 * h)),
    }


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

    from open_loop_testing.optimum_trajectory import find_optimum

    d = np.load("open_loop_testing/steady_state_map.npz")
    T_R_grid = d["T_R"]
    F_at_bound = np.array([35.0])
    betas = [1.0, 0.8, 0.6, 0.4, 0.385, 0.3, 0.2]

    print(f"{'beta':>6} {'T_R*(F=35)':>11} {'dC_b/dF':>10} {'dPi/dF':>10}")
    results = []
    for beta in betas:
        try:
            at_bound = find_optimum(F_at_bound, T_R_grid, beta, **kw, **prices)
        except ValueError:
            results.append({"beta": beta, "T_R_used": None, "dC_b_dF": None, "dPi_dF": None,
                             "note": "no feasible T_R at F=35 for this beta"})
            print(f"{beta:6.3f} {'--':>11} {'infeasible at F=35 for this beta':>22}")
            continue
        grad = gradient_at_F(35.0, at_bound["T_R_star"], beta, **kw, **prices)
        results.append({"beta": beta, "T_R_used": at_bound["T_R_star"], **grad})
        print(f"{beta:6.3f} {at_bound['T_R_star']:11.2f} {grad['dC_b_dF']:10.5f} {grad['dPi_dF']:10.2f}")

    out_path = "open_loop_testing/boundary_gradient.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    sidecar = save_with_provenance(
        out_path, data=None,
        params={"F": 35.0, "h": 1e-3, **prices},
        script=__file__, start_time=start_time, extra={"results": results},
    )
    print(f"Saved {out_path}")
    print(f"Saved {sidecar}")

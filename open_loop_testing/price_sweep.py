import numpy as np

from open_loop_testing.optimum_trajectory import find_optimum


def sweep_price_ratio(F_grid, T_R_grid, ratios, beta, price_B, price_energy,
                        alpha, C_A0, T_in, C):
    rows = []
    for ratio in ratios:
        price_A0 = ratio * price_B
        result = find_optimum(F_grid, T_R_grid, beta, alpha, C_A0, T_in, C,
                                price_B, price_A0, price_energy)
        rows.append({
            "price_A0_over_price_B": ratio, "price_A0": price_A0,
            "F_star": result["F_star"], "T_R_star": result["T_R_star"],
            "profit_star": result["profit_star"],
            "at_F_upper_bound": bool(result["F_star"] >= F_grid[-1] - 1e-6),
        })
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

    d = np.load("open_loop_testing/steady_state_map.npz")
    F_grid, T_R_grid = d["F"], d["T_R"]

    nominal_ratio = econ["price_A0"] / econ["price_B"]
    ratios = np.logspace(np.log10(nominal_ratio) - 1, np.log10(nominal_ratio) + 1, 11).tolist()

    rows = sweep_price_ratio(F_grid, T_R_grid, ratios, beta=1.0,
                               price_B=econ["price_B"], price_energy=econ["price_energy"], **kw)

    print(f"{'p_A0/p_B':>10} {'F*':>7} {'T_R*':>8} {'profit*':>10} {'at F bound?':>12}")
    for row in rows:
        print(f"{row['price_A0_over_price_B']:10.4f} {row['F_star']:7.2f} {row['T_R_star']:8.2f} "
              f"{row['profit_star']:10.1f} {str(row['at_F_upper_bound']):>12}")

    out_path = "open_loop_testing/price_sweep.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)

    sidecar = save_with_provenance(
        out_path, data=None,
        params={"ratios": ratios, "nominal_ratio": nominal_ratio, "beta": 1.0,
                "price_B": econ["price_B"], "price_energy": econ["price_energy"]},
        script=__file__, start_time=start_time, extra={"rows": rows},
    )
    print(f"Saved {out_path}")
    print(f"Saved {sidecar}")

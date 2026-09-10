import json
import time

import numpy as np

from models.params import load_params
from models.deactivation import make_severity_fn
from open_loop_testing.campaign_integrated_gap import (
    find_best_always_feasible_point, campaign_total_profit_adaptive,
)
from open_loop_testing.partial_campaign_gap import time_at_beta
from utils.provenance import save_with_provenance

BETA_MINS = [0.20, 0.25, 0.30, 0.35]


def run():
    PARAMS = load_params()
    C = PARAMS["certain_params"]
    feed = PARAMS["nominal_feed"]
    econ = PARAMS["economics"]
    kw = dict(alpha=PARAMS["uncertain_params"]["alpha_nominal"], C_A0=feed["C_A0"], T_in=feed["T_in"], C=C)
    prices = dict(price_B=econ["price_B"], price_A0=econ["price_A0"], price_energy=econ["price_energy"])

    CAMPAIGN_H = 75.0
    schedule = make_severity_fn("severe", CAMPAIGN_H, profile="exponential")

    d = np.load("open_loop_testing/steady_state_map.npz")
    F_grid, T_R_grid = d["F"], d["T_R"]

    rows = []
    for beta_min in BETA_MINS:
        t_end = time_at_beta(schedule, beta_min, CAMPAIGN_H)
        frac_below = (CAMPAIGN_H - t_end) / CAMPAIGN_H

        best_fixed = find_best_always_feasible_point(F_grid, T_R_grid, schedule, t_end, **kw, **prices)
        adaptive_total = campaign_total_profit_adaptive(F_grid, T_R_grid, schedule, t_end, **kw, **prices)
        gap_pct = 100.0 * (adaptive_total - best_fixed["campaign_total_profit"]) / adaptive_total

        row = {
            "beta_min": beta_min,
            "t_end_h": t_end,
            "campaign_fraction_below_beta_min": frac_below,
            "fixed_F": best_fixed["F"],
            "fixed_T_R": best_fixed["T_R"],
            "fixed_total_profit": best_fixed["campaign_total_profit"],
            "adaptive_total_profit": adaptive_total,
            "gap_pct": gap_pct,
        }
        rows.append(row)
        print(f"beta_min={beta_min:.2f}  t_end={t_end:6.2f}h  frac_below={frac_below*100:5.2f}%  "
              f"fixed=(F={best_fixed['F']:.2f}, T_R={best_fixed['T_R']:.2f})  "
              f"fixed_profit={best_fixed['campaign_total_profit']:.1f}  "
              f"adaptive_profit={adaptive_total:.1f}  gap={gap_pct:.2f}%")

    return rows, CAMPAIGN_H


if __name__ == "__main__":
    start_time = time.time()
    rows, campaign_h = run()

    out_path = "open_loop_testing/beta_min_sensitivity.json"
    result = {"rows": rows, "campaign_h": campaign_h, "severity": "severe", "profile": "exponential"}
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    sidecar = save_with_provenance(
        out_path, data=None,
        params={"campaign_h": campaign_h, "profile": "exponential", "severity": "severe",
                "beta_mins": BETA_MINS},
        script=__file__, start_time=start_time, extra=result,
    )
    print(f"\nSaved {out_path}")
    print(f"Saved {sidecar}")

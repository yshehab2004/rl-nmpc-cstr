import json
import sys
import time

import numpy as np

from models.deactivation import make_deactivation_fn
from models.disturbances import build_feed_disturbance, step_disturbance
from models.params import load_params
from supervisors.campaign import run_campaign
from supervisors.tiers import RTOSupervisor, T0a_naive, T0b_prudent

PARAMS = load_params()
FEED = PARAMS["nominal_feed"]
RL = PARAMS["rl"]

CAMPAIGN_H = 75.0
BETA_FINAL = 0.20
STEP_MAG, STEP_AT = 0.6, 30.0
T0B_MARGIN = 2.0
SEED = 7000
ARM = "rl/runs_d028/A1_new_withmargin"

CASES = [
    ("T0a_naive",   None),
    ("T0b_prudent", None),
    ("T2_rto",      None),
    ("T3_best",     "sac_with_margin_seed4_w330000_ub"),
    ("T3_median",   "sac_with_margin_seed7_w330000_ub"),
    ("T3_worst",    "sac_with_margin_seed3_w330000_ub"),
]


def make_sup(kind, rl_run):
    if kind == "T0a_naive":
        return T0a_naive()
    if kind == "T0b_prudent":
        return T0b_prudent(margin=T0B_MARGIN)
    if kind == "T2_rto":
        return RTOSupervisor(cadence_h=2.0, margin=T0B_MARGIN)
    from rl.policy_supervisor import RLSupervisor
    return RLSupervisor(f"{ARM}/{rl_run}/model.zip", campaign_h=CAMPAIGN_H,
                         observe_cooling_margin=True, deterministic=True)


def main():
    start = time.time()
    out = {"scenario": "S3_severe", "campaign_h": CAMPAIGN_H,
           "beta_final": BETA_FINAL, "step_mag": STEP_MAG, "step_at_h": STEP_AT,
           "seed": SEED, "t0b_margin": T0B_MARGIN, "arm": ARM, "cases": {}}

    for kind, rl_run in CASES:
        t0 = time.time()
        ageing = make_deactivation_fn(BETA_FINAL, CAMPAIGN_H)
        dist = build_feed_disturbance(
            C_A0_gen=step_disturbance(FEED["C_A0"], STEP_MAG, STEP_AT))
        run = run_campaign(make_sup(kind, rl_run), campaign_h=CAMPAIGN_H,
                            ageing_fn=ageing, disturbance_fn=dist, seed=SEED,
                            n_horizon=20, use_estimator=True,
                            decision_interval_h=RL["decision_interval_h"])
        x = np.asarray(run["x"]); u = np.asarray(run["u"])
        out["cases"][kind] = {
            "rl_run": rl_run,
            "t":         np.asarray(run["t"]).tolist(),
            "C_b":       x[:, 1].tolist(),
            "T_R":       x[:, 2].tolist(),
            "F":         u[:, 0].tolist(),
            "Q_dot":     u[:, 1].tolist(),
            "beta_true": np.asarray(run["beta_true"]).tolist(),
            "beta_hat":  np.asarray(run["beta_hat"]).tolist(),
            "decisions": [{k: float(d[k]) for k in
                           ("t", "F_ref", "T_R_ref", "margin", "F_s", "T_R_s")}
                          for d in run["decisions"]],
            "metrics":   {k: float(v) for k, v in run["metrics"].items()},
        }
        m = run["metrics"]
        print("  %-12s profit/h=%8.1f  viol=%.4f  frac=%.3f  (%.0fs)" % (
            kind, m["total_profit"] / CAMPAIGN_H,
            m["violation_integral_degC_h"], m["fraction_time_violating"],
            time.time() - t0), flush=True)

    path = "supervisors/campaign_traces_S3.json"
    with open(path, "w") as f:
        json.dump(out, f)
    from utils.provenance import save_with_provenance
    save_with_provenance(path, None,
                          {"scenario": "S3_severe", "cases": [c[0] for c in CASES],
                           "arm": ARM, "seed": SEED, "campaign_h": CAMPAIGN_H,
                           "beta_final": BETA_FINAL, "t0b_margin": T0B_MARGIN},
                          script=__file__, start_time=start)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

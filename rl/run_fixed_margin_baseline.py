import json
import multiprocessing as mp
import time

import numpy as np

from models.params import load_params
from rl.env import RandomAgeing
from rl.run_baselines import (CAMPAIGN_H, ConstantActionSupervisor,
                              N_EPISODES, _training_feed, episode_return)
from supervisors.campaign import run_campaign
from utils.provenance import save_with_provenance

PARAMS = load_params()
RL = PARAMS["rl"]

MARGINS = (2.5, 2.0)

FIXED_TARGET = (18.25, 140.0)

EXPECTED_AT_2P5 = 325.4044358849304
REPRODUCTION_TOL = 1e-6


def _run_one(job):
    margin, ep = job
    rng = np.random.default_rng(5000 + ep)
    ageing_fn = RandomAgeing((RL["train_beta_final_low"],
                              RL["train_beta_final_high"]))(rng, CAMPAIGN_H)
    disturbance_fn = _training_feed()(rng, CAMPAIGN_H)

    supervisor = ConstantActionSupervisor(target=(*FIXED_TARGET, margin))
    t0 = time.time()
    run = run_campaign(supervisor, campaign_h=CAMPAIGN_H, ageing_fn=ageing_fn,
                       disturbance_fn=disturbance_fn, seed=5000 + ep,
                       n_horizon=20, use_estimator=True,
                       decision_interval_h=RL["decision_interval_h"])
    m = run["metrics"]
    return margin, {
        "episode": ep,
        "return": episode_return(m),
        "profit_per_hour": m["total_profit"] / CAMPAIGN_H,
        "violation_integral": m["violation_integral_degC_h"],
        "fraction_time_violating": m["fraction_time_violating"],
        "beta_final": float(run["beta_true"][-1]),
        "wall_s": time.time() - t0,
    }


def main(n_workers=None):
    t_start = time.time()
    jobs = [(mg, ep) for mg in MARGINS for ep in range(N_EPISODES)]
    n_workers = n_workers or min(len(jobs), mp.cpu_count())
    print(f"{len(jobs)} campaigns on {n_workers} workers", flush=True)

    results = {mg: [] for mg in MARGINS}
    with mp.Pool(processes=n_workers) as pool:
        for margin, row in pool.imap_unordered(_run_one, jobs):
            results[margin].append(row)
            print(f"  margin={margin} ep{row['episode']:<2} "
                  f"beta_final={row['beta_final']:.3f} "
                  f"return={row['return']:8.1f} "
                  f"profit/h={row['profit_per_hour']:8.1f} "
                  f"viol={row['violation_integral']:.4f} "
                  f"({row['wall_s']:.0f}s)", flush=True)

    out = {}
    for mg in MARGINS:
        eps = sorted(results[mg], key=lambda r: r["episode"])
        rets = [e["return"] for e in eps]
        out[f"margin_{mg}"] = {
            "margin": mg,
            "episodes": eps,
            "mean_return": float(np.mean(rets)),
            "std_return": float(np.std(rets, ddof=1)),
            "mean_profit_per_hour": float(np.mean([e["profit_per_hour"] for e in eps])),
            "mean_violation_integral": float(np.mean([e["violation_integral"] for e in eps])),
            "episodes_violating": int(sum(e["violation_integral"] > 0 for e in eps)),
        }

    print()
    print(f"{'margin':>7} {'return':>10} {'sd':>8} {'profit/h':>10} {'viol':>9} {'nviol':>6}")
    for mg in MARGINS:
        r = out[f"margin_{mg}"]
        print(f"{mg:>7} {r['mean_return']:10.2f} {r['std_return']:8.2f} "
              f"{r['mean_profit_per_hour']:10.1f} "
              f"{r['mean_violation_integral']:9.4f} "
              f"{r['episodes_violating']:5d}/{N_EPISODES}")

    got = out["margin_2.5"]["mean_return"]
    delta = abs(got - EXPECTED_AT_2P5)
    reproduced = delta < REPRODUCTION_TOL
    print()
    print(f"reproduction of baseline_scale.json `constant` at margin 2.5:")
    print(f"  expected {EXPECTED_AT_2P5:.6f}")
    print(f"  got      {got:.6f}")
    print(f"  delta    {delta:.2e}   -> {'MATCH' if reproduced else 'MISMATCH'}")
    if not reproduced:
        print("  DO NOT substitute the margin-2.0 figure into Table 5.2 until "
              "this is explained.")

    out["reproduction_check"] = {
        "expected_at_2.5": EXPECTED_AT_2P5,
        "got_at_2.5": got,
        "abs_delta": float(delta),
        "tolerance": REPRODUCTION_TOL,
        "reproduced": bool(reproduced),
    }

    path = "rl/fixed_margin_baseline.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    save_with_provenance(
        path, None,
        params={"margins": list(MARGINS), "fixed_target": list(FIXED_TARGET),
                "n_episodes": N_EPISODES, "campaign_h": CAMPAIGN_H,
                "episode_seeds": [5000 + i for i in range(N_EPISODES)],
                "decision_interval_h": RL["decision_interval_h"],
                "n_horizon": 20, "use_estimator": True},
        script=__file__, start_time=t_start,
        extra={"mean_return_at_2.0": out["margin_2.0"]["mean_return"],
               "mean_return_at_2.5": out["margin_2.5"]["mean_return"],
               "reproduction_check": out["reproduction_check"]})
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

import json
import time
from pathlib import Path

import numpy as np

from models.deactivation import make_deactivation_fn
from models.disturbances import build_feed_disturbance, step_disturbance
from models.params import load_params
from supervisors.campaign import run_campaign
from supervisors.tiers import FixedSupervisor, RTOSupervisor, T0a_naive

PARAMS = load_params()
FEED = PARAMS["nominal_feed"]
RL = PARAMS["rl"]
T_LIMIT = PARAMS["state_bounds"]["T_R"]["upper_soft"]

CAMPAIGN_H = 75.0
STEP_MAG, STEP_AT = 0.6, 30.0
T0B_MARGIN = 2.0
N_SEEDS = 5

RUNS_ROOT = "rl/runs"
RL_RUNS = ("sac_with_margin_seed1_w330000",
           "sac_with_margin_seed3_w330000",
           "sac_with_margin_seed5_w330000")


def discover_arm(arm_dir):
    from pathlib import Path as _P
    d = _P(arm_dir)
    if not d.is_dir():
        raise SystemExit(f"no such arm directory: {d}")
    runs = sorted(x.name for x in d.iterdir() if (x / "model.zip").exists())
    if not runs:
        raise SystemExit(f"no model.zip under {d}")
    return runs


def _step_feed(mag=STEP_MAG, at=STEP_AT):
    return build_feed_disturbance(
        C_A0_gen=step_disturbance(FEED["C_A0"], mag, at))


def scenarios():
    return [
        ("S1_no_ageing", CAMPAIGN_H, None, _step_feed(), "holdout_extrapolation"),
        ("S2_moderate", CAMPAIGN_H, 0.35, _step_feed(), "holdout_interpolation"),
        ("S3_severe", CAMPAIGN_H, 0.20, _step_feed(), "holdout_interpolation"),
        ("OOD_beta010_100h", 100.0, 0.10, _step_feed(), "ood"),
        ("OOD_simultaneous", CAMPAIGN_H, 0.25,
         build_feed_disturbance(
             C_A0_gen=step_disturbance(FEED["C_A0"], STEP_MAG, STEP_AT),
             T_in_gen=step_disturbance(FEED["T_in"], 3.0, STEP_AT)), "ood"),
    ]


def _make_tier(tier, rl_run=None, campaign_h=CAMPAIGN_H, runs_root=None):
    if tier == "T0a_naive":
        return T0a_naive()
    if tier == "T0b_prudent":
        return FixedSupervisor(18.25, 140.0, T0B_MARGIN, name="T0b_prudent")
    if tier == "T2_rto":
        return RTOSupervisor(cadence_h=2.0, margin=T0B_MARGIN)
    if tier == "T3_rl":
        from rl.policy_supervisor import RLSupervisor
        root = runs_root or RUNS_ROOT
        return RLSupervisor(f"{root}/{rl_run}/model.zip", campaign_h=CAMPAIGN_H,
                             observe_cooling_margin=True, deterministic=True,
                             allow_overrun=(campaign_h > CAMPAIGN_H))
    raise ValueError(tier)


def _run_one(job):
    (name, horizon, beta_final, dist_fn, kind, tier, rl_run, seed,
     runs_root) = job
    ageing_fn = (None if beta_final is None
                 else make_deactivation_fn(beta_final=beta_final,
                                            campaign_h=horizon))
    sup = _make_tier(tier, rl_run, horizon, runs_root=runs_root)
    t0 = time.time()
    run = run_campaign(sup, campaign_h=horizon, ageing_fn=ageing_fn,
                        disturbance_fn=dist_fn, seed=seed, n_horizon=20,
                        decision_interval_h=RL["decision_interval_h"])
    m = run["metrics"]
    label = tier if rl_run is None else f"{tier}[{rl_run.split('_')[-2]}]"
    return name, tier, {
        "seed": seed, "rl_run": rl_run, "label": label,
        "profit_per_hour": m["total_profit"] / horizon,
        "violation_integral": m["violation_integral_degC_h"],
        "violation_max": m["violation_max"],
        "fraction_time_violating": m["fraction_time_violating"],
        "wall_s": time.time() - t0,
    }


def main():
    import multiprocessing as mp

    jobs = []
    for name, horizon, beta_final, dist_fn, kind in scenarios():
        for tier in ("T0a_naive", "T0b_prudent", "T2_rto"):
            for s in range(N_SEEDS):
                jobs.append((name, horizon, beta_final, dist_fn, kind, tier,
                              None, 7000 + s, RUNS_ROOT))
        for rl_run in RL_RUNS:
            for s in range(N_SEEDS):
                jobs.append((name, horizon, beta_final, dist_fn, kind, "T3_rl",
                              rl_run, 7000 + s, RUNS_ROOT))

    n = min(len(jobs), mp.cpu_count())
    print(f"{len(jobs)} campaigns on {n} workers", flush=True)
    raw = {}
    with mp.Pool(processes=n) as pool:
        for name, tier, row in pool.imap_unordered(_run_one, jobs):
            raw.setdefault(name, {}).setdefault(tier, []).append(row)
            print(f"  {name:18} {row['label']:14} seed{row['seed']} "
                  f"profit/h={row['profit_per_hour']:8.1f} "
                  f"viol={row['violation_integral']:.4f} "
                  f"({row['wall_s']:.0f}s)", flush=True)

    results = {}
    kinds = {n: k for n, _, _, _, k in scenarios()}
    for name, tiers in raw.items():
        results[name] = {"kind": kinds[name], "tiers": {}}
        for tier, rows in tiers.items():
            p = [r["profit_per_hour"] for r in rows]
            v = [r["violation_integral"] for r in rows]
            results[name]["tiers"][tier] = {
                "n": len(rows), "runs": rows,
                "mean_profit_per_hour": float(np.mean(p)),
                "std_profit_per_hour": float(np.std(p)),
                "mean_violation_integral": float(np.mean(v)),
                "max_violation_integral": float(max(v)),
                "mean_fraction_time_violating":
                    float(np.mean([r["fraction_time_violating"] for r in rows])),
                "runs_violating": int(sum(x > 0 for x in v)),
            }
    return results


if __name__ == "__main__":
    import argparse

    from utils.provenance import save_with_provenance

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", default=None,
                    help="D-028 arm directory, e.g. rl/runs_d028/A1_new_withmargin. "
                         "Every seed under it is evaluated. Omit to reproduce the "
                         "pre-retrain run from rl/runs.")
    ap.add_argument("--tag", default=None,
                    help="suffix for the output filename; defaults to the arm name")
    args = ap.parse_args()

    if args.arm:
        RUNS_ROOT = args.arm.rstrip("/")
        RL_RUNS = tuple(discover_arm(RUNS_ROOT))
        TAG = args.tag or Path(RUNS_ROOT).name
        print(f"arm {RUNS_ROOT}: {len(RL_RUNS)} models", flush=True)
    else:
        TAG = args.tag or "legacy"

    start = time.time()
    results = main()

    order = [n for n, _, _, _, _ in scenarios()]
    tiers = ("T0a_naive", "T0b_prudent", "T2_rto", "T3_rl")

    print("\n" + "=" * 92)
    print("PHASE 3 -- profit and safety reported SEPARATELY (D-013)")
    print("=" * 92)
    for name in order:
        r = results[name]
        print(f"\n{name}   [{r['kind']}]")
        print(f"{'tier':>14} {'profit/h':>10} {'+/-':>8} {'viol degC.h':>12} "
              f"{'runs violating':>15}")
        for t in tiers:
            m = r["tiers"].get(t)
            if not m:
                continue
            print(f"{t:>14} {m['mean_profit_per_hour']:10.1f} "
                  f"{m['std_profit_per_hour']:8.1f} "
                  f"{m['mean_violation_integral']:12.4f} "
                  f"{m['runs_violating']:8d}/{m['n']}")

    print("\n" + "=" * 92)
    print("THE THESIS QUESTION: T3 (RL) against T2 (RTO)")
    print("=" * 92)
    for name in order:
        t2 = results[name]["tiers"].get("T2_rto")
        t3 = results[name]["tiers"].get("T3_rl")
        if not (t2 and t3):
            continue
        d = 100.0 * (t3["mean_profit_per_hour"] - t2["mean_profit_per_hour"]) \
            / t2["mean_profit_per_hour"]
        print(f"{name:>20}  T3 {t3['mean_profit_per_hour']:8.1f} vs "
              f"T2 {t2['mean_profit_per_hour']:8.1f}   {d:+6.1f}%   "
              f"viol {t3['mean_violation_integral']:.4f} vs "
              f"{t2['mean_violation_integral']:.4f}")
    print("\nRead with the seed spread (+/- column): a difference smaller than")
    print("it is not a result. And profit only counts among tiers whose")
    print("violations are comparable -- D-013 is a GATE, not a tiebreak.")

    print("\n" + "=" * 92)
    print("THE GATE: did T3 generalise, or memorise?")
    print("=" * 92)
    print("in-distribution reference (rl/policy_eval.json, training "
          "randomisation): 2352.0 profit/h")
    for name in order:
        t3 = results[name]["tiers"].get("T3_rl")
        if not t3:
            continue
        rel = 100.0 * (t3["mean_profit_per_hour"] - 2352.0) / 2352.0
        print(f"{name:>20}  [{results[name]['kind']:<22}] "
              f"{t3['mean_profit_per_hour']:8.1f}  {rel:+6.1f}% vs in-distribution")
    print("\nMarkedly worse on the HOLDOUTS than in-distribution => memorised.")
    print("S1 is EXTRAPOLATION (training ages every episode), so it is not")
    print("comparable to S2/S3 and a deficit there means something different.")

    out = f"supervisors/phase3_results_{TAG}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    sidecar = save_with_provenance(
        out, data=None,
        params={"scenarios": order, "n_seeds": N_SEEDS,
                "eval_seeds": [7000 + s for s in range(N_SEEDS)],
                "t0b_margin": T0B_MARGIN, "rl_runs": list(RL_RUNS),
                "runs_root": RUNS_ROOT, "arm": args.arm,
                "step_magnitude": STEP_MAG, "step_at_h": STEP_AT,
                "campaign_h": CAMPAIGN_H,
                "decision_interval_h": RL["decision_interval_h"]},
        script=__file__, start_time=start, extra={"results": results})
    print(f"\nSaved {out}, {sidecar}")

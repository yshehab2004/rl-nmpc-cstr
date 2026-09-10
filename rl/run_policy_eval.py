import json
import time
from pathlib import Path

import numpy as np

from models.params import load_params
from rl.env import RandomAgeing
from rl.policy_supervisor import RLSupervisor
from rl.run_baselines import CAMPAIGN_H, episode_return
from rl.train import _training_feed
from supervisors.campaign import run_campaign

RL = load_params()["rl"]
N_EPISODES = 10
RUNS_DIR = Path("rl/runs")


def _policies():
    out = []
    for d in sorted(RUNS_DIR.iterdir()):
        model = d / "model.zip"
        meta_path = d / "train_meta.json"
        if not model.exists():
            continue
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        out.append({
            "name": d.name,
            "model": str(model),
            "observe_cooling_margin": meta.get("observe_cooling_margin", True),
            "train_weight": meta.get("safety_penalty_weight"),
            "lagrangian": meta.get("lagrangian", False),
            "final_weight": meta.get("lambda_final") or meta.get("safety_penalty_weight"),
        })
    return out


def _run_one(job):
    spec, ep = job
    rng = np.random.default_rng(5000 + ep)
    ageing_fn = RandomAgeing((RL["train_beta_final_low"],
                              RL["train_beta_final_high"]))(rng, CAMPAIGN_H)
    disturbance_fn = _training_feed()(rng, CAMPAIGN_H)

    sup = RLSupervisor(spec["model"], campaign_h=CAMPAIGN_H,
                        observe_cooling_margin=spec["observe_cooling_margin"],
                        deterministic=True)
    t0 = time.time()
    run = run_campaign(sup, campaign_h=CAMPAIGN_H, ageing_fn=ageing_fn,
                        disturbance_fn=disturbance_fn, seed=5000 + ep,
                        n_horizon=20, use_estimator=True,
                        decision_interval_h=RL["decision_interval_h"])
    m = run["metrics"]
    margins = [d["margin"] for d in run["decisions"]]
    return spec["name"], {
        "episode": ep,
        "profit_per_hour": m["total_profit"] / CAMPAIGN_H,
        "violation_integral": m["violation_integral_degC_h"],
        "violation_max": m["violation_max"],
        "fraction_time_violating": m["fraction_time_violating"],
        "return_at_250": episode_return(m),
        "mean_margin_requested": float(np.mean(margins)),
        "beta_final": float(run["beta_true"][-1]),
        "wall_s": time.time() - t0,
    }


def main(n_workers=None):
    import multiprocessing as mp

    specs = _policies()
    if not specs:
        raise SystemExit(f"no trained models under {RUNS_DIR}")
    jobs = [(s, ep) for s in specs for ep in range(N_EPISODES)]
    n_workers = n_workers or min(len(jobs), mp.cpu_count())
    print(f"{len(specs)} policies x {N_EPISODES} episodes "
          f"= {len(jobs)} campaigns on {n_workers} workers", flush=True)

    rows = {s["name"]: [] for s in specs}
    with mp.Pool(processes=n_workers) as pool:
        for name, row in pool.imap_unordered(_run_one, jobs):
            rows[name].append(row)
            print(f"  {name:34} ep{row['episode']:<2} "
                  f"profit/h={row['profit_per_hour']:8.1f} "
                  f"viol={row['violation_integral']:.4f} "
                  f"margin={row['mean_margin_requested']:.2f} "
                  f"({row['wall_s']:.0f}s)", flush=True)

    results = {}
    for s in specs:
        eps = sorted(rows[s["name"]], key=lambda r: r["episode"])
        results[s["name"]] = {
            **{k: s[k] for k in ("train_weight", "final_weight",
                                  "lagrangian", "observe_cooling_margin")},
            "episodes": eps,
            "mean_profit_per_hour": float(np.mean([e["profit_per_hour"] for e in eps])),
            "std_profit_per_hour": float(np.std([e["profit_per_hour"] for e in eps])),
            "mean_violation_integral": float(np.mean([e["violation_integral"] for e in eps])),
            "max_violation_integral": float(max(e["violation_integral"] for e in eps)),
            "max_violation_peak": float(max(e["violation_max"] for e in eps)),
            "episodes_violating": int(sum(e["violation_integral"] > 0 for e in eps)),
            "mean_margin_requested": float(np.mean([e["mean_margin_requested"] for e in eps])),
        }
    return results


if __name__ == "__main__":
    from utils.provenance import save_with_provenance

    start = time.time()
    results = main()

    print("\n" + "=" * 96)
    print("TRAINED POLICIES ON ONE SCALE -- profit and safety reported "
          "SEPARATELY (D-013)")
    print("=" * 96)
    print(f"{'policy':>32} {'train w':>10} {'profit/h':>10} {'+/-':>7} "
          f"{'viol':>8} {'eps viol':>9} {'margin':>7}")
    for name, r in sorted(results.items(),
                           key=lambda kv: (kv[1]["final_weight"] or 0)):
        w = r["final_weight"]
        wtxt = f"{w:.0f}" + ("*" if r["lagrangian"] else "")
        print(f"{name:>32} {wtxt:>10} {r['mean_profit_per_hour']:10.1f} "
              f"{r['std_profit_per_hour']:7.1f} "
              f"{r['mean_violation_integral']:8.4f} "
              f"{r['episodes_violating']:5d}/{N_EPISODES} "
              f"{r['mean_margin_requested']:7.2f}")
    print("\n* = weight solved for by the Lagrangian, not chosen")
    print("\nThe rule (D-011, applied to the price): take the SMALLEST weight")
    print("achieving the constraint. Read the 'eps viol' column for that, and")
    print("'profit/h' for what it costs.")

    out = "rl/policy_eval.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    sidecar = save_with_provenance(
        out, data=None,
        params={"n_episodes": N_EPISODES, "campaign_h": CAMPAIGN_H,
                "episode_seeds": [5000 + i for i in range(N_EPISODES)],
                "decision_interval_h": RL["decision_interval_h"],
                "deterministic": True, "use_estimator": True},
        script=__file__, start_time=start, extra={"results": results})
    print(f"\nSaved {out}, {sidecar}")

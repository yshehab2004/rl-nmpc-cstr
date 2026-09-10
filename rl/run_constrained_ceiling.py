import json
import time

import numpy as np

from models.params import load_params
from open_loop_testing.optimum_trajectory import find_optimum
from rl.env import RandomAgeing
from rl.run_baselines import CAMPAIGN_H, N_EPISODES, episode_return
from rl.train import _training_feed
from supervisors.campaign import run_campaign

PARAMS = load_params()
C = PARAMS["certain_params"]
ECON = PARAMS["economics"]
PRICES = dict(price_B=ECON["price_B"], price_A0=ECON["price_A0"],
              price_energy=ECON["price_energy"])
T_LIMIT = PARAMS["state_bounds"]["T_R"]["upper_soft"]
ALPHA = PARAMS["uncertain_params"]["alpha_nominal"]
RL = PARAMS["rl"]

MARGINS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)


class ConstrainedOptimumSupervisor:

    def __init__(self, margin):
        self.margin = float(margin)
        self.name = f"ceiling_m{margin}"
        self.F_grid = np.linspace(5.0, 35.0, 121)
        self.T_R_grid = np.linspace(100.0, 150.0, 201)
        self.n_solves = 0
        self._cache = {}

    def get_target(self, t_now, beta_hat=None, C_A0=None, T_in=None, **kw):
        self.n_solves += 1
        key = (round(float(beta_hat), 3), round(float(C_A0), 2),
               round(float(T_in), 1))
        if key not in self._cache:
            opt = find_optimum(self.F_grid, self.T_R_grid, key[0], ALPHA,
                                key[1], key[2], C, **PRICES,
                                T_R_max=T_LIMIT - self.margin)
            self._cache[key] = (opt["F_star"], opt["T_R_star"], self.margin)
        return self._cache[key]


USE_ESTIMATOR = False


def _run_one(job):
    margin, ep = job
    rng = np.random.default_rng(5000 + ep)
    ageing_fn = RandomAgeing((RL["train_beta_final_low"],
                              RL["train_beta_final_high"]))(rng, CAMPAIGN_H)
    disturbance_fn = _training_feed()(rng, CAMPAIGN_H)
    t0 = time.time()
    run = run_campaign(ConstrainedOptimumSupervisor(margin),
                        campaign_h=CAMPAIGN_H, ageing_fn=ageing_fn,
                        disturbance_fn=disturbance_fn, seed=5000 + ep,
                        n_horizon=20, use_estimator=USE_ESTIMATOR,
                        decision_interval_h=RL["decision_interval_h"])
    m = run["metrics"]
    return margin, {"episode": ep, "return": episode_return(m),
                     "profit_per_hour": m["total_profit"] / CAMPAIGN_H,
                     "violation_integral": m["violation_integral_degC_h"],
                     "violation_max": m["violation_max"],
                     "beta_final": float(run["beta_true"][-1]),
                     "wall_s": time.time() - t0}


def main(n_workers=None):
    global MARGINS, N_EPISODES, USE_ESTIMATOR
    import multiprocessing as mp

    jobs = [(m, ep) for m in MARGINS for ep in range(N_EPISODES)]
    n_workers = n_workers or min(len(jobs), mp.cpu_count())
    print(f"{len(jobs)} campaigns on {n_workers} workers", flush=True)

    by_margin = {m: [] for m in MARGINS}
    with mp.Pool(processes=n_workers) as pool:
        for margin, row in pool.imap_unordered(_run_one, jobs):
            by_margin[margin].append(row)
            print(f"  m={margin:.1f} ep{row['episode']:<2} "
                  f"return={row['return']:8.1f} "
                  f"profit/h={row['profit_per_hour']:8.1f} "
                  f"viol={row['violation_integral']:.4f} "
                  f"({row['wall_s']:.0f}s)", flush=True)

    results = {}
    for m in MARGINS:
        rows = sorted(by_margin[m], key=lambda r: r["episode"])
        results[f"margin_{m}"] = {
            "margin": m, "episodes": rows,
            "mean_return": float(np.mean([r["return"] for r in rows])),
            "std_return": float(np.std([r["return"] for r in rows])),
            "mean_profit_per_hour": float(np.mean([r["profit_per_hour"] for r in rows])),
            "mean_violation_integral": float(np.mean([r["violation_integral"] for r in rows])),
            "max_violation_integral": float(max(r["violation_integral"] for r in rows)),
            "max_violation_peak": float(max(r["violation_max"] for r in rows)),
            "episodes_violating": int(sum(r["violation_integral"] > 0 for r in rows)),
        }
    return results


if __name__ == "__main__":
    import argparse

    from utils.provenance import save_with_provenance

    ap = argparse.ArgumentParser()
    ap.add_argument("--margins", type=float, nargs="+", default=list(MARGINS))
    ap.add_argument("--episodes", type=int, default=N_EPISODES)
    ap.add_argument("--estimator", action="store_true",
                    help="run on the EKF estimate rather than the true beta. "
                         "The gap to the perfect-information frontier is the "
                         "PRICE OF ESTIMATION, and it is also the reference a "
                         "real tier's margin must be derived against -- an "
                         "oracle's sufficient margin is a LOWER BOUND on what "
                         "a tier running on an estimate needs (F-038 measured "
                         "the EKF's lag at 0.040 in beta; D-011 already "
                         "required evaluating at beta_hat - 0.04).")
    ap.add_argument("--out", default="rl/constrained_ceiling.json")
    args = ap.parse_args()
    MARGINS = tuple(args.margins)
    N_EPISODES = args.episodes
    USE_ESTIMATOR = args.estimator

    start = time.time()
    results = main()

    print("\n" + "=" * 76)
    info = "PERFECT INFORMATION" if not USE_ESTIMATOR else "AN ESTIMATED beta"
    print(f"PRICE OF SAFETY UNDER {info}")
    print("=" * 76)
    print(f"{'margin':>7} {'return':>9} {'sd':>7} {'profit/h':>10} "
          f"{'max viol':>9} {'eps viol':>9}")
    for m in MARGINS:
        r = results[f"margin_{m}"]
        print(f"{m:7.1f} {r['mean_return']:9.1f} {r['std_return']:7.1f} "
              f"{r['mean_profit_per_hour']:10.1f} "
              f"{r['max_violation_integral']:9.4f} "
              f"{r['episodes_violating']:5d}/{N_EPISODES}")

    clean = [m for m in MARGINS if results[f"margin_{m}"]["episodes_violating"] == 0]
    unconstrained = results[f"margin_{MARGINS[0]}"]["mean_return"]
    if clean:
        m_star = min(clean)
        constrained = results[f"margin_{m_star}"]["mean_return"]
        results["constrained_ceiling"] = {
            "margin": m_star, "return": constrained,
            "profit_per_hour": results[f"margin_{m_star}"]["mean_profit_per_hour"],
            "unconstrained_return": unconstrained,
            "price_of_safety_return": unconstrained - constrained,
            "price_of_safety_pct": 100.0 * (unconstrained - constrained) / unconstrained,
        }
        print(f"\nCONSTRAINED CEILING: margin {m_star:.1f} degC -> return "
              f"{constrained:.1f} ({results[f'margin_{m_star}']['mean_profit_per_hour']:.1f} profit/h)")
        print(f"  smallest margin with ZERO violation in all {N_EPISODES} episodes")
        print(f"\nPRICE OF SAFETY (under {info}): "
              f"{unconstrained - constrained:.1f} return units, "
              f"{100.0 * (unconstrained - constrained) / unconstrained:.1f}%")
        if USE_ESTIMATOR:
            print("  This is the cost of honouring the limit WITH an estimator,")
            print("  so it bundles the constraint and the estimator's lag. The")
            print("  PRICE OF ESTIMATION is this frontier against the")
            print("  perfect-information one at MATCHED episode counts -- the")
            print("  two must not be differenced across different episode sets.")
        else:
            print("  what honouring the limit costs when beta is known exactly --")
            print("  the cost of the CONSTRAINT, with the cost of ESTIMATION removed.")
    else:
        print("\nNo margin in the swept range achieves zero violation in every")
        print("episode. Either the range is too narrow or margin alone cannot")
        print("deliver the constraint -- which is OQ-013's question.")

    out = args.out
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    sidecar = save_with_provenance(
        out, data=None,
        params={"margins": list(MARGINS), "n_episodes": N_EPISODES,
                "campaign_h": CAMPAIGN_H, "use_estimator": USE_ESTIMATOR,
                "decision_interval_h": RL["decision_interval_h"],
                "episode_seeds": [5000 + i for i in range(N_EPISODES)]},
        script=__file__, start_time=start, extra={"results": results})
    print(f"\nSaved {out}, {sidecar}")

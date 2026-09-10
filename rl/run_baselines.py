import json
import time

import numpy as np

from models.params import load_params
from open_loop_testing.optimum_trajectory import find_optimum
from rl.env import ActionMapper, RandomAgeing
from rl.train import _training_feed
from supervisors.campaign import run_campaign

PARAMS = load_params()
C = PARAMS["certain_params"]
ECON = PARAMS["economics"]
PRICES = dict(price_B=ECON["price_B"], price_A0=ECON["price_A0"],
              price_energy=ECON["price_energy"])
W = ECON["safety_penalty_weight"]
T_LIMIT = PARAMS["state_bounds"]["T_R"]["upper_soft"]
ALPHA = PARAMS["uncertain_params"]["alpha_nominal"]
RL = PARAMS["rl"]
CAMPAIGN_H = 75.0
N_EPISODES = 10


def episode_return(metrics):
    return (metrics["total_profit"]
            - W * metrics["violation_integral_degC_h"]) / (
                RL["decision_interval_h"] * 1000.0)


class RandomActionSupervisor:

    name = "random"

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.mapper = ActionMapper()
        self.n_solves = 0

    def get_target(self, t_now, **kw):
        self.n_solves += 1
        return self.mapper.to_target(self.rng.uniform(-1.0, 1.0, size=3))


class ConstantActionSupervisor:

    name = "constant"

    def __init__(self, target=(18.25, 140.0, 2.5)):
        self.target = tuple(float(v) for v in target)
        self.n_solves = 0

    def get_target(self, t_now, **kw):
        self.n_solves += 1
        return self.target


class PerfectOptimumSupervisor:

    name = "ceiling"

    def __init__(self):
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
                                key[1], key[2], C, **PRICES, T_R_max=T_LIMIT)
            self._cache[key] = (opt["F_star"], opt["T_R_star"], 0.0)
        return self._cache[key]


def _build(policy_name, seed):
    if policy_name == "random":
        return RandomActionSupervisor(seed), True
    if policy_name == "constant":
        return ConstantActionSupervisor(), True
    if policy_name == "ceiling":
        return PerfectOptimumSupervisor(), False
    raise ValueError(policy_name)


def _run_one(job):
    policy_name, ep = job
    rng = np.random.default_rng(5000 + ep)
    ageing_fn = RandomAgeing((RL["train_beta_final_low"],
                              RL["train_beta_final_high"]))(rng, CAMPAIGN_H)
    disturbance_fn = _training_feed()(rng, CAMPAIGN_H)

    supervisor, use_est = _build(policy_name, seed=5000 + ep)
    t0 = time.time()
    run = run_campaign(supervisor, campaign_h=CAMPAIGN_H, ageing_fn=ageing_fn,
                        disturbance_fn=disturbance_fn, seed=5000 + ep,
                        n_horizon=20, use_estimator=use_est,
                        decision_interval_h=RL["decision_interval_h"])
    m = run["metrics"]
    return policy_name, {
        "episode": ep,
        "return": episode_return(m),
        "profit_per_hour": m["total_profit"] / CAMPAIGN_H,
        "violation_integral": m["violation_integral_degC_h"],
        "fraction_time_violating": m["fraction_time_violating"],
        "beta_final": float(run["beta_true"][-1]),
        "wall_s": time.time() - t0,
    }


def main(n_workers=None):
    import multiprocessing as mp

    names = ["random", "constant", "ceiling"]
    jobs = [(n, ep) for n in names for ep in range(N_EPISODES)]
    n_workers = n_workers or min(len(jobs), mp.cpu_count())
    print(f"{len(jobs)} campaigns on {n_workers} workers", flush=True)

    results = {n: {"episodes": []} for n in names}
    with mp.Pool(processes=n_workers) as pool:
        for name, row in pool.imap_unordered(_run_one, jobs):
            results[name]["episodes"].append(row)
            print(f"  {name:>9} ep{row['episode']:<2} "
                  f"beta_final={row['beta_final']:.3f} "
                  f"return={row['return']:8.1f} "
                  f"profit/h={row['profit_per_hour']:8.1f} "
                  f"viol={row['violation_integral']:.4f} "
                  f"({row['wall_s']:.0f}s)", flush=True)

    for name in names:
        rows = sorted(results[name]["episodes"], key=lambda r: r["episode"])
        results[name]["episodes"] = rows
        for key, field in (("mean_return", "return"),
                            ("mean_profit_per_hour", "profit_per_hour"),
                            ("mean_violation_integral", "violation_integral")):
            results[name][key] = float(np.mean([r[field] for r in rows]))
        results[name]["std_return"] = float(np.std([r["return"] for r in rows]))
    return results


if __name__ == "__main__":
    from utils.provenance import save_with_provenance

    start = time.time()
    results = main()

    print("\n" + "=" * 72)
    print("THE SCALE  (training distribution, identical episodes)")
    print("=" * 72)
    print(f"{'policy':>12} {'return':>10} {'+/- sd':>8} {'profit/h':>11} {'viol degC.h':>13}")
    for name in ("random", "constant", "ceiling"):
        r = results[name]
        print(f"{name:>12} {r['mean_return']:10.1f} {r['std_return']:8.1f} "
              f"{r['mean_profit_per_hour']:11.1f} "
              f"{r['mean_violation_integral']:13.4f}")
    lo, hi = results["random"]["mean_return"], results["ceiling"]["mean_return"]
    span = hi - lo
    print(f"\nrandom -> ceiling span: {span:.1f} return units")
    print(f"trained agent plateaued at 741 (pilot 11471382)")
    if span > 1e-9:
        print(f"  -> agent sits {100 * (741 - lo) / span:.0f}% of the way "
              f"from random to the perfect-information ceiling")
    print("\nIf that span is small, the decision problem is flat and no")
    print("supervisory result can mean much. If it is large, the spread is")
    print("real and a policy that climbs it has done something.")

    out = "rl/baseline_scale.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    sidecar = save_with_provenance(
        out, data=None,
        params={"campaign_h": CAMPAIGN_H, "n_episodes": N_EPISODES,
                "decision_interval_h": RL["decision_interval_h"],
                "safety_penalty_weight": W, "n_horizon": 20,
                "episode_seeds": [5000 + i for i in range(N_EPISODES)]},
        script=__file__, start_time=start, extra={"results": results})
    print(f"\nSaved {out}, {sidecar}")

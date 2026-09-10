import argparse
import json
import time
from pathlib import Path

import numpy as np

from models.params import load_params

PARAMS = load_params()
RL = PARAMS["rl"]


class LagrangianWeightCallback:

    def __init__(self, budget, intervals_per_episode, eta=0.15, scale=0.01,
                  w_min=250.0, w_max=1.0e6, w0=1000.0, update_every=4800,
                  tag=""):
        self.budget = float(budget)
        self.intervals_per_episode = float(intervals_per_episode)
        self.eta, self.scale = float(eta), float(scale)
        self.w_min, self.w_max = float(w_min), float(w_max)
        self.weight = float(w0)
        self.update_every = int(update_every)
        self.tag = tag
        self.history = []

    def attach(self, model, vec):
        from stable_baselines3.common.callbacks import BaseCallback

        outer = self

        class _CB(BaseCallback):
            def _on_step(self) -> bool:
                if self.num_timesteps % outer.update_every != 0:
                    return True
                vs = [i["violation_integral"] for i in self.locals.get("infos", [])
                      if isinstance(i, dict) and "violation_integral" in i]
                if not vs:
                    return True
                viol = float(np.mean(vs)) * outer.intervals_per_episode
                err = (viol - outer.budget) / outer.scale
                outer.weight = float(np.clip(
                    outer.weight * np.exp(outer.eta * np.clip(err, -3.0, 3.0)),
                    outer.w_min, outer.w_max))
                vec.env_method("set_safety_weight", outer.weight)
                margins = [i["margin"] for i in self.locals.get("infos", [])
                           if isinstance(i, dict) and "margin" in i]
                outer.history.append({"timesteps": int(self.num_timesteps),
                                       "violation": viol,
                                       "weight": outer.weight,
                                       "mean_margin_requested":
                                           float(np.mean(margins)) if margins else None})
                print(f"[{outer.tag}] lambda: viol/ep={viol:.5f} "
                      f"budget={outer.budget:.5f} -> weight={outer.weight:.0f}",
                      flush=True)
                return True

        return _CB()


def _training_feed():
    from rl.env import RandomFeedDisturbance

    return RandomFeedDisturbance(
        C_A0_noise_std=RL["feed_noise_std"],
        T_in_noise_std=RL["train_T_in_noise_std"],
        noise_tau_h=RL["feed_noise_tau_h"],
        p_event=RL["train_p_event"],
        C_A0_mag_range=(RL["train_C_A0_event_mag_low"],
                        RL["train_C_A0_event_mag_high"]),
        T_in_mag_range=(RL["train_T_in_event_mag_low"],
                        RL["train_T_in_event_mag_high"]),
        step_onset_h=tuple(RL["train_step_onset_h"]),
        ramp_onset_h=tuple(RL["train_ramp_onset_h"]),
        ramp_duration_h=tuple(RL["train_ramp_duration_h"]))


def build_vec_env(n_envs, seed, campaign_h, observe_cooling_margin, n_horizon,
                    safety_weight=None, monitor_path=None,
                    uniform_in_beta=False, margin_hi=None):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

    from rl.env import RandomAgeing, make_env

    fns = [make_env(rank=i, campaign_h=campaign_h,
                     decision_interval_h=RL["decision_interval_h"],
                     observe_cooling_margin=observe_cooling_margin,
                     n_horizon=n_horizon, seed=seed,
                     safety_penalty_weight=safety_weight,
                     ageing=RandomAgeing((RL["train_beta_final_low"],
                                           RL["train_beta_final_high"]),
                                          uniform_in_beta=uniform_in_beta),
                     margin_hi=margin_hi,
                     feed=_training_feed())
            for i in range(n_envs)]
    vec = DummyVecEnv(fns) if n_envs == 1 else SubprocVecEnv(fns)
    return VecMonitor(vec, filename=monitor_path)


def random_policy_baseline(campaign_h, seed, observe_cooling_margin, n_horizon,
                            safety_weight=None,                            n_episodes=3):
    from rl.env import RandomAgeing, SupervisoryEnv

    env = SupervisoryEnv(campaign_h=campaign_h,
                          decision_interval_h=RL["decision_interval_h"],
                          observe_cooling_margin=observe_cooling_margin,
                          n_horizon=n_horizon,
                          safety_penalty_weight=safety_weight,
                          ageing=RandomAgeing((RL["train_beta_final_low"],
                                                RL["train_beta_final_high"])),
                          feed=_training_feed())
    returns, profits, violations = [], [], []
    for ep in range(n_episodes):
        env.reset(seed=seed + 1000 + ep)
        total_r = total_p = total_v = 0.0
        done = False
        while not done:
            obs, r, terminated, truncated, info = env.step(env.action_space.sample())
            total_r += r
            total_p += info["profit"]
            total_v += info["violation_integral"]
            done = terminated or truncated
        returns.append(total_r)
        profits.append(total_p / campaign_h)
        violations.append(total_v)
    return {"mean_return": float(np.mean(returns)),
            "mean_profit_per_hour": float(np.mean(profits)),
            "mean_violation_integral": float(np.mean(violations)),
            "n_episodes": n_episodes}


def fixed_tier_baseline(campaign_h, seed, observe_cooling_margin, n_horizon,
                            safety_weight=None,                         n_episodes=3):
    from supervisors.tiers import T0b_prudent
    from rl.env import ActionMapper, RandomAgeing, SupervisoryEnv

    mapper = ActionMapper()
    F_ref, T_R_ref, margin = T0b_prudent().get_target(0.0)
    action = np.array([
        2.0 * (F_ref - mapper.F_lo) / (mapper.F_hi - mapper.F_lo) - 1.0,
        2.0 * (T_R_ref - mapper.T_R_lo) / (mapper.T_R_hi - mapper.T_R_lo) - 1.0,
        2.0 * (margin - mapper.margin_lo) / (mapper.margin_hi - mapper.margin_lo) - 1.0,
    ], dtype=np.float32)

    env = SupervisoryEnv(campaign_h=campaign_h,
                          decision_interval_h=RL["decision_interval_h"],
                          observe_cooling_margin=observe_cooling_margin,
                          n_horizon=n_horizon,
                          safety_penalty_weight=safety_weight,
                          ageing=RandomAgeing((RL["train_beta_final_low"],
                                                RL["train_beta_final_high"])),
                          feed=_training_feed())
    returns, profits, violations = [], [], []
    for ep in range(n_episodes):
        env.reset(seed=seed + 2000 + ep)
        total_r = total_p = total_v = 0.0
        done = False
        while not done:
            _, r, terminated, truncated, info = env.step(action)
            total_r += r
            total_p += info["profit"]
            total_v += info["violation_integral"]
            done = terminated or truncated
        returns.append(total_r)
        profits.append(total_p / campaign_h)
        violations.append(total_v)
    return {"mean_return": float(np.mean(returns)),
            "mean_profit_per_hour": float(np.mean(profits)),
            "mean_violation_integral": float(np.mean(violations)),
            "target": [F_ref, T_R_ref, margin], "n_episodes": n_episodes}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--campaign-h", type=float, default=75.0)
    p.add_argument("--n-horizon", type=int, default=20)
    p.add_argument("--algo", choices=("sac", "ppo"), default="sac")
    p.add_argument("--no-cooling-margin", action="store_true",
                    help="ablation arm: withhold cooling margin from the observation")
    p.add_argument("--out", default="rl/runs")
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--uniform-in-beta", action="store_true",
                    help="sample a STARTING activity as well as a final one, so "
                         "training time is spread across beta instead of "
                         "concentrated where exponential decay happens to "
                         "linger. Legacy sampling gives 0.48%% of time below "
                         "beta=0.20; this gives ~10%%.")
    p.add_argument("--margin-hi", type=float, default=None,
                    help="ceiling of the margin action, degC (default 5.0). "
                         "Recorded in train_meta.json because it DEFINES what "
                         "an action means -- a policy trained at 5.0 and "
                         "decoded at 8.0 has every margin it requests inflated "
                         "by 60%%. DELIBERATELY NOT RAISED despite 9 of 24 "
                         "low-beta probes pinning at 5.0: F-051 measured that "
                         "margin >= 2.0 already unsaturates the cooler and "
                         "drives violation to exactly zero there, so 5.0 is "
                         "2.5x what the physics needs. A policy at the ceiling "
                         "is saying it does not know what to do, not that it "
                         "needs more room -- a symptom of the coverage defect, "
                         "not a separate one. Raising it would give a confused "
                         "policy more scope to lose profit AND confound the "
                         "coverage experiment.")
    p.add_argument("--gamma", type=float, default=0.99,
                    help="discount. 0.99 gives an effective horizon of 100 "
                         "steps = 25h of a 75h campaign, so the agent plans a "
                         "third of the way through a decay it is supposed to "
                         "anticipate.")
    p.add_argument("--lagrangian", action="store_true",
                    help="solve for the violation price rather than fixing it: "
                         "adapt the multiplier until the violation budget is met")
    p.add_argument("--violation-budget", type=float, default=0.001,
                    help="degC.h of violation allowed per 75h episode. Small "
                         "but non-zero so the price can RELAX when the "
                         "constraint is comfortably met; at exactly zero the "
                         "multiplier can only ever increase.")
    p.add_argument("--safety-weight", type=float, default=None,
                    help="override economics.safety_penalty_weight. The "
                         "weight is SWEPT, not chosen: take the smallest "
                         "value achieving zero violations (D-011's rule for "
                         "margin, applied to the price). No single weight is "
                         "principled (D-013), so the sweep and its "
                         "return-vs-violation curve are the evidence.")
    p.add_argument("--learner-threads", type=int, default=8,
                    help="torch threads for the LEARNER only; the env workers "
                         "stay single-threaded via OMP_NUM_THREADS")
    args = p.parse_args()

    from rl.env import DEFAULT_MARGIN_HI
    margin_hi = DEFAULT_MARGIN_HI if args.margin_hi is None else args.margin_hi

    try:
        import tensorboard
        tb_log = str(out_dir / "tb")
    except ImportError:
        tb_log = None
        print("tensorboard not installed -- CSV monitor only", flush=True)
    observe_cooling_margin = not args.no_cooling_margin
    arm = "with_margin" if observe_cooling_margin else "no_margin"
    w_tag = "_lag" if args.lagrangian else (
        "" if args.safety_weight is None else f"_w{args.safety_weight:g}")
    if args.uniform_in_beta:
        w_tag += "_ub"
    tag = f"{args.algo}_{arm}_seed{args.seed}{w_tag}"
    out_dir = Path(args.out) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    print(f"[{tag}] {args.timesteps} steps, {args.n_envs} envs, "
          f"{args.campaign_h:g}h episodes, cooling margin observed: "
          f"{observe_cooling_margin}", flush=True)

    baseline = t0b_ref = None
    if not args.skip_baseline:
        t0 = time.time()
        baseline = random_policy_baseline(args.campaign_h, args.seed,
                                           observe_cooling_margin, args.n_horizon,
                                           safety_weight=args.safety_weight)
        print(f"[{tag}] random-policy floor:  return={baseline['mean_return']:.2f} "
              f"profit/h={baseline['mean_profit_per_hour']:.1f} "
              f"viol={baseline['mean_violation_integral']:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        t0 = time.time()
        t0b_ref = fixed_tier_baseline(args.campaign_h, args.seed,
                                       observe_cooling_margin, args.n_horizon,
                                       safety_weight=args.safety_weight)
        print(f"[{tag}] T0b fixed-tier bar:   return={t0b_ref['mean_return']:.2f} "
              f"profit/h={t0b_ref['mean_profit_per_hour']:.1f} "
              f"viol={t0b_ref['mean_violation_integral']:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        print(f"[{tag}] the agent must clear BOTH: random shows it is not "
              f"flailing, T0b shows it learned adaptation rather than one "
              f"good fixed point.", flush=True)

    vec = build_vec_env(args.n_envs, args.seed, args.campaign_h,
                         observe_cooling_margin, args.n_horizon,
                         safety_weight=args.safety_weight,
                         monitor_path=str(out_dir / "monitor"),
                         uniform_in_beta=args.uniform_in_beta,
                         margin_hi=margin_hi)

    try:
        import torch
        torch.set_num_threads(args.learner_threads)
        print(f"[{tag}] learner torch threads: {torch.get_num_threads()}",
              flush=True)
    except Exception as exc:
        print(f"[{tag}] could not set torch threads: {exc}", flush=True)

    if args.algo == "sac":
        from stable_baselines3 import SAC
        model = SAC("MlpPolicy", vec, seed=args.seed, verbose=1,
                     tensorboard_log=tb_log,
                     learning_rate=3e-4, buffer_size=200_000,
                     batch_size=256, tau=0.005, gamma=args.gamma,
                     train_freq=(1, "step"), gradient_steps=args.n_envs,
                     learning_starts=max(1000, 2 * args.n_envs))
    else:
        from stable_baselines3 import PPO
        model = PPO("MlpPolicy", vec, seed=args.seed, verbose=1,
                     tensorboard_log=tb_log,
                     learning_rate=3e-4, n_steps=256, batch_size=256,
                     gamma=args.gamma, gae_lambda=0.95)

    lagrange = None
    callback = None
    if args.lagrangian:
        lagrange = LagrangianWeightCallback(
            budget=args.violation_budget,
            intervals_per_episode=round(args.campaign_h / RL["decision_interval_h"]),
            update_every=max(1, 100 * args.n_envs), tag=tag)
        vec.env_method("set_safety_weight", lagrange.weight)
        callback = lagrange.attach(model, vec)
        print(f"[{tag}] Lagrangian: budget {args.violation_budget} degC.h/episode, "
              f"initial weight {lagrange.weight:.0f}", flush=True)

    model.learn(total_timesteps=args.timesteps, progress_bar=False,
                 callback=callback)
    model_path = out_dir / "model.zip"
    model.save(model_path)
    vec.close()

    wall = time.time() - start_time
    meta = {"tag": tag, "algo": args.algo, "seed": args.seed,
             "timesteps": args.timesteps, "n_envs": args.n_envs,
             "uniform_in_beta": bool(args.uniform_in_beta),
             "margin_hi": margin_hi,
             "gamma": args.gamma,
             "uniform_in_beta": bool(args.uniform_in_beta),
             "margin_hi": margin_hi,
             "gamma": args.gamma,
             "campaign_h": args.campaign_h, "n_horizon": args.n_horizon,
             "observe_cooling_margin": observe_cooling_margin,
             "decision_interval_h": RL["decision_interval_h"],
             "safety_penalty_weight": (args.safety_weight
                                       if args.safety_weight is not None
                                       else PARAMS["economics"]["safety_penalty_weight"]),
             "beta_final_range": [RL["train_beta_final_low"],
                                   RL["train_beta_final_high"]],
             "feed_noise_std": RL["feed_noise_std"],
             "feed_noise_tau_h": RL["feed_noise_tau_h"],
             "training_distribution": {k: v for k, v in RL.items()
                                       if k.startswith("train") or "noise" in k},
             "lagrangian": bool(args.lagrangian),
             "violation_budget": args.violation_budget if args.lagrangian else None,
             "lambda_history": (lagrange.history if lagrange else None),
             "lambda_final": (lagrange.weight if lagrange else None),
             "random_policy_baseline": baseline,
             "t0b_fixed_tier_baseline": t0b_ref,
             "wall_time_s": wall}
    with open(out_dir / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    from utils.provenance import save_with_provenance
    sidecar = save_with_provenance(str(model_path), data=None, params=meta,
                                    script=__file__, start_time=start_time)
    print(f"[{tag}] done in {wall / 60:.1f} min -> {model_path}, {sidecar}", flush=True)


if __name__ == "__main__":
    main()

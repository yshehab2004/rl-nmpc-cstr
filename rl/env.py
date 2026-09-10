import gymnasium as gym
import numpy as np
from gymnasium import spaces

from models.deactivation import make_deactivation_fn
from models.disturbances import (build_feed_disturbance, ou_noise_disturbance,
                                  ramp_disturbance, step_disturbance)
from models.params import load_params
from open_loop_testing.optimum_trajectory import compute_profit
from supervisors.campaign import CampaignEngine, T_STEP

PARAMS = load_params()
FEED = PARAMS["nominal_feed"]
ECON = PARAMS["economics"]
PRICES = dict(price_B=ECON["price_B"], price_A0=ECON["price_A0"],
              price_energy=ECON["price_energy"])
T_LIMIT = PARAMS["state_bounds"]["T_R"]["upper_soft"]
Q_LOWER = PARAMS["input_bounds"]["Q_dot"]["lower"]
Q_UPPER = PARAMS["input_bounds"]["Q_dot"]["upper"]
F_LOWER = PARAMS["input_bounds"]["F"]["lower"]
F_UPPER = PARAMS["input_bounds"]["F"]["upper"]
SAFETY_WEIGHT = ECON["safety_penalty_weight"]
MEAS = PARAMS["measurement"]

REWARD_SCALE = 1000.0

_C_A_RANGE = 6.0
_C_B_RANGE = MEAS["C_b_range"]
_T_LO, _T_SPAN = 100.0, 50.0


DEFAULT_MARGIN_HI = 5.0


class ActionMapper:

    F_lo, F_hi = F_LOWER, F_UPPER
    T_R_lo, T_R_hi = 100.0, T_LIMIT
    margin_lo = 0.0

    def __init__(self, margin_hi=DEFAULT_MARGIN_HI):
        self.margin_hi = float(margin_hi)

    def to_target(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        unit = 0.5 * (a + 1.0)
        return (float(self.F_lo + unit[0] * (self.F_hi - self.F_lo)),
                float(self.T_R_lo + unit[1] * (self.T_R_hi - self.T_R_lo)),
                float(self.margin_lo + unit[2] * (self.margin_hi - self.margin_lo)))


def reward_from(profit, violation_integral, weight, interval_h):
    return float((profit - weight * violation_integral)
                 / (interval_h * REWARD_SCALE))


class ConstantAgeing:

    def __init__(self, beta=1.0):
        self.beta = float(beta)

    def __call__(self, rng, campaign_h):
        if self.beta >= 1.0:
            return None
        return make_deactivation_fn(beta_final=self.beta, campaign_h=campaign_h)


class RandomAgeing:

    def __init__(self, beta_final_range=(0.15, 0.95), uniform_in_beta=False,
                  beta_start_range=(0.15, 1.0)):
        self.lo, self.hi = beta_final_range
        self.uniform_in_beta = bool(uniform_in_beta)
        self.start_lo, self.start_hi = beta_start_range

    def __call__(self, rng, campaign_h):
        if not self.uniform_in_beta:
            beta_final = float(rng.uniform(self.lo, self.hi))
            return make_deactivation_fn(beta_final=beta_final,
                                         campaign_h=campaign_h)
        beta_start = float(rng.uniform(self.start_lo, self.start_hi))
        beta_final = float(rng.uniform(self.lo, beta_start))
        return make_deactivation_fn(beta_final=beta_final,
                                     campaign_h=campaign_h,
                                     beta_start=beta_start)


class BackgroundNoiseFeed:

    def __init__(self, std=0.05, tau_h=0.5):
        self.std = float(std)
        self.tau_h = float(tau_h)

    def __call__(self, rng, campaign_h):
        seed = int(rng.integers(0, 2 ** 31 - 1))
        return build_feed_disturbance(
            C_A0_gen=ou_noise_disturbance(FEED["C_A0"], std=self.std,
                                           tau_h=self.tau_h, seed=seed,
                                           t_max=campaign_h))


class RandomFeedDisturbance:

    _NOMINAL_CAMPAIGN_H = 75.0

    def __init__(self, C_A0_noise_std=0.05, T_in_noise_std=0.5,
                  noise_tau_h=1.0, p_event=0.5,
                  C_A0_mag_range=(0.2, 0.8), T_in_mag_range=(1.0, 5.0),
                  step_onset_h=(15.0, 50.0), ramp_onset_h=(10.0, 40.0),
                  ramp_duration_h=(5.0, 25.0)):
        self.C_A0_noise_std = float(C_A0_noise_std)
        self.T_in_noise_std = float(T_in_noise_std)
        self.noise_tau_h = float(noise_tau_h)
        self.p_event = float(p_event)
        self.C_A0_mag_range = tuple(C_A0_mag_range)
        self.T_in_mag_range = tuple(T_in_mag_range)
        self.step_onset_h = tuple(step_onset_h)
        self.ramp_onset_h = tuple(ramp_onset_h)
        self.ramp_duration_h = tuple(ramp_duration_h)

    def _scaled(self, hours, campaign_h):
        return float(hours) / self._NOMINAL_CAMPAIGN_H * campaign_h

    def __call__(self, rng, campaign_h):
        C_A0_base = FEED["C_A0"]
        T_in_base = FEED["T_in"]

        if rng.random() < self.p_event:
            kind = int(rng.integers(0, 3))
            if kind == 0:
                mag = float(rng.uniform(*self.C_A0_mag_range))
                onset = self._scaled(rng.uniform(*self.step_onset_h), campaign_h)
                C_A0_base = step_disturbance(FEED["C_A0"], mag, onset)
            elif kind == 1:
                mag = float(rng.uniform(*self.C_A0_mag_range))
                onset = self._scaled(rng.uniform(*self.ramp_onset_h), campaign_h)
                dur = self._scaled(rng.uniform(*self.ramp_duration_h), campaign_h)
                C_A0_base = ramp_disturbance(FEED["C_A0"], mag, onset, onset + dur)
            else:
                mag = float(rng.uniform(*self.T_in_mag_range))
                onset = self._scaled(rng.uniform(*self.step_onset_h), campaign_h)
                T_in_base = step_disturbance(FEED["T_in"], mag, onset)

        C_A0_gen = ou_noise_disturbance(
            C_A0_base, std=self.C_A0_noise_std, tau_h=self.noise_tau_h,
            seed=int(rng.integers(0, 2 ** 31 - 1)), t_max=campaign_h)
        T_in_gen = ou_noise_disturbance(
            T_in_base, std=self.T_in_noise_std, tau_h=self.noise_tau_h,
            seed=int(rng.integers(0, 2 ** 31 - 1)), t_max=campaign_h)
        return build_feed_disturbance(C_A0_gen=C_A0_gen, T_in_gen=T_in_gen)


class SupervisoryEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self, campaign_h=75.0, decision_interval_h=0.25,
                  observe_cooling_margin=True, ageing=None, feed=None,
                  safety_penalty_weight=None, n_horizon=20, noise=True,
                  seed=None, margin_hi=DEFAULT_MARGIN_HI):
        super().__init__()
        self.campaign_h = float(campaign_h)
        self.decision_interval_h = float(decision_interval_h)
        self.observe_cooling_margin = bool(observe_cooling_margin)
        self.ageing = ageing
        self.feed = feed
        self.weight = (SAFETY_WEIGHT if safety_penalty_weight is None
                       else float(safety_penalty_weight))
        self.n_horizon = n_horizon
        self.noise = noise
        self._seed = seed

        self.steps_per_decision = max(1, int(round(decision_interval_h / T_STEP)))
        self.max_steps = int(round(self.campaign_h / self.decision_interval_h))

        self.margin_hi = float(margin_hi)
        self.mapper = ActionMapper(margin_hi=self.margin_hi)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        n_obs = 9 + (1 if self.observe_cooling_margin else 0)
        self.observation_space = spaces.Box(-2.0, 2.0, shape=(n_obs,),
                                             dtype=np.float32)

        self.engine = None
        self._step_count = 0


    def build_observation(self, ctx):
        x_hat = np.asarray(ctx["x_hat"], dtype=float).flatten()
        u_last = ctx["u_last"]
        F_last = (u_last[0] if u_last is not None else F_LOWER)

        obs = [
            ctx["beta_hat"],
            x_hat[0] / _C_A_RANGE,
            x_hat[1] / _C_B_RANGE,
            (x_hat[2] - _T_LO) / _T_SPAN,
            (x_hat[3] - _T_LO) / _T_SPAN,
            (ctx["C_A0"] - FEED["C_A0"]) / 1.0,
            (ctx["T_in"] - FEED["T_in"]) / 10.0,
            (F_last - F_LOWER) / (F_UPPER - F_LOWER),
            min(1.0, ctx["t"] / self.campaign_h),
        ]
        if self.observe_cooling_margin:
            if u_last is None:
                margin = 1.0
            else:
                margin = (float(u_last[1]) - Q_LOWER) / (Q_UPPER - Q_LOWER)
            obs.append(margin)
        return np.asarray(obs, dtype=np.float32)


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        seed = self._seed if seed is None else seed
        rng = np.random.default_rng(seed)

        ageing_fn = self.ageing(rng, self.campaign_h) if self.ageing else None
        disturbance_fn = self.feed(rng, self.campaign_h) if self.feed else None

        self.engine = CampaignEngine(
            ageing_fn=ageing_fn, disturbance_fn=disturbance_fn,
            seed=None if seed is None else int(rng.integers(0, 2 ** 31 - 1)),
            noise=self.noise, use_estimator=True, n_horizon=self.n_horizon)
        self._step_count = 0
        return self.build_observation(self.engine.context()), {}

    def step(self, action):
        F_ref, T_R_ref, margin = self.mapper.to_target(action)
        k_before = self.engine.k

        self.engine.set_target(F_ref, T_R_ref, margin)
        remaining = int(round(self.campaign_h / T_STEP)) - self.engine.k
        self.engine.advance(min(self.steps_per_decision, remaining))
        self._step_count += 1

        profit, violation = self._interval_outcome(k_before)
        reward = reward_from(profit, violation, self.weight,
                              self.decision_interval_h)

        decision = self.engine.decisions[-1]
        ctx = self.engine.context()
        obs = self.build_observation(ctx)

        x_now = np.asarray(self.engine.x).flatten()
        terminated = bool(not np.all(np.isfinite(x_now))
                          or x_now[2] < 50.0 or x_now[2] > 200.0)
        truncated = self._step_count >= self.max_steps

        info = {"profit": profit, "violation_integral": violation,
                "F_ref": F_ref, "T_R_ref": T_R_ref, "margin": margin,
                "F_s": decision["F_s"], "T_R_s": decision["T_R_s"],
                "beta_hat": ctx["beta_hat"], "t": ctx["t"]}
        if terminated:
            info["termination"] = "diverged"
        return obs, reward, terminated, truncated, info

    def _interval_outcome(self, k_before):
        h = self.engine.hist
        x = np.asarray(h["x"][k_before:])
        u = np.asarray(h["u"][k_before:])
        C_A0 = np.asarray(h["C_A0"][k_before:])
        if len(x) == 0:
            return 0.0, 0.0
        profit_rate = compute_profit(u[:, 0], x[:, 1], C_A0, u[:, 1], **PRICES)
        violation = np.maximum(0.0, x[:, 2] - T_LIMIT)
        return float(np.sum(profit_rate) * T_STEP), float(np.sum(violation) * T_STEP)

    def set_safety_weight(self, weight):
        self.weight = float(weight)
        return self.weight

    def result(self):
        return self.engine.result()


class make_env:

    def __init__(self, rank=0, **kwargs):
        self.rank = int(rank)
        self.kwargs = kwargs

    def __call__(self):
        kwargs = dict(self.kwargs)
        if kwargs.get("seed") is not None:
            kwargs["seed"] = int(kwargs["seed"]) + self.rank
        return SupervisoryEnv(**kwargs)

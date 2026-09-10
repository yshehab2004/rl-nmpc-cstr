import json
from pathlib import Path

import numpy as np

from rl.env import (ActionMapper, DEFAULT_MARGIN_HI, SupervisoryEnv)


class RLSupervisor:

    def __init__(self, model_path, campaign_h, observe_cooling_margin=True,
                  algo="sac", deterministic=True, allow_overrun=False,
                  name="T3_rl", margin_hi=None):
        self.name = name
        self.campaign_h = float(campaign_h)
        self.observe_cooling_margin = bool(observe_cooling_margin)
        self.deterministic = bool(deterministic)
        self.allow_overrun = bool(allow_overrun)
        self.n_solves = 0

        if algo == "sac":
            from stable_baselines3 import SAC as _Algo
        elif algo == "ppo":
            from stable_baselines3 import PPO as _Algo
        else:
            raise ValueError(f"unknown algo {algo!r}")
        self.model = _Algo.load(model_path, device="cpu")

        if margin_hi is None:
            meta_path = Path(model_path).parent / "train_meta.json"
            if meta_path.exists():
                margin_hi = json.loads(meta_path.read_text()).get("margin_hi")
            margin_hi = DEFAULT_MARGIN_HI if margin_hi is None else margin_hi
        self.margin_hi = float(margin_hi)

        self._encoder = SupervisoryEnv(
            campaign_h=self.campaign_h,
            observe_cooling_margin=self.observe_cooling_margin,
            margin_hi=self.margin_hi)
        self.mapper = ActionMapper(margin_hi=self.margin_hi)

        expected = self.model.observation_space.shape[0]
        actual = self._encoder.observation_space.shape[0]
        if expected != actual:
            raise ValueError(
                f"policy expects {expected} observations but this encoder "
                f"produces {actual}. observe_cooling_margin="
                f"{self.observe_cooling_margin} probably does not match the "
                f"arm this policy was trained on.")

    def observation_for(self, ctx):
        return self._encoder.build_observation(ctx)

    def get_target(self, t_now, beta_hat=None, C_A0=None, T_in=None,
                    x_hat=None, u_last=None):
        t_now = float(t_now)
        if t_now > self.campaign_h and not self.allow_overrun:
            raise ValueError(
                f"t={t_now:.2f}h exceeds the campaign_h={self.campaign_h:.2f}h "
                f"this policy was trained on. Episode progress (t/campaign_h) "
                f"is an observation entry, so running longer silently "
                f"redefines it -- at t=campaign_h the policy would read a "
                f"value it only ever saw at the very end of training. Pass "
                f"allow_overrun=True to do this deliberately, as the "
                f"out-of-distribution probes do.")

        ctx = {"t": t_now, "beta_hat": beta_hat, "C_A0": C_A0, "T_in": T_in,
               "x_hat": x_hat, "u_last": u_last}
        obs = self.observation_for(ctx)
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        self.n_solves += 1
        return self.mapper.to_target(np.asarray(action, dtype=float))

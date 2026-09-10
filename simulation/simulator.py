import numpy as np

from models.cstr_model import build_model
from models.params import load_params
from do_mpc.simulator import Simulator

PARAMS = load_params()


def build_simulator(
    model=None,
    c_b_set: float = None,
    disturbance_fn=None,
    ageing_fn=None,
) -> Simulator:
    if model is None:
        model = build_model()

    uncertain = PARAMS["uncertain_params"]
    feed = PARAMS["nominal_feed"]
    if c_b_set is None:
        c_b_set = PARAMS["nmpc"]["C_b_setpoint_default"]

    simulator = Simulator(model)
    simulator.set_param(t_step=PARAMS["nmpc"]["t_step"])

    p_template = simulator.get_p_template()

    def p_fun(t_now):
        if ageing_fn is not None:
            alpha, beta = ageing_fn(np.asarray(t_now).item())
        else:
            alpha, beta = uncertain["alpha_nominal"], uncertain["beta_nominal"]
        p_template["alpha"] = alpha
        p_template["beta"] = beta
        return p_template

    simulator.set_p_fun(p_fun)

    tvp_template = simulator.get_tvp_template()

    def tvp_fun(t_now):
        if disturbance_fn is not None:
            C_A0, T_in = disturbance_fn(np.asarray(t_now).item())
        else:
            C_A0, T_in = feed["C_A0"], feed["T_in"]
        tvp_template["C_b_set"] = c_b_set
        tvp_template["F_ref"] = 0.0
        tvp_template["d_T_R"] = 0.0
        tvp_template["T_R_ref"] = 0.0
        tvp_template["C_A0_feed"] = C_A0
        tvp_template["T_in_feed"] = T_in
        return tvp_template

    simulator.set_tvp_fun(tvp_fun)

    simulator.setup()
    return simulator


def initial_state() -> np.ndarray:
    init = PARAMS["initial_state"]
    return np.array([[init["C_a"]], [init["C_b"]], [init["T_R"]], [init["T_K"]]])

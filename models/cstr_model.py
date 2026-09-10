import do_mpc
from casadi import exp

from models.params import load_params

PARAMS = load_params()
CERTAIN = PARAMS["certain_params"]


def build_model(model_type: str = "continuous", for_controller: bool = False,
                certain: dict = None) -> do_mpc.model.Model:
    C = CERTAIN if certain is None else certain
    model = do_mpc.model.Model(model_type)

    C_a = model.set_variable(var_type="_x", var_name="C_a", shape=(1, 1))
    C_b = model.set_variable(var_type="_x", var_name="C_b", shape=(1, 1))
    T_R = model.set_variable(var_type="_x", var_name="T_R", shape=(1, 1))
    T_K = model.set_variable(var_type="_x", var_name="T_K", shape=(1, 1))

    F = model.set_variable(var_type="_u", var_name="F")
    Q_dot = model.set_variable(var_type="_u", var_name="Q_dot")

    alpha = model.set_variable(var_type="_p", var_name="alpha")
    beta = model.set_variable(var_type="_p", var_name="beta")

    C_b_set = model.set_variable(var_type="_tvp", var_name="C_b_set", shape=(1, 1))
    F_ref = model.set_variable(var_type="_tvp", var_name="F_ref", shape=(1, 1))
    d_T_R = model.set_variable(var_type="_tvp", var_name="d_T_R", shape=(1, 1))
    T_R_ref = model.set_variable(var_type="_tvp", var_name="T_R_ref", shape=(1, 1))

    if for_controller:
        beta_belief = model.set_variable(var_type="_tvp", var_name="beta_belief", shape=(1, 1))
        beta = beta_belief
    C_A0_feed = model.set_variable(var_type="_tvp", var_name="C_A0_feed", shape=(1, 1))
    T_in_feed = model.set_variable(var_type="_tvp", var_name="T_in_feed", shape=(1, 1))

    K0_ab, K0_bc, K0_ad = C["K0_ab"], C["K0_bc"], C["K0_ad"]
    E_A_ab, E_A_bc, E_A_ad = C["E_A_ab"], C["E_A_bc"], C["E_A_ad"]
    H_R_ab, H_R_bc, H_R_ad = C["H_R_ab"], C["H_R_bc"], C["H_R_ad"]
    rho, Cp, Cp_k = C["rho"], C["Cp"], C["Cp_k"]
    A_R, V_R, m_k, K_w = C["A_R"], C["V_R"], C["m_k"], C["K_w"]

    K_1 = beta * K0_ab * exp(-E_A_ab / (T_R + 273.15))
    K_2 = beta * K0_bc * exp(-E_A_bc / (T_R + 273.15))
    K_3 = K0_ad * exp(-alpha * E_A_ad / (T_R + 273.15))

    T_dif = model.set_expression(expr_name="T_dif", expr=T_R - T_K)

    model.set_rhs("C_a", F * (C_A0_feed - C_a) - K_1 * C_a - K_3 * (C_a ** 2))
    model.set_rhs("C_b", -F * C_b + K_1 * C_a - K_2 * C_b)
    model.set_rhs(
        "T_R",
        ((K_1 * C_a * H_R_ab + K_2 * C_b * H_R_bc + K_3 * (C_a ** 2) * H_R_ad) / (-rho * Cp))
        + F * (T_in_feed - T_R)
        + ((K_w * A_R) * (-T_dif)) / (rho * Cp * V_R),
    )
    model.set_rhs("T_K", (Q_dot + K_w * A_R * T_dif) / (m_k * Cp_k))

    model.setup()
    return model

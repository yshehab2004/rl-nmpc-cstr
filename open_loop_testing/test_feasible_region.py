import numpy as np
import pytest

from open_loop_testing.feasible_region import classify, feasible_area, FEASIBLE, COOLING_FLOOR, HEATING_LIMIT, TEMP_LIMIT


def test_classify_feasible_point():
    Q_dot = np.array([-500.0])
    T_R = np.array([[120.0]])
    code = classify(Q_dot.reshape(1, 1), T_R, q_lower=-8500.0, q_upper=0.0, T_R_max=140.0)
    assert code[0, 0] == FEASIBLE


def test_classify_cooling_floor():
    Q_dot = np.array([[-9000.0]])
    T_R = np.array([[120.0]])
    code = classify(Q_dot, T_R, q_lower=-8500.0, q_upper=0.0, T_R_max=140.0)
    assert code[0, 0] == COOLING_FLOOR


def test_classify_heating_limit():
    Q_dot = np.array([[500.0]])
    T_R = np.array([[120.0]])
    code = classify(Q_dot, T_R, q_lower=-8500.0, q_upper=0.0, T_R_max=140.0)
    assert code[0, 0] == HEATING_LIMIT


def test_classify_temperature_limit_takes_priority_over_q_dot():
    Q_dot = np.array([[-9000.0]])
    T_R = np.array([[145.0]])
    code = classify(Q_dot, T_R, q_lower=-8500.0, q_upper=0.0, T_R_max=140.0)
    assert code[0, 0] == TEMP_LIMIT


def test_feasible_area_full_grid_feasible():
    F_grid = np.linspace(5.0, 35.0, 4)
    T_R_grid = np.linspace(100.0, 150.0, 6)
    code = np.full((4, 6), FEASIBLE)
    area = feasible_area(F_grid, T_R_grid, code)
    assert area == pytest.approx(30.0 * 50.0)


def test_feasible_area_half_grid_feasible():
    F_grid = np.linspace(5.0, 35.0, 4)
    T_R_grid = np.linspace(100.0, 150.0, 6)
    code = np.full((4, 6), FEASIBLE)
    code[2:, :] = COOLING_FLOOR
    area = feasible_area(F_grid, T_R_grid, code)
    assert area == pytest.approx(30.0 * 50.0 / 2)


def test_real_map_feasible_area_shrinks_monotonically_as_beta_falls():
    d = np.load("open_loop_testing/steady_state_map.npz")
    F, T_R_grid, betas, Q_dot = d["F"], d["T_R"], d["beta"], d["Q_dot"]
    T_R_mesh = np.broadcast_to(T_R_grid[None, :], Q_dot[0].shape)

    areas = []
    for i in range(len(betas)):
        code = classify(Q_dot[i], T_R_mesh, q_lower=-8500.0, q_upper=0.0, T_R_max=140.0)
        areas.append(feasible_area(F, T_R_grid, code))

    order = np.argsort(-betas)
    ordered_areas = np.array(areas)[order]
    assert np.all(np.diff(ordered_areas) <= 0), (
        f"expected feasible area to shrink as beta falls, got {ordered_areas} "
        f"for betas {betas[order]}"
    )
    assert ordered_areas[0] > ordered_areas[-1]

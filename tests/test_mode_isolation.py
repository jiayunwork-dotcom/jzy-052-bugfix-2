"""板厚模式隔离：三维点源律与二维线源律各自独立，绝不混用同一条衰减律。"""

import pytest

from app.physics import thick, thin
from handcalc import hand_thick, hand_thin

Q, K, ALPHA, V, D = 2400.0, 50.0, 1.2e-5, 0.005, 0.002
X, Y, Z = -0.01, 0.003, 0.004


def test_thick_mode_uses_3d_point_source_law():
    got = thick.temperature_rise(X, Y, Z, q=Q, k=K, alpha=ALPHA, v=V)
    assert got == pytest.approx(hand_thick(X, Y, Z, Q, K, ALPHA, V), rel=1e-12)
    # 若误套二维衰减律，结果必然明显偏离
    wrong = hand_thin(X, Y, Q, K, ALPHA, V, D)
    assert abs(got - wrong) / wrong > 0.05


def test_thin_mode_uses_2d_line_source_law():
    got = thin.temperature_rise(X, Y, q=Q, k=K, alpha=ALPHA, v=V, thickness=D)
    assert got == pytest.approx(hand_thin(X, Y, Q, K, ALPHA, V, D), rel=1e-12)
    # 若误套三维衰减律，结果必然明显偏离
    wrong = hand_thick(X, Y, Z, Q, K, ALPHA, V)
    assert abs(got - wrong) / wrong > 0.05


def test_api_modes_are_not_cross_wired(client):
    """同一组数值参数走两个模式，各自命中各自的闭式公式。"""
    process = {
        "mode": "thick",
        "travel_speed": V,
        "conductivity": K,
        "diffusivity": ALPHA,
        "efficiency": 1.0,
        "initial_temperature": 25.0,
        "power": Q,
        "thickness": D,  # thick 模式下应被忽略
    }
    point = {"x": X, "y": Y, "z": Z}
    thick_body = client.post("/v1/calculate", json={"process": process, "point": point}).json()
    thin_body = client.post(
        "/v1/calculate",
        json={"process": {**process, "mode": "thin"}, "point": {"x": X, "y": Y}},
    ).json()
    r_thick = thick_body["point"]["temperature_rise"]
    r_thin = thin_body["point"]["temperature_rise"]
    assert r_thick == pytest.approx(hand_thick(X, Y, Z, Q, K, ALPHA, V), rel=1e-9)
    assert r_thin == pytest.approx(hand_thin(X, Y, Q, K, ALPHA, V, D), rel=1e-9)
    assert abs(r_thick - r_thin) / r_thin > 0.05

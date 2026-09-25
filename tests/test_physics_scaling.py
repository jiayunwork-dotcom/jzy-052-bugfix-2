"""缩放规律：服务输出必须与物理定律逐条对上。"""

import pytest

from app.physics import thick, thin

Q, K, ALPHA, V = 2400.0, 50.0, 1.2e-5, 0.005
D = 0.002


def test_zero_power_gives_zero_rise_everywhere():
    """有效功率归零时任意点温升都归零。"""
    points = [(-0.01, 0.002, 0.003), (0.0, 0.0, 0.0), (1.0, -0.5, 0.1)]
    for x, y, z in points:
        assert thick.temperature_rise(x, y, z, q=0.0, k=K, alpha=ALPHA, v=V) == 0.0
        assert thin.temperature_rise(x, y, q=0.0, k=K, alpha=ALPHA, v=V, thickness=D) == 0.0


def test_power_doubling_doubles_rise():
    """固定观察点，单把有效功率翻倍，温升随之翻倍。"""
    for x, y, z in [(-0.01, 0.002, 0.003), (0.02, -0.004, 0.001)]:
        a = thick.temperature_rise(x, y, z, q=Q, k=K, alpha=ALPHA, v=V)
        b = thick.temperature_rise(x, y, z, q=2 * Q, k=K, alpha=ALPHA, v=V)
        assert b == pytest.approx(2.0 * a, rel=1e-12)
        a2 = thin.temperature_rise(x, y, q=Q, k=K, alpha=ALPHA, v=V, thickness=D)
        b2 = thin.temperature_rise(x, y, q=2 * Q, k=K, alpha=ALPHA, v=V, thickness=D)
        assert b2 == pytest.approx(2.0 * a2, rel=1e-12)


def test_conductivity_doubling_halves_rise():
    """单把热导率翻倍而其余不变，同一点温升减半。"""
    x, y, z = -0.01, 0.002, 0.003
    a = thick.temperature_rise(x, y, z, q=Q, k=K, alpha=ALPHA, v=V)
    b = thick.temperature_rise(x, y, z, q=Q, k=2 * K, alpha=ALPHA, v=V)
    assert b == pytest.approx(0.5 * a, rel=1e-12)
    a2 = thin.temperature_rise(x, y, q=Q, k=K, alpha=ALPHA, v=V, thickness=D)
    b2 = thin.temperature_rise(x, y, q=Q, k=2 * K, alpha=ALPHA, v=V, thickness=D)
    assert b2 == pytest.approx(0.5 * a2, rel=1e-12)


def test_speed_doubling_cools_point_behind_source_thick():
    """厚板模式单把行走速度翻倍，指数衰减更急，热源后方同一点更凉。"""
    x, y, z = -0.01, 0.002, 0.003  # 热源后方、偏离中线
    slow = thick.temperature_rise(x, y, z, q=Q, k=K, alpha=ALPHA, v=V)
    fast = thick.temperature_rise(x, y, z, q=Q, k=K, alpha=ALPHA, v=2 * V)
    assert 0.0 < fast < slow


def test_far_ahead_of_source_rise_vanishes():
    """观察点挪到热源正前方远处，温升应当很小。"""
    assert thick.temperature_rise(5.0, 0.0, 0.0, q=Q, k=K, alpha=ALPHA, v=V) < 1e-9
    assert thin.temperature_rise(5.0, 0.0, q=Q, k=K, alpha=ALPHA, v=V, thickness=D) < 1e-9


def test_far_away_temperature_returns_to_t0():
    """距离趋于无穷时温度回到 T0（温升趋于零）。"""
    rise = thick.temperature_rise(-1.0e5, 1.0, 1.0, q=Q, k=K, alpha=ALPHA, v=V)
    assert 0.0 <= rise < 1e-3
    rise2 = thin.temperature_rise(-1.0e5, 1.0, q=Q, k=K, alpha=ALPHA, v=V, thickness=D)
    assert 0.0 <= rise2 < 1e-3

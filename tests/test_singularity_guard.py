"""奇点判定边界：观察线穿过或紧贴热源奇点时，接口必须按"峰值无界"处理，
且不得误伤离热源有正常距离的查询。

三类坐标穿插覆盖：严格落在零点、偏移极小（~1e-9 量级）、偏移正常（毫米量级）；
另核对功率翻倍的等比例性与批量条目间的隔离。
"""

import math

import pytest
from scipy.optimize import minimize_scalar

from app.physics import thick
from app.solver import MAX_PHYSICAL_RISE, scan_peak
from handcalc import hand_thick

Q, K, ALPHA, V, T0 = 2400.0, 50.0, 1.2e-5, 0.005, 25.0  # demo_process 的有效参数
MELT = 1500.0

ZERO = {"x": 0.0, "y": 0.0, "z": 0.0}
TINY = [
    {"x": 0.0, "y": 1e-9, "z": 0.0},
    {"x": 0.0, "y": 0.0, "z": 1e-9},
    {"x": 0.0, "y": -1e-9, "z": 1e-9},
    {"x": 0.0, "y": 1e-7, "z": 0.0},
    {"x": 0.0, "y": 1e-11, "z": 0.0},
    {"x": 1e-9, "y": 1e-9, "z": 0.0},
]
NORMAL = {"x": -0.01, "y": 0.002, "z": 0.003}


def _calc(client, process, point, power=None, melt=MELT):
    proc = dict(process)
    if power is not None:
        proc["power"] = power
    body = {"process": proc, "point": point}
    if melt is not None:
        body["melting_temperature"] = melt
    resp = client.post("/v1/calculate", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _reference_peak_rise(y, z, q):
    """独立手算公式 + scipy 寻峰，与服务实现互不相干。"""
    res = minimize_scalar(
        lambda x: -hand_thick(x, y, z, q, K, ALPHA, V),
        bounds=(-0.1, 0.0),
        method="bounded",
        options={"xatol": 1e-15},
    )
    return res.x, -res.fun


def _walk_numbers(payload):
    if isinstance(payload, dict):
        for v in payload.values():
            yield from _walk_numbers(v)
    elif isinstance(payload, list):
        for v in payload:
            yield from _walk_numbers(v)
    elif isinstance(payload, (int, float)):
        yield payload


def _assert_singular_payload(body):
    """峰值与观察点均按奇点处理：温度为 None、标记 singular、给出原因。"""
    assert body["point"]["singular"] is True
    assert body["point"]["temperature"] is None
    assert body["point"]["temperature_rise"] is None
    assert body["peak"]["singular"] is True
    assert body["peak"]["temperature"] is None
    assert body["peak"]["temperature_rise"] is None
    assert body["peak"]["x"] is None
    assert body["peak"]["note"]  # 必须给出失败原因
    # 整个响应里不得出现任何脱离物理量级的数值
    for value in _walk_numbers(body):
        assert math.isfinite(value)
        assert abs(value) < MAX_PHYSICAL_RISE


def test_exact_zero_reports_singularity(client, demo_process):
    """坐标严格落在热源上：维持原有正确行为（回归锁定）。"""
    body = _calc(client, demo_process, ZERO)
    _assert_singular_payload(body)
    # 中线穿过热源，熔化温升必然被越过：熔池照常形成
    assert body["melt_pool"]["formed"] is True


@pytest.mark.parametrize("point", TINY)
def test_tiny_offset_treated_same_as_exact_zero(client, demo_process, point):
    """偏移极小但非零：必须与严格零点走同一条奇点路径，不得放出失控数值。"""
    body = _calc(client, demo_process, point)
    _assert_singular_payload(body)
    zero_body = _calc(client, demo_process, ZERO)
    assert body["peak"] == zero_body["peak"]  # 与零点响应完全一致
    # 熔宽不受奇点判定影响：紧贴中线与严格中线的熔化边界一致
    assert body["melt_pool"]["formed"] is True
    assert body["melt_pool"]["width"] == pytest.approx(
        zero_body["melt_pool"]["width"], rel=1e-6
    )


def test_tiny_offset_stays_singular_when_power_doubles(client, demo_process):
    """失控区不随功率放大而消失：偏移 1e-9 时功率翻倍仍必须被拦截。"""
    body = _calc(client, demo_process, {"x": 0.0, "y": 1e-9, "z": 0.0}, power=6000.0)
    _assert_singular_payload(body)


def test_normal_offset_results_unchanged(client, demo_process):
    """正常距离：峰值位置/温升与独立手算一致，熔宽与独立二分一致。"""
    body = _calc(client, demo_process, NORMAL)
    assert body["point"]["singular"] is False
    assert body["peak"]["singular"] is False

    # 观察点温升对独立闭式手算
    assert body["point"]["temperature_rise"] == pytest.approx(
        hand_thick(NORMAL["x"], NORMAL["y"], NORMAL["z"], Q, K, ALPHA, V), rel=1e-9
    )
    # 峰值对独立寻峰
    ref_x, ref_rise = _reference_peak_rise(NORMAL["y"], NORMAL["z"], Q)
    assert body["peak"]["x"] == pytest.approx(ref_x, abs=1e-6)
    assert body["peak"]["temperature_rise"] == pytest.approx(ref_rise, rel=1e-6)
    assert body["peak"]["temperature"] == pytest.approx(T0 + ref_rise, rel=1e-6)

    # 熔宽对独立二分（手算峰值 = 熔化温升的横向位置）
    delta_tm = MELT - T0
    lo, hi = 0.0, 1e-1
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _reference_peak_rise(mid, NORMAL["z"], Q)[1] > delta_tm:
            lo = mid
        else:
            hi = mid
    assert body["melt_pool"]["formed"] is True
    assert body["melt_pool"]["width"] == pytest.approx(lo + hi, rel=1e-4)
    assert body["melt_pool"]["y_boundaries"] == pytest.approx(
        [-body["melt_pool"]["width"] / 2, body["melt_pool"]["width"] / 2]
    )


def test_power_doubling_scales_normal_results_exactly(client, demo_process):
    """正常距离：功率翻倍，观察点与峰值温升严格等比例，峰位不动。"""
    for point in [NORMAL, {"x": 0.0, "y": 1e-4, "z": 0.0}, {"x": -0.02, "y": 0.0, "z": 0.006}]:
        base = _calc(client, demo_process, point)
        doubled = _calc(client, demo_process, point, power=6000.0)
        assert base["peak"]["singular"] is False
        assert doubled["peak"]["singular"] is False
        assert doubled["point"]["temperature_rise"] == pytest.approx(
            2.0 * base["point"]["temperature_rise"], rel=1e-12
        )
        assert doubled["peak"]["temperature_rise"] == pytest.approx(
            2.0 * base["peak"]["temperature_rise"], rel=1e-12
        )
        assert doubled["peak"]["x"] == pytest.approx(base["peak"]["x"], abs=1e-12)


def test_guard_band_between_tiny_and_normal(client, demo_process):
    """判定边界：rho=1e-6（峰值温升 ~7.6e6 K）拦截；rho=1e-4（~7.6e4 K）放行。"""
    inside = _calc(client, demo_process, {"x": 0.0, "y": 1e-6, "z": 0.0})
    _assert_singular_payload(inside)
    outside = _calc(client, demo_process, {"x": 0.0, "y": 1e-4, "z": 0.0})
    assert outside["peak"]["singular"] is False
    assert 0.0 < outside["peak"]["temperature_rise"] < MAX_PHYSICAL_RISE


def test_zero_tiny_normal_interleaved_on_same_preset(client):
    """同一份具名工艺档，三类坐标穿插查询，各自命中各自的行为。"""
    cases = [
        (ZERO, True),
        ({"x": 0.0, "y": 1e-9, "z": 0.0}, True),
        (NORMAL, False),
        ({"x": 0.0, "y": 0.0, "z": 1e-9}, True),
        ({"x": -0.02, "y": 0.004, "z": 0.001}, False),
        (ZERO, True),
    ]
    for point, expect_singular in cases:
        resp = client.post(
            "/v1/calculate",
            json={"preset": "demo-butt-thick", "point": point, "melting_temperature": MELT},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["peak"]["singular"] is expect_singular
        assert body["point"]["singular"] is expect_singular
        if expect_singular:
            assert body["peak"]["temperature"] is None
        else:
            assert 0.0 < body["peak"]["temperature_rise"] < MAX_PHYSICAL_RISE


def test_batch_items_are_isolated(client, demo_process):
    """批量中极小偏移/零点/非法参数各自落各自的结果，不影响其余条目。"""
    items = [
        {"process": demo_process, "point": NORMAL},
        {"process": demo_process, "point": {"x": 0.0, "y": 1e-9, "z": 0.0}},
        {"process": demo_process, "point": ZERO},
        {"process": {**demo_process, "conductivity": -50.0}, "point": NORMAL},
        {"process": demo_process, "point": {"x": -0.02, "y": 0.004, "z": 0.001}},
    ]
    resp = client.post("/v1/calculate/batch", json={"items": items})
    assert resp.status_code == 200, resp.text
    results = resp.json()["results"]
    assert [r["ok"] for r in results] == [True, True, True, False, True]

    assert results[1]["result"]["peak"]["singular"] is True
    assert results[1]["result"]["peak"]["temperature"] is None
    assert results[2]["result"]["peak"]["singular"] is True
    assert results[3]["error"]["type"] == "invalid_input"

    # 正常条目与单独查询结果一致
    solo = _calc(client, demo_process, NORMAL, melt=None)
    assert results[0]["result"]["peak"] == solo["peak"]
    assert results[0]["result"]["point"] == solo["point"]
    assert results[4]["result"]["peak"]["singular"] is False


def test_scan_peak_marks_tiny_rho_singular_directly():
    """求解器层：rho 极小时 scan_peak 直接判 singular，正常 rho 不受影响。"""
    lam = V / (2.0 * ALPHA)

    def field_at(rho):
        return lambda x: thick.temperature_rise(x, rho, 0.0, q=Q, k=K, alpha=ALPHA, v=V)

    tiny = scan_peak(field_at(1e-9), rho=1e-9, lam=lam, q=Q, xtol=1e-10, max_iter=200)
    assert tiny.singular is True
    assert tiny.x is None and tiny.rise is None

    normal = scan_peak(field_at(2e-3), rho=2e-3, lam=lam, q=Q, xtol=1e-10, max_iter=200)
    assert normal.singular is False
    assert 0.0 < normal.rise < MAX_PHYSICAL_RISE

    # q = 0 的合法退化不受奇点判定影响，即使 rho 严格为零
    zero_q = scan_peak(field_at(0.0), rho=0.0, lam=lam, q=0.0, xtol=1e-10, max_iter=200)
    assert zero_q.singular is False
    assert zero_q.rise == 0.0

"""奇点判断边界：严格零点与极小偏移必须同样被识别，正常点结果不受影响。"""

import pytest


def _calculate(client, process, point, **extra):
    return client.post(
        "/v1/calculate",
        json={"process": process, "point": point, **extra},
    )


@pytest.mark.parametrize(
    "point",
    [
        {"x": 0.0, "y": 0.0, "z": 0.0},
        {"x": 0.0, "y": 1.0e-9, "z": 0.0},
        {"x": -1.0e-9, "y": 0.0, "z": 0.0},
        {"x": -1.0e-9, "y": 1.0e-9, "z": 1.0e-9},
    ],
)
def test_exact_and_tiny_offset_points_are_treated_as_singular(
    client, demo_process, point
):
    resp = _calculate(client, demo_process, point, melting_temperature=1500.0)
    assert resp.status_code == 200
    body = resp.json()

    assert body["point"]["singular"] is True
    assert body["point"]["temperature_rise"] is None
    assert body["point"]["temperature"] is None
    assert "奇点" in body["point"]["reason"]

    assert body["peak"]["singular"] is True
    assert body["peak"]["x"] is None
    assert body["peak"]["temperature_rise"] is None
    assert body["peak"]["temperature"] is None
    assert "无界" in body["peak"]["reason"]

    # 近奇点仍应能继续利用“中线无界”的信息给出熔池，而不是返回失控温度。
    assert body["melt_pool"]["formed"] is True
    assert body["melt_pool"]["width"] > 0.0


@pytest.mark.parametrize("offset", [1.0e-9, 1.0e-7])
def test_tiny_transverse_offset_marks_unbounded_peak_but_keeps_valid_point(
    client, demo_process, offset
):
    # x 仍在正常毫米尺度；横向偏移很小时，该点本身约为 764 K，
    # 但这条观察线的沿程峰值会逼近热源奇点，不能返回失控的巨大“峰值”。
    point = {"x": -0.01, "y": offset, "z": 0.0}
    resp = _calculate(client, demo_process, point)
    assert resp.status_code == 200
    body = resp.json()

    assert body["point"]["singular"] is False
    assert body["point"]["temperature_rise"] == pytest.approx(763.9437268410977, rel=1e-8)

    assert body["peak"]["singular"] is True
    assert body["peak"]["temperature_rise"] is None
    assert body["peak"]["temperature"] is None
    assert "无界" in body["peak"]["reason"]


def test_normal_offset_query_remains_accurate(client, demo_process):
    point = {"x": -0.01, "y": 0.003, "z": 0.0}
    resp = _calculate(client, demo_process, point, melting_temperature=1500.0)
    assert resp.status_code == 200
    body = resp.json()

    assert body["point"]["singular"] is False
    assert body["point"]["temperature_rise"] == pytest.approx(667.5903651920434, rel=1e-10)
    assert body["point"]["temperature"] == pytest.approx(692.5903651920434, rel=1e-10)

    assert body["peak"]["singular"] is False
    assert body["peak"]["temperature_rise"] == pytest.approx(1550.2404743706802, rel=1e-8)
    assert body["peak"]["temperature"] == pytest.approx(1575.2404743706802, rel=1e-8)

    assert body["melt_pool"]["formed"] is True
    assert body["melt_pool"]["width"] == pytest.approx(0.006215335262706503, rel=1e-8)


def test_power_doubling_preserves_finite_proportional_scaling(client, demo_process):
    point = {"x": -0.01, "y": 0.003, "z": 0.0}
    first = _calculate(client, demo_process, point).json()
    doubled_process = {**demo_process, "power": demo_process["power"] * 2.0}
    second = _calculate(client, doubled_process, point).json()

    assert first["effective_power"] * 2.0 == second["effective_power"]
    assert second["point"]["temperature_rise"] == pytest.approx(
        2.0 * first["point"]["temperature_rise"], rel=1e-10
    )
    assert second["peak"]["temperature_rise"] == pytest.approx(
        2.0 * first["peak"]["temperature_rise"], rel=1e-10
    )
    assert second["point"]["singular"] is False
    assert second["peak"]["singular"] is False


def test_power_doubling_does_not_release_near_singular_value(client, demo_process):
    point = {"x": 0.0, "y": 1.0e-9, "z": 0.0}
    first = _calculate(client, demo_process, point).json()
    doubled_process = {**demo_process, "power": demo_process["power"] * 2.0}
    second = _calculate(client, doubled_process, point).json()

    assert first["peak"]["singular"] is True
    assert second["peak"]["singular"] is True
    assert first["peak"]["temperature"] is None
    assert second["peak"]["temperature"] is None


def test_batch_singular_item_does_not_affect_other_items(client, demo_process):
    body = {
        "items": [
            {"process": demo_process, "point": {"x": 0.0, "y": 1.0e-9, "z": 0.0}},
            {
                "process": demo_process,
                "point": {"x": -0.01, "y": 0.003, "z": 0.0},
            },
        ]
    }
    resp = client.post("/v1/calculate/batch", json=body)
    assert resp.status_code == 200
    results = resp.json()["results"]

    assert [item["ok"] for item in results] == [True, True]
    assert results[0]["result"]["peak"]["singular"] is True
    assert results[0]["result"]["peak"]["temperature"] is None
    assert results[1]["result"]["peak"]["singular"] is False
    assert results[1]["result"]["point"]["temperature_rise"] == pytest.approx(
        667.5903651920434, rel=1e-10
    )

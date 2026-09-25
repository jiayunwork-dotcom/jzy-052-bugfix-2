"""非法输入必须被挡回并给出原因；q = 0 的退化情形必须算得出零。"""

import pytest


def _post(client, process, point=None, **extra):
    body = {"process": process, "point": point or {"x": -0.01, "y": 0.0, "z": 0.0}}
    body.update(extra)
    return client.post("/v1/calculate", json=body)


@pytest.mark.parametrize(
    "patch,keyword",
    [
        ({"travel_speed": 0.0}, "travel_speed"),
        ({"travel_speed": -0.005}, "travel_speed"),
        ({"conductivity": 0.0}, "conductivity"),
        ({"conductivity": -50.0}, "conductivity"),
        ({"diffusivity": 0.0}, "diffusivity"),
        ({"diffusivity": -1.2e-5}, "diffusivity"),
        ({"efficiency": 1.5}, "efficiency"),
        ({"efficiency": -0.2}, "efficiency"),
        ({"power": -100.0}, "power"),
        ({"heat_input": -1.0, "power": None}, "heat_input"),
        ({"mode": "medium"}, "mode"),
        ({"mode": "3d"}, "mode"),
    ],
)
def test_invalid_process_parameters_rejected(client, demo_process, patch, keyword):
    proc = {**demo_process, **patch}
    resp = _post(client, proc)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "invalid_input"
    assert any(keyword in reason for reason in detail["reasons"])


def test_thick_mode_requires_z(client, demo_process):
    resp = _post(client, demo_process, point={"x": -0.01, "y": 0.0})
    assert resp.status_code == 422
    assert any("z" in r for r in resp.json()["detail"]["reasons"])


def test_thin_mode_requires_thickness(client, demo_process):
    proc = {**demo_process, "mode": "thin"}
    resp = _post(client, proc, point={"x": -0.01, "y": 0.0})
    assert resp.status_code == 422
    assert any("thickness" in r for r in resp.json()["detail"]["reasons"])


def test_power_sources_are_mutually_exclusive(client, demo_process):
    both = {**demo_process, "heat_input": 6.0e5}
    assert _post(client, both).status_code == 422
    neither = {k: v for k, v in demo_process.items() if k != "power"}
    assert _post(client, neither).status_code == 422


def test_melting_temperature_must_exceed_initial(client, demo_process):
    resp = _post(client, demo_process, melting_temperature=20.0)
    assert resp.status_code == 422
    assert any("melting_temperature" in r for r in resp.json()["detail"]["reasons"])


def test_zero_power_is_legal_and_yields_zero_rise(client, demo_process):
    """有效功率为零：全场温升为零、温度恒等于 T0，不报错。"""
    proc = {**demo_process, "power": 0.0}
    resp = _post(client, proc, point={"x": 0.0, "y": 0.0, "z": 0.0}, melting_temperature=1500.0)
    assert resp.status_code == 200
    body = resp.json()
    assert body["effective_power"] == 0.0
    assert body["point"]["temperature_rise"] == 0.0
    assert body["point"]["temperature"] == 25.0
    assert body["peak"]["temperature_rise"] == 0.0
    assert body["melt_pool"]["formed"] is False


def test_zero_efficiency_is_legal_and_yields_zero_rise(client, demo_process):
    proc = {**demo_process, "efficiency": 0.0}
    resp = _post(client, proc)
    assert resp.status_code == 200
    assert resp.json()["point"]["temperature_rise"] == 0.0

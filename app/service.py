"""核算编排：温度场 → 峰值扫描 → 熔宽估计。纯函数实现，无框架依赖，天然可并发。"""

from __future__ import annotations

import math
from collections.abc import Callable

from .physics import thick as thick_field
from .physics import thin as thin_field
from .schemas import PointInput, SolverOptions
from .solver import MAX_PHYSICAL_RISE, PeakResult, estimate_melt_width, scan_peak
from .validation import THICK, Process


def _field_and_rho(proc: Process, y: float, z: float | None) -> tuple[Callable[[float], float], float]:
    """构造固定 (y, z) 下只随 x 变化的温升函数，以及观察线到中线的距离 ρ。"""
    q = proc.effective_power
    k = proc.conductivity
    alpha = proc.diffusivity
    v = proc.travel_speed
    if proc.mode == THICK:

        def field(x: float) -> float:
            return thick_field.temperature_rise(x, y, z, q=q, k=k, alpha=alpha, v=v)

        return field, math.hypot(y, z)

    def field(x: float) -> float:
        return thin_field.temperature_rise(
            x, y, q=q, k=k, alpha=alpha, v=v, thickness=proc.thickness
        )

    return field, abs(y)


def calculate(
    proc: Process,
    point: PointInput,
    melting_temperature: float | None,
    solver: SolverOptions,
) -> dict:
    """对单组工艺参数与观察点做一次核算，返回可 JSON 序列化的结果字典。"""
    q = proc.effective_power
    t0 = proc.initial_temperature
    lam = proc.travel_speed / (2.0 * proc.diffusivity)
    z = point.z if proc.mode == THICK else None

    # 观察点温升；温升非有限或越过物理上限（紧贴热源奇点）时按奇点处理，
    # 与坐标严格落在热源上（温升 inf）走同一条路径
    field_at_point, rho = _field_and_rho(proc, point.y, z)
    rise = field_at_point(point.x)
    point_singular = not math.isfinite(rise) or rise > MAX_PHYSICAL_RISE

    # 沿 x 的峰值温度（观察点所在的 y、z 处）
    peak = scan_peak(
        field_at_point,
        rho=rho,
        lam=lam,
        q=q,
        xtol=solver.tolerance,
        max_iter=solver.max_iterations,
    )

    # 可选：按熔化等温线估算熔宽
    melt_payload = None
    if melting_temperature is not None:
        delta_tm = melting_temperature - t0

        def peak_at(y: float) -> PeakResult:
            field, r = _field_and_rho(proc, y, z)
            return scan_peak(
                field,
                rho=r,
                lam=lam,
                q=q,
                xtol=solver.tolerance,
                max_iter=solver.max_iterations,
            )

        melt = estimate_melt_width(
            peak_at,
            delta_tm=delta_tm,
            xtol=solver.tolerance,
            max_iter=solver.max_iterations,
        )
        melt_payload = {
            "formed": melt.formed,
            "melting_temperature": melting_temperature,
            "width": 2.0 * melt.half_width if melt.formed else None,
            "y_boundaries": [-melt.half_width, melt.half_width] if melt.formed else None,
            "depth_z": z,
        }
        if not melt.formed:
            melt_payload["reason"] = "整条温度曲线均低于熔化温度，该工艺下不会形成熔池"

    result = {
        "mode": proc.mode,
        "effective_power": q,
        "initial_temperature": t0,
        "point": {
            "x": point.x,
            "y": point.y,
            "z": z,
            "temperature_rise": None if point_singular else rise,
            "temperature": None if point_singular else t0 + rise,
            "singular": point_singular,
        },
        "peak": {
            "x": peak.x,
            "temperature_rise": peak.rise,
            "temperature": None if peak.singular else t0 + peak.rise,
            "singular": peak.singular,
        },
        "melt_pool": melt_payload,
    }
    if peak.singular:
        result["peak"]["note"] = "移动热源理想化解在热源正后方中线上发散，峰值温度无界（奇点）"
    return result

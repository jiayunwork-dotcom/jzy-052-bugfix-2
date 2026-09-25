"""物理约束校验：把结构合法的输入进一步挡在物理规则之外，并给出全部原因。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import InvalidInputError
from .schemas import PointInput, ProcessInput

THICK = "thick"
THIN = "thin"
VALID_MODES = (THICK, THIN)


@dataclass(frozen=True)
class Process:
    """校验通过、可直接用于求解的归一化工艺参数。"""

    mode: str
    travel_speed: float
    conductivity: float
    diffusivity: float
    efficiency: float
    initial_temperature: float
    effective_power: float  # 有效热功率 q；q = 0 为合法退化（全场温升为零）
    thickness: float | None


def _is_positive_finite(value: float) -> bool:
    return math.isfinite(value) and value > 0.0


def validate_process(p: ProcessInput) -> Process:
    reasons: list[str] = []

    if p.mode not in VALID_MODES:
        reasons.append(
            f"板厚模式 mode 只能取 'thick'（厚板三维点源）或 'thin'（薄板二维线源），收到 {p.mode!r}"
        )

    if not _is_positive_finite(p.travel_speed):
        reasons.append(f"行走速度 travel_speed 必须为正数，收到 {p.travel_speed!r}")
    if not _is_positive_finite(p.conductivity):
        reasons.append(f"热导率 conductivity 必须为正数，收到 {p.conductivity!r}")
    if not _is_positive_finite(p.diffusivity):
        reasons.append(f"热扩散率 diffusivity 必须为正数，收到 {p.diffusivity!r}")
    if not math.isfinite(p.efficiency) or not 0.0 <= p.efficiency <= 1.0:
        reasons.append(f"热效率 efficiency 必须位于 [0, 1] 区间，收到 {p.efficiency!r}")
    if not math.isfinite(p.initial_temperature):
        reasons.append(f"初始温度 initial_temperature 必须为有限数值，收到 {p.initial_temperature!r}")

    if (p.power is None) == (p.heat_input is None):
        reasons.append("必须且只能提供 power（总功率 W）或 heat_input（线能量 J/m）其中之一")
    else:
        source = p.power if p.power is not None else p.heat_input
        name = "power" if p.power is not None else "heat_input"
        if not math.isfinite(source) or source < 0.0:
            reasons.append(
                f"{name} 不得为负（有效热功率必须非负；取零时全场温升为零，属合法退化），收到 {source!r}"
            )

    thickness: float | None = None
    if p.mode == THIN:
        if p.thickness is None:
            reasons.append("薄板 thin 模式必须提供板厚 thickness（二维线源解按板厚分摊功率）")
        elif not _is_positive_finite(p.thickness):
            reasons.append(f"板厚 thickness 必须为正数，收到 {p.thickness!r}")
        else:
            thickness = p.thickness

    if reasons:
        raise InvalidInputError(reasons)

    gross = p.power if p.power is not None else p.heat_input * p.travel_speed
    return Process(
        mode=p.mode,
        travel_speed=p.travel_speed,
        conductivity=p.conductivity,
        diffusivity=p.diffusivity,
        efficiency=p.efficiency,
        initial_temperature=p.initial_temperature,
        effective_power=p.efficiency * gross,
        thickness=thickness,
    )


def validate_point(mode: str, point: PointInput) -> None:
    reasons: list[str] = []
    if not math.isfinite(point.x):
        reasons.append(f"观察点 x 必须为有限数值，收到 {point.x!r}")
    if not math.isfinite(point.y):
        reasons.append(f"观察点 y 必须为有限数值，收到 {point.y!r}")
    if mode == THICK:
        if point.z is None:
            reasons.append("厚板 thick 模式必须给出深度坐标 z")
        elif not math.isfinite(point.z):
            reasons.append(f"观察点 z 必须为有限数值，收到 {point.z!r}")
    if reasons:
        raise InvalidInputError(reasons)


def validate_melting_temperature(melting_temperature: float | None, initial_temperature: float) -> None:
    if melting_temperature is None:
        return
    if not math.isfinite(melting_temperature) or melting_temperature <= initial_temperature:
        raise InvalidInputError(
            [
                f"熔化温度 melting_temperature 必须高于初始温度 {initial_temperature}，"
                f"收到 {melting_temperature!r}"
            ]
        )

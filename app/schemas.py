"""HTTP 层数据结构：只做类型与结构约束，物理约束见 validation.py。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ProcessInput(BaseModel):
    """一套完整的材料与工艺参数。

    功率来源二选一：power（总功率 W）或 heat_input（线能量 J/m）；
    有效热功率 q = efficiency × power = efficiency × heat_input × travel_speed。
    """

    model_config = ConfigDict(extra="forbid")

    mode: str  # "thick" 厚板三维点源 / "thin" 薄板二维线源
    travel_speed: float  # 行走速度 v，m/s
    conductivity: float  # 热导率 k，W/(m·K)
    diffusivity: float  # 热扩散率 α，m²/s
    efficiency: float = 1.0  # 热效率 η，[0, 1]
    initial_temperature: float = 20.0  # 母材初始温度 T0
    power: float | None = None  # 总功率 W
    heat_input: float | None = None  # 线能量 J/m
    thickness: float | None = None  # 板厚 m，thin 模式必填


class PointInput(BaseModel):
    """随热源平动坐标系中的观察点。"""

    model_config = ConfigDict(extra="forbid")

    x: float  # 沿焊接方向、以热源为原点的相对坐标（热源后方为负）
    y: float = 0.0  # 板面内横向偏移
    z: float | None = None  # 板厚方向深度，thick 模式必填


class SolverOptions(BaseModel):
    """峰值扫描 / 熔宽求解的数值控制。"""

    model_config = ConfigDict(extra="forbid")

    max_iterations: int = Field(default=200, ge=1, le=100000)
    tolerance: float = Field(default=1e-10, gt=0.0, le=1.0)


class CalcRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    process: ProcessInput | None = None  # 临时带入整套参数
    preset: str | None = None  # 或点名已登记的工艺档
    point: PointInput
    melting_temperature: float | None = None  # 给出则估算熔宽
    solver: SolverOptions = Field(default_factory=SolverOptions)


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[Any]  # 逐条独立校验，个别非法或不收敛不影响其余条目


class PresetPutRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    process: ProcessInput
    description: str = ""

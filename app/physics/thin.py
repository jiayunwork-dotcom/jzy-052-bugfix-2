"""薄板二维线热源 Rosenthal 准稳态解（功率沿板厚均布，温度沿板厚方向抹平）。

    T - T0 = q / (2πk·d) · exp(-vx / (2α)) · K0(v·r / (2α)),  r = √(x² + y²)

K0 为零阶第二类修正贝塞尔函数，随距离约按 1/√r 衰减，慢于三维点源的 1/R。
两支公式各自独立，不得混用同一条衰减律。
"""

from __future__ import annotations

import math

from scipy.special import k0e


def temperature_rise(
    x: float, y: float, *, q: float, k: float, alpha: float, v: float, thickness: float
) -> float:
    """返回观察点温升。q = 0 时恒为零；原点为线源奇点，返回 math.inf。"""
    if q == 0.0:
        return 0.0
    rho_sq = y * y
    r = math.sqrt(x * x + rho_sq)
    if r == 0.0:
        return math.inf
    lam = v / (2.0 * alpha)
    if x < 0.0:
        r_plus_x = rho_sq / (r - x) if rho_sq > 0.0 else 0.0
    else:
        r_plus_x = r + x
    # K0(λr)·exp(-λx) = k0e(λr)·exp(-λ(r+x))；k0e 为指数放缩的 K0，
    # 避免 exp(-λx) 在热源远后方溢出、同时保持远场精度
    return (
        q
        / (2.0 * math.pi * k * thickness)
        * float(k0e(lam * r))
        * math.exp(-lam * r_plus_x)
    )

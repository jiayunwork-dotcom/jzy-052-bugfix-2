"""厚板三维点热源 Rosenthal 准稳态解。

坐标系随热源平动：热源恒位于原点、向 +x 方向匀速前进。

    T - T0 = q / (2πkR) · exp(-v(R + x) / (2α)),  R = √(x² + y² + z²)

q 为有效热功率，k 为热导率，α 为热扩散率，v 为行走速度。
"""

from __future__ import annotations

import math


def temperature_rise(
    x: float, y: float, z: float, *, q: float, k: float, alpha: float, v: float
) -> float:
    """返回观察点温升。

    q = 0 时全场温升为零（合法退化）；原点为点源奇点，返回 math.inf。
    """
    if q == 0.0:
        return 0.0
    rho_sq = y * y + z * z
    r = math.sqrt(x * x + rho_sq)
    if r == 0.0:
        return math.inf
    lam = v / (2.0 * alpha)
    # x < 0 时 R + x = ρ² / (R - x)，避免 |x| >> ρ 时的灾难性抵消
    if x < 0.0:
        r_plus_x = rho_sq / (r - x) if rho_sq > 0.0 else 0.0
    else:
        r_plus_x = r + x
    return q / (2.0 * math.pi * k * r) * math.exp(-lam * r_plus_x)

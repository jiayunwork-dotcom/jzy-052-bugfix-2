"""沿焊接方向的一维峰值扫描与熔池宽度估计。

所有搜索均带步数上限：在允许步数内不收敛即抛 NonConvergenceError，
绝不返回一个凑数的值。
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from .errors import NonConvergenceError

_BRACKET_EXPANSION_LIMIT = 60


@dataclass(frozen=True)
class PeakResult:
    x: float | None  # 峰值位置（singular 时无意义）
    rise: float | None  # 峰值温升（singular 时为 None）
    singular: bool  # 观察线穿过热源奇点，峰值无界


@dataclass(frozen=True)
class MeltResult:
    formed: bool
    half_width: float | None  # 横向熔化边界 |y*|；未形成熔池时为 None


def _golden_section_max(
    f: Callable[[float], float], a: float, b: float, xtol: float, max_iter: int
) -> tuple[float, float]:
    """在 [a, b] 上求单峰函数 f 的最大值点（黄金分割搜索）。"""
    scale = abs(b - a) or 1.0
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(max_iter):
        if abs(b - a) <= xtol * scale:
            x = 0.5 * (a + b)
            return x, f(x)
        if fc < fd:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
        else:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
    raise NonConvergenceError(
        f"峰值一维搜索在 {max_iter} 步内未收敛"
        f"（当前区间宽度 {abs(b - a):.3e}，目标 {xtol * scale:.3e}）"
    )


def scan_peak(
    field: Callable[[float], float], *, rho: float, lam: float, q: float, xtol: float, max_iter: int
) -> PeakResult:
    """固定 (y, z)，沿 x 扫描温升曲线取最大值。

    field 为只随 x 变化的温升函数；rho 为观察线到焊道中线的距离
    （厚板 √(y²+z²)，薄板 |y|）；lam = v/(2α)。
    rho = 0 时观察线穿过热源奇点，峰值无界，标记 singular。
    """
    if q == 0.0:
        return PeakResult(x=0.0, rise=0.0, singular=False)
    if rho == 0.0:
        return PeakResult(x=None, rise=None, singular=True)
    # 峰值位于热源后方 x* ≈ -λρ²/2 附近；给出覆盖性左端点，必要时向外扩张
    a = -(4.0 * lam * rho * rho + 4.0 * rho + 1e-12)
    for _ in range(_BRACKET_EXPANSION_LIMIT):
        if field(0.5 * a) >= field(a):
            break
        a *= 2.0
    else:
        raise NonConvergenceError("峰值搜索无法在给定步数内围住极大值点（区间扩张失败）")
    x_star, f_star = _golden_section_max(field, a, 0.0, xtol, max_iter)
    return PeakResult(x=x_star, rise=f_star, singular=False)


def estimate_melt_width(
    peak_at: Callable[[float], PeakResult], *, delta_tm: float, xtol: float, max_iter: int
) -> MeltResult:
    """在中线附近求横向熔化边界。

    peak_at(y) 给出横向偏移 y 处沿 x 方向的峰值温升。边界 y* 满足
    峰值温升 = 熔化温升 delta_tm，熔宽估计 = 2·y*。
    整条曲线都达不到 delta_tm 时返回 formed=False（不会形成熔池）。
    """
    center = peak_at(0.0)
    center_rise = math.inf if center.singular else center.rise
    if not center_rise > delta_tm:
        return MeltResult(formed=False, half_width=None)

    # 向外扩张，找到峰值低于熔化温升的外侧边界
    y_hi = 1e-3
    for _ in range(_BRACKET_EXPANSION_LIMIT):
        pk = peak_at(y_hi)
        rise = math.inf if pk.singular else pk.rise
        if rise < delta_tm:
            break
        y_hi *= 2.0
    else:
        raise NonConvergenceError("熔宽估计：横向边界外扩搜索超出允许步数，熔化区异常宽广")

    # 二分求峰值温升 = delta_tm 的横向位置
    scale = y_hi
    y_lo = 0.0
    for _ in range(max_iter):
        if y_hi - y_lo <= xtol * scale:
            break
        mid = 0.5 * (y_lo + y_hi)
        pk = peak_at(mid)
        rise = math.inf if pk.singular else pk.rise
        if rise > delta_tm:
            y_lo = mid
        else:
            y_hi = mid
    else:
        raise NonConvergenceError(f"熔宽二分搜索在 {max_iter} 步内未收敛")

    return MeltResult(formed=True, half_width=0.5 * (y_lo + y_hi))

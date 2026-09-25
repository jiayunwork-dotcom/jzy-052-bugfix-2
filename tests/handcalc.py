"""测试内独立手算实现，用于交叉核对服务输出（与 app.physics 相互独立）。"""

import math

from scipy.special import k0


def hand_thick(x, y, z, q, k, alpha, v):
    """三维点源：ΔT = q/(2πkR)·exp(-v(R+x)/(2α))。"""
    R = math.sqrt(x * x + y * y + z * z)
    return q / (2.0 * math.pi * k * R) * math.exp(-v * (R + x) / (2.0 * alpha))


def hand_thin(x, y, q, k, alpha, v, d):
    """二维线源：ΔT = q/(2πk·d)·exp(-vx/(2α))·K0(v·r/(2α))。"""
    r = math.hypot(x, y)
    return (
        q
        / (2.0 * math.pi * k * d)
        * math.exp(-v * x / (2.0 * alpha))
        * float(k0(v * r / (2.0 * alpha)))
    )

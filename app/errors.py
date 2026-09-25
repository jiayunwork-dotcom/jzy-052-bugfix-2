"""服务级错误类型：输入非法、求解不收敛、工艺档缺失。"""

from __future__ import annotations


class InvalidInputError(Exception):
    """输入参数违反物理约束或组合规则。reasons 为面向调用方的原因列表。"""

    def __init__(self, reasons: list[str]):
        self.reasons = list(reasons)
        super().__init__("; ".join(self.reasons))


class NonConvergenceError(Exception):
    """峰值扫描或熔宽求解在允许步数内未能收敛。"""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class PresetNotFoundError(Exception):
    """点名的工艺档不存在。"""

    def __init__(self, name: str):
        self.name = name
        super().__init__(f"工艺档不存在: {name}")

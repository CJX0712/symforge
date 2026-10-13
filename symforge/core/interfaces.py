"""符号回归模型协议（所有方法：旗舰 / 消融 / 外部基线均实现）。

作者：晨星
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SymbolicModel(Protocol):
    """符号回归模型协议：fit(train) → predict(holdout) → 可导出表达式。"""

    name: str

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """仅在训练集上拟合（禁止触碰 holdout）。"""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测。"""
        ...

    def expression(self) -> str:
        """人类可读表达式（不可用时返回 ""）。"""
        ...

    def to_sympy(self, symbols: list[Any]) -> Any | None:
        """sympy 表达式（不可解析时返回 None）。"""
        ...

    def complexity(self) -> int:
        """表达式节点数（-1 表示不可得）。"""
        ...

    def metadata(self) -> dict[str, Any]:
        """附加信息（预算、后端等）。"""
        ...


__all__ = ["SymbolicModel"]

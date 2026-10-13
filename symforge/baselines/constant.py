"""基线 0：常数预测（mean(y_train)）—— 恢复率地板。

作者：晨星
"""

from __future__ import annotations

from typing import Any

import numpy as np


class ConstantModel:
    name = "constant"

    def __init__(self, seed: int = 0) -> None:
        self._mean = 0.0

    def available(self) -> bool:
        return True

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._mean = float(np.mean(np.asarray(y, dtype=float)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(np.asarray(X).shape[0], self._mean)

    def expression(self) -> str:
        return f"{self._mean:.6g}"

    def to_sympy(self, symbols: list[Any] | None = None):
        import sympy as sp

        return sp.Float(self._mean)

    def complexity(self) -> int:
        return 1

    def metadata(self) -> dict[str, Any]:
        return {"mean": self._mean}


__all__ = ["ConstantModel"]

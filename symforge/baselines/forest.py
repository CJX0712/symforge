"""基线 2：随机森林回归（sklearn Tier-0）/ kNN 回归（numpy Tier-1 兜底）。

作者：晨星
"""

from __future__ import annotations

from typing import Any

import numpy as np


class ForestBaseline:
    """通用非线性回归基线（不可符号化 → symbolic_exact = None）。"""

    name = "forest"

    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)
        self._backend = "none"
        self._model = None
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def available(self) -> bool:
        return True

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        try:
            import sklearn  # noqa: F401
            from sklearn.ensemble import RandomForestRegressor

            self._backend = "sklearn"
            self._model = RandomForestRegressor(
                n_estimators=200,
                random_state=self.seed,
                n_jobs=1,
            )
            self._model.fit(X, y)
        except ImportError:
            self._backend = "numpy_knn"
            self._X, self._y = X, y

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self._backend == "sklearn":
            return np.asarray(self._model.predict(X), dtype=float)
        if self._backend == "numpy_knn":
            assert self._X is not None and self._y is not None
            # k=5 距离加权 kNN（纯 numpy 离线兜底）
            d2 = ((X[:, None, :] - self._X[None, :, :]) ** 2).sum(axis=2)
            k = min(5, self._X.shape[0])
            nn = np.argpartition(d2, k - 1, axis=1)[:, :k]
            out = np.empty(X.shape[0])
            for i in range(X.shape[0]):
                dist = d2[i, nn[i]]
                w = 1.0 / (dist + 1e-12)
                out[i] = float(np.sum(w * self._y[nn[i]]) / np.sum(w))
            return out
        raise RuntimeError("model not fitted")

    def expression(self) -> str:
        return ""

    def to_sympy(self, symbols: list[Any] | None = None):
        return None

    def complexity(self) -> int:
        return -1

    def metadata(self) -> dict[str, Any]:
        return {"backend": self._backend}


__all__ = ["ForestBaseline"]

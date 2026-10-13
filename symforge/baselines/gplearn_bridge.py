"""基线 3：gplearn 0.4.3 SymbolicRegressor（外部 Tier-0 开源 GP-SR）。

公平对比约定：**同一函数集**（add/sub/mul/div + sqrt/log/sin/cos/abs/neg，
双方均无 exp）、**同一评估预算**（population_size = 旗舰 pop×n_islands，
generations 一致）、同一数据切分、同一 tournament_size、同一 ERC 常量
终端范围 const_range=(-8,8)（gplearn 支持 const_range，此前的
「结构性缺失常量终端」判断有误，已修正并对齐）。

作者：晨星
"""

from __future__ import annotations

from typing import Any

import numpy as np

from symforge.symreg.operators import gplearn_function_set
from symforge.symreg.simplify import program_str_to_sympy
from symforge.symreg.tree import var_symbols

_GPLEARN_OK = True
try:
    from gplearn.genetic import SymbolicRegressor
except ImportError:  # pragma: no cover - 离线环境
    _GPLEARN_OK = False


def available() -> bool:
    return _GPLEARN_OK


class GplearnModel:
    """gplearn 基线（等预算）。"""

    name = "gplearn"

    def __init__(
        self,
        seed: int = 0,
        population_size: int = 150,
        generations: int = 25,
        tournament_size: int = 4,
    ) -> None:
        self.seed = int(seed)
        self.population_size = int(population_size)
        self.generations = int(generations)
        self.tournament_size = int(tournament_size)
        self._model = None
        self._symbols: list[Any] = []
        self._program_str = ""
        self._backend = "gplearn" if _GPLEARN_OK else "unavailable"

    def available(self) -> bool:
        return _GPLEARN_OK

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if not _GPLEARN_OK:
            raise ImportError("gplearn 不可用")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._symbols = var_symbols(X.shape[1])
        self._model = SymbolicRegressor(
            population_size=self.population_size,
            generations=self.generations,
            tournament_size=self.tournament_size,
            function_set=gplearn_function_set(),
            const_range=(-8.0, 8.0),  # 与旗舰 ERC 范围对齐
            parsimony_coefficient=0.0001,  # 与旗舰 parsimony_coef 对齐
            metric="mse",
            p_crossover=0.70,
            p_subtree_mutation=0.15,
            p_hoist_mutation=0.0,
            p_point_mutation=0.10,
            p_point_replace=0.05,
            max_samples=1.0,
            random_state=self.seed,
            n_jobs=1,
        )
        self._model.fit(X, y)
        self._program_str = str(self._model._program)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("model not fitted")
        return np.asarray(self._model.predict(np.asarray(X, dtype=float)), dtype=float)

    def expression(self) -> str:
        return self._program_str

    def to_sympy(self, symbols: list[Any] | None = None):
        if not self._program_str:
            return None
        syms = symbols if symbols is not None else self._symbols
        try:
            return program_str_to_sympy(self._program_str, len(syms))
        except Exception:
            return None

    def complexity(self) -> int:
        if self._model is None:
            return -1
        return int(self._model._program.length_)

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self._backend,
            "population_size": self.population_size,
            "generations": self.generations,
        }


__all__ = ["GplearnModel", "available"]

"""GP 模型封装：GeneticProgrammer → SymbolicModel 协议。

作者：晨星
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from symforge.core.config import SRConfig
from symforge.symreg.gp import GeneticProgrammer, predict_individual
from symforge.symreg.program import program_str, program_to_sympy
from symforge.symreg.tree import var_symbols


class GPModel:
    """GP 引擎模型（旗舰与全部消融配置共用同一实现，仅 config 不同）。"""

    def __init__(self, name: str, config: SRConfig, seed: int) -> None:
        self.name = name
        self.cfg = config
        self.seed = int(seed)
        self._engine = GeneticProgrammer(config, seed)
        self._result = None
        self._fit_sec = float("nan")
        self._symbols: list[Any] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        t0 = time.perf_counter()
        self._result = self._engine.fit(X, np.asarray(y, dtype=float))
        self._fit_sec = time.perf_counter() - t0
        self._symbols = var_symbols(X.shape[1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._result is None:
            raise RuntimeError("model not fitted")
        return predict_individual(self._result.best, np.asarray(X, dtype=float))

    def expression(self) -> str:
        if self._result is None:
            return ""
        return program_str(self._result.best.prog)

    def to_sympy(self, symbols: list[Any] | None = None):
        if self._result is None:
            return None
        syms = symbols if symbols is not None else self._symbols
        if not syms:
            return None
        return program_to_sympy(self._result.best.prog, syms)

    def complexity(self) -> int:
        if self._result is None:
            return -1
        return self._result.best.size

    def metadata(self) -> dict[str, Any]:
        if self._result is None:
            return {}
        return {
            "fit_sec": self._fit_sec,
            "n_evals": self._result.n_evals,
            "generations_run": self._result.generations_run,
            "converged": self._result.converged,
            "best_mse": self._result.best.mse,
            "linear_scaling": self.cfg.linear_scaling,
            "n_islands": self.cfg.n_islands,
            "population_size": self.cfg.population_size,
            "generations": self.cfg.generations,
        }


__all__ = ["GPModel"]

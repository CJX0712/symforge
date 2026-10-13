"""基线 1：多项式回归（sklearn Tier-0 / numpy 岭回归 Tier-1 兜底）。

公平性：多项式阶数经**训练集内 3 折 CV** 选择（{1..6}），不触碰 holdout。
可符号化：系数 + 幂次矩阵 → sympy 表达式。

作者：晨星
"""

from __future__ import annotations

from typing import Any

import numpy as np

from symforge.symreg.simplify import poly_to_sympy
from symforge.symreg.tree import var_symbols

MAX_DEGREE = 6


def _poly_features_np(X: np.ndarray, degree: int) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """numpy 多项式特征（degree ≤ 2 的完整交互；degree ≥ 3 用逐列幂 + 成对交互）。

    为保证与 sklearn PolynomialFeatures(degree=d) 语义接近（含全部 ≤d 阶单项），
     Tier-0 路径直接用 sklearn；此 numpy 路径为离线兜底，语义 = 全部单项式（含交互）。
    """
    n, d = X.shape
    powers: list[tuple[int, ...]] = [tuple([0] * d)]
    # 单项式：逐列 1..degree
    for j in range(d):
        for p in range(1, degree + 1):
            powers.append(tuple(p if k == j else 0 for k in range(d)))
    # 成对交互（次数 2）
    if degree >= 2:
        for a in range(d):
            for b in range(a + 1, d):
                powers.append(tuple(1 if k in (a, b) else 0 for k in range(d)))
    # 三阶交互（次数 3，封顶防止组合爆炸）
    if degree >= 3:
        for a in range(d):
            for b in range(a + 1, d):
                for c in range(b + 1, d):
                    powers.append(tuple(1 if k in (a, b, c) else 0 for k in range(d)))
    feats = np.empty((n, len(powers)), dtype=float)
    for i, pw in enumerate(powers):
        col = np.ones(n)
        for k, p in enumerate(pw):
            if p:
                col = col * X[:, k] ** p
        feats[:, i] = col
    return feats, powers


def _fit_ridge(feats: np.ndarray, y: np.ndarray, lam: float = 1e-8) -> tuple[np.ndarray, float]:
    """闭式岭回归（含截距列）。"""
    design = np.hstack([feats, np.ones((feats.shape[0], 1))])
    ata = design.T @ design + lam * np.eye(design.shape[1])
    aty = design.T @ y
    coef = np.linalg.solve(ata, aty)
    return coef[:-1], float(coef[-1])


def _select_degree(X: np.ndarray, y: np.ndarray, backend: str) -> int:
    """训练集内 3 折 CV 选阶（fold 只在 train 上切；不触碰 holdout）。"""
    n = X.shape[0]
    rng = np.random.Generator(np.random.PCG64(12345))
    idx = rng.permutation(n)
    folds = np.array_split(idx, 3)
    best_deg, best_err = 1, float("inf")
    for deg in range(1, MAX_DEGREE + 1):
        errs = []
        for f in range(3):
            val = folds[f]
            tr = np.concatenate([folds[k] for k in range(3) if k != f])
            try:
                if backend == "sklearn":
                    from sklearn.linear_model import LinearRegression
                    from sklearn.preprocessing import PolynomialFeatures

                    pf = PolynomialFeatures(degree=deg, include_bias=False)
                    lr = LinearRegression()
                    lr.fit(pf.fit_transform(X[tr]), y[tr])
                    pred = lr.predict(pf.transform(X[val]))
                else:
                    feats, _ = _poly_features_np(X[tr], deg)
                    coef, intercept = _fit_ridge(feats, y[tr])
                    vfeats, _ = _poly_features_np(X[val], deg)
                    pred = vfeats @ coef + intercept
                errs.append(float(np.mean((pred - y[val]) ** 2)))
            except Exception:
                errs.append(float("inf"))
        mean_err = float(np.mean(errs))
        if mean_err < best_err - 1e-12:
            best_deg, best_err = deg, mean_err
    return best_deg


class PolyRegression:
    """多项式回归基线（符号可判等）。"""

    name = "poly"

    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)
        self._backend = "none"
        self._coef: np.ndarray | None = None
        self._intercept = 0.0
        self._powers: list[tuple[int, ...]] = []
        self._degree = 1
        self._symbols: list[Any] = []

    def available(self) -> bool:
        return True  # sklearn 与 numpy 双路径均可用

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        try:
            import sklearn  # noqa: F401

            self._backend = "sklearn"
        except ImportError:
            self._backend = "numpy"
        self._degree = _select_degree(X, y, self._backend)
        if self._backend == "sklearn":
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import PolynomialFeatures

            pf = PolynomialFeatures(degree=self._degree, include_bias=False)
            feats = pf.fit_transform(X)
            lr = LinearRegression()
            lr.fit(feats, y)
            self._coef = np.asarray(lr.coef_, dtype=float)
            self._intercept = float(lr.intercept_)
            self._powers = [tuple(int(v) for v in row) for row in pf.powers_]
        else:
            feats, powers = _poly_features_np(X, self._degree)
            self._coef, self._intercept = _fit_ridge(feats, y)
            self._powers = powers
        self._symbols = var_symbols(X.shape[1])

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        feats = np.empty((n, len(self._powers)), dtype=float)
        for i, pw in enumerate(self._powers):
            col = np.ones(n)
            for k, p in enumerate(pw):
                if p:
                    col = col * X[:, k] ** p
            feats[:, i] = col
        return feats

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._coef is None:
            raise RuntimeError("model not fitted")
        return self._transform(X) @ self._coef + self._intercept

    def expression(self) -> str:
        if self._coef is None:
            return ""
        return f"poly(degree={self._degree}, {len(self._powers)} terms)"

    def to_sympy(self, symbols: list[Any] | None = None):
        if self._coef is None:
            return None
        syms = symbols if symbols is not None else self._symbols
        return poly_to_sympy(np.array(self._powers), self._coef, self._intercept, syms)

    def complexity(self) -> int:
        return len(self._powers)

    def metadata(self) -> dict[str, Any]:
        return {"backend": self._backend, "degree": self._degree, "n_terms": len(self._powers)}


__all__ = ["PolyRegression"]

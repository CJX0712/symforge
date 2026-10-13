"""基准数据套件：10 个带符号真值的公式发现任务 + 噪声鲁棒性子套件。

真值来源（世界顶级基准，零杜撰）：
- Nguyen-1/2/4/5：White et al. 2013, "Better GP benchmarks for symbolic
  regression"（Nguyen et al. 2009 原套件，SRBench/La Cava 2021 沿用）
- pendulum / projectile / coulomb / lens / recip_sum3：经典物理公式
  （AI Feynman, Udrescu & Tegmark 2020, SciML 基准同款任务族）
- rational：x + 1/x（保护除法标准测试）

数据切分：train 40 点（SR 低数据范式）/ holdout 200 点，seeded 生成器，
无 shuffle 需求（非时序），无泄漏：y 仅由真值函数生成，holdout 不参与 fit。

作者：晨星
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp

from symforge.core.errors import DataError
from symforge.core.seed import seed_of
from symforge.symreg.tree import var_symbols

PI = sp.pi


@dataclass(frozen=True)
class DatasetSpec:
    """一个公式发现任务。"""

    name: str
    ndim: int
    truth: sp.Expr
    lows: tuple[float, ...]
    highs: tuple[float, ...]
    difficulty: str
    source: str
    n_train: int = 40
    n_holdout: int = 200


def _spec(name: str, expr: sp.Expr, lows, highs, difficulty: str, source: str) -> DatasetSpec:
    return DatasetSpec(
        name=name,
        ndim=(expr.free_symbols and len(expr.free_symbols)) or 1,
        truth=expr,
        lows=tuple(lows),
        highs=tuple(highs),
        difficulty=difficulty,
        source=source,
    )


def _build_suite() -> dict[str, DatasetSpec]:
    x0, x1, x2 = var_symbols(3)
    suite = {
        # ---- Nguyen 套件（White et al. 2013）----
        "nguyen1": _spec("nguyen1", x0**3 + x0**2 + x0, [-1], [1], "easy", "White2013/Nguyen-1"),
        "nguyen2": _spec("nguyen2", x0**4 + x0**3 + x0**2 + x0, [-1], [1], "easy", "White2013/Nguyen-2"),
        "nguyen4": _spec("nguyen4", x0**6 - 2 * x0**4 + x0**2, [-1], [1], "medium", "White2013/Nguyen-4"),
        "nguyen5": _spec("nguyen5", x0**5 - 2 * x0**3 + x0, [-1], [1], "medium", "White2013/Nguyen-5"),
        # ---- 经典函数/保护除法 ----
        "rational": _spec("rational", x0 + 1 / x0, [0.5], [2.0], "medium", "protected-division classic"),
        # ---- 物理公式（AI Feynman 任务族）----
        "pendulum": _spec(
            "pendulum",
            2 * PI * sp.sqrt(x0 / 10),
            [0.5],
            [2.0],
            "medium",
            "simple pendulum T=2π√(L/g), g≈10",
        ),
        "projectile": _spec(
            "projectile",
            x0**2 * sp.sin(2 * x1) / 10,
            [10.0, 0.2],
            [20.0, 1.2],
            "medium",
            "projectile range R=v²sin(2θ)/g, g≈10",
        ),
        "coulomb": _spec(
            "coulomb",
            x0 * x1 / x2**2,
            [1.0, 1.0, 1.0],
            [3.0, 3.0, 3.0],
            "hard",
            "Coulomb F=k·q₁q₂/r², k=1",
        ),
        "lens": _spec("lens", x0 * x1 / (x0 + x1), [1.0, 1.0], [4.0, 4.0], "hard", "thin lens f=uv/(u+v)"),
        "recip_sum3": _spec(
            "recip_sum3",
            3 / x0 + 3 / x1 + 3 / x2,
            [1.0, 1.0, 1.0],
            [3.0, 3.0, 3.0],
            "hard",
            "Keijzer-15 family: Σ 3/xᵢ",
        ),
    }
    # 修正 ndim（free_symbols 计数对常数项不敏感，按 lows/highs 长度为准）
    for spec in suite.values():
        object.__setattr__(spec, "ndim", len(spec.lows))
    return suite


SUITE: dict[str, DatasetSpec] = _build_suite()

# 噪声鲁棒性子套件（σ = noise_frac × std(y_train)）
NOISY_SUBSET: dict[str, float] = {"nguyen4": 0.05, "pendulum": 0.05, "lens": 0.05}

DIFFICULTY_ORDER = ("easy", "medium", "hard")


def list_datasets() -> list[str]:
    return list(SUITE.keys())


def get_spec(name: str) -> DatasetSpec:
    if name not in SUITE:
        raise DataError(f"E101: 未知数据集 {name!r}，可选 {sorted(SUITE)}")
    return SUITE[name]


@dataclass
class Dataset:
    """一次具体的数据实现（train / holdout / truth 函数）。"""

    name: str
    difficulty: str
    ndim: int
    X_train: np.ndarray
    y_train: np.ndarray
    X_holdout: np.ndarray
    y_holdout: np.ndarray
    truth: sp.Expr
    noise_frac: float = 0.0


def _sample(rng: np.random.Generator, spec: DatasetSpec, n: int) -> np.ndarray:
    cols = [rng.uniform(lo, hi, size=n) for lo, hi in zip(spec.lows, spec.highs, strict=True)]
    return np.column_stack(cols)


def make_dataset(name: str, seed: int, noise_frac: float = 0.0) -> Dataset:
    """生成一个数据集的 train/holdout（确定性：seed 派生）。"""
    spec = get_spec(name)
    rng = np.random.Generator(np.random.PCG64(seed_of(f"data::{name}") + int(seed)))
    X_train = _sample(rng, spec, spec.n_train)
    X_holdout = _sample(rng, spec, spec.n_holdout)
    fn = sp.lambdify(var_symbols(spec.ndim), spec.truth, "numpy")
    y_train = np.asarray(fn(*[X_train[:, i] for i in range(spec.ndim)]), dtype=float)
    y_holdout = np.asarray(fn(*[X_holdout[:, i] for i in range(spec.ndim)]), dtype=float)
    if noise_frac > 0:
        sd = float(np.std(y_train)) or 1.0
        y_train = y_train + rng.normal(0.0, noise_frac * sd, size=y_train.shape)
    return Dataset(
        name=spec.name,
        difficulty=spec.difficulty,
        ndim=spec.ndim,
        X_train=X_train,
        y_train=y_train,
        X_holdout=X_holdout,
        y_holdout=y_holdout,
        truth=spec.truth,
        noise_frac=noise_frac,
    )


def difficulty_of(name: str) -> str:
    return get_spec(name).difficulty


__all__ = [
    "NOISY_SUBSET",
    "SUITE",
    "Dataset",
    "DatasetSpec",
    "difficulty_of",
    "get_spec",
    "list_datasets",
    "make_dataset",
]

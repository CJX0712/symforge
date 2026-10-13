"""tests/test_gp.py — 进化引擎端到端契约。

锁定：
1. 确定性：同 seed 两次 fit 全程逐位一致（表达式 / 适应度 / 评估数）；
2. 恢复能力：简单目标（3·x0−1）小预算下 R² ≥ 0.99；
3. Pareto 档案支配规则（含等号——全等个体不得累积）；
4. 预算中立热启动：开/关播种均正常收敛（等评估数契约由基准层验证）；
5. 线性缩放系数闭式解与 numpy 参照一致。

作者：晨星
"""

from __future__ import annotations

import numpy as np
import pytest

from symforge.core.config import SRConfig
from symforge.symreg.gp import (
    GeneticProgrammer,
    GPResult,
    Individual,
    ParetoArchive,
    _linear_scaling_coef,
    predict_individual,
)
from symforge.symreg.program import flatten


def _small_cfg(**kw) -> SRConfig:
    base = dict(
        population_size=80,
        generations=8,
        n_islands=2,
        polish_top=2,
        migration_interval=4,
    )
    base.update(kw)
    cfg = SRConfig(**base)
    cfg.validate()
    return cfg


def _data():
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(60, 2))
    y = 3.0 * X[:, 0] - 1.0
    return X, y


def test_fit_deterministic() -> None:
    X, y = _data()
    r1 = GeneticProgrammer(_small_cfg(), seed=7).fit(X, y)
    r2 = GeneticProgrammer(_small_cfg(), seed=7).fit(X, y)
    assert r1.expression == r2.expression
    assert r1.best.fitness == r2.best.fitness
    assert r1.n_evals == r2.n_evals
    assert r1.generations_run == r2.generations_run


def test_simple_recovery() -> None:
    X, y = _data()
    res = GeneticProgrammer(_small_cfg(population_size=120, generations=15), seed=3).fit(X, y)
    pred = predict_individual(res.best, X)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    assert r2 > 0.99, f"简单目标未恢复: R²={r2:.4f} expr={res.expression}"


def test_gpresult_properties() -> None:
    X, y = _data()
    res = GeneticProgrammer(_small_cfg(generations=2), seed=1).fit(X, y)
    assert isinstance(res, GPResult)
    assert res.size >= 1 and res.depth >= 0
    assert isinstance(res.expression, str) and len(res.expression) > 0
    assert res.best.fitness < float("inf")


def test_pareto_equal_individuals_not_accumulated() -> None:
    """全等个体被支配（规则含等号）：档案不得被同一 (fitness,size) 占满。"""
    arch = ParetoArchive(cap=16)
    p = flatten(("var", 0))
    for _ in range(8):
        arch.update(Individual(prog=p, a=1.0, b=0.0, mse=0.5, fitness=0.5))
    assert len(arch.front) == 1
    # 严格更优者进入，旧者被移除
    arch.update(Individual(prog=p, a=1.0, b=0.0, mse=0.1, fitness=0.1))
    assert len(arch.front) == 1
    assert arch.front[0].fitness == 0.1


def test_pareto_front_no_dominated_members() -> None:
    rng = np.random.default_rng(5)
    arch = ParetoArchive(cap=64)
    for _ in range(200):
        fit = float(rng.uniform(0, 2))
        arch.update(Individual(prog=flatten(("var", 0)), fitness=fit, mse=fit))
    # size 均相同（=1），支配退化为 fitness 维：front 应只剩最小 fitness
    fits = [ind.fitness for ind in arch.front]
    assert len(fits) == 1 and fits[0] == min(fits)


def test_seeding_on_off_both_converge() -> None:
    X, y = _data()
    for use_seed in (True, False):
        res = GeneticProgrammer(_small_cfg(use_seeding=use_seed, generations=10), seed=11).fit(X, y)
        assert res.best.fitness < 0.5, f"use_seeding={use_seed} 未收敛"
        assert res.n_evals > 0


def test_linear_scaling_coef_reference() -> None:
    rng = np.random.default_rng(6)
    f = rng.uniform(-3, 3, size=50)
    y = 2.0 * f - 0.5 + rng.normal(0, 0.01, size=50)
    a, b = _linear_scaling_coef(f, y)
    ref_a, ref_b = np.polyfit(f, y, 1)
    assert abs(a - ref_a) < 1e-6 and abs(b - ref_b) < 1e-6
    # 恒定 f 退化：a=0, b=mean(y)
    a0, b0 = _linear_scaling_coef(np.full(10, 3.0), np.arange(10.0))
    assert a0 == 0.0 and abs(b0 - 4.5) < 1e-12


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

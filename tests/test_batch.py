"""tests/test_batch.py — 波次批量评估引擎契约。

锁定：
1. batch_fitness 与逐个体单路径评分一致（含 LS 系数 / MSE / fitness 公式）；
2. 分块路径（内存守卫触发）与单块路径逐位一致；
3. 确定性：同输入两次运行逐位一致；
4. 叶操作码（OP_VAR/OP_CONST）必须参与波次分组（漏叶事故回归测试）。

作者：晨星
"""

from __future__ import annotations

import numpy as np
import pytest

import symforge.symreg.batch_eval as be
from symforge.symreg.program import flatten, program_to_tree
from symforge.symreg.tree import eval_tree, random_tree


def _make_progs(rng, n=12, ndim=3):
    return [flatten(random_tree(rng, d, ndim, method="grow", use_erc=True)) for d in rng.integers(2, 6, size=n)]


def test_batch_vs_single_bitwise_consistent() -> None:
    rng = np.random.default_rng(0)
    progs = _make_progs(rng)
    X = rng.uniform(-2, 2, size=(64, 3))
    y = rng.uniform(-5, 5, size=64)
    for ls in (True, False):
        a_b, b_b, fit_b, mse_b = be.batch_fitness(progs, X, y, linear_scaling=ls, parsimony_coef=1e-4)
        for k, p in enumerate(progs):
            f = eval_tree(program_to_tree(p), X)
            if ls:
                s_ff, s_f, s_fy = float(np.sum(f * f)), float(np.sum(f)), float(np.sum(f * y))
                s_y = float(np.sum(y))
                det = 64 * s_ff - s_f * s_f
                a = (64 * s_fy - s_f * s_y) / det if abs(det) >= 1e-12 else 0.0
                b = (s_y - a * s_f) / 64
            else:
                a, b = 1.0, 0.0
            mse = float(np.mean((a * f + b - y) ** 2))
            assert a_b[k] == a, f"LS 系数 a 不一致 [{k}]"
            assert b_b[k] == b, f"LS 系数 b 不一致 [{k}]"
            assert mse_b[k] == mse, f"MSE 不一致 [{k}]"
            assert fit_b[k] == mse / (float(np.var(y)) + 1e-12) + 1e-4 * p.size


def test_chunked_equals_single_chunk_bitwise() -> None:
    """内存守卫分块后结果与单块逐位一致（数值不变性）。"""
    rng = np.random.default_rng(1)
    progs = _make_progs(rng, n=8)
    X = rng.uniform(-2, 2, size=(40, 3))
    y = rng.uniform(-5, 5, size=40)
    a1, b1, f1, m1 = be.batch_fitness(progs, X, y, True, 0.0)
    limit_backup = be._BUF_CELL_LIMIT
    try:
        be._BUF_CELL_LIMIT = 64  # 强制每 1-2 个程序分一块
        a2, b2, f2, m2 = be.batch_fitness(progs, X, y, True, 0.0)
    finally:
        be._BUF_CELL_LIMIT = limit_backup
    for x1, x2, name in ((a1, a2, "a"), (b1, b2, "b"), (f1, f2, "fitness"), (m1, m2, "mse")):
        assert np.array_equal(x1, x2), f"分块路径 {name} 与单块不一致"


def test_deterministic_repeated_runs() -> None:
    rng = np.random.default_rng(2)
    progs = _make_progs(rng)
    X = rng.uniform(-2, 2, size=(32, 3))
    y = rng.uniform(-5, 5, size=32)
    r1 = be.batch_fitness(progs, X, y, True, 1e-4)
    r2 = be.batch_fitness(progs, X, y, True, 1e-4)
    for x1, x2 in zip(r1, r2, strict=False):
        assert np.array_equal(x1, x2)


def test_leaf_ops_in_waves_regression() -> None:
    """回归：_OP_VALUES 必须含叶操作码 0/1——否则所有程序输出恒 0。

    事故：曾用 range(2,14) 分组，OP_VAR/OP_CONST 指令从未写入缓冲，
    np.empty 零页使全种群 fitness 恒等、搜索退化为纯 size 最小化。
    """
    assert 0 in be._OP_VALUES and 1 in be._OP_VALUES
    np.random.default_rng(3)
    # 纯叶程序：单变量 / 单常量，输出必须等于变量列 / 常量本身
    p_var = flatten(("var", 1))
    p_const = flatten(("const", 2.5))
    X = np.arange(12.0).reshape(6, 2)
    y = np.zeros(6)
    _a, _b, _f, mse = be.batch_fitness([p_var, p_const], X, y, False, 0.0)
    # y=0 且无缩放：mse = mean(f²)；var 程序 f=X[:,1] → mean(X[:,1]²)
    np.testing.assert_allclose(mse[0], float(np.mean(X[:, 1] ** 2)), rtol=1e-12)
    np.testing.assert_allclose(mse[1], 2.5**2, rtol=1e-12)


def test_empty_population() -> None:
    X = np.zeros((4, 2))
    y = np.zeros(4)
    out = be.batch_fitness([], X, y, True, 0.0)
    assert all(len(x) == 0 for x in out)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

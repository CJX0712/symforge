"""不变量自检（selftest）：受保护算子 / 树求值一致性 / 线性缩放 / 符号判定。

作者：晨星
"""

from __future__ import annotations

import numpy as np

from symforge.core.seed import make_rng, set_thread_env
from symforge.symreg.gp import _linear_scaling_coef
from symforge.symreg.operators import op_div, op_exp, op_log, op_sqrt
from symforge.symreg.simplify import program_str_to_sympy, symbolic_equal
from symforge.symreg.tree import (
    eval_tree,
    eval_tree_scalar,
    make_const,
    make_op,
    make_var,
    to_sympy,
    var_symbols,
)

OK = "✅"
BAD = "❌"


def _check(name: str, cond: bool, detail: str = "") -> bool:
    print(f"  [{OK if cond else BAD}] {name}" + (f"  {detail}" if detail else ""))
    return cond


def run_selftest(verbose: bool = False) -> int:
    set_thread_env()
    print("SymForge selftest — 不变量自检")
    print("=" * 60)
    results: list[bool] = []

    # 1) 受保护算子
    a = np.array([1.0, -4.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])
    results.append(_check("div 保护：除零返回 1.0", bool(np.all(op_div(a, b) == 1.0))))
    results.append(_check("sqrt 保护：负域取绝对值开方", bool(np.allclose(op_sqrt(a), np.sqrt(np.abs(a))))))
    results.append(_check("log 保护：0 处有限", bool(np.all(np.isfinite(op_log(np.array([0.0])))))))
    big = op_exp(np.array([1e5]))
    results.append(_check("exp 保护：clip 后有限", bool(np.all(np.isfinite(big)))))

    # 2) 向量化 vs 标量参照（SeqMineForge 式「严格晚于」交叉断言）
    rng = make_rng(7)
    from symforge.symreg.tree import random_tree

    tree = random_tree(rng, 5, 3)
    X = rng.uniform(-2, 2, size=(25, 3))
    v = eval_tree(tree, X)
    s = np.array([eval_tree_scalar(tree, X[i]) for i in range(X.shape[0])])
    results.append(_check("向量化 ≡ 标量参照（25 点）", bool(np.allclose(v, s, atol=1e-12))))

    # 3) 线性缩放：y = 3x + 2 在树 = x 上应被精确拟合
    X1 = np.linspace(-1, 1, 20).reshape(-1, 1)
    y1 = 3.0 * X1[:, 0] + 2.0
    f = eval_tree(make_var(0), X1)
    a_c, b_c = _linear_scaling_coef(f, y1)
    resid = a_c * f + b_c - y1
    results.append(
        _check("线性缩放闭式解精确（y=3x+2）", bool(np.max(np.abs(resid)) < 1e-9), f"a={a_c:.6f} b={b_c:.6f}")
    )

    # 4) 符号等价判定
    x = var_symbols(1)[0]
    truth = x**2 + 1
    results.append(_check("符号等价：(x+1)²−2x ≡ x²+1", symbolic_equal((x + 1) ** 2 - 2 * x, truth)))
    results.append(_check("符号不等：x²+2 ≢ x²+1", not symbolic_equal(x**2 + 2, truth)))
    # gplearn 程序串解析（X 命名变量）
    expr2 = program_str_to_sympy("add(sub(X0, X1), mul(X0, X1))", 2)
    results.append(_check("gplearn 程序串 → sympy", bool(expr2.free_symbols == set(var_symbols(2)))))

    # 5) 树 → sympy 往返
    t = make_op("add", make_op("mul", make_var(0), make_const(2.0)), make_var(0))
    e = to_sympy(t, var_symbols(1))
    results.append(_check("树 → sympy 结构正确", bool(e.equals(2 * x + x) or e == 3 * x)))

    n_ok = sum(results)
    n_all = len(results)
    print("=" * 60)
    print(f"selftest: {n_ok}/{n_all} 通过")
    return 0 if n_ok == n_all else 1


if __name__ == "__main__":
    raise SystemExit(run_selftest())

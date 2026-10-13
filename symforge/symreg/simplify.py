"""sympy 桥接：树/gplearn 程序串/多项式 → sympy，符号等价判定（带超时）。

作者：晨星
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

import numpy as np
import sympy as sp

from symforge.symreg.tree import to_sympy as tree_to_sympy
from symforge.symreg.tree import var_symbols

_SYMPY_FN = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "sqrt": sp.sqrt,
    "log": sp.log,
    "exp": sp.exp,
    "sin": sp.sin,
    "cos": sp.cos,
    "abs": sp.Abs,
    "neg": lambda a: -a,
    "inv": lambda a: 1 / a,
    "max": sp.Max,
    "min": sp.Min,
}

_TOKEN = re.compile(r"\s*(\(|\)|,|[A-Za-z_][A-Za-z_0-9]*|-?\d+\.\d+|-?\d+)")


def _tokenize(s: str) -> list[str]:
    pos = 0
    out: list[str] = []
    while pos < len(s):
        m = _TOKEN.match(s, pos)
        if not m:
            break
        out.append(m.group(1))
        pos = m.end()
    return out


def _parse_expr(tokens: list[str], i: int, symbols: list[sp.Symbol]):
    tok = tokens[i]
    if tok == "(":
        # 不应出现
        raise ValueError("unexpected (")
    if tok == ")":
        raise ValueError("unexpected )")
    # 变量 X0..Xn
    if re.fullmatch(r"X\d+", tok):
        idx = int(tok[1:])
        return symbols[idx], i + 1
    # 数值常量
    if re.fullmatch(r"-?\d+(\.\d+)?", tok):
        return sp.Float(float(tok)), i + 1
    # 函数调用
    if tok in _SYMPY_FN:
        if i + 1 >= len(tokens) or tokens[i + 1] != "(":
            raise ValueError(f"expected ( after {tok}")
        j = i + 2
        args: list[sp.Expr] = []
        while True:
            arg, j = _parse_expr(tokens, j, symbols)
            args.append(arg)
            if tokens[j] == ",":
                j += 1
                continue
            if tokens[j] == ")":
                return _SYMPY_FN[tok](*args), j + 1
            raise ValueError(f"unexpected token {tokens[j]!r}")
    raise ValueError(f"unknown token {tok!r}")


def program_str_to_sympy(s: str, ndim: int) -> sp.Expr:
    """gplearn 程序串（如 'add(sub(X0, X1), mul(X0, X1))'）→ sympy 表达式。"""
    tokens = _tokenize(s)
    if not tokens:
        raise ValueError("empty program")
    expr, end = _parse_expr(tokens, 0, var_symbols(ndim))
    if end != len(tokens):
        raise ValueError(f"trailing tokens: {tokens[end:]}")
    return expr


def poly_to_sympy(powers: np.ndarray, coefs: np.ndarray, intercept: float, symbols: list[sp.Symbol]) -> sp.Expr:
    """PolynomialFeatures 幂次矩阵 + 线性系数 → sympy 表达式。"""
    expr = sp.Float(float(intercept))
    for row, c in zip(powers, coefs, strict=False):
        if abs(c) < 1e-14:
            continue
        term = sp.Float(float(c))
        for p, sym in zip(row, symbols, strict=False):
            if p:
                term *= sym ** int(p)
        expr += term
    return sp.expand(expr)


def _rationalize_constants(expr: sp.Expr, denom: int = 64, tol: float = 1e-6) -> sp.Expr:
    """把 Float 常量替换为最接近的简单有理数（limit_denominator），
    使 2.0000001 之类数值恢复出的常量能与符号真值判等。"""
    floats = set(expr.atoms(sp.Float))
    if not floats:
        return expr
    subs: dict[sp.Float, sp.Rational] = {}
    for f in floats:
        v = float(f)
        r = sp.Rational(v).limit_denominator(denom)
        if abs(float(r) - v) <= tol * max(1.0, abs(v)):
            subs[f] = r
    return expr.subs(subs) if subs else expr


def _symbolic_equal_impl(expr: sp.Expr, truth: sp.Expr) -> bool:
    try:
        e2 = _rationalize_constants(expr)
        diff = sp.simplify(e2 - truth)
        if diff == 0:
            return True
        # 常数未完全对齐时的二次尝试：残差为极小数也算等价
        return bool(diff.is_number and abs(float(diff)) < 1e-9)
    except Exception:
        return False


def _expr_node_count(expr: sp.Expr) -> int:
    """sympy 表达式节点数（大表达式 sp.simplify 可能数分钟不归——预过滤）。"""
    return sum(1 for _ in sp.preorder_traversal(expr))


def symbolic_equal(expr: sp.Expr | None, truth: sp.Expr, timeout: float = 5.0, max_nodes: int = 80) -> bool:
    """符号等价判定（简化可能很慢 → 双重护栏）：

    1. 节点数预过滤：超过 max_nodes 直接 False（symbolic_exact 是附加指标，
       numeric_match 才是主门禁；SRBench 同样对 simplify 设界）；
    2. 线程超时 + **非阻塞 shutdown**——曾有事故：``with ThreadPoolExecutor``
       退出时隐式 join 工作线程，超时形同虚设，大表达式 simplify 挂住
       整个基准十余分钟。
    """
    if expr is None:
        return False
    try:
        if _expr_node_count(expr) > max_nodes:
            return False
    except Exception:
        return False
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        fut = pool.submit(_symbolic_equal_impl, expr, truth)
        return fut.result(timeout=timeout)
    except FutureTimeout:
        return False
    except Exception:
        return False
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def simplify_expression(expr: sp.Expr) -> str:
    try:
        return str(sp.simplify(expr))
    except Exception:
        return str(expr)


__all__ = [
    "_rationalize_constants",
    "poly_to_sympy",
    "program_str_to_sympy",
    "simplify_expression",
    "symbolic_equal",
    "tree_to_sympy",
]

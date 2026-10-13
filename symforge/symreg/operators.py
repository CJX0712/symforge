"""受保护算子集（GP 函数集，与 gplearn 基线**完全一致**，保证公平对比）。

集合（binary）：add / sub / mul / div
集合（unary）：sqrt / log / sin / cos / abs / neg

注：gplearn 0.4.3 的 function_set 不支持 ``exp``，为保证与外部 Tier-0
基线的**同一函数集**公平对比，``exp`` 不进共享集（套件真值均不需要 exp）。

保护约定（全程文档化，单测锁定）：
- div：|b| < 1e-12 → 返回 1.0（gplearn 同款约定，避免除零爆炸）
- sqrt：对 |a| 开方（负数域返回实值）
- log：log(|a| + 1e-12)（a=0 处有限）
- exp：clip 到 [-20, 20] 防溢出
- neg/sin/cos/abs：原生

作者：晨星
"""

from __future__ import annotations

from typing import Any

import numpy as np

DIV_EPS = 1e-12
LOG_EPS = 1e-12
EXP_CLIP = 20.0


def op_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b


def op_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a - b


def op_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a * b


def op_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(np.abs(b) < DIV_EPS, 1.0, a / np.where(np.abs(b) < DIV_EPS, 1.0, b))


def op_sqrt(a: np.ndarray) -> np.ndarray:
    return np.sqrt(np.abs(a))


def op_log(a: np.ndarray) -> np.ndarray:
    return np.log(np.abs(a) + LOG_EPS)


def op_exp(a: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(a, -EXP_CLIP, EXP_CLIP))


def op_sin(a: np.ndarray) -> np.ndarray:
    return np.sin(a)


def op_cos(a: np.ndarray) -> np.ndarray:
    return np.cos(a)


def op_abs(a: np.ndarray) -> np.ndarray:
    return np.abs(a)


def op_neg(a: np.ndarray) -> np.ndarray:
    return -a


BINARY_OPS: dict[str, Any] = {
    "add": op_add,
    "sub": op_sub,
    "mul": op_mul,
    "div": op_div,
}

UNARY_OPS: dict[str, Any] = {
    "sqrt": op_sqrt,
    "log": op_log,
    "sin": op_sin,
    "cos": op_cos,
    "abs": op_abs,
    "neg": op_neg,
}

ALL_OPS: dict[str, Any] = {**BINARY_OPS, **UNARY_OPS}

# gplearn function_set 名称映射（gplearn 用 'div' 等同名，直接对齐）
GPLEARN_NAMES: dict[str, str] = {
    "add": "add",
    "sub": "sub",
    "mul": "mul",
    "div": "div",
    "sqrt": "sqrt",
    "log": "log",
    "exp": "exp",
    "sin": "sin",
    "cos": "cos",
    "abs": "abs",
    "neg": "neg",
}

# sympy 函数映射（树 → sympy 表达式）
SYMPY_FUNCS: dict[str, str] = {
    "add": "Add",
    "sub": "Add",  # 特殊处理：a + (-b)
    "mul": "Mul",
    "div": "Mul",  # 特殊处理：a * (1/b)
    "sqrt": "sqrt",
    "log": "log",
    "sin": "sin",
    "cos": "cos",
    "abs": "Abs",
    "neg": "Mul",  # 特殊处理：-1 * a
}


def gplearn_function_set() -> tuple[str, ...]:
    """返回与 gplearn 对齐的 function_set 元组（gplearn 无 exp，故共享集不含 exp）。"""
    return ("add", "sub", "mul", "div", "sqrt", "log", "sin", "cos", "abs", "neg")


def apply_op(op: str, args: list[np.ndarray]) -> np.ndarray:
    """按名应用算子；未知算子抛 KeyError（由调用方转 SymRegError）。"""
    fn = ALL_OPS[op]
    return fn(*args)


__all__ = [
    "ALL_OPS",
    "BINARY_OPS",
    "DIV_EPS",
    "EXP_CLIP",
    "LOG_EPS",
    "UNARY_OPS",
    "apply_op",
    "gplearn_function_set",
]

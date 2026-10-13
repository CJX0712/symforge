"""表达式树：不可变元组表示 + 向量化求值 + sympy 互转。

表示约定（模块级常量锁定，SeqMineForge 教训：约定必须显式）：
- 变量叶：``("var", i)``（i 为整数列索引）
- 常量叶：``("const", float)``
- 二元内部节点：``(op, left, right)``，op ∈ BINARY_OPS
- 一元内部节点：``(op, child)``，op ∈ UNARY_OPS

作者：晨星
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import sympy as sp

from symforge.core.errors import SymRegError
from symforge.symreg.operators import ALL_OPS, BINARY_OPS, UNARY_OPS

Node = tuple

_VARS_CACHE: dict[int, list[sp.Symbol]] = {}


def var_symbols(ndim: int) -> list[sp.Symbol]:
    """缓存化的 x0..x{ndim-1} 符号（确定性）。"""
    if ndim not in _VARS_CACHE:
        _VARS_CACHE[ndim] = [sp.Symbol(f"x{i}") for i in range(ndim)]
    return _VARS_CACHE[ndim]


# ---------------------------------------------------------------- 构造
def make_var(i: int) -> Node:
    return ("var", int(i))


def make_const(v: float) -> Node:
    return ("const", float(v))


def make_op(op: str, *children: Node) -> Node:
    if op in BINARY_OPS and len(children) == 2:
        return (op, children[0], children[1])
    if op in UNARY_OPS and len(children) == 1:
        return (op, children[0])
    raise SymRegError(f"E202: 算子 {op!r} 与子节点数 {len(children)} 不匹配")


def is_leaf(node: Node) -> bool:
    return node[0] in ("var", "const")


def op_of(node: Node) -> str:
    return str(node[0])


# ---------------------------------------------------------------- 随机生成
def random_tree(
    rng: np.random.Generator,
    depth: int,
    ndim: int,
    method: str = "grow",
    use_erc: bool = True,
    const_min: float = -8.0,
    const_max: float = 8.0,
) -> Node:
    """Koza 式 grow/full 随机树。depth=0 时必为叶。"""
    if depth <= 0 or (method == "grow" and rng.random() < 0.35):
        if rng.random() < 0.5 or ndim <= 0:
            if use_erc:
                return make_const(float(rng.uniform(const_min, const_max)))
            return make_const(1.0)
        return make_var(int(rng.integers(0, ndim)))
    if rng.random() < 0.75:
        op = BINARY_OPS_NAMES[int(rng.integers(0, len(BINARY_OPS_NAMES)))]
        return make_op(
            op,
            random_tree(rng, depth - 1, ndim, method, use_erc, const_min, const_max),
            random_tree(rng, depth - 1, ndim, method, use_erc, const_min, const_max),
        )
    op = UNARY_OPS_NAMES[int(rng.integers(0, len(UNARY_OPS_NAMES)))]
    return make_op(op, random_tree(rng, depth - 1, ndim, method, use_erc, const_min, const_max))


BINARY_OPS_NAMES: tuple[str, ...] = tuple(BINARY_OPS.keys())
UNARY_OPS_NAMES: tuple[str, ...] = tuple(UNARY_OPS.keys())


# ---------------------------------------------------------------- 结构度量
def tree_size(node: Node) -> int:
    """节点总数（含叶）。"""
    if is_leaf(node):
        return 1
    return 1 + sum(tree_size(c) for c in node[1:])


def tree_depth(node: Node) -> int:
    if is_leaf(node):
        return 0
    return 1 + max(tree_depth(c) for c in node[1:])


def iter_nodes(node: Node) -> Iterator[Node]:
    yield node
    if not is_leaf(node):
        for c in node[1:]:
            yield from iter_nodes(c)


def iter_paths(node: Node, prefix: tuple[int, ...] = ()) -> Iterator[tuple[int, ...]]:
    """所有节点的路径（根为 ()；子索引为 tuple 元素）。"""
    yield prefix
    if not is_leaf(node):
        for i, c in enumerate(node[1:]):
            yield from iter_paths(c, (*prefix, i))


def get_at(node: Node, path: tuple[int, ...]) -> Node:
    cur = node
    for i in path:
        cur = cur[i + 1]
    return cur


def replace_at(node: Node, path: tuple[int, ...], new: Node) -> Node:
    """返回把 path 处子树替换为 new 的新树（不可变）。"""
    if not path:
        return new
    i = path[0]
    children = list(node[1:])
    children[i] = replace_at(children[i], path[1:], new)
    return (node[0], *children)


def random_path(rng: np.random.Generator, node: Node) -> tuple[int, ...]:
    paths = list(iter_paths(node))
    return paths[int(rng.integers(0, len(paths)))]


# ---------------------------------------------------------------- 求值
def eval_tree(node: Node, X: np.ndarray) -> np.ndarray:
    """向量化求值：X 为 (n, ndim)，返回 (n,)。

    SeqMineForge 教训：任何向量化/标量双实现必须有「严格晚于」交叉断言
    （见 tests/test_tree.py 的 vectorized-vs-scalar 对照）。
    """
    op = node[0]
    if op == "var":
        return X[:, int(node[1])]
    if op == "const":
        return np.full(X.shape[0], float(node[1]))
    if op not in ALL_OPS:
        raise SymRegError(f"E203: 未知算子 {op!r}")
    args = [eval_tree(c, X) for c in node[1:]]
    return ALL_OPS[op](*args)


def eval_tree_scalar(node: Node, x: np.ndarray) -> float:
    """标量参照实现（单点），仅供交叉验证，不参与主链路。"""
    op = node[0]
    if op == "var":
        return float(x[int(node[1])])
    if op == "const":
        return float(node[1])
    args = [eval_tree_scalar(c, x) for c in node[1:]]
    fn = ALL_OPS[op]
    if op == "div":
        b = args[1]
        return 1.0 if abs(b) < 1e-12 else args[0] / (1.0 if abs(b) < 1e-12 else b)
    return float(fn(*args))


# ---------------------------------------------------------------- 变异算子
def perturb_constants(node: Node, rng: np.random.Generator, scale: float = 0.5) -> Node:
    """对所有常量叶加性高斯扰动（point mutation 的常量变体）。"""
    if node[0] == "const":
        return make_const(float(np.clip(node[1] + rng.normal(0.0, scale), -1e3, 1e3)))
    if is_leaf(node):
        return node
    return (node[0], *[perturb_constants(c, rng, scale) for c in node[1:]])


def mutate_point(node: Node, rng: np.random.Generator, ndim: int) -> Node:
    """点变异：把单个节点替换为同元算子/同类叶。"""
    path = random_path(rng, node)
    target = get_at(node, path)
    if target[0] in ("var", "const"):
        if rng.random() < 0.5 and ndim > 0:
            new: Node = make_var(int(rng.integers(0, ndim)))
        else:
            new = make_const(float(rng.uniform(-8.0, 8.0)))
    elif target[0] in BINARY_OPS:
        names = list(BINARY_OPS_NAMES)
        new = make_op(names[int(rng.integers(0, len(names)))], target[1], target[2])
    else:
        names = list(UNARY_OPS_NAMES)
        new = make_op(names[int(rng.integers(0, len(names)))], target[1])
    return replace_at(node, path, new)


# ---------------------------------------------------------------- sympy 互转
def to_sympy(node: Node, symbols: list[sp.Symbol] | None = None) -> sp.Expr:
    """树 → sympy 表达式（未简化；简化由 simplify 模块负责）。"""
    if symbols is None:
        symbols = var_symbols(_max_var_index(node) + 1)
    op = node[0]
    if op == "var":
        return symbols[int(node[1])]
    if op == "const":
        return sp.Float(float(node[1]))
    kids = [to_sympy(c, symbols) for c in node[1:]]
    if op == "add":
        return kids[0] + kids[1]
    if op == "sub":
        return kids[0] - kids[1]
    if op == "mul":
        return kids[0] * kids[1]
    if op == "div":
        return kids[0] / kids[1]
    if op == "neg":
        return -kids[0]
    return getattr(sp, SYMPY_NAME[op])(*kids)


SYMPY_NAME: dict[str, str] = {
    "sqrt": "sqrt",
    "log": "log",
    "exp": "exp",
    "sin": "sin",
    "cos": "cos",
    "abs": "Abs",
}


def _max_var_index(node: Node) -> int:
    best = 0
    for n in iter_nodes(node):
        if n[0] == "var":
            best = max(best, int(n[1]))
    return best


def to_str(node: Node) -> str:
    """中缀可读形式（调试用）。"""
    if node[0] == "var":
        return f"x{int(node[1])}"
    if node[0] == "const":
        v = float(node[1])
        return f"{v:.4g}"
    if node[0] in BINARY_OPS:
        return f"({to_str(node[1])} {node[0]} {to_str(node[2])})"
    return f"{node[0]}({to_str(node[1])})"


__all__ = [
    "Node",
    "eval_tree",
    "eval_tree_scalar",
    "get_at",
    "is_leaf",
    "iter_nodes",
    "iter_paths",
    "make_const",
    "make_op",
    "make_var",
    "mutate_point",
    "op_of",
    "perturb_constants",
    "random_path",
    "random_tree",
    "replace_at",
    "to_str",
    "to_sympy",
    "tree_depth",
    "tree_size",
    "var_symbols",
]

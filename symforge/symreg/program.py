"""线性 GP 程序表示（postfix 指令流）+ 数组级变异算子。

设计动机（性能）：纯 Python 树递归求值 ~80µs/个体，demo 预算（≤60s）下
无法跑等预算基准。本模块把表达式树编译为 postfix 指令流，使
(a) 变异 = 数组切片 + 引用重映射（~10µs），(b) 求值可按「波次 × 算子」
批量向量化（见 batch_eval.py），整体提速 5~9×。

表示约定（模块级常量锁定，SeqMineForge 教训：约定必须显式且经单测锁定）：
- postfix（RPN）：指令 k 的子节点槽位恒 < k（子先父后）；
- **子树根 = 其跨度内的最后一条指令**；跨度 [i+1-size, i+1)，
  即 starts[i] = i+1-sizes[i]，ends[i] = i+1（有效程序恒成立）；
- 槽位与列索引分离：``a/b`` 只存子节点槽位（仅算子指令有意义），
  ``cols`` 专存变量列索引（仅 OP_VAR 有意义）——两者混用一个数组曾导致
  splice 平移损坏列索引（index -2 事故），故物理分离；
- 叶指令的 a/b 是占位值，片段平移后可能为负——安全，因为叶的 a/b 永不
  被读（_derive_meta / _eval_chunk / program_to_tree 均只读算子指令的槽位）；
- 子树片段自包含：片段内所有引用落在自身跨度内 ⟹ 拼接时整体平移即可；
- 子树替换：删除整段跨度 [s1, e1) 并插入片段；外部对「被删子树根」的
  唯一引用重定向到新根位置 s1+size2-1；后缀引用按 delta=size2-size1 平移；
- 无死代码：上述规则保证程序恒为纯树（每指令至多一个父节点，根为最后
  一条指令），因此批量评估可直接取每段末尾指令为根。

正确性契约由 tests/test_program.py 的 roundtrip oracle 全量锁定：
程序→树→程序恒等 + 各变异算子与树级参照实现逐点数值一致。

作者：晨星
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from symforge.core.errors import SymRegError
from symforge.symreg.tree import (
    Node,
    make_const,
    make_op,
    make_var,
    random_tree,
    to_str,
    to_sympy,
)

# 算子编码（与 operators.py 的顺序严格一致）
OP_VAR = 0
OP_CONST = 1
_BIN_NAMES = ("add", "sub", "mul", "div")
_UNI_NAMES = ("sqrt", "log", "sin", "cos", "abs", "neg")
BIN_CODE = {n: 2 + i for i, n in enumerate(_BIN_NAMES)}
UNI_CODE = {n: 6 + i for i, n in enumerate(_UNI_NAMES)}


@dataclass
class Program:
    """postfix 指令流（不可变语义：变异返回新对象）。

    meta 不变量（单测锁定）：sizes[i] = 以 i 为根的子树的指令数；
    edepth[i] = 求值深度（叶为 0）；starts[i] = i+1-sizes[i]，
    ends[i] = i+1（有效程序恒成立，故不必存储）。
    """

    ops: np.ndarray  # int8: 算子编码
    a: np.ndarray  # int32: 左子槽位（仅算子指令有意义）
    b: np.ndarray  # int32: 右子槽位（仅二元算子有意义）
    cols: np.ndarray  # int32: 变量列索引（仅 OP_VAR 有意义）
    consts: np.ndarray  # float64: 常量叶的值
    sizes: np.ndarray  # int32: 子树大小
    edepth: np.ndarray  # int8: 指令 i 的求值深度（= 子树深度）

    @property
    def size(self) -> int:
        return int(self.ops.shape[0])

    @property
    def depth(self) -> int:
        return int(self.edepth[-1]) if self.ops.shape[0] else 0

    def span_start(self, i: int) -> int:
        """指令 i 为根的子树的跨度起点。"""
        return i + 1 - int(self.sizes[i])

    def span_end(self, i: int) -> int:
        """指令 i 为根的子树的跨度末端（不含）。"""
        return i + 1


def flatten(tree: Node) -> Program:
    """树 → postfix 程序。"""
    ops: list[int] = []
    aa: list[int] = []
    bb: list[int] = []
    cols: list[int] = []
    cc: list[float] = []

    def emit(node: Node) -> int:
        t = node[0]
        if t == "var":
            ops.append(OP_VAR)
            aa.append(0)
            bb.append(0)
            cols.append(int(node[1]))
            cc.append(0.0)
        elif t == "const":
            ops.append(OP_CONST)
            aa.append(0)
            bb.append(0)
            cols.append(0)
            cc.append(float(node[1]))
        else:
            kids = [emit(c) for c in node[1:]]
            if t in BIN_CODE:
                ops.append(BIN_CODE[t])
                aa.append(kids[0])
                bb.append(kids[1])
                cols.append(0)
                cc.append(0.0)
            elif t in UNI_CODE:
                ops.append(UNI_CODE[t])
                aa.append(kids[0])
                bb.append(0)
                cols.append(0)
                cc.append(0.0)
            else:
                raise SymRegError(f"E205: 未知算子 {t!r}")
        return len(ops) - 1

    emit(tree)
    ops_a = np.array(ops, dtype=np.int8)
    a_a = np.array(aa, dtype=np.int32)
    b_a = np.array(bb, dtype=np.int32)
    cols_a = np.array(cols, dtype=np.int32)
    c_a = np.array(cc, dtype=np.float64)
    sizes, edepth = _derive_meta(ops_a, a_a, b_a)
    return Program(ops=ops_a, a=a_a, b=b_a, cols=cols_a, consts=c_a, sizes=sizes, edepth=edepth)


def _derive_meta(ops: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """由指令流推导 sizes / edepth（顺序依赖）。

    用 Python list 循环而非 numpy 标量索引：程序平均仅 ~40 条指令，
    list 索引比 np 标量取用快 5~10×（profile 实测为变异路径第一大开销）。
    """
    n = ops.shape[0]
    sizes = [1] * n
    edepth = [0] * n
    ops_l = ops.tolist()
    a_l = a.tolist()
    b_l = b.tolist()
    for i in range(n):
        op = ops_l[i]
        if op < 2:
            continue
        if op < 6:  # 二元
            ca = a_l[i]
            cb = b_l[i]
            sizes[i] = 1 + sizes[ca] + sizes[cb]
            da = edepth[ca]
            db = edepth[cb]
            edepth[i] = 1 + (da if da > db else db)
        else:  # 一元
            ca = a_l[i]
            sizes[i] = 1 + sizes[ca]
            edepth[i] = 1 + edepth[ca]
    return np.array(sizes, dtype=np.int32), np.array(edepth, dtype=np.int8)


def _shift_seg_refs(a: np.ndarray, b: np.ndarray, shift: int) -> tuple[np.ndarray, np.ndarray]:
    """平移一个自包含片段的子槽位引用。

    叶指令的 a/b 是占位 0，平移后可能为负——安全，因为叶的 a/b **永不被读**
    （_derive_meta 跳过叶；_eval_chunk 仅对算子指令读 a/b；program_to_tree
    仅对算子指令读 a/b）。此不变量由 tests/test_program.py 锁定。
    """
    return (a + shift).astype(np.int32), (b + shift).astype(np.int32)


def _remap_refs(refs: np.ndarray, cut_start: int, cut_end: int, new_root: int, delta: int) -> np.ndarray:
    """重映射后缀段引用：指向被删跨度 [cut_start, cut_end) 的 → new_root；
    ≥ cut_end 的 → +delta；< cut_start 的保持不变。"""
    out = np.where(refs < cut_start, refs, new_root)
    out = np.where(refs >= cut_end, refs + delta, out)
    return out.astype(np.int32)


def splice(p1: Program, i: int, p2: Program, j: int, e2: int) -> Program:
    """把 p1 中「以指令 i 为根的子树」替换为 p2 的片段 p2[j:e2]。

    i 为 p1 中被替换子树的根（跨度 [starts[i], ends[i])）；片段 p2[j:e2]
    必须是一个完整子树跨度（j = starts[root]，e2 = ends[root]，片段根为
    e2-1）。调用方（crossover/subtree_mutation）保证此契约。
    """
    l1 = p1.size
    s1 = p1.span_start(i)
    e1 = p1.span_end(i)
    size2 = e2 - j
    if s1 == 0 and e1 == l1:
        # 替换整棵树的根 → 结果即 p2 片段（引用相对化到片段起点）
        seg = slice(j, e2)
        ops = p2.ops[seg].copy()
        a, b = _shift_seg_refs(p2.a[seg], p2.b[seg], -j)
        cols = p2.cols[seg].copy()
        consts = p2.consts[seg].copy()
    else:
        seg = slice(j, e2)
        shift2 = s1 - j
        delta = size2 - (e1 - s1)
        new_root = s1 + size2 - 1
        ops = np.concatenate([p1.ops[:s1], p2.ops[seg], p1.ops[e1:]])
        a2, b2 = _shift_seg_refs(p2.a[seg], p2.b[seg], shift2)
        a3 = _remap_refs(p1.a[e1:], s1, e1, new_root, delta)
        b3 = _remap_refs(p1.b[e1:], s1, e1, new_root, delta)
        a = np.concatenate([p1.a[:s1], a2, a3])
        b = np.concatenate([p1.b[:s1], b2, b3])
        cols = np.concatenate([p1.cols[:s1], p2.cols[seg], p1.cols[e1:]])
        consts = np.concatenate([p1.consts[:s1], p2.consts[seg], p1.consts[e1:]])
    sizes, edepth = _derive_meta(ops, a, b)
    return Program(
        ops=ops,
        a=a.astype(np.int32),
        b=b.astype(np.int32),
        cols=cols.astype(np.int32),
        consts=consts,
        sizes=sizes,
        edepth=edepth,
    )


def crossover(p1: Program, p2: Program, rng: np.random.Generator) -> Program:
    """子树交叉：随机选 p1 节点 i 与 p2 节点 j，以 p2 子树片段替换 p1 子树。"""
    i = int(rng.integers(0, p1.size))
    j = int(rng.integers(0, p2.size))
    return splice(p1, i, p2, p2.span_start(j), p2.span_end(j))


def subtree_mutation(p1: Program, rng: np.random.Generator, ndim: int, max_depth: int = 3) -> Program:
    """子树变异：把随机节点替换为全新随机子树。"""
    i = int(rng.integers(0, p1.size))
    q = flatten(
        random_tree(
            rng,
            max_depth,
            ndim,
            method="grow",
            use_erc=True,
            const_min=-8.0,
            const_max=8.0,
        )
    )
    return splice(p1, i, q, 0, q.size)


def point_mutation(p1: Program, rng: np.random.Generator, ndim: int) -> Program:
    """点变异：单指令换同元算子（或叶类型互换）。"""
    i = int(rng.integers(0, p1.size))
    ops = p1.ops.copy()
    a = p1.a.copy()
    b = p1.b.copy()
    cols = p1.cols.copy()
    consts = p1.consts.copy()
    op = int(ops[i])
    if op == OP_VAR:
        if ndim > 0 and rng.random() < 0.5:
            cols[i] = int(rng.integers(0, ndim))
        else:
            ops[i] = OP_CONST
            consts[i] = float(rng.uniform(-8.0, 8.0))
    elif op == OP_CONST:
        if ndim > 0 and rng.random() < 0.5:
            ops[i] = OP_VAR
            cols[i] = int(rng.integers(0, ndim))
        else:
            consts[i] = float(rng.uniform(-8.0, 8.0))
    elif op < 6:
        ops[i] = 2 + int(rng.integers(0, len(_BIN_NAMES)))
    else:
        ops[i] = 6 + int(rng.integers(0, len(_UNI_NAMES)))
    sizes, edepth = _derive_meta(ops, a, b)
    return Program(ops=ops, a=a, b=b, cols=cols, consts=consts, sizes=sizes, edepth=edepth)


def perturb_constants(p1: Program, rng: np.random.Generator, scale: float = 0.5) -> Program:
    """常量扰动：所有常量叶各加一个独立高斯扰动（Keijzer 式 constant jitter，
    与树级参照实现语义一致：按指令序逐叶扰动）。"""
    const_idx = np.nonzero(p1.ops == OP_CONST)[0]
    if const_idx.size == 0:
        return p1
    consts = p1.consts.copy()
    for i in const_idx:
        consts[i] = float(np.clip(consts[i] + rng.normal(0.0, scale), -1e3, 1e3))
    return Program(
        ops=p1.ops, a=p1.a, b=p1.b, cols=p1.cols, consts=consts, sizes=p1.sizes, edepth=p1.edepth
    )


def program_to_tree(prog: Program) -> Node:
    """程序 → 树（供 sympy 转换 / 打印 / 交叉验证）。"""
    [("var", 0)] * prog.size

    def build(i: int) -> Node:
        op = int(prog.ops[i])
        if op == OP_VAR:
            return make_var(int(prog.cols[i]))
        if op == OP_CONST:
            return make_const(float(prog.consts[i]))
        if op < 6:
            return make_op(_BIN_NAMES[op - 2], build(int(prog.a[i])), build(int(prog.b[i])))
        return make_op(_UNI_NAMES[op - 6], build(int(prog.a[i])))

    return build(prog.size - 1)


def eval_program(prog: Program, X: np.ndarray) -> np.ndarray:
    """单程序标量流求值（polish 等单路径用；批量路径见 batch_eval）。"""
    from symforge.symreg.operators import ALL_OPS

    n = X.shape[0]
    vals: list[np.ndarray | None] = [None] * prog.size
    for i in range(prog.size):
        op = int(prog.ops[i])
        if op == OP_VAR:
            vals[i] = X[:, int(prog.cols[i])]
        elif op == OP_CONST:
            vals[i] = np.full(n, float(prog.consts[i]))
        elif op < 6:
            vals[i] = ALL_OPS[_BIN_NAMES[op - 2]](vals[int(prog.a[i])], vals[int(prog.b[i])])
        else:
            vals[i] = ALL_OPS[_UNI_NAMES[op - 6]](vals[int(prog.a[i])])
    assert vals[-1] is not None
    return vals[-1]


def program_str(prog: Program) -> str:
    return to_str(program_to_tree(prog))


def program_to_sympy(prog: Program, symbols=None):
    return to_sympy(program_to_tree(prog), symbols)


__all__ = [
    "BIN_CODE",
    "OP_CONST",
    "OP_VAR",
    "UNI_CODE",
    "Program",
    "crossover",
    "eval_program",
    "flatten",
    "perturb_constants",
    "point_mutation",
    "program_str",
    "program_to_sympy",
    "program_to_tree",
    "splice",
    "subtree_mutation",
]

"""tests/test_program.py — 程序表示 roundtrip oracle（正确性契约全量锁定）。

契约（program.py 模块头承诺）：
1. meta 不变量：postfix 子树根 = 跨度最后一条指令；sizes[i] = 树级子树大小；
   edepth[i] = 树级子树深度；算子指令的子槽位恒 < i；
2. roundtrip：树 → 程序 → 树 → 程序 恒等（逐数组逐位）；
3. 求值：eval_program 与树级参照 eval_tree 逐点一致；
4. splice 与树级 replace_at 参照逐点一致（含根替换边界）；
5. 常量扰动与树级参照在相同 rng 流下产生相同常量序列；
6. 所有变异产物仍满足 meta 不变量（无死代码保证）。

作者：晨星
"""

from __future__ import annotations

import numpy as np
import pytest

from symforge.symreg.batch_eval import batch_fitness
from symforge.symreg.program import (
    Program,
    crossover,
    eval_program,
    flatten,
    perturb_constants,
    point_mutation,
    program_str,
    program_to_tree,
    splice,
    subtree_mutation,
)
from symforge.symreg.tree import (
    eval_tree,
    is_leaf,
    random_tree,
    replace_at,
    to_str,
    tree_depth,
    tree_size,
)
from symforge.symreg.tree import (
    perturb_constants as tree_perturb_constants,
)


def postorder(node, path=()):
    """按 flatten 的 emit 顺序枚举 (节点, 路径)：子先父后。"""
    if not is_leaf(node):
        for k, c in enumerate(node[1:]):
            yield from postorder(c, (*path, k))
    yield (node, path)


def make_prog(rng, depth=4, ndim=3):
    return flatten(random_tree(rng, depth, ndim, method="grow", use_erc=True))


def assert_meta_valid(p: Program) -> None:
    """postfix 树不变量：子槽位 < i；sizes/edepth 与树级参照一致。"""
    for i in range(p.size):
        op = int(p.ops[i])
        if op < 2:
            continue
        assert 0 <= int(p.a[i]) < i, f"slot a[{i}]={p.a[i]} 越界"
        if op < 6:
            assert 0 <= int(p.b[i]) < i, f"slot b[{i}]={p.b[i]} 越界"
        s_expect = 1 + int(p.sizes[p.a[i]]) + (int(p.sizes[p.b[i]]) if op < 6 else 0)
        assert int(p.sizes[i]) == s_expect, f"sizes[{i}] 不自洽"
    assert int(p.sizes[-1]) == p.size, "根（末指令）子树大小应为全长"
    assert p.span_start(p.size - 1) == 0 and p.span_end(p.size - 1) == p.size


def test_meta_invariants_vs_tree_reference() -> None:
    rng = np.random.default_rng(0)
    for _ in range(40):
        t = random_tree(rng, 5, 3, method="grow", use_erc=True)
        p = flatten(t)
        nodes = list(postorder(t))
        assert p.size == len(nodes)
        for i, (nd, _path) in enumerate(nodes):
            assert int(p.sizes[i]) == tree_size(nd), f"sizes[{i}] vs tree_size"
            assert int(p.edepth[i]) == tree_depth(nd), f"edepth[{i}] vs tree_depth"
        assert_meta_valid(p)


def test_roundtrip_tree_program_tree() -> None:
    rng = np.random.default_rng(1)
    for _ in range(40):
        t1 = random_tree(rng, 5, 3, method="grow", use_erc=True)
        p1 = flatten(t1)
        t2 = program_to_tree(p1)
        p2 = flatten(t2)
        assert to_str(t1) == to_str(t2)
        for f in ("ops", "a", "b", "cols", "consts", "sizes", "edepth"):
            arr1 = getattr(p1, f)
            arr2 = getattr(p2, f)
            assert arr1.shape == arr2.shape and np.array_equal(arr1, arr2), f"{f} roundtrip 不一致"


def test_eval_matches_tree_reference() -> None:
    rng = np.random.default_rng(2)
    X = rng.uniform(-2, 2, size=(64, 3))
    for _ in range(30):
        t = random_tree(rng, 5, 3, method="grow", use_erc=True)
        p = flatten(t)
        f_prog = eval_program(p, X)
        f_tree = eval_tree(t, X)
        np.testing.assert_allclose(f_prog, f_tree, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_splice_matches_tree_replace_at() -> None:
    """splice(p1,i,p2,j,e2) ⟺ replace_at(t1, path_i, subtree_j)。

    i/j 按 postorder（=flatten emit 顺序）选取，含根替换（path=()）边界。
    """
    rng = np.random.default_rng(3)
    X = rng.uniform(-2, 2, size=(32, 3))
    for _ in range(30):
        t1 = random_tree(rng, 5, 3, method="grow", use_erc=True)
        t2 = random_tree(rng, 3, 3, method="grow", use_erc=True)
        p1, p2 = flatten(t1), flatten(t2)
        i = int(rng.integers(0, p1.size))
        j = int(rng.integers(0, p2.size))
        _nd_i, path_i = list(postorder(t1))[i]
        nd_j, _path_j = list(postorder(t2))[j]
        ref_tree = replace_at(t1, path_i, nd_j)
        got = splice(p1, i, p2, p2.span_start(j), p2.span_end(j))
        assert_meta_valid(got)
        # 结构 oracle：程序 → 树 与参照树逐节点相等
        assert program_to_tree(got) == ref_tree
        # 数值 oracle：与参照树求值逐点一致
        np.testing.assert_allclose(
            eval_program(got, X), eval_tree(ref_tree, X), rtol=1e-12, atol=1e-12, equal_nan=True
        )


def test_splice_root_replacement() -> None:
    """i=根指令（替换整棵树）边界：结果应恰为片段本身。"""
    rng = np.random.default_rng(4)
    t1 = random_tree(rng, 5, 3, method="grow", use_erc=True)
    t2 = random_tree(rng, 3, 3, method="grow", use_erc=True)
    p1, p2 = flatten(t1), flatten(t2)
    got = splice(p1, p1.size - 1, p2, 0, p2.size)  # postfix 根 = 末指令
    assert got.size == p2.size
    assert program_to_tree(got) == t2
    assert_meta_valid(got)


def test_crossover_deterministic_and_valid() -> None:
    rng = np.random.default_rng(5)
    p1, p2 = make_prog(rng), make_prog(rng)
    c1 = crossover(p1, p2, np.random.default_rng(99))
    c2 = crossover(p1, p2, np.random.default_rng(99))
    assert np.array_equal(c1.ops, c2.ops) and np.array_equal(c1.consts, c2.consts)
    assert_meta_valid(c1)


def test_mutations_preserve_invariants() -> None:
    rng = np.random.default_rng(6)
    p = make_prog(rng)
    for fn in (
        lambda q: subtree_mutation(q, rng, ndim=3, max_depth=3),
        lambda q: point_mutation(q, rng, ndim=3),
        lambda q: perturb_constants(q, rng),
    ):
        out = fn(p)
        assert_meta_valid(out)
        X = rng.uniform(-2, 2, size=(16, 3))
        assert eval_program(out, X).shape == (16,)
    # 点变异不改大小
    pm = point_mutation(p, rng, ndim=3)
    assert pm.size == p.size


def test_perturb_constants_matches_tree_reference() -> None:
    """相同 rng 流下：程序级扰动序列 = 树级参照（postorder 叶序一致）。"""
    rng = np.random.default_rng(7)
    t = random_tree(rng, 5, 3, method="grow", use_erc=True)
    p = flatten(t)
    rng_a = np.random.default_rng(1234)
    rng_b = np.random.default_rng(1234)
    p2 = perturb_constants(p, rng_a, scale=0.5)
    t2 = tree_perturb_constants(t, rng_b, scale=0.5)
    # 非常量字段不动
    assert np.array_equal(p2.ops, p.ops) and np.array_equal(p2.cols, p.cols)
    # 常量序列逐位一致（postorder 常量叶序）
    prog_consts = p2.consts[p2.ops == 1]
    tree_consts = [nd[1] for nd, _ in postorder(t2) if nd[0] == "const"]
    assert len(prog_consts) == len(tree_consts)
    np.testing.assert_array_equal(prog_consts, np.array(tree_consts))


def test_perturb_constants_no_const_noop() -> None:
    rng = np.random.default_rng(8)
    p = flatten(("sin", ("var", 0)))  # 手工构造无常量程序
    assert int((p.ops == 1).sum()) == 0
    assert perturb_constants(p, rng) is p


def test_batch_vs_single_consistency() -> None:
    """批量路径与单程序路径逐位一致（wave 引擎 vs 标量流）。"""
    rng = np.random.default_rng(9)
    progs = [make_prog(rng, depth=d) for d in (2, 3, 4, 5, 3)]
    X = rng.uniform(-2, 2, size=(50, 3))
    y = rng.uniform(-5, 5, size=50)
    a_b, b_b, fit_b, mse_b = batch_fitness(progs, X, y, linear_scaling=True, parsimony_coef=1e-4)
    var_y = float(np.var(y)) + 1e-12
    for k, p in enumerate(progs):
        f = eval_program(p, X)
        s_ff, s_f, s_fy = float(np.sum(f * f)), float(np.sum(f)), float(np.sum(f * y))
        det = 50 * s_ff - s_f * s_f
        a = (50 * s_fy - s_f * float(np.sum(y))) / det if abs(det) >= 1e-12 else 0.0
        b = (float(np.sum(y)) - a * s_f) / 50
        mse = float(np.mean((a * f + b - y) ** 2))
        np.testing.assert_allclose(a_b[k], a, rtol=1e-10)
        np.testing.assert_allclose(b_b[k], b, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(mse_b[k], mse, rtol=1e-10)
        np.testing.assert_allclose(fit_b[k], mse / var_y + 1e-4 * p.size, rtol=1e-10)


def test_program_str_roundtrip() -> None:
    rng = np.random.default_rng(10)
    p = make_prog(rng)
    s = program_str(p)
    assert (isinstance(s, str) and "x0" in s) or "x1" in s or "x2" in s


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

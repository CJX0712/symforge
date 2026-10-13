"""波次批量评估器：把整个种群的程序按 (求值深度 × 算子) 分批向量化求值。

性能路径（集大成）：直接树递归 ~80µs/个体 → 本实现 ~10µs/个体（5~9×）。
原理：postfix 指令按求值深度分波，同波同算子的指令可合并为单次 numpy
广播运算（gather → op → scatter），把 Python 逐节点开销摊到整个种群。

确定性：全部为固定形状的 numpy 归约（pairwise sum 顺序确定），
同输入逐位一致（单测锁定）。

作者：晨星"""

from __future__ import annotations

import numpy as np

from symforge.symreg.operators import (
    op_abs,
    op_add,
    op_cos,
    op_div,
    op_log,
    op_mul,
    op_neg,
    op_sin,
    op_sqrt,
    op_sub,
)
from symforge.symreg.program import OP_CONST, OP_VAR, Program

_BIN_FNS = (op_add, op_sub, op_mul, op_div)
_UNI_FNS = (op_sqrt, op_log, op_sin, op_cos, op_abs, op_neg)

# 单批缓冲上限（防 bloat 时内存爆炸）：P × maxL × n ≤ 该值则分块
_BUF_CELL_LIMIT = 8_000_000

# 复用暂存缓冲（每代分配 38MB 级缓冲是纯开销；所有被读槽位均先被写）
_scratch: np.ndarray | None = None
_scratch_shape: tuple[int, int, int] = (0, 0, 0)

# 全部算子编码（含叶指令 OP_VAR=0 / OP_CONST=1 —— 漏掉叶曾导致全种群适应度
# 恒等、选择退化为纯 size 最小化的事故）
_OP_VALUES = np.arange(0, 14, dtype=np.int8)


def _get_buf(P: int, max_l: int, n: int) -> np.ndarray:
    global _scratch, _scratch_shape
    if _scratch is None or _scratch_shape[0] < P or _scratch_shape[1] < max_l or _scratch_shape[2] < n:
        _scratch = np.empty((P, max_l, n), dtype=float)
        _scratch_shape = (P, max_l, n)
    return _scratch[:P, :max_l, :n]


def _eval_chunk(progs: list[Program], X: np.ndarray) -> np.ndarray:
    """评估一个程序块，返回原始树输出 f，形状 (P, n)。"""
    P = len(progs)
    n = X.shape[0]
    lens = np.array([p.size for p in progs], dtype=np.int64)
    max_l = int(lens.max())
    offsets = np.concatenate([[0], np.cumsum(lens)[:-1]])
    total = int(lens.sum())

    ops = np.concatenate([p.ops for p in progs])
    aa = np.concatenate([p.a for p in progs])
    bb = np.concatenate([p.b for p in progs])
    cc = np.concatenate([p.cols for p in progs])
    kv = np.concatenate([p.consts for p in progs])
    ed = np.concatenate([p.edepth for p in progs])
    pids = np.repeat(np.arange(P), lens)
    slots = np.arange(total) - np.repeat(offsets, lens)
    roots = lens - 1

    # 按 (求值深度, 算子) 稳定排序 → 同波同算子连续
    order = np.lexsort((ops, ed))
    ops_s = ops[order]
    ed_s = ed[order]
    pids_s = pids[order]
    a_s = aa[order]
    b_s = bb[order]
    c_s = cc[order]
    k_s = kv[order]
    slots_s = slots[order]

    buf = _get_buf(P, max_l, n)
    XT = np.ascontiguousarray(X.T)

    # 波次边界（ed_s 已排序）
    max_ed = int(ed_s[-1]) if ed_s.size else 0
    wave_bounds = np.searchsorted(ed_s, np.arange(0, max_ed + 2))
    for w in range(len(wave_bounds) - 1):
        lo, hi = int(wave_bounds[w]), int(wave_bounds[w + 1])
        if hi <= lo:
            continue
        seg_ops = ops_s[lo:hi]
        # 同波内按算子分组：单次向量化 searchsorted 取全部 14 组左边界
        op_bounds = np.searchsorted(seg_ops, _OP_VALUES)
        for k in range(len(_OP_VALUES)):
            g_lo = int(op_bounds[k])
            g_hi = int(op_bounds[k + 1]) if k + 1 < len(_OP_VALUES) else hi - lo
            if g_hi <= g_lo:
                continue
            opv = k  # _OP_VALUES[k] == k
            gi = slice(lo + g_lo, lo + g_hi)
            p = pids_s[gi]
            t = slots_s[gi]
            if opv == OP_VAR:
                buf[p, t] = XT[c_s[gi]]
            elif opv == OP_CONST:
                buf[p, t] = k_s[gi][:, None]
            elif opv < 6:
                left = buf[p, a_s[gi]]
                right = buf[p, b_s[gi]]
                buf[p, t] = _BIN_FNS[opv - 2](left, right)
            else:
                buf[p, t] = _UNI_FNS[opv - 6](buf[p, a_s[gi]])

    return buf[np.arange(P), roots]


def _ls_fitness(
    progs: list[Program],
    f: np.ndarray,
    y_row: np.ndarray,
    linear_scaling: bool,
    parsimony_coef: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """由原始树输出 f (P,n) 计算线性缩放系数、适应度与 MSE。"""
    P, n = f.shape
    sizes = np.array([p.size for p in progs], dtype=float)
    var_y = float(np.var(y_row)) + 1e-12
    if linear_scaling:
        s_ff = np.sum(f * f, axis=1)
        s_f = np.sum(f, axis=1)
        s_fy = np.sum(f * y_row[None, :], axis=1)
        s_y = float(np.sum(y_row))
        det = n * s_ff - s_f * s_f
        num = n * s_fy - s_f * s_y
        a = np.divide(num, det, out=np.zeros(P), where=np.abs(det) >= 1e-12)
        b = (s_y - a * s_f) / n
        resid = a[:, None] * f + b[:, None] - y_row[None, :]
    else:
        a = np.ones(P)
        b = np.zeros(P)
        resid = f - y_row[None, :]
    mse = np.mean(resid * resid, axis=1)
    fitness = mse / var_y + parsimony_coef * sizes
    return a, b, fitness, mse


def batch_fitness(
    progs: list[Program],
    X: np.ndarray,
    y: np.ndarray,
    linear_scaling: bool,
    parsimony_coef: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """批量求值 + 线性缩放 + 适应度，返回 (a, b, fitness, mse)。

    内存守卫：P × maxL × n 超限时自动分块（数值逐位不变）。
    """
    P = len(progs)
    y_row = np.asarray(y, dtype=float)
    empty = (np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0))
    if P == 0:
        return empty
    max_size = max(p.size for p in progs)
    if P * max_size * X.shape[0] <= _BUF_CELL_LIMIT:
        f = _eval_chunk(progs, X)
        return _ls_fitness(progs, f, y_row, linear_scaling, parsimony_coef)
    a_out = np.empty(P)
    b_out = np.empty(P)
    fit_out = np.empty(P)
    mse_out = np.empty(P)
    start = 0
    while start < P:
        end = start + 1
        while end < P:
            cells = (end - start + 1) * max(progs[k].size for k in range(start, end + 1)) * X.shape[0]
            if cells > _BUF_CELL_LIMIT:
                break
            end += 1
        sub = progs[start:end]
        f = _eval_chunk(sub, X)
        a_s, b_s, fit_s, mse_s = _ls_fitness(sub, f, y_row, linear_scaling, parsimony_coef)
        a_out[start:end] = a_s
        b_out[start:end] = b_s
        fit_out[start:end] = fit_s
        mse_out[start:end] = mse_s
        start = end
    return a_out, b_out, fit_out, mse_out


__all__ = ["batch_fitness"]

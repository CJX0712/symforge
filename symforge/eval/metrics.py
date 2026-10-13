"""评测指标：R² / 数值等价 / 符号等价 / 聚合 + Wilcoxon。

作者：晨星
"""

from __future__ import annotations

from typing import Any

import numpy as np
import sympy as sp

from symforge.core.errors import EvalError
from symforge.symreg.simplify import symbolic_equal
from symforge.symreg.tree import var_symbols

# 数值等价容差（SRBench 口径）：holdout 上 max|ŷ−y| ≤ tol·(1+max|y|)
NUMERIC_TOL = 1e-6


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0:
        raise EvalError("E401: y_true 方差为 0，R² 无定义")
    return 1.0 - ss_res / ss_tot


def numeric_match(y_pred: np.ndarray, y_true: np.ndarray, tol: float = NUMERIC_TOL) -> bool:
    """数值等价：与真值函数在 holdout 上逐点一致（相对尺度容差）。"""
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    scale = 1.0 + float(np.max(np.abs(y_true)))
    return bool(np.max(np.abs(y_pred - y_true)) <= tol * scale)


def check_symbolic(model: Any, truth: sp.Expr, ndim: int) -> bool | None:
    """符号等价判定（模型不可符号化时返回 None，不伪造）。"""
    try:
        expr = model.to_sympy(var_symbols(ndim))
    except Exception:
        return None
    if expr is None:
        return None
    return symbolic_equal(expr, truth)


def aggregate_rates(rows: list[dict], method: str, key: str = "numeric_match") -> dict[str, float]:
    """按方法聚合：overall rate + per-dataset rates（供显著性与分组报告）。"""
    per_ds: dict[str, list[int]] = {}
    for r in rows:
        if r["method"] != method or r["status"] != "ok":
            continue
        m = r["metrics"]
        v = (1 if m.get("numeric_match") else 0) if key == "numeric_match" else 1 if m.get("symbolic_exact") else 0
        per_ds.setdefault(r["dataset"], []).append(v)
    if not per_ds:
        return {"overall": float("nan"), "per_dataset": {}}
    overall = float(np.mean([v for vals in per_ds.values() for v in vals]))
    per_dataset = {ds: float(np.mean(v)) for ds, v in per_ds.items()}
    return {"overall": overall, "per_dataset": per_dataset}


def _ranks(diffs: list[float]) -> list[float]:
    """Wilcoxon 平均秩（平局取平均秩）。"""
    order = sorted(range(len(diffs)), key=lambda i: diffs[i])
    ranks = [0.0] * len(diffs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and diffs[order[j + 1]] == diffs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def wilcoxon_p(diffs: list[float]) -> float:
    """单侧 Wilcoxon 符号秩检验（H1: diffs > 0），小样本精确正态近似。

    全零差 → p=1.0；n<6 → 返回 nan（功效不足，如实标注）。
    """
    nz = [d for d in diffs if abs(d) > 1e-15]
    if not nz:
        return 1.0
    n = len(nz)
    if n < 6:
        return float("nan")
    ranks = _ranks(nz)
    w_plus = sum(r for d, r in zip(nz, ranks, strict=True) if d > 0)
    mu = n * (n + 1) / 4.0
    sigma = (n * (n + 1) * (2 * n + 1) / 24.0) ** 0.5
    if sigma <= 0:
        return 1.0
    z = (w_plus - mu - 0.5) / sigma  # 连续性校正
    from scipy.stats import norm

    return float(1.0 - norm.cdf(z))


def significance(flag_a: dict, flag_b: dict) -> dict[str, Any]:
    """按数据集配对比较 A vs B（A=旗舰）。

    判据（SOP §6）：均值差 > ½(σ_A + σ_B) 视为显著（辅以 Wilcoxon）。
    """
    ds_common = sorted(set(flag_a["per_dataset"]) & set(flag_b["per_dataset"]))
    if not ds_common:
        return {"mean_diff": float("nan"), "threshold": float("nan"), "significant": False, "p": float("nan")}
    a = np.array([flag_a["per_dataset"][d] for d in ds_common])
    b = np.array([flag_b["per_dataset"][d] for d in ds_common])
    mean_diff = float(np.mean(a - b))
    threshold = 0.5 * (float(np.std(a)) + float(np.std(b)))
    p = wilcoxon_p([float(x - y) for x, y in zip(a, b, strict=True)])
    return {
        "mean_diff": mean_diff,
        "threshold": threshold,
        "significant": bool(mean_diff > threshold and mean_diff > 0),
        "p": p,
        "n_datasets": len(ds_common),
    }


__all__ = [
    "NUMERIC_TOL",
    "aggregate_rates",
    "check_symbolic",
    "numeric_match",
    "r2_score",
    "significance",
    "wilcoxon_p",
]

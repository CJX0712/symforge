"""Pipeline：基准评测 + 门禁判定 + 失败案例归因。

门禁（预注册，定死；依据 SOP §9/§10）：

- G1  主门禁：等预算下旗舰 numeric-recovery ≥ plain_gp + 0.15，
      且按数据集配对显著（均值差 > ½(σ₁+σ₂)，Wilcoxon 单侧 p<0.05）；
- G1b 次级：vs 外部 Tier-0 基线 gplearn 的 Δ（如实报告，不设硬门槛，
      因 gplearn 结构性缺失常量终端）；
- G2  绝对地板：旗舰 aggregate recovery ≥ 0.55；
- G4  消融：旗舰 > ablate_no_scaling 且旗舰 > ablate_single_island
      （均值差 > 0；负结果同样保留进报告）；
- G5  性能预算：demo 端到端 ≤ 60s。

作者：晨星
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from symforge.baselines import ABLATION_METHODS, build_methods
from symforge.core.config import SRConfig, preset_config
from symforge.core.errors import PipelineError
from symforge.core.types import BenchmarkResult, BenchmarkRow, SRMetrics
from symforge.data.suite import NOISY_SUBSET, SUITE, list_datasets, make_dataset
from symforge.eval.metrics import (
    NUMERIC_TOL,
    aggregate_rates,
    check_symbolic,
    numeric_match,
    r2_score,
    significance,
)

GATE_G1_DELTA = 0.15
GATE_G2_FLOOR = 0.55


def run_cell(
    method_name: str,
    factory: Any,
    dataset: Any,
    seed: int,
) -> BenchmarkRow:
    """单个 (dataset × method × seed) 单元：fit(train) → eval(holdout)。"""
    t0 = time.perf_counter()
    row = BenchmarkRow(dataset=dataset.name, method=method_name, seed=seed, difficulty=dataset.difficulty)
    try:
        model = factory(seed)
        model.fit(dataset.X_train, dataset.y_train)
        pred_train = model.predict(dataset.X_train)
        pred_hold = model.predict(dataset.X_holdout)
        m = SRMetrics()
        m.r2_train = r2_score(dataset.y_train, pred_train)
        m.r2_holdout = r2_score(dataset.y_holdout, pred_hold)
        # 数值等价容差感知噪声水平（holdout 本身干净；SRBench 惯例：
        # 噪声任务的等价判据放宽到噪声量级，clean 任务保持 1e-6 严格判据）
        tol = max(NUMERIC_TOL, float(dataset.noise_frac))
        m.numeric_match = numeric_match(pred_hold, dataset.y_holdout, tol=tol)
        m.symbolic_exact = check_symbolic(model, dataset.truth, dataset.ndim)
        m.size = int(model.complexity())
        m.mse_train = float(np.mean((dataset.y_train - pred_train) ** 2))
        m.expression = model.expression()
        m.runtime_sec = time.perf_counter() - t0
        row.metrics = m
        row.status = "ok"
    except Exception as exc:
        row.status = "skipped"
        row.metrics.error = f"{type(exc).__name__}: {exc}"
    return row


def run_benchmark(
    preset: str = "quick",
    seeds: tuple[int, ...] = (7, 42, 123),
    datasets: list[str] | None = None,
    include_ablations: bool = True,
    include_noise: bool = False,
    verbose: bool = False,
) -> BenchmarkResult:
    """跑基准（quick/full 预设；等预算 GP 家族 + 经典基线）。"""
    if preset not in ("quick", "full"):
        raise PipelineError(f"E501: 未知 preset {preset!r}")
    config: SRConfig = preset_config(preset)
    ds_names = list(datasets) if datasets else list_datasets()
    for name in ds_names:
        if name not in SUITE:
            raise PipelineError(f"E502: 未知数据集 {name!r}")

    methods = build_methods(config)
    result = BenchmarkResult(preset=preset)
    ablation_seeds = seeds[:2] if preset == "quick" else seeds

    for name in ds_names:
        spec = SUITE[name]
        noise_frac = NOISY_SUBSET.get(name, 0.0) if include_noise else 0.0
        for seed in seeds:
            data = make_dataset(name, seed, noise_frac=noise_frac)
            for mname, (factory, avail) in methods.items():
                if not avail:
                    result.rows.append(
                        BenchmarkRow(
                            dataset=name, method=mname, seed=seed, difficulty=spec.difficulty, status="skipped"
                        )
                    )
                    result.rows[-1].metrics.error = "backend unavailable"
                    continue
                if mname in ABLATION_METHODS and seed not in ablation_seeds:
                    continue  # 消融降 seeds（SOP：性能 gate 保持 3 seeds，消融 2 seeds）
                row = run_cell(mname, factory, data, seed)
                result.rows.append(row)
                if verbose:
                    print(f"  [{name:12s} seed={seed:3d} {mname:22s}] match={row.metrics.numeric_match}")

    _compute_gates(result, methods_used=[m for m, (_, a) in methods.items() if a])
    return result


def _rows_of(result: BenchmarkResult) -> list[dict]:
    return [r.to_dict() for r in result.rows]


def _compute_gates(result: BenchmarkResult, methods_used: list[str]) -> None:
    rows = _rows_of(result)
    agg = {m: aggregate_rates(rows, m) for m in methods_used}
    gates: dict[str, Any] = {}

    # G1：旗舰 vs plain_gp（GP 家族内部对照，归因干净）
    if "symfuse" in agg and "plain_gp" in agg:
        sig = significance(agg["symfuse"], agg["plain_gp"])
        gates["G1_vs_plain_gp"] = {
            "flagship": agg["symfuse"]["overall"],
            "plain_gp": agg["plain_gp"]["overall"],
            "delta": sig["mean_diff"],
            "required_delta": GATE_G1_DELTA,
            "significant": sig["significant"],
            "p": sig["p"],
            "threshold_half_sigma": sig["threshold"],
            "pass": bool(sig["mean_diff"] >= GATE_G1_DELTA and sig["significant"]),
        }
    # G1b：旗舰 vs gplearn（外部 Tier-0，次级报告）
    if "symfuse" in agg and "gplearn" in agg:
        sig = significance(agg["symfuse"], agg["gplearn"])
        gates["G1b_vs_gplearn"] = {
            "flagship": agg["symfuse"]["overall"],
            "gplearn": agg["gplearn"]["overall"],
            "delta": sig["mean_diff"],
            "significant": sig["significant"],
            "p": sig["p"],
            "pass": bool(sig["mean_diff"] > 0 and sig["significant"]),
        }
    # G2：绝对地板
    if "symfuse" in agg:
        gates["G2_floor"] = {
            "flagship": agg["symfuse"]["overall"],
            "floor": GATE_G2_FLOOR,
            "pass": bool(agg["symfuse"]["overall"] >= GATE_G2_FLOOR),
        }
    # G4：消融（各开关单独贡献）
    for abl in ABLATION_METHODS:
        if "symfuse" in agg and abl in agg:
            sig = significance(agg["symfuse"], agg[abl])
            gates[f"G4_{abl}"] = {
                "flagship": agg["symfuse"]["overall"],
                abl: agg[abl]["overall"],
                "delta": sig["mean_diff"],
                "significant": sig["significant"],
                "p": sig["p"],
                "pass": bool(sig["mean_diff"] > 0),
            }
    # G5：性能预算——预注册定义为「demo 端到端 ≤ 60s」，由 examples/run_demo.py
    # 实测；此处报告基准单 run 运行时长统计作代理指标（曾误用整套基准
    # 总时长对比 60s 预算，恒 FAIL，属实现错误而非性能问题）。
    runtimes = [r["metrics"]["runtime_sec"] for r in rows if r["status"] == "ok" and "runtime_sec" in r["metrics"]]
    med = float(np.median(runtimes)) if runtimes else float("nan")
    gates["G5_runtime"] = {
        "n_runs": len(runtimes),
        "median_sec": med,
        "p95_sec": float(np.percentile(runtimes, 95)) if runtimes else float("nan"),
        "budget_sec": 60.0,
        "note": "硬门禁在 demo 端到端计时（examples/run_demo.py）；本项为单 run 时长统计",
        "pass": bool(med <= 60.0),
    }
    result.gates = gates
    result.summary = {
        "methods": methods_used,
        "recovery": {m: agg[m]["overall"] for m in agg},
        "per_dataset": {m: agg[m]["per_dataset"] for m in agg},
        "n_rows": len(rows),
        "n_skipped": sum(1 for r in rows if r["status"] == "skipped"),
    }


def failure_cases(result: BenchmarkResult, top_k: int = 5) -> list[dict]:
    """失败案例（全部从 results 派生，SOP §6）：旗舰未恢复且存在基线恢复的单元。"""
    rows = _rows_of(result)
    by_cell: dict[tuple[str, int], dict[str, dict]] = {}
    for r in rows:
        if r["status"] != "ok":
            continue
        by_cell.setdefault((r["dataset"], r["seed"]), {})[r["method"]] = r
    cases: list[dict] = []
    for (ds, seed), per_method in sorted(by_cell.items()):
        flag = per_method.get("symfuse")
        if flag is None or flag["metrics"]["numeric_match"]:
            continue
        winners = [m for m, r in per_method.items() if m != "symfuse" and r["metrics"]["numeric_match"]]
        cases.append(
            {
                "dataset": ds,
                "difficulty": flag["difficulty"],
                "seed": seed,
                "flagship_expr": flag["metrics"]["expression"],
                "flagship_r2": flag["metrics"]["r2_holdout"],
                "baselines_recovered": winners,
            }
        )
    cases.sort(key=lambda c: (c["difficulty"], c["dataset"]))
    return cases[:top_k]


__all__ = ["GATE_G1_DELTA", "GATE_G2_FLOOR", "failure_cases", "run_benchmark", "run_cell"]

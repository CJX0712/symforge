"""分块全量基准 runner：绕过单进程超时上限，按 seed 分块 + 合并算门禁。

用法（顺序执行，前台）：
  python scripts/bench_runner.py --mode seed --seed 7
  python scripts/bench_runner.py --mode seed --seed 42
  python scripts/bench_runner.py --mode seed --seed 123
  python scripts/bench_runner.py --mode noise
  python scripts/bench_runner.py --mode merge

产物（results/）：
  rows_full_s{seed}.json   每块原始行（可复现审计）
  rows_noise.json          噪声子套件行
  full_report.json         合并门禁 + 汇总（clean 套件）
  noise_report.json        噪声子套件汇总

作者：晨星
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from symforge.core.types import BenchmarkResult
from symforge.data.suite import NOISY_SUBSET
from symforge.pipeline.benchmark import _compute_gates, run_benchmark

RESULTS = Path(__file__).resolve().parent.parent / "results"


class _DictRow:
    """把 dict 伪装成 BenchmarkRow（_compute_gates 只调 to_dict）。"""

    def __init__(self, d: dict) -> None:
        self._d = d

    def to_dict(self) -> dict:
        return self._d


def _save(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"  saved -> {path}")


def _rows_to_dicts(rows) -> list[dict]:
    return [r.to_dict() if hasattr(r, "to_dict") else dict(r) for r in rows]


def mode_seed(seed: int, datasets: list[str] | None = None, tag: str = "") -> None:
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    res = run_benchmark(preset="full", seeds=(seed,), datasets=datasets)
    _save(RESULTS / f"rows_full_s{seed}{tag}.json", _rows_to_dicts(res.rows))
    ok = sum(1 for r in res.rows if r.status == "ok")
    print(f"seed {seed}{tag}: {ok}/{len(res.rows)} ok, {time.time() - t0:.0f}s")


def mode_noise(datasets: list[str] | None = None, tag: str = "") -> None:
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    res = run_benchmark(
        preset="full",
        seeds=(7, 42, 123),
        datasets=datasets,
        include_noise=True,
    )
    _save(RESULTS / f"rows_noise{tag}.json", _rows_to_dicts(res.rows))
    ok = sum(1 for r in res.rows if r.status == "ok")
    print(f"noise{tag}: {ok}/{len(res.rows)} ok, {time.time() - t0:.0f}s")


def mode_merge() -> None:
    rows: list[dict] = []
    import glob

    for path in sorted(glob.glob(str(RESULTS / "rows_full_s*.json"))):
        rows.extend(json.loads(Path(path).read_text(encoding="utf-8")))
    methods = sorted({r["method"] for r in rows})
    res = BenchmarkResult(preset="full")
    res.rows = [_DictRow(d) for d in rows]
    _compute_gates(res, methods_used=methods)
    report = {"gates": res.gates, "summary": res.summary}
    _save(RESULTS / "full_report.json", report)

    # 门禁速览
    for k, v in res.gates.items():
        extra = ""
        if "flagship" in v and "plain_gp" in v:
            extra = f"  flagship={v['flagship']:.2f} vs plain={v['plain_gp']:.2f}"
        elif "flagship" in v and "floor" in v:
            extra = f"  flagship={v['flagship']:.2f}"
        print(f"  {k}: {'PASS' if v['pass'] else 'FAIL'}{extra}")

    # 噪声子套件汇总
    import glob as _glob

    noise_rows: list[dict] = []
    for path in sorted(_glob.glob(str(RESULTS / "rows_noise*.json"))):
        noise_rows.extend(json.loads(Path(path).read_text(encoding="utf-8")))
    if noise_rows:
        nres = BenchmarkResult(preset="full")
        nres.rows = [_DictRow(d) for d in noise_rows]
        nmethods = sorted({r["method"] for r in noise_rows})
        _compute_gates(nres, methods_used=nmethods)
        _save(RESULTS / "noise_report.json", {"gates": nres.gates, "summary": nres.summary})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["seed", "noise", "merge"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--datasets", help="逗号分隔数据集子集")
    ap.add_argument("--tag", default="", help="输出文件名后缀（分块去重）")
    args = ap.parse_args()
    if args.mode == "seed":
        ds = args.datasets.split(",") if args.datasets else None
        mode_seed(args.seed, datasets=ds, tag=args.tag)
    elif args.mode == "noise":
        ds = args.datasets.split(",") if args.datasets else list(NOISY_SUBSET)
        mode_noise(datasets=ds, tag=args.tag)
    else:
        mode_merge()

"""SymForge CLI。

作者：晨星
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from symforge import __version__
from symforge.core.seed import set_thread_env


def _cmd_suite(args: argparse.Namespace) -> int:
    from symforge.data.suite import SUITE

    print(f"{'dataset':<14}{'ndim':>5}  {'difficulty':<10}  source")
    print("-" * 78)
    for name, spec in SUITE.items():
        print(f"{name:<14}{spec.ndim:>5}  {spec.difficulty:<10}  {spec.source}")
    return 0


def _cmd_recover(args: argparse.Namespace) -> int:

    from symforge.baselines import build_methods
    from symforge.core.config import preset_config
    from symforge.data.suite import make_dataset
    from symforge.eval.metrics import check_symbolic, numeric_match, r2_score

    set_thread_env()
    config = preset_config("full")
    methods = build_methods(config)
    if args.method not in methods:
        print(f"未知方法 {args.method!r}，可选: {sorted(methods)}", file=sys.stderr)
        return 2
    data = make_dataset(args.dataset, args.seed)
    factory, avail = methods[args.method]
    if not avail:
        print(f"方法 {args.method} 后端不可用", file=sys.stderr)
        return 3
    model = factory(args.seed)
    model.fit(data.X_train, data.y_train)
    pred = model.predict(data.X_holdout)
    print(f"dataset : {args.dataset} (ndim={data.ndim}, seed={args.seed})")
    print(f"method  : {args.method}")
    print(f"expr    : {model.expression()}")
    print(f"r2 hold : {r2_score(data.y_holdout, pred):.6f}")
    print(f"numeric : {'MATCH' if numeric_match(pred, data.y_holdout) else 'no'}")
    sym = check_symbolic(model, data.truth, data.ndim)
    print(f"symbolic: {sym}")
    print(f"truth   : {data.truth}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from symforge.pipeline.benchmark import failure_cases, run_benchmark

    set_thread_env()
    seeds = tuple(int(s) for s in args.seeds.split(","))
    datasets = args.datasets.split(",") if args.datasets else None
    result = run_benchmark(
        preset=args.preset,
        seeds=seeds,
        datasets=datasets,
        include_ablations=not args.no_ablations,
        include_noise=args.noise,
        verbose=args.verbose,
    )
    payload = result.to_dict()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"benchmark written -> {out}")

    # 汇总表
    print()
    print(f"{'method':<24}{'recovery':>10}{'r2(mean)':>12}")
    print("-" * 50)
    rows = [r.to_dict() for r in result.rows]
    for m in result.summary["methods"]:
        ok = [r for r in rows if r["method"] == m and r["status"] == "ok"]
        if not ok:
            continue
        rec = float(sum(1 for r in ok if r["metrics"]["numeric_match"]) / len(ok))
        r2 = float(sum(r["metrics"]["r2_holdout"] for r in ok) / len(ok))
        print(f"{m:<24}{rec:>10.3f}{r2:>12.4f}")

    print()
    print("GATES")
    print("-" * 50)
    for gname, g in result.gates.items():
        icon = "PASS" if g.get("pass") else "FAIL"
        print(f"  [{icon}] {gname}: {g}")

    if args.failures:
        print()
        print("FAILURE CASES (derived from results)")
        print("-" * 50)
        for c in failure_cases(result):
            print(f"  {c['dataset']} (seed={c['seed']}, {c['difficulty']}): "
                  f"baselines recovered={c['baselines_recovered']}")
            print(f"    flagship: {c['flagship_expr']}  r2={c['flagship_r2']:.4f}")
    return 0


def _cmd_selftest(args: argparse.Namespace) -> int:
    from symforge.selftest import run_selftest

    return run_selftest(verbose=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="symforge",
        description="SymForge: world-class symbolic regression / formula discovery (author: 晨星)",
    )
    parser.add_argument("--version", action="version", version=f"symforge {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_suite = sub.add_parser("suite", help="列出基准数据集")
    p_suite.set_defaults(func=_cmd_suite)

    p_rec = sub.add_parser("recover", help="单数据集公式恢复")
    p_rec.add_argument("--dataset", default="nguyen1")
    p_rec.add_argument("--seed", type=int, default=7)
    p_rec.add_argument("--method", default="symfuse")
    p_rec.set_defaults(func=_cmd_recover)

    p_run = sub.add_parser("run", help="跑基准评测")
    p_run.add_argument("--preset", default="quick", choices=["quick", "full"])
    p_run.add_argument("--seeds", default="7,42,123")
    p_run.add_argument("--datasets", default="")
    p_run.add_argument("--out", default="examples/benchmark.json")
    p_run.add_argument("--noise", action="store_true", help="启用噪声子套件")
    p_run.add_argument("--no-ablations", action="store_true")
    p_run.add_argument("--failures", action="store_true")
    p_run.add_argument("--verbose", action="store_true")
    p_run.set_defaults(func=_cmd_run)

    p_self = sub.add_parser("selftest", help="不变量自检")
    p_self.set_defaults(func=_cmd_selftest)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

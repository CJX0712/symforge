"""SymForge 端到端演示：3 个代表性数据集 × quick 预设，端到端 ≤ 60s（G5 门禁）。

运行：python examples/run_demo.py
作者：晨星
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from symforge.core.config import preset_config
from symforge.data.suite import make_dataset
from symforge.eval.metrics import numeric_match, r2_score
from symforge.symreg.gp import GeneticProgrammer, predict_individual

DEMO_DATASETS = ("nguyen1", "pendulum", "coulomb")
G5_BUDGET_SEC = 60.0


def main() -> int:
    cfg = preset_config("quick")
    cfg.validate()
    print(f"SymForge demo — quick preset (pop={cfg.population_size}, gens={cfg.generations}, islands={cfg.n_islands})")
    print("=" * 72)

    t_all = time.perf_counter()
    recovered = 0
    for name in DEMO_DATASETS:
        data = make_dataset(name, 7)
        t0 = time.perf_counter()
        res = GeneticProgrammer(cfg, seed=7).fit(data.X_train, data.y_train)
        pred = predict_individual(res.best, data.X_holdout)
        r2 = r2_score(data.y_holdout, pred)
        ok = numeric_match(pred, data.y_holdout)
        recovered += int(ok)
        print(
            f"{name:12s}  {'✅ 恢复' if ok else '— 未恢复'}  R²={r2:+.4f}"
            f"  size={res.best.size:3d}  {time.perf_counter() - t0:5.1f}s"
        )
        print(f"{'':12s}  expr: {res.expression}")

    total = time.perf_counter() - t_all
    print("=" * 72)
    verdict = "✅ PASS" if total <= G5_BUDGET_SEC else "❌ FAIL"
    print(f"恢复 {recovered}/{len(DEMO_DATASETS)}   端到端 {total:.1f}s / {G5_BUDGET_SEC:.0f}s 预算 {verdict}")
    return 0 if total <= G5_BUDGET_SEC else 1


if __name__ == "__main__":
    raise SystemExit(main())

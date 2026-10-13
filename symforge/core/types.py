"""核心数据类型：SRMetrics / BenchmarkRow / BenchmarkResult。

作者：晨星
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SRMetrics:
    """单次符号回归运行的评测指标。

    语义约定（跨模块统一）：
    - ``numeric_match``：数值等价（SRBench 口径），holdout 上
      ``max|ŷ−y| ≤ 1e-6·(1+max|y|)`` 记 True；
    - ``symbolic_exact``：sympy 符号等价（严格，仅树/多项式可判）；
    - ``size``/``depth``：表达式节点数 / 深度（越小越简洁）。
    """

    r2_train: float = float("nan")
    r2_holdout: float = float("nan")
    numeric_match: bool = False
    symbolic_exact: bool | None = None
    size: int = -1
    depth: int = -1
    mse_train: float = float("nan")
    runtime_sec: float = float("nan")
    expression: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkRow:
    """基准评测的一行：(dataset × method × seed)。"""

    dataset: str
    method: str
    seed: int
    difficulty: str = ""
    metrics: SRMetrics = field(default_factory=SRMetrics)
    status: str = "ok"  # ok | skipped

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "method": self.method,
            "seed": self.seed,
            "difficulty": self.difficulty,
            "status": self.status,
            "metrics": self.metrics.to_dict(),
        }


@dataclass
class BenchmarkResult:
    """基准评测总结果：rows + 门禁 + 汇总。"""

    preset: str = ""
    rows: list[BenchmarkRow] = field(default_factory=list)
    gates: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "preset": self.preset,
            "rows": [r.to_dict() for r in self.rows],
            "gates": self.gates,
            "summary": self.summary,
        }


__all__ = ["BenchmarkResult", "BenchmarkRow", "SRMetrics"]

"""方法注册表：旗舰 SymFuse + 消融 + 全部基线。

公平性总约定（写死，报告引用）：
- GP 家族（旗舰/消融/gplearn）共用同一函数集、同一**总评估预算**
  （population_size × n_islands × generations，旗舰的热启动池评估
  以等量削减进化代数偿还）、同一数据切分、同一 tournament_size；
- 消融只改一个开关：linear_scaling / n_islands / seeding / polish /
  parsimony；
- 单岛消融与 plain_gp 的 population_size 乘以 n_islands，保证与旗舰
  **总评估数相等**（否则 G1 的差值会被预算优势虚假抬高）；
- sklearn 基线（poly/forest）训练集内 CV 调参，不触碰 holdout；
- 任何后端缺失 → available()=False，benchmark 记 skipped，不伪造数字。

作者：晨星
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from symforge.baselines.constant import ConstantModel
from symforge.baselines.forest import ForestBaseline
from symforge.baselines.gp_model import GPModel
from symforge.baselines.gplearn_bridge import GplearnModel
from symforge.baselines.gplearn_bridge import available as gplearn_available
from symforge.baselines.poly import PolyRegression
from symforge.core.config import SRConfig

MethodFactory = Callable[[int], Any]

ABLATION_METHODS = (
    "ablate_no_scaling",
    "ablate_single_island",
    "ablate_no_seeding",
    "ablate_no_polish",
    "ablate_no_parsimony",
)
GP_METHODS = ("symfuse", "plain_gp", "gplearn", *ABLATION_METHODS)


def _gp_factory(name: str, **overrides: Any) -> MethodFactory:
    def factory(seed: int) -> GPModel:
        cfg = SRConfig(**{**overrides})
        return GPModel(name, cfg, seed)

    return factory


def build_methods(config: SRConfig) -> dict[str, tuple[MethodFactory, bool]]:
    """返回 {method_name: (factory, available)}。

    config 提供 GP 家族预算（population_size / generations / tournament_size）。
    总评估预算 = population_size × n_islands × generations，对所有 GP
    家族方法一致（旗舰的热启动池评估由代数削减偿还，见 gp.fit）。
    """
    gp_kw = dict(
        population_size=config.population_size,
        generations=config.generations,
        tournament_size=config.tournament_size,
    )
    total_pop = config.population_size * max(1, config.n_islands)
    methods: dict[str, tuple[MethodFactory, bool]] = {
        # ---- 旗舰 SymFuse：LS + 4 岛 + 热启动 + Pareto + 抛光 + parsimony ----
        "symfuse": (_gp_factory("symfuse", **gp_kw, linear_scaling=True, n_islands=4), True),
        # ---- 消融 A1：关线性缩放（其余全开）----
        "ablate_no_scaling": (
            _gp_factory("ablate_no_scaling", **gp_kw, linear_scaling=False, n_islands=4),
            True,
        ),
        # ---- 消融 A2：关岛屿（单岛，总预算不变：pop×4）----
        "ablate_single_island": (
            _gp_factory(
                "ablate_single_island",
                linear_scaling=True,
                n_islands=1,
                population_size=total_pop,
                generations=config.generations,
                tournament_size=config.tournament_size,
            ),
            True,
        ),
        # ---- 消融 A3：关热启动播种 ----
        "ablate_no_seeding": (
            _gp_factory("ablate_no_seeding", **gp_kw, linear_scaling=True, n_islands=4, use_seeding=False),
            True,
        ),
        # ---- 消融 A4：关常数抛光 ----
        "ablate_no_polish": (
            _gp_factory("ablate_no_polish", **gp_kw, linear_scaling=True, n_islands=4, polish_top=0),
            True,
        ),
        # ---- 消融 A5：关 parsimony 压力 ----
        "ablate_no_parsimony": (
            _gp_factory("ablate_no_parsimony", **gp_kw, linear_scaling=True, n_islands=4, parsimony_coef=0.0),
            True,
        ),
        # ---- plain GP：全部机制关闭（家族内部对照；总预算=旗舰）----
        "plain_gp": (
            _gp_factory(
                "plain_gp",
                population_size=total_pop,
                generations=config.generations,
                tournament_size=config.tournament_size,
                linear_scaling=False,
                n_islands=1,
                use_seeding=False,
                polish_top=0,
                parsimony_coef=0.0,
            ),
            True,
        ),
        # ---- 外部 Tier-0 基线（等预算：pop×n_islands 单种群）----
        "gplearn": (
            lambda seed: GplearnModel(
                seed=seed,
                population_size=total_pop,
                generations=config.generations,
                tournament_size=config.tournament_size,
            ),
            gplearn_available(),
        ),
        # ---- 经典 ML 基线 ----
        "poly": (lambda seed: PolyRegression(seed=seed), True),
        "forest": (lambda seed: ForestBaseline(seed=seed), True),
        # ---- 地板 ----
        "constant": (lambda seed: ConstantModel(seed=seed), True),
    }
    return methods


__all__ = ["ABLATION_METHODS", "GP_METHODS", "build_methods"]

"""SRConfig：符号回归运行配置 + ENV_SYMFORGE_* 覆盖 + schema 校验。

MatrixForge 教训：``from __future__ import annotations`` 使 ``f.type`` 为**字符串**，
``target is int`` 永不命中 → env 值残留字符串静默失效。故按**字段名字符串映射**转换。

作者：晨星
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields

from symforge.core.errors import ConfigError

ENV_PREFIX = "SYMFORGE_"


@dataclass
class SRConfig:
    """符号回归引擎配置（所有字段均可被 ENV_SYMFORGE_* 覆盖）。"""

    # ---- 进化预算 ----
    population_size: int = 300
    generations: int = 40
    tournament_size: int = 4
    elitism: int = 2

    # ---- 结构约束 ----
    max_depth_init: int = 5
    max_depth: int = 10
    const_min: float = -8.0
    const_max: float = 8.0
    use_erc: bool = True

    # ---- 算子概率 ----
    p_crossover: float = 0.70
    p_subtree_mutation: float = 0.15
    p_point_mutation: float = 0.10
    p_constant_perturb: float = 0.05

    # ---- 旗舰机制开关（消融用）----
    linear_scaling: bool = True
    n_islands: int = 4
    migration_interval: int = 5
    migration_top: int = 1

    # ---- 适应度塑形 ----
    # 甜点校准（nguyen1 网格实测）：1e-3 过压缩（小近似解压过精确解），
    # 0 无压缩（膨胀等价解），1e-4 兼顾「精确优先」与「简洁偏好」
    parsimony_coef: float = 0.0001
    polish_top: int = 2
    tol_fitness: float = 1e-9
    seed_fraction: float = 0.3
    use_seeding: bool = True

    def validate(self) -> None:
        """schema 校验；非法抛 ConfigError（含 from None 抑制链）。"""
        ints = {
            "population_size": (10, 5000),
            "generations": (1, 500),
            "tournament_size": (2, 20),
            "elitism": (0, 50),
            "max_depth_init": (1, 15),
            "max_depth": (2, 30),
            "n_islands": (1, 16),
            "migration_interval": (1, 100),
            "migration_top": (0, 10),
        }
        for name, (lo, hi) in ints.items():
            v = getattr(self, name)
            if not isinstance(v, int) or not (lo <= v <= hi):
                raise ConfigError(f"E301: {name}={v!r} 超出 [{lo},{hi}]") from None
        floats = {
            "p_crossover": (0.0, 1.0),
            "p_subtree_mutation": (0.0, 1.0),
            "p_point_mutation": (0.0, 1.0),
            "p_constant_perturb": (0.0, 1.0),
        }
        for name, (lo, hi) in floats.items():
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or not (lo <= float(v) <= hi):
                raise ConfigError(f"E302: {name}={v!r} 超出 [{lo},{hi}]") from None
        total_p = self.p_crossover + self.p_subtree_mutation + self.p_point_mutation + self.p_constant_perturb
        if total_p > 1.0 + 1e-9:
            raise ConfigError(f"E303: 算子概率之和 {total_p:.4f} > 1.0") from None
        if self.max_depth < self.max_depth_init:
            raise ConfigError(f"E304: max_depth({self.max_depth}) < max_depth_init({self.max_depth_init})") from None
        if self.const_min >= self.const_max:
            raise ConfigError(f"E305: const_min({self.const_min}) >= const_max({self.const_max})") from None
        if self.tol_fitness < 0:
            raise ConfigError(f"E306: tol_fitness({self.tol_fitness}) < 0") from None
        if not (0.0 <= self.parsimony_coef <= 1.0):
            raise ConfigError(f"E309: parsimony_coef({self.parsimony_coef}) 超出 [0,1]") from None
        if not (0 <= self.polish_top <= 20):
            raise ConfigError(f"E310: polish_top({self.polish_top}) 超出 [0,20]") from None
        if not (0.0 <= self.seed_fraction <= 1.0):
            raise ConfigError(f"E311: seed_fraction({self.seed_fraction}) 超出 [0,1]") from None


def _coerce(name: str, raw: str):
    """按字段名把 env 字符串转为正确类型（MatrixForge 修法）。"""
    int_fields = {
        "population_size",
        "generations",
        "tournament_size",
        "elitism",
        "max_depth_init",
        "max_depth",
        "n_islands",
        "migration_interval",
        "migration_top",
    }
    float_fields = {
        "const_min",
        "const_max",
        "p_crossover",
        "p_subtree_mutation",
        "p_point_mutation",
        "p_constant_perturb",
        "tol_fitness",
        "parsimony_coef",
    }
    bool_fields = {"use_erc", "linear_scaling"}
    if name in int_fields:
        return int(raw)
    if name in float_fields:
        return float(raw)
    if name in bool_fields:
        low = raw.strip().lower()
        if low in ("1", "true", "yes", "on"):
            return True
        if low in ("0", "false", "no", "off"):
            return False
        raise ConfigError(f"E307: {name} 无法解析为 bool: {raw!r}")
    return raw


def from_env(base: SRConfig | None = None) -> SRConfig:
    """应用 ENV_SYMFORGE_* 覆盖（返回新对象，不修改入参）。"""
    cfg = SRConfig() if base is None else SRConfig(**{f.name: getattr(base, f.name) for f in fields(base)})
    names = {f.name for f in fields(cfg)}
    for key, raw in os.environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        name = key[len(ENV_PREFIX):].lower()
        if name not in names:
            continue
        setattr(cfg, name, _coerce(name, raw))
    cfg.validate()
    return cfg


# ---- 预设 ----
# quick：demo 预设（≤60s 预算）；full：完整基准（CLI 可用）
PRESETS: dict[str, dict] = {
    "quick": dict(
        population_size=150,
        generations=25,
        n_islands=4,
        n_seeds=3,
        datasets="all",
        include_ablations=True,
        ablation_seeds=2,
    ),
    "full": dict(
        population_size=400,
        generations=50,
        n_islands=4,
        n_seeds=3,
        datasets="all",
        include_ablations=True,
        ablation_seeds=3,
    ),
}


def preset_config(preset: str) -> SRConfig:
    """取预设的引擎配置（不包含数据集/种子等流水线字段）。"""
    if preset not in PRESETS:
        raise ConfigError(f"E308: 未知 preset {preset!r}，可选 {sorted(PRESETS)}") from None
    spec = PRESETS[preset]
    return SRConfig(
        population_size=spec["population_size"],
        generations=spec["generations"],
        n_islands=spec["n_islands"],
    )


__all__ = ["ENV_PREFIX", "PRESETS", "SRConfig", "from_env", "preset_config"]

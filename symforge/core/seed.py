"""全局确定性：唯一 seed 入口。

规则（SOP §3）：
- 所有随机性必须经由 ``make_rng(seed)`` 派生的 ``numpy.random.Generator``；
- 禁止在库内使用全局 ``np.random``；
- 线程环境变量在进程入口（cli / run_demo / conftest）统一设置，
  保证 BLAS 求和顺序稳定（MatrixForge 实测多线程 BLAS 可致 1e-14 级抖动）。

作者：晨星
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np

_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def set_thread_env() -> None:
    """把数值库线程数钉为 1（幂等；已设置则不覆盖）。"""
    for var in _THREAD_VARS:
        os.environ.setdefault(var, "1")


def make_rng(seed: int) -> np.random.Generator:
    """从整数 seed 派生确定性 PCG64 生成器。"""
    if not isinstance(seed, (int, np.integer)):
        raise SymRegError(f"E201: seed 必须为整数，得到 {type(seed).__name__}")
    return np.random.Generator(np.random.PCG64(int(seed)))


def set_all(seed: int) -> np.random.Generator:
    """设置全局随机源（random 模块 + numpy legacy）并返回 Generator。"""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    return make_rng(seed)


def seed_of(obj: Any, default: int = 0) -> int:
    """从任意对象稳定派生一个 seed（用于按数据集名派生数据 seed）。"""
    import zlib

    if isinstance(obj, (int, np.integer)):
        return int(obj)
    payload = repr(obj).encode("utf-8")
    return zlib.crc32(payload) % (2**31)


# 延迟导入避免环
from symforge.core.errors import SymRegError  # noqa: E402

__all__ = ["make_rng", "seed_of", "set_all", "set_thread_env"]

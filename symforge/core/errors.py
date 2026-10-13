"""错误目录 E100~E500。

ValuaForge 教训：Python 重定义同名类会**静默遮蔽**先定义者（导入期零信号）。
``_discover_registry`` 同时检测**重码与重名**，撞名即 RuntimeError 立即炸。

作者：晨星
"""

from __future__ import annotations


class SymForgeError(Exception):
    """所有 SymForge 错误的基类。"""

    code = "E000"


class DataError(SymForgeError):
    """E1xx：数据生成 / 加载 / 契约错误。"""

    code = "E100"


class SymRegError(SymForgeError):
    """E2xx：符号回归引擎错误（树 / GP / 简化）。"""

    code = "E200"


class ConfigError(SymForgeError):
    """E3xx：配置错误。"""

    code = "E300"


class EvalError(SymForgeError):
    """E4xx：评测错误。"""

    code = "E400"


class PipelineError(SymForgeError):
    """E5xx：流水线错误。"""

    code = "E500"


def _discover_registry() -> dict[str, type[SymForgeError]]:
    registry: dict[str, type[SymForgeError]] = {}
    for name, obj in list(globals().items()):
        if not (isinstance(obj, type) and issubclass(obj, SymForgeError)):
            continue
        if obj is SymForgeError:
            continue
        if name in registry:
            raise RuntimeError(f"错误类重名（静默遮蔽风险）: {name}")
        registry[name] = obj
    codes: dict[str, str] = {}
    for name, cls in registry.items():
        if cls.code in codes:
            raise RuntimeError(f"错误码重复: {cls.code} 被 {codes[cls.code]} 与 {name} 共用")
        codes[cls.code] = name
    return registry


ERROR_CATALOG = _discover_registry()


def error_class(code: str) -> type[SymForgeError]:
    """按错误码取类；未注册抛 KeyError。"""
    for cls in ERROR_CATALOG.values():
        if cls.code == code:
            return cls
    raise KeyError(code)


__all__ = [
    "ERROR_CATALOG",
    "ConfigError",
    "DataError",
    "EvalError",
    "PipelineError",
    "SymForgeError",
    "SymRegError",
    "error_class",
]

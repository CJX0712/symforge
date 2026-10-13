"""SymForge：符号回归 / 公式发现系统（SymFuse 旗舰）。

模块划分（单向无环：cli → pipeline → {data, baselines, eval, symreg} → core）：

- ``core``      类型 / 错误 / 配置 / 种子 / 接口
- ``data``      基准套件（Nguyen + 物理公式，符号真值）
- ``symreg``    受保护算子 / 表达式树 / GP 引擎 / sympy 桥接
- ``baselines`` 旗舰封装 + 消融 + gplearn/poly/forest/constant 基线
- ``eval``      R² / 数值等价 / 符号等价 / 显著性
- ``pipeline``  基准评测 + 门禁 + 失败案例

作者：晨星
"""

__version__ = "0.1.0"

__all__ = ["__version__"]

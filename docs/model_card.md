# SymForge Model Card

作者：晨星 · v0.1.0

## 模型概述

SymFuse：符号回归（公式发现）旗舰引擎。输入表格数据 (X, y)，输出可读数学
表达式（postfix 指令流 ↔ sympy 双向）。面向科学公式发现、可解释建模、
特征机理挖掘。

## 预期用途

- ✅ 低维（1~8 变量）、几百样本内、存在紧凑解析式（可能含噪声）的回归任务
- ✅ 可解释性优先场景：输出即为公式，非黑箱
- ⚠️ 高维/百万样本/强非解析结构 → 用 GBDT/神经网络（本工具的 constant/forest 基线即为对照）
- ❌ 不用于：因果推断（发现的公式是相关性拟合，非因果）、安全关键决策的唯一依据

## 训练数据

无需训练数据集——引擎在用户提供的数据上自举。内置基准套件
（Nguyen-1/2/4/5、有理分式、摆/抛体/库仑/透镜物理公式、倒数和）仅用于评测，
采样区间见 `data/suite.py`。

## 性能数字（等评估预算，330 单元，0 失败）

- clean 套件 numeric-recovery：**0.633**（plain_gp 0.333 / gplearn 0.300 / poly 0.400）
- 5% 标签噪声子套件：**1.0**（9/9）
- demo 端到端：3.1s（预算 60s）
- 完整数字与门禁：`results/full_report.json`、`results/noise_report.json`

## 局限与偏差

- 预注册 G1 显著性判据中 half-σ 项在 10 数据集二元恢复率上功效不足
  （Δ=0.30、p=0.021 已达标，仅该项未过）——如实报告，未事后改判据；
- 消融显示线性缩放与 parsimony 在本套件上为负贡献（诚实负结果，
  作为配置优化方向记录）；
- 符号等价判定对 >80 节点表达式直接放弃（防 simplify 挂死），
  此时仅报数值等价。

## 硬件/运行时

纯 CPU。quick 预设单 run ≈1s；full 预设（80000 evals）单 run ≈5-20s。
无 GPU 依赖；Python ≥3.12，核心依赖 numpy/scipy/sympy/scikit-learn。

## 引用

```
@software{symforge2026,
  author = {晨星},
  title  = {SymForge: world-class symbolic regression engine},
  year   = {2026},
  url    = {https://github.com/CJX0712/symforge}
}
```

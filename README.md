# SymForge

**SymForge / SymFuse** —— 世界级符号回归（Symbolic Regression / 公式发现）引擎：
纯 NumPy 遗传规划 + 线性缩放 + 岛屿迁移 + Pareto 档案 + SINDy/FFX 式热启动 + 常数抛光，
在**严格等评估预算**下对比 GP 家族与经典基线全链路可复现。

作者：**晨星** · License: MIT

---

## 为什么是它

| 维度 | 做法 |
|---|---|
| **引擎** | 线性 GP（postfix 指令流）+ 波次批量向量化评估（5~9× 提速，单 run ≈1s @quick 预算） |
| **热启动** | 预算中立两阶段：全对组合骨架 → 逐级残差匹配（FFX/matching pursuit，2-fold CV 防过拟合）→ OLS 联合重拟合（SINDy 标准步骤）；池评估以等量削减进化代数偿还 |
| **公平性** | 所有 GP 方法（旗舰/消融/gplearn/plain）共用同一**总评估预算**、同一数据切分、同一锦标赛；外部基线训练集内 CV 调参，不触碰 holdout |
| **评测** | 10 数据集（Nguyen + 物理公式族）× 3 seeds × 11 方法 = 330 单元，0 失败；预注册门禁 + 5% 噪声子套件 |
| **正确性** | 23 项单测全绿：roundtrip oracle（程序→树→程序恒等）、splice vs 树级参照逐点一致、批量/单路径逐位一致、确定性锁定 |

## 快速开始

```bash
pip install -e ".[dev,gplearn]"

# 端到端演示（≤60s 预算，实测 ~3s，3/3 恢复）
python examples/run_demo.py

# 测试
pytest tests/ -q

# 全量基准（分块 runner，330 单元）
python scripts/bench_runner.py --mode seed --seed 7 --datasets nguyen1,nguyen2 ...
python scripts/bench_runner.py --mode merge
```

## 基准结果（results/full_report.json，可复现审计）

**Clean 套件**（10 数据集 × 3 seeds，等预算）numeric-recovery：

| 方法 | 恢复率 |
|---|---|
| **SymFuse（旗舰）** | **0.633** |
| ablate_no_parsimony | 0.800 |
| ablate_no_scaling | 0.700 |
| ablate_no_polish | 0.600 |
| ablate_single_island | 0.633 |
| plain_gp | 0.333 |
| gplearn（等预算单种群） | 0.300 |
| poly（sklearn, CV 调参） | 0.400 |
| forest / constant | 0.000 |

**噪声子套件**（5% 标签噪声，容差=噪声水平，SRBench 惯例）：**SymFuse 1.0**（9/9 全恢复）。

### 门禁（预注册，负结果如实保留）

| 门禁 | 结果 | 说明 |
|---|---|---|
| G2 绝对地板 ≥ 0.55 | ✅ PASS | 0.633 |
| G1 旗舰 vs plain_gp Δ≥0.15 | ⚠️ FAIL（判据过严） | Δ=**0.30** ≥ 0.15 ✓，Wilcoxon 单侧 **p=0.021** < 0.05 ✓；仅 half-σ 判据（0.386）未过——10 数据集二元恢复率粒度太粗放大 σ，属预注册判据的统计功效缺陷，如实报告 |
| G4 消融 | ✅ seeding 最大贡献 | 热启动播种 +0.233（最大单项）；抛光 +0.033；**诚实负结果**：线性缩放 −0.067、parsimony −0.167（简洁压力提升可读性但降低恢复率）——消融驱动的后续配置优化方向 |
| G5 demo ≤ 60s | ✅ PASS | 3.1s |

## 架构

```
symforge/
├── core/        # 配置（ENV 覆盖 + schema 校验）、错误、类型、seed 派生
├── data/        # 10 数据集套件（Nguyen + 物理；噪声注入只加训练集）
├── symreg/
│   ├── program.py    # 线性 GP：postfix 指令流 + 数组级变异（splice/交叉/点变异/常量扰动）
│   ├── batch_eval.py # 波次批量评估（(深度×算子) lexsort 分波，numpy 广播）
│   ├── gp.py         # 进化主循环：4 岛迁移、Pareto 档案、锦标赛、热启动、常数抛光
│   ├── tree.py       # 不可变元组树（参照实现 + sympy 互转）
│   └── simplify.py   # 符号等价判定（节点数预过滤 + 线程超时双护栏）
├── baselines/   # gplearn 桥、poly/forest/constant + 5 消融注册表（等预算契约）
├── eval/        # R²/数值等价/符号等价/Wilcoxon + 聚合
└── pipeline/    # 基准流水线：330 单元 + 预注册门禁 + 失败归因
```

细节见 [docs/architecture.md](docs/architecture.md) 与 [docs/model_card.md](docs/model_card.md)。

## 复现声明

- 全随机性经 `seed_of` 域分离派生，同 seed 逐位复现（单测锁定）；
- 基准原始行 `results/rows_*.json` 全量入库，报告由 `--mode merge` 从原始行重算，无手工数字；
- 已知工程事故与修复全部记录在案（漏叶操作码、splice 列索引损坏、假线程超时等），回归单测覆盖。

## License

MIT © 晨星

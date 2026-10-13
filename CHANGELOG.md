# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/)，版本号遵循 SemVer。

## [0.1.0] - 2026-10-13

首个公开版本。作者：晨星。

### Added
- 线性 GP 引擎：postfix 指令流表示 + 数组级变异（splice/交叉/点变异/常量扰动）
- 波次批量向量化评估引擎（5~9× 提速，分块数值不变性）
- SymFuse 旗舰机制：线性缩放、4 岛迁移、Pareto 档案、预算中立两阶段热启动
  （FFX/matching pursuit + 2-fold CV + OLS 联合重拟合）、常数细网格抛光
- 基准流水线：10 数据集 × 3 seeds × 11 方法（等评估预算契约）+ 预注册门禁
  + 5% 噪声子套件 + 失败归因
- 23 项单测：roundtrip oracle、splice vs 树级参照、批量/单路径逐位一致、
  确定性锁定、漏叶操作码回归
- 示例 demo（≤60s 预算，实测 3.1s，3/3 恢复）
- CI（ruff + pytest + demo smoke，Python 3.12/3.13 矩阵）

### 基准结果
- clean 恢复率：SymFuse 0.633 vs plain_gp 0.333（Δ=0.30，Wilcoxon p=0.021）
- 噪声子套件（5%）：SymFuse 1.0
- 诚实负结果：线性缩放 −0.067、parsimony −0.167（消融驱动配置优化方向）

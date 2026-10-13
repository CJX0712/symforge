# SymForge 架构

作者：晨星

## 1. 表示层：线性 GP（postfix 指令流）

树表达式编译为 postfix 数组（`Program`：`ops/a/b/cols/consts/sizes/edepth`），
变异 = 数组切片 + 引用重映射（~10µs），求值可跨种群批量向量化（~10µs/个体，5~9×）。

**表示约定（单测锁定）**：
- postfix（RPN）：指令 k 的子节点槽位恒 < k；**子树根 = 跨度内最后一条指令**；
  `starts[i] = i+1-sizes[i]`、`ends[i] = i+1`；
- **槽位/列索引物理分离**：`a/b` 只存子槽位，`cols` 专存变量列索引——
  混用曾致 splice 平移损坏列索引（index -2 事故）；
- 叶指令的 a/b 为占位，片段平移后可为负——安全（叶的 a/b 永不被读）；
- 无死代码保证：程序恒为纯树，批量评估直接取段末指令为根。

## 2. 评估层：波次批量引擎（batch_eval.py）

按 `(求值深度, 算子)` lexsort 分波；同波同算子合并为单次 numpy 广播
（gather → op → scatter）。`_OP_VALUES` 必须含叶操作码 0/1
（曾漏叶致全种群适应度恒等、搜索退化为纯 size 最小化——回归单测锁定）。
缓冲复用 + `P×maxL×n` 上限自动分块（分块与单块逐位一致，单测锁定）。

## 3. 进化层（gp.py）

- **4 岛 + 环迁移**（每 5 代迁最优 1 个）；
- **Pareto 档案**（fitness ↓ × size ↓），支配规则含等号（防全等个体无限累积）；
- **锦标赛**（size=4）+ 精英保留；
- 变异算子：交叉 0.70 / 子树变异 0.15 / 点变异 0.10 / 常量扰动 0.05；
- **常数抛光**：批量评估逐常量试探 ±delta，终抛光用细网格（至 1e-6）；
- **确定性**：全部 RNG 走 `np.random.Generator`，同 seed 逐位复现。

## 4. 热启动：预算中立两阶段（集大成 SINDy/FFX）

1. **阶段 A**：单变量块 + 全对组合（add/sub/mul/div）批量评估；
2. **阶段 B**：逐级残差匹配（matching pursuit / FX 式贪心）只在骨架基上做，
   **2-fold CV 打分**防过拟合基带偏；
3. **阶段 C**：**OLS 联合重拟合**（SINDy 标准步骤，`np.linalg.lstsq`，
   列数≤样本数良态守卫）。

池评估数计入总预算：`gens_eff = generations - ceil(extra / evo_per_gen)`，
与 plain_gp **总评估数严格相等**（公平对比契约，基准层验证）。

## 5. 评测层

- `numeric_match`：holdout 上 `max|ŷ−y| ≤ tol·(1+max|y|)`；
  clean 任务 tol=1e-6（SRBench 口径），噪声任务 tol=noise_frac（容差感知噪声尺度）；
- `symbolic_equal`：常量有理化 → `sp.simplify(差)==0`；
  节点数预过滤（>80 直接 False）+ 线程超时**非阻塞 shutdown** 双护栏
  （曾有 `with ThreadPoolExecutor` 隐式 join 致假超时挂住基准十余分钟的事故）；
- `significance`：按数据集配对 Wilcoxon 单侧 + half-σ 判据。

## 6. 工程事故档案（全部有回归测试）

| 事故 | 根因 | 修复 |
|---|---|---|
| splice IndexError -2 | 列索引与槽位混用 + 跨度端点取错 | `cols` 分离 + `span_start/span_end` |
| 全种群适应度恒等 | `_OP_VALUES` 漏叶操作码 | `arange(0,14)` + 回归测试 |
| Pareto 档案被同表达式占满 | 支配规则不含等号 | 规则加等号 |
| 恢复率 0.9998 近失误 | 贪心烘系数在相关基上偏斜 | OLS 联合重拟合 + 2-fold CV 选基 |
| 噪声套件全员 0 恢复 | match 容差不感知噪声 | tol = max(1e-6, noise_frac) |
| 基准挂死十余分钟 | 线程超时隐式 join | 节点预过滤 + 非阻塞 shutdown |
| G5 恒 FAIL | 误用整套基准总时长对比 demo 预算 | demo 端到端计时 + 单 run 统计 |

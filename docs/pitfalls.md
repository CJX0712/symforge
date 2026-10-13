# SymForge 事故与修复档案（pitfalls.md）

全部事故均有回归测试或流程修正。作者：晨星

## 表示层

1. **splice IndexError -2**：变量列索引与子槽位混在 `a` 数组，splice 平移损坏列索引；
   且 crossover 传 `[j, ends[j])` 但 postfix 子树跨度是 `[starts[j], ends[j])`（根是跨度最后一条指令）。
   修：`cols` 物理分离 + `span_start/span_end`。
2. **叶占位可为负**：`_shift_seg_refs` 平移叶 a/b 占位——安全，叶的 a/b 永不被读（文档化 + 单测锁定）。

## 评估层

3. **全种群适应度恒等、搜索退化为纯 size 最小化**：`_OP_VALUES = range(2,14)` 漏叶操作码 0/1，
   OP_VAR/OP_CONST 指令从未写入缓冲，np.empty 零页使所有 f≡0。修：`arange(0,14)` + 回归测试。
4. **噪声套件全员 0 恢复**：match 容差 1e-6 不感知噪声尺度（R²=0.9997 仍挂）。
   修：tol = max(1e-6, noise_frac)（SRBench 惯例）。

## 进化层

5. **Pareto 档案被同一表达式占满 5+ 席**：支配规则不含等号，全等个体不被支配。修：规则加等号。
6. **恢复率 0.9998 近失误**（nguyen2/4/5、recip_sum3）：贪心逐级烘系数在相关基上偏斜。
   修：OLS 联合重拟合 + 2-fold CV 选基 + 骨架只含 singles + 种子加 x⁵/x⁶。
7. **公平对比契约**：单岛方法 population_size 必须乘 n_islands（否则 G1 被预算优势虚假抬高）；
   gplearn 有 const_range（勿想当然）。

## 工程

8. **假线程超时挂死基准十余分钟**：`with ThreadPoolExecutor` 退出隐式 join，
   `fut.result(timeout=5)` 超时后仍等 simplify 跑完。修：节点数预过滤（>80 直接 False）
   + `shutdown(wait=False, cancel_futures=True)`。
9. **G5 门禁恒 FAIL**：实现错误——拿整套基准总时长（2563s）对比 demo 60s 预算。
   修：demo 端到端实测（3.1s）+ 基准内只报单 run 时长统计。
10. **后台任务跑 venv python 报 `No module named 'encodings'`**：长任务必须前台跑；
    超 600s 的命令会 SIGTERM——基准必须分块（scripts/bench_runner.py 按 seed×数据集切）。
11. **Python 环境灾备**：托管 python Lib/ 整目录消失——embeddable zip 解包 python313.zip 到
    `versions/<ver>/Lib/` 可恢复；包装在 `envs/default` venv。
12. **`keyword argument repeated`**：sed 补丁只删一半导致工厂函数 kw 重复——补丁后必须 import 冒烟。

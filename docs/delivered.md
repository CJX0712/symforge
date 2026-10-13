# SymForge 交付台账（delivered.md）

作者：晨星 · 2026-10-13 · v0.1.0 已开源

## 交付物

| 项 | 状态 | 位置 |
|---|---|---|
| 线性 GP 引擎（postfix 指令流 + 数组级变异） | ✅ | `symforge/symreg/program.py` |
| 波次批量向量化评估（5~9×，分块数值不变） | ✅ | `symforge/symreg/batch_eval.py` |
| SymFuse 旗舰（LS+4 岛+Pareto+热启动+抛光+parsimony） | ✅ | `symforge/symreg/gp.py` |
| 预算中立两阶段热启动（FFX+CV+OLS，SINDy 式） | ✅ | `gp.py::_warm_start_pool` |
| 等预算基准流水线（330 单元，0 失败） | ✅ | `symforge/pipeline/benchmark.py` + `results/` |
| 预注册门禁（G1/G1b/G2/G4/G5） | ✅ | `benchmark.py::_compute_gates` |
| 单测 23 项（roundtrip oracle 等） | ✅ 全绿 | `tests/` |
| selftest 10/10、ruff 全绿 | ✅ | `symforge/selftest.py` |
| demo（≤60s 预算） | ✅ 3.1s / 3 数据集全恢复 | `examples/run_demo.py` |
| CI（ruff+pytest+demo，py3.12/3.13） | ✅ 已触发 | `.github/workflows/ci.yml` |
| 文档（README/architecture/model_card/CHANGELOG） | ✅ | `README.md` `docs/` |
| GitHub 开源 + tag v0.1.0 + Release | ✅ | https://github.com/CJX0712/symforge |

## 基准数字（等评估预算，可复现审计于 results/）

- clean 恢复率：**SymFuse 0.633** / plain_gp 0.333 / gplearn 0.300 / poly 0.400
  （Δ=0.30 ≥ 0.15，Wilcoxon p=0.021）
- G2 地板 ✅（0.633 ≥ 0.55）；G5 demo ✅（3.1s）
- G1 ⚠️ 仅 half-σ 判据未过（0.386 > 0.30）——预注册判据在 10 数据集二元率上功效不足，如实保留 FAIL，未事后改判据
- 消融（诚实负结果保留）：播种 +0.233（最大贡献）、抛光 +0.033、岛屿 ±0、线性缩放 −0.067、parsimony −0.167
- 噪声子套件（5%，容差=噪声水平）：SymFuse **1.0**（9/9）

## 遗留优化方向（消融驱动，未塞进 v0.1.0）

1. 关 parsimony（0.80 > 0.63）与关线性缩放（0.70 > 0.63）提示旗舰默认配置有提升空间——需新一轮预注册后全量重跑，避免事后调参污染；
2. G1 判据建议换 per-dataset 配对二项检验或增加数据集数；
3. gplearn 缺常量终端属结构性劣势，G1b 只作次级报告（已实现）。

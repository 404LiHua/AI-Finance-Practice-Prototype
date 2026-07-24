# 阶段 C 当前推荐模型 v2 报告

运行日期：2026-07-24  
推荐模型：`fixed_control_ensemble_v2`  
范围：固定 30 只股票、价格特征、三个冻结种子、validation 开发比较

## 1. 改进过程

根据前序消融，首先实现了“固定因果时间图 + 时域图传播 + 等权残差融合”的单模型 v2。该结构取消 Gumbel、FFT 和可学习门控，但三种子 MAE 为 0.040009，较原始图频 v1 变差 1.36%，因此该直接组合被拒绝，没有提升为推荐模型。

随后采用不增加训练参数的固定控制组集成：

```text
最终预测 = 0.5 × 仅时域 Transformer
         + 0.5 × 固定时间图对照
```

两个分支使用相同数据、特征、split 和种子。权重固定为 0.5/0.5，不利用 validation 学习融合参数，避免重现可学习门控的不稳定问题。

## 2. 三种子结果

| 种子 | Validation MAE | Validation RMSE | 方向 Accuracy | 方向 F1 |
|---:|---:|---:|---:|---:|
| 20260723 | 0.034534 | 0.053879 | 65.83% | 0.3051 |
| 20260724 | 0.036107 | 0.051498 | 57.50% | 0.3704 |
| 20260725 | 0.034683 | 0.052196 | 59.17% | 0.1967 |
| 均值 ± 标准差 | **0.035108 ± 0.000868** | **0.052524 ± 0.001224** | 60.83% | 0.2907 |

相对原始动态图频域 v1：

- MAE 降低 11.05%。
- RMSE 降低 5.40%。
- 方向 Accuracy 从 51.94% 提升到 60.83%。
- MAE 标准差从 0.001654 降至 0.000868。

因此固定控制组集成满足“同时改善 v1 的三种子平均 MAE 和 RMSE”这一提升门槛，正式替代图频 v1 成为当前推荐开发模型。

## 3. 与现有基线比较

| 模型 | MAE | RMSE | 方向 Accuracy | 方向 F1 |
|---|---:|---:|---:|---:|
| 固定控制组集成 v2 | **0.035108** | 0.052524 | **60.83%** | 0.2907 |
| FreTS 4 周 | 0.035475 | **0.050712** | 57.22% | **0.4377** |
| 价格版 Minimalist Transformer | 0.035717 | 0.052308 | 57.22% | 0.3591 |
| Naive | 0.036745 | 0.051702 | 69.17% | 0.0000 |
| 原始图频 v1 | 0.039471 | 0.055525 | 51.94% | 0.3431 |

v2 取得当前开发比较中的最低平均 MAE，但 RMSE 仍落后于 FreTS、Naive 和价格 Transformer，方向 F1 也低于 FreTS。因此它是当前阶段 C 的推荐工程模型，不代表在所有指标上全面领先。

## 4. 当前模型状态

- 当前推荐：固定等权控制组集成 v2。
- 稳定基础分支：仅时域 Transformer。
- 图结构分支：固定因果时间邻接图。
- 原始图频 v1：保留用于可追溯和消融，不再作为推荐运行结构。
- 固定时域图传播单模型：实验拒绝，不推广。
- 股票横截面节点：继续暂缓。

本模型来自已经多次使用的开发期 validation，属于选择暴露结果。test 没有参与训练、早停、指标计算或结构选择；在新的独立数据到来前，不把该结果解释为独立泛化证据。

## 5. 产物

- 推荐配置：`stage_c/configs/recommended_v2.json`
- 固定集成组件：`stage_c/models/fixed_ensemble.py`
- 构建与评估入口：`stage_c/build_recommended_v2.py`
- 三种子结果：`outputs/experiments/stage_c_30stocks_recommended_v2/recommended_v2_results.csv`
- 统一比较：`outputs/experiments/stage_c_30stocks_recommended_v2/recommended_v2_unified_comparison.csv`
- 决策：`outputs/experiments/stage_c_30stocks_recommended_v2/recommended_v2_decision.json`
- 每个种子的 `model_manifest.json` 记录两个组件模型、固定权重和权重文件路径。

工程成本、逐股票误差、收益分组和组件分歧的完整诊断见：

- `reports/STAGE_C_V2_ENGINEERING_AND_ERROR_DIAGNOSTICS.md`
- `outputs/diagnostics/stage_c_recommended_v2/STAGE_C_V2_DIAGNOSTICS.xlsx`

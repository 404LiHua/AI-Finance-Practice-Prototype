# 阶段 D-3：跨折稳健诊断与唯一候选建议

报告日期：2026-07-24

诊断 ID：`stage_d_d3_robust_diagnostics_v1`

来源实验：`stage_d_d2_bounded_baselines_v1`
协议：`stage_d_rolling_origin_v1`

## 1. 诊断边界

- 候选集合严格等于 D-2 登记的 17 个模型结果。
- 未训练新模型、未增加候选、未修改 `0.25/0.50/0.75` 收缩系数。
- 复算并通过 D-2 六项汇总文件 SHA-256。
- 读取了 153 行模型—折—种子指标和 22,950 行 D-2 开发期预测。
- 未读取 C-4 逐样本数据，未读取未来 D-SCREENING。
- 本报告是选择暴露的滚动开发证据，不是独立验证结论。

## 2. 预定义稳健门槛与排序

准入门槛沿用阶段 D 计划：

1. 至少 2/3 折的平均 MAE 优于同折 Naive。
2. 最差折 MAE 相对 Naive 的恶化不得超过 2%。
3. 跨折平均 MAE 相对 Naive 必须严格改善。

通过门槛后，按以下固定顺序形成唯一建议：平均 MAE、最差折差距、MAE 变异系数、平均 RMSE、模型名，均按升序排列。Naive 仅作基线，不参与推荐。

## 3. 唯一候选建议

唯一建议为：`frets_return_l4__fixed_shrink_a075`。

| 指标 | 结果 |
|---|---:|
| 跨折平均 MAE | 0.031242 |
| 跨折平均 RMSE | 0.044065 |
| 相对 Naive 平均 MAE 改善 | 1.780% |
| 优于 Naive 的折数 | 3/3 |
| 最差折相对 Naive | 改善 1.110% |
| MAE 变异系数 | 0.083620 |

共有 6 个结果通过门槛，冻结排序如下：

1. FreTS L4 + 收缩 0.75
2. FreTS L4 + 收缩 0.50
3. FreTS L4 原始预测
4. FreTS L4 + 收缩 0.25
5. Minimalist Transformer + 收缩 0.25
6. 仅时域控制组 + 收缩 0.25

该排序仅用于形成 D-4 的唯一冻结建议，不允许在未来数据上重新选择上述 6 个结果。

## 4. 最差折诊断

建议候选的最差折为 `D_RO_03`：

- MAE：`0.031538`
- Naive MAE：`0.031892`
- 相对 Naive 改善：`1.110%`
- RMSE 改善：`0.299%`
- 三种子 MAE 标准差：`0.000041`

即使在最差折中，候选仍满足保护门槛并优于 Naive。

## 5. 种子稳定性

- 三种子平均 MAE 范围：`0.031197–0.031270`
- 种子 MAE 极差：`0.000073`
- 种子 MAE 变异系数：`0.001255`
- 单样本跨种子预测标准差均值：`0.004051`
- 单样本跨种子预测标准差最大值：`0.026058`

总体 MAE 对种子稳定，但少量单样本预测仍存在较大的跨种子偏移，D-4 应保留三种子固定聚合或明确冻结单一推理方式。

## 6. 逐股票误差

- 30 只股票中，19 只的 MAE 优于对应 Naive。
- 误差最大的股票为 `000021.SZ`：MAE `0.079166`，比 Naive 恶化 `5.811%`，占候选绝对误差总量 `8.447%`。
- 误差贡献最高的 5 只股票合计占绝对误差 `30.702%`。

候选的总体优势并非覆盖所有股票，D-4 失败条件应明确监控最差股票误差集中度，而不能只看总 MAE。

## 7. 收益分组误差

| 实际收益组 | 样本数 | 候选 MAE | 相对 Naive 改善 |
|---|---:|---:|---:|
| 负向尾部 `< -3%` | 288 | 0.049990 | +8.795% |
| 负向中等 | 285 | 0.019609 | +1.128% |
| 近零 `[-1%, 1%]` | 309 | 0.006633 | -36.849% |
| 正向中等 | 228 | 0.017995 | +0.243% |
| 正向尾部 `> 3%` | 240 | 0.066828 | -0.917% |

主要优势来自负向尾部；近零收益预测明显不如 Naive，正向尾部也小幅落后。这是候选当前最重要的结构性风险，不能通过增加新阈值或改收缩系数在 D-3 内修补。

## 8. 组件分歧

FreTS L4 与三个价格类控制组的预测相关性仅为 `0.019–0.050`，方向分歧率为 `46.0%–48.9%`。FreTS 的逐样本绝对误差低于：

- Minimalist Transformer：`54.59%` 的样本；
- 仅时域控制组：`56.15%` 的样本；
- 固定时间图控制组：`57.93%` 的样本。

价格版 Minimalist Transformer 与仅时域控制组相关性为 `0.921`，说明两者高度重叠；固定图没有形成足以抵消其额外误差的稳定优势。当前证据不支持把这些分歧直接转化为新的组合候选。

## 9. D-3 结论

D-3 已完成。建议将 `frets_return_l4__fixed_shrink_a075` 作为 D-4 唯一冻结候选，并保留 Naive 作为冻结比较基线。

D-4 尚需冻结：三种子推理聚合方式、逐股票/收益分组失败条件、最差折保护、指标门槛、代码和模型文件 SHA-256，以及独立复算入口。完成 D-4 并取得明确授权前，不得获取未来 D-SCREENING。

## 10. 证据文件

- 固定诊断规则：`stage_d/configs/d3_diagnostics.json`
- 唯一建议：`outputs/stage_d/d3_robust_diagnostics_v1/unique_candidate_recommendation.json`
- 准入与排名：`outputs/stage_d/d3_robust_diagnostics_v1/eligible_candidate_ranking.csv`
- 逐股票诊断：`outputs/stage_d/d3_robust_diagnostics_v1/per_stock_diagnostics.csv`
- 收益分组诊断：`outputs/stage_d/d3_robust_diagnostics_v1/return_group_diagnostics.csv`
- 最差折诊断：`outputs/stage_d/d3_robust_diagnostics_v1/worst_fold_diagnostics.csv`
- 种子稳定性：`outputs/stage_d/d3_robust_diagnostics_v1/seed_stability_summary.csv`
- 组件分歧：`outputs/stage_d/d3_robust_diagnostics_v1/component_disagreement_summary.csv`
- SHA-256 清单：`outputs/stage_d/d3_robust_diagnostics_v1/sha256_manifest.json`

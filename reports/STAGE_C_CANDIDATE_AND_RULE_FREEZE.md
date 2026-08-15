# 阶段 C 推荐 v2 候选与评估规则冻结记录

冻结日期：2026-07-24  
冻结编号：`stage_c_recommended_v2_c3_20260724`  
状态：`LOCKED_BEFORE_SCREENING`

## 1. 冻结结论

推荐候选固定为：

```text
fixed_control_ensemble_v2
= 0.5 × temporal_only_control
+ 0.5 × fixed_temporal_graph_control
```

固定使用 30 只股票、价格特征、8 周序列和种子 `20260723/20260724/20260725`。集成层没有可训练参数。股票横截面节点设计继续暂缓。

冻结后不得更改结构、组件、权重、特征及顺序、序列长度、训练种子、检查点、损失函数、优化设置、指标定义或判定门槛。任何变更都必须形成新的候选版本，且不得使用本次未来 SCREENING 数据进行开发。

## 2. 冻结基线

独立 SCREENING 必须在完全相同的样本行上一次性运行以下基线：

| 基线 | 固定作用 |
|---|---|
| Naive | 零收益参考 |
| FreTS return-only L4 | 频域基线 |
| Minimalist Transformer price-only L8 | 数值时序 Transformer 基线 |
| Temporal-only control | 推荐模型的稳定时域组件对照 |
| Fixed temporal graph control | 推荐模型的固定图组件对照 |
| Graph-frequency v1 | 已被替代的阶段 C 历史参考 |

所有可训练模型均固定三个种子，不允许根据 SCREENING 结果挑选单个种子。

## 3. 指标优先级

1. 平均 MAE：主要晋级指标。
2. 平均 RMSE：大误差保护指标。
3. Direction Accuracy 与 Direction F1：方向性保护指标。
4. 三种子 MAE 变异系数：稳定性保护指标。
5. 高绝对收益组误差和最差五只股票误差占比：风险集中度诊断。

开发期 validation 数值只作为选择暴露的参考，不作为未来 SCREENING 的绝对目标值。

## 4. SCREENING 通过门槛

以下条件必须全部满足才记为 `PASS`：

- 候选平均 MAE 不高于 Naive、FreTS L4、价格版 Minimalist Transformer 三个核心基线中的最佳值。
- 候选平均 RMSE 不超过三个核心基线最佳值的 1.05 倍。
- Direction Accuracy 不低于 0.50，Direction F1 不低于 0.15。
- 三种子 MAE 变异系数不超过 0.10。
- 高绝对收益组 MAE / 总体 MAE 不超过 2.25。
- 最差五只股票贡献的绝对误差占比不超过 45%。
- 覆盖固定 30 只股票，每只至少 4 个合格样本，且候选与全部基线使用完全一致的评估行。

## 5. 失败与不确定条件

任一完整性问题均直接记为 `INVALID_INTEGRITY_FAILURE`，包括文件缺失或哈希变化、模型不能加载、产生非有限预测、样本行不一致、冻结项变化、未授权读取未来数据、读取 SCREENING 后重新训练/调权/选种子/改门槛，或未使用冻结的统一评估器。

任一性能条件触发即记为 `FAIL`：

- 候选 MAE 差于 Naive。
- 候选 RMSE 超过最佳核心基线的 1.10 倍。
- Direction Accuracy 低于 0.45，或 Direction F1 低于 0.10。
- 三种子 MAE 变异系数高于 0.15。
- 高绝对收益组 MAE / 总体 MAE 高于 2.50。
- 最差五只股票误差占比高于 50%。

如果没有触发失败条件，但未满足全部通过条件，则记为 `INCONCLUSIVE`。该结果不能晋级，也不能在同一份 SCREENING 数据上继续调参。

## 6. SHA-256 冻结包

冻结清单覆盖候选配置与代码、统一评估器、开发期数据契约和 train/validation 文件、三个候选种子的组件检查点与结果，以及所有冻结基线的配置、检查点和结果。

生成或重建清单：

```powershell
python -m stage_c.freeze_stage_c_candidate
```

只读复核：

```powershell
python -m stage_c.freeze_stage_c_candidate --verify
```

冻结文件位于 `stage_c/frozen/recommended_v2_c3/`。在获得用户明确授权前，不获取、不读取、不推理未来 SCREENING 数据。

# 阶段 E：E-5.4 十模型统一诊断报告

日期：2026-07-25  
状态：`PASS / READ-ONLY CROSS-MODEL TRAIN-VALIDATION DIAGNOSTICS`  
限制：不重新训练、不写checkpoint、不排名、不删模、不形成候选、不访问SCREENING。

## 1. 冻结规则与数据来源

统一诊断配置：`stage_e/configs/e5_unified_diagnostics_v1.json`  
配置SHA-256：`57c703206d29cab7eb66d52867e9dca247354e766690c4c93e34217ef2c9b3db`

配置在同时读取E-5.2与E-5.3结果前冻结，锁定：

- E-5.2六个低成本基线与E-5.3四个神经/图基线，共10个模型；
- 三折 `E_RO_01/E_RO_02/E_RO_03`；
- 三种子 `20260723/20260724/20260725`；
- 45个模型两两分歧组合；
- 模型分歧使用三种子预测算术均值；
- 最差折定义为“三种子在折内池化后MAE最大”；
- 收益分组固定为模型×种子内按冻结目标收益秩划分10组；
- 成本字段只读取既有工程回执，不补测或推断缺失推理时间。

输入批次分别为E-5.2批次 `c2344853...` 和E-5.3批次 `44cc5262...`，其元数据、预测、回执与机器验收SHA均在诊断配置中固定。

## 2. 机器验收

- 合并预测：45,000行；
- 合并工程回执：90行；
- 模型×种子契约格：30组，每组1500个冻结验证键；
- 缺失键0、额外键0；
- 工程成本：10行；
- 折池化指标：30行；最差折：10行；
- 逐股票诊断：1000行；
- 收益十分位诊断：100行；
- 种子两两稳定性：30行；
- 模型两两分歧：45行；
- 所有数值指标有限，独立复算及SHA一致；
- 测试 `51/51 PASS`；机器验收 `23/23 PASS`；
- `training_executed=false`，`checkpoint_written=false`；
- 未排名、删模或晋级候选，未读取未来/封存数据，未访问SCREENING。

## 3. 总体误差与最差折

十模型三种子池化MAE约为 `0.03162~0.03659`，RMSE约为 `0.04599~0.05202`。这些值仅用于描述，不在本节点形成排名。

所有10个模型的最差折均为 `E_RO_03`，说明后段验证区间存在跨模型一致的难度上升，而非某一架构的孤立问题。E_RO_03最差折MAE约为 `0.03496~0.04155`。

MAPE受接近零的周收益分母影响，模型间可达到约 `0.97~504.06` 的极大跨度；因此当前MAPE只保留为完整性指标，不应单独用于后续候选门槛。

## 4. 工程成本

九次“折×种子”合计训练时间记录：

- Naive与行业VAR接近零；
- 股票节点固定行业图约32.35秒；
- RF约49.92秒；
- LSTM约72.42秒；
- Minimalist Transformer约129.33秒；
- FreTS约148.54秒；
- SVM约195.09秒；
- TCN约219.57秒；
- 确定性Time-GNN约257.96秒。

参数量从Naive的0、行业VAR的182，到FreTS的197,761。第一批模型没有冻结统一推理计时，因此推理时间只对E-5.3四模型有机器记录，报告不对缺失值做估算。

## 5. 逐股票与分组误差

相对Naive逐股票MAE更低的股票数仅作覆盖诊断：

- 固定行业股票节点图：78只；
- FreTS：74只；
- Minimalist Transformer：70只；
- TCN：68只；
- LSTM：65只；
- 确定性Time-GNN：57只；
- RF：40只；
- 行业VAR：30只；
- SVM：28只。

所有模型的最高行业MAE均出现在信息技术行业，说明该行业是共同风险分组。市值分组中，除SVM的最高MAE位于小市值组外，其余模型最高MAE均位于中市值组。

收益尾部方面，8个模型的最高十分位误差位于正收益尾部 `D10`；行业VAR与RF的最高误差位于负收益尾部 `D1`。这表明后续失败条件不能只看总体平均误差，必须同时约束双侧收益尾部。

## 6. 种子稳定性

种子MAE变异系数整体较低，约为 `0~0.00466`，但逐样本预测标准差揭示了不同结构的稳定性差异：

- RF平均预测种子标准差约 `0.000605`；
- FreTS和股票节点固定图约 `0.00142`；
- LSTM约 `0.00215`；
- Minimalist Transformer约 `0.00260`；
- 确定性Time-GNN约 `0.00318`；
- TCN约 `0.00384`；
- Naive、行业VAR和当前SVM实现为0。

因此“种子平均MAE稳定”不能替代逐样本预测稳定性门槛。

## 7. 模型与组件分歧

45组模型对全部生成。较高的一组预测相关包括：

- LSTM与Minimalist Transformer：Pearson约 `0.7916`；
- LSTM与股票节点固定行业图：约 `0.7912`；
- LSTM与确定性Time-GNN：约 `0.7515`。

行业VAR与RF/SVM接近零相关，说明统计行业聚合与非线性机器学习预测提供了明显不同的预测方向。Naive为常数预测，与其他模型的相关系数按冻结规则记为0，不能解释为普通线性负相关。

模型分歧结果是结构诊断，不代表应组合高分歧模型，也没有自动生成集成候选。

## 8. 结论与下一节点

E-5.4统一诊断已经完成。现有证据表明：

1. E_RO_03是共同最差折；
2. 信息技术、中市值和双侧收益尾部需要进入正式失败条件；
3. 聚合MAE稳定不足以证明逐样本预测稳定；
4. 真实股票节点图在多数股票上相对Naive具有更低MAE，但仍需经过预冻结门槛评审，不能在本节点晋级；
5. 工程成本跨度较大，后续门槛需明确成本上限及缺失推理计时的处理方式。

下一执行节点为E-6.1候选门槛冻结：在读取任何“按指标排序的候选建议”前，冻结相对Naive与FreTS的最低改善、最差折、逐股票覆盖、信息技术行业、中市值、D1/D10尾部、种子预测相关/方差、工程成本上限、失败条件和三种子推理聚合方式。门槛冻结前不得形成唯一候选或申请SCREENING。

## 9. 核心产物

- 统一预测：`outputs/stage_e/e5_unified_diagnostics_v1/unified_predictions_ten_models.csv.gz`；
- 工程成本：`outputs/stage_e/e5_unified_diagnostics_v1/engineering_cost_summary.csv`；
- 最差折：`outputs/stage_e/e5_unified_diagnostics_v1/worst_fold_summary.csv`；
- 逐股票：`outputs/stage_e/e5_unified_diagnostics_v1/diagnostics_per_stock.csv`；
- 行业/市值/收益：对应 `diagnostics_industry.csv`、`diagnostics_market_cap.csv`、`diagnostics_return_decile.csv`；
- 种子离散度：`outputs/stage_e/e5_unified_diagnostics_v1/seed_prediction_dispersion.csv`；
- 模型分歧：`outputs/stage_e/e5_unified_diagnostics_v1/model_disagreement.csv`；
- 组件分歧：`outputs/stage_e/e5_unified_diagnostics_v1/component_disagreement.csv`；
- 批次SHA-256：`fea38e3c09ede4ab0c462bed89f5db87e8e0a1ffb04246b9f0d74acd792b3240`；
- 机器验收：`outputs/stage_e/e5_unified_diagnostics_acceptance_v1.json`；
- 验收文件SHA-256：`fbec826196a2b3cc8a78d1079fa2caf5fe5bf37f01b767462cda6a7fef4c6991`。

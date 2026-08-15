# 阶段 E：E-5.2 第一批低成本基线单种子工程回执

日期：2026-07-25  
状态：`PASS / TRAIN-VALIDATION SINGLE-SEED ENGINEERING EXPOSURE`  
用途限制：本回执仅证明冻结基线可训练、可保存、可独立加载和可统一评价；不得据此排名、删模、形成候选或申请 SCREENING。

## 1. 运行前冻结

冻结配置：`stage_e/configs/e5_low_cost_baselines_v1.json`  
配置 SHA-256：`9468df9d4ee492f1ace7c77f4a9313ccdca930d38ffc8723869f90fd0053ec4c`

冻结模型集合恰好为六个：

1. `naive`；
2. `frets_return_l4__fixed_shrink_a075`；
3. `minimalist_price_only_l8`；
4. `random_forest_price_l12`；
5. `svm_rbf_price_l12`；
6. `industry_var1_ridge`。

冻结特征视图分别为零收益预测、4周收益序列、8周价格数值序列、12周展平价格数值序列和行业均值周收益。三折固定为 `E_RO_01/E_RO_02/E_RO_03`，工程种子固定为 `20260725`，开发截止日保持 `2023-06-02`。

训练参数、早停规则、损失、优化器、RF/SVM参数以及行业VAR(1)+ridge系数均已写入冻结配置。FreTS继续使用 Stage D 冻结源码，未新增网格搜索或自由候选。

## 2. 失败条件与独立加载

以下任一条件均登记为工程失败：模型或特征视图偏离冻结配置、冻结键缺失或新增、非有限损失/预测、独立加载预测差超过 `1e-7`、checkpoint或配置SHA缺失、依赖或FreTS冻结源码缺失、失败后静默删模、读取未来/封存数据或访问SCREENING。

独立加载入口集中在 `stage_e/e5/low_cost.py`：

- PyTorch/FreTS/Minimalist Transformer：`load_predict_torch`；
- RF/SVM：`load_predict_sklearn`；
- 行业VAR：`load_predict_industry_var`。

统一运行入口为 `stage_e/run_e5_low_cost_single_seed.py`，机器验收入口为 `stage_e/accept_e5_low_cost_single_seed.py`。

## 3. 工程运行结果

- 三折 × 六模型，共18次运行，`18/18 PASS`；
- 失败回执：0；
- 每折冻结验证键：500，总计1500；
- 每个模型均输出1500行统一预测；
- 缺失键：0；额外键：0；目标最大差：0；
- 所有checkpoint独立加载最大预测差：0；
- 所有指标有限，落盘预测的独立复算CSV哈希一致；
- 未删模、未选择候选、未读取未来或封存数据、未访问SCREENING。

三折合计训练时间仅作为工程成本记录：FreTS约58.29秒、Minimalist Transformer约36.50秒、RF约15.39秒、SVM约57.65秒；Naive和封闭式行业VAR训练成本接近零。该成本不得用于本节点的模型淘汰。

## 4. 机器产物

- 批次元数据：`outputs/stage_e/e5_low_cost_baselines_single_seed_v1/metadata.json`；
- 统一预测：`outputs/stage_e/e5_low_cost_baselines_single_seed_v1/unified_predictions.csv.gz`；
- 工程回执：`outputs/stage_e/e5_low_cost_baselines_single_seed_v1/engineering_receipts.csv`；
- 机器验收：`outputs/stage_e/e5_low_cost_single_seed_acceptance_v1.json`；
- 批次 SHA-256：`7d3f1b07f07b316888bbf6fb7de44e43de1b778945dff02af419b6bd0fa074ba`；
- 验收文件 SHA-256：`b14b193c7dd3986281dccd090631c85bf410879f7d1ab16b3f4d41d2a4bc7709`。

## 5. 验收结论与下一节点

E-5.2第一批低成本基线的“运行前冻结 + 单种子工程回执”已经完成并通过机器验收。当前没有形成任何性能排序或候选建议。

下一节点保持为E-5.2三种子正式复核：不得改变本批模型集合、特征视图和训练协议；补运行 `20260723/20260724`，与现有 `20260725` 形成冻结三种子统一汇总。完成前不进入候选选择，也不读取E-SCREENING。

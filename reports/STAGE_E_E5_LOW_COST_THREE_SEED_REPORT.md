# 阶段 E：E-5.2 第一批低成本基线三种子统一复核

日期：2026-07-25  
状态：`PASS / THREE-SEED TRAIN-VALIDATION STABILITY REVIEW`  
暴露限制：不排名、不删模、不形成候选、不访问SCREENING。

## 1. 协议保持情况

本次没有修改E-5.2冻结的六模型集合、特征视图、三折定义、训练参数、失败条件或独立加载入口。基础协议配置SHA-256保持：

`9468df9d4ee492f1ace7c77f4a9313ccdca930d38ffc8723869f90fd0053ec4c`

三种子复核配置为 `stage_e/configs/e5_low_cost_three_seed_review_v1.json`，SHA-256为：

`728171962ad91873e78cff1759af201ea106cb9e886257de9c986e63a125d02a`

新增训练仅包含 `20260723/20260724`；已验收的工程种子 `20260725` 按原批次SHA复用，没有重新训练。

## 2. 工程与数据契约验收

- 三折 × 六模型 × 三种子，共54份回执；
- 新增训练36次，`36/36 PASS`；三种子合计 `54/54 PASS`；
- 失败回执0；
- 每个“模型×种子”覆盖1500个冻结验证键；
- 缺失键0、额外键0、目标差0；
- checkpoint独立加载最大预测差0；
- 三种子两两稳定性表共18行；
- 所有评价指标有限；
- 统一预测、指标、回执和批次SHA独立复算一致；
- 未删模、未选择或晋级候选，未读取未来/封存数据，未访问SCREENING。

单元测试更新后为 `42/42 PASS`，机器验收为 `19/19 PASS`。

## 3. 稳定性描述

以下结果仅作工程诊断，不构成性能排名或候选选择：

- Naive、行业VAR和当前固定实现的SVM在三种子间预测完全一致；
- RF三组Pearson相关约为 `0.9494~0.9545`；
- Minimalist Transformer三组Pearson相关约为 `0.6142~0.8052`；
- FreTS三组Pearson相关约为 `0.5349~0.7769`；
- 各模型种子MAE变异系数均较低，但FreTS与Minimalist Transformer仍表现出“聚合误差稳定、逐样本预测不完全稳定”的特征。

本节点没有预登记候选门槛，因此不得把上述诊断转化为淘汰或晋级结论。

## 4. 核心产物

- 三种子配置：`stage_e/configs/e5_low_cost_three_seed_review_v1.json`；
- 执行入口：`stage_e/run_e5_low_cost_three_seed_review.py`；
- 验收入口：`stage_e/accept_e5_low_cost_three_seed_review.py`；
- 统一预测：`outputs/stage_e/e5_low_cost_baselines_three_seed_v1/unified_predictions_three_seed.csv.gz`；
- 三种子回执：`outputs/stage_e/e5_low_cost_baselines_three_seed_v1/engineering_receipts_three_seed.csv`；
- 稳定性表：`outputs/stage_e/e5_low_cost_baselines_three_seed_v1/pairwise_seed_stability.csv`；
- 批次SHA-256：`c234485365354333bd43a6396a83552644e40c92b8993d6f17463b1a2cfaf5a5`；
- 机器验收：`outputs/stage_e/e5_low_cost_three_seed_acceptance_v1.json`；
- 验收文件SHA-256：`42911b554b2f020e2e9c5cc5d91afda5f4d8dca998d27fd70329ebcc2d3aa1d4`。

## 5. 下一节点

E-5.2第一批低成本基线三种子复核已经完成。下一节点更新为E-5.3第二批神经网络与真实股票节点图基线的运行前冻结：先登记LSTM、TCN、Time-GNN稳定分支以及一个固定实现的MTGNN/Graph WaveNet类基线，再进行单种子工程回执。E-5.3冻结前不得依据本批结果修改第一批模型或新增自由候选；E-5.4统一诊断和E-6候选门槛尚未启动。

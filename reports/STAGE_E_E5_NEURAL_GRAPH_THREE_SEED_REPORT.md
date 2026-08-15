# 阶段 E：E-5.3 神经网络与真实股票节点图三种子统一复核

日期：2026-07-25  
状态：`PASS / THREE-SEED TRAIN-VALIDATION STABILITY REVIEW`  
限制：本节点不排名、不删模、不形成候选、不访问SCREENING。

## 1. 协议保持

基础冻结配置及SHA保持不变：

- 配置：`stage_e/configs/e5_neural_graph_baselines_v1.json`；
- SHA-256：`6ee08295cd508f97218390b2e2564ae638bb5b58ecbd86047e0565a0393a6a43`。

三种子复核配置：`stage_e/configs/e5_neural_graph_three_seed_review_v1.json`，SHA-256：

`1c499328cacec1828e90e8a043115ee087fa0d729096db7e00adf6bcea573294`

新增训练仅包含 `20260723/20260724`；原验收种子 `20260725` 按原批次SHA复用。模型集合、特征视图、训练参数、Time-GNN确定性Top-k、股票节点顺序和固定行业邻接均未修改。

## 2. 工程验收

- 三折 × 四模型 × 三种子，共36份回执；
- 新增24次运行全部通过；三种子合计 `36/36 PASS`；
- 失败回执0；
- 每个“模型×种子”覆盖1500个冻结验证键；
- 缺失键0、额外键0、目标差0；
- checkpoint独立加载差均低于 `1e-7`；
- 9份Time-GNN回执均确认确定性稀疏化；
- 9份真实股票图登记均为100个股票节点、13个行业和100个自环；
- 九个“种子×折”的股票顺序SHA和邻接SHA完全一致；
- 指标有限，统一预测和指标SHA独立复算一致；
- 机器验收 `22/22 PASS`；
- 未选择、删除或晋级候选，未读取未来/封存数据，未访问SCREENING。

## 3. 稳定性描述

以下只作工程诊断，不构成排名：

- LSTM三组Pearson相关约 `0.5734~0.9171`；
- TCN约 `0.5249~0.6259`；
- 确定性Time-GNN约 `0.5539~0.6518`；
- 固定行业股票节点图约 `0.8443~0.8558`；
- 四模型MAE种子变异系数约 `0.0017~0.0039`。

结果说明四模型聚合误差较稳定，但逐样本预测稳定性仍有差异。固定行业股票节点图在本批逐样本相关上更一致，但E-5.3没有预登记候选门槛，不能据此晋级或淘汰其他模型。

## 4. 产物

- 运行入口：`stage_e/run_e5_neural_graph_three_seed_review.py`；
- 验收入口：`stage_e/accept_e5_neural_graph_three_seed_review.py`；
- 输出目录：`outputs/stage_e/e5_neural_graph_baselines_three_seed_v1`；
- 三种子预测：`outputs/stage_e/e5_neural_graph_baselines_three_seed_v1/unified_predictions_three_seed.csv.gz`；
- 股票图登记：`outputs/stage_e/e5_neural_graph_baselines_three_seed_v1/real_stock_graph_registry_three_seed.json`；
- 稳定性表：`outputs/stage_e/e5_neural_graph_baselines_three_seed_v1/pairwise_seed_stability.csv`；
- 批次SHA-256：`44cc5262afad388eb130b49d5eb305adefa995fc9012e4dc70f9c1159211a60c`；
- 机器验收：`outputs/stage_e/e5_neural_graph_three_seed_acceptance_v1.json`；
- 验收文件SHA-256：`2d60f322662af4d287791e82a6a409808864d42dc3c66765466dec0ff0345c08`。

## 5. 下一节点

E-5.3已经完成。下一执行节点更新为E-5.4统一诊断：合并E-5.2六个低成本基线和E-5.3四个神经/股票图基线的三种子预测，在不新增模型和不改变任何训练结果的条件下，统一计算工程成本、最差折、逐股票、行业、市值、收益分组、种子稳定性和模型间预测分歧。E-5.4开始前应先冻结统一诊断规则和失败条件；候选选择及SCREENING仍未授权。

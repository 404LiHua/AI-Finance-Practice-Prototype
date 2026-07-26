# 阶段 E：E-5.3 神经网络与真实股票节点图基线单种子工程回执

日期：2026-07-25  
状态：`PASS / TRAIN-VALIDATION SINGLE-SEED ENGINEERING EXPOSURE`  
用途限制：不排名、不删模、不形成候选、不访问SCREENING。

## 1. 运行前冻结

冻结配置：`stage_e/configs/e5_neural_graph_baselines_v1.json`  
配置 SHA-256：`6ee08295cd508f97218390b2e2564ae638bb5b58ecbd86047e0565a0393a6a43`

冻结模型集合恰好为四个：

1. `lstm_price_l8`；
2. `tcn_price_l8`；
3. `timegnn_deterministic_topk_l8`；
4. `stock_node_gwnet_fixed_industry_l8`。

四个模型均读取相同E-5数值视图和冻结样本键。LSTM、TCN和Time-GNN按单股票8周窗口训练；Graph-WaveNet类基线以折内固定顺序的100个 `stock_code` 作为图节点，使用冻结行业映射构造固定邻接。

训练种子固定为 `20260725`，三折固定为 `E_RO_01/E_RO_02/E_RO_03`，开发截止日保持 `2023-06-02`。模型结构、隐藏维度、卷积扩张率、Top-k、温度、图传播阶数、批量大小、学习率、早停和失败条件均在首次训练前冻结。

## 2. Time-GNN稳定分支与许可

Time-GNN上游仓库采用MIT许可。项目登记了上游ZIP、模型文件和许可证SHA，并保存适配说明 `stage_e/e5/TIMEGNN_ATTRIBUTION.md`。

稳定分支固定移除硬Gumbel采样，改为温度缩放的确定性Top-k稀疏化；图传播使用纯PyTorch稠密计算，不依赖未安装的PyG。该分支的节点仍是单股票窗口内时间位置，不能与真实股票节点图混淆。

## 3. 真实股票节点图

`stock_node_gwnet_fixed_industry_l8` 使用真实股票代码作为100个节点：

- 三折股票顺序SHA一致：`c815049174ae151ca9f9acad87375f1c69db4df9802b954ebb324da2d50ab601`；
- 三折邻接SHA一致：`24b646482db8a3db2fa4cde47c6f0a61bf2c83d6ade1f1836db7043c50672f6d`；
- 行业数量：13；
- 自环数量：100；
- 邻接规则：同行业连接、保留自环、逐行归一化；
- 模型结构：节点时序空洞卷积 + 固定图一/二阶传播 + 节点级收益预测。

## 4. 单种子工程结果

- 三折 × 四模型，共12次运行，`12/12 PASS`；
- 失败回执0；
- 每个模型覆盖1500个冻结验证键；缺失键0、额外键0；
- checkpoint独立加载最大预测差为 `3.13e-9`，低于 `1e-7`门槛；
- Time-GNN三折均确认使用确定性图稀疏化；
- 股票图三折节点顺序和邻接完全一致；
- 指标有限，落盘预测的指标与SHA独立复算一致；
- 测试 `46/46 PASS`；机器验收 `18/18 PASS`；
- 未排名、删模或晋级候选，未读取未来/封存数据，未访问SCREENING。

三折训练时间仅作工程成本记录：LSTM约17.95秒、TCN约65.59秒、稳定Time-GNN约71.08秒、股票节点Graph-WaveNet类基线约12.01秒。该成本不用于本节点淘汰模型。

## 5. 产物与SHA

- 实现：`stage_e/e5/neural_graph.py`；
- 运行入口：`stage_e/run_e5_neural_graph_single_seed.py`；
- 验收入口：`stage_e/accept_e5_neural_graph_single_seed.py`；
- 输出目录：`outputs/stage_e/e5_neural_graph_baselines_single_seed_v1`；
- 股票图登记：`outputs/stage_e/e5_neural_graph_baselines_single_seed_v1/real_stock_graph_registry.json`；
- 批次SHA-256：`ac3b5e885d0f23ca8540edac8c317a64fb2aa782e3adcf196bb35938370844e2`；
- 机器验收：`outputs/stage_e/e5_neural_graph_single_seed_acceptance_v1.json`；
- 验收文件SHA-256：`b2651c29b3351b710f0436ce9e25261024ca4f7db543b37b1fe4675f92394a1b`。

## 6. 下一节点

E-5.3运行前冻结和单种子工程回执已经完成。下一节点为保持本配置完全不变，补运行种子 `20260723/20260724`，形成四模型三折三种子统一复核。三种子复核前不得修改稳定Time-GNN稀疏化、股票节点顺序或固定行业邻接，也不得根据当前单种子指标删模或晋级。

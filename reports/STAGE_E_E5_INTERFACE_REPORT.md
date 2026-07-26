# Stage E E-5.1 统一接口验收报告

报告日期：2026-07-25  
接口版本：`e5_unified_interface_v1`  
状态：`PASS / NO NEW BASELINE TRAINING`

## 1. 已完成内容

- 冻结E-5预测行契约；
- 建立 `E5FoldView` 统一折数据视图；
- 建立 `E5ModelAdapter` 模型适配抽象接口；
- 建立统一预测文件校验器；
- 建立回归、方向、三种子、最差折和分组评价器；
- 建立逐股票、行业、市值、收益十分位和文本可用性诊断；
- 建立配置、checkpoint、预测、环境和诊断SHA-256追溯；
- 使用E-4S.2既有预测完成无训练机器验收。

## 2. 冻结预测契约

每行必须包含：

`model_id / seed / fold_id / sample_row_id / trade_date / target_date / stock_code / target_return / prediction / sample_valid / text_available / checkpoint_sha256 / config_sha256`

任何模型不得遗漏冻结验证键、增加额外键、修改目标、删除文本缺失样本或改变三折三种子网格。

## 3. 验收夹具

使用现有两个控制组作为接口夹具，不构成E-5候选：

- `temporal_no_graph_no_text`；
- `time_industry_no_text`。

每个模型、每个种子均精确覆盖1500个冻结验证键：缺失0、额外0、最大目标差约 `9.76e-17`。三折三种子共9000行预测通过契约。

三折的无文本、TF-IDF/SVD和BGE共9个文本视图均与数值窗口共享完全相同的 `sample_row_id`。

## 4. 统一评价器

已统一输出：

- MSE、MAE、RMSE、MAPE；
- Accuracy、Precision、Recall、F1、MCC；
- 三种子均值、标准差、MAE变异系数；
- 两两Pearson、Spearman和预测方向一致率；
- 最差折MAE/RMSE；
- 逐股票、行业、市值、收益十分位、文本可用性诊断。

MAPE在收益接近零时会显著放大，本项目保留该指标满足申报要求，但候选选择不得单独依赖MAPE。

## 5. 独立复算

评价器从落盘统一预测文件再次独立复算全部指标，所有CSV哈希与首次计算完全一致。机器验收12项检查全部通过。

- 接口配置 SHA-256：`f8ba7f21d8820d784eba36985f764130baf6074de52d67ac85ebe4086e8555d4`；
- 预测契约 SHA-256：`61c6865426f3ed3115a09f12f153c2f1802b192554d158101fd47c27c88849bf`；
- 接口批次 SHA-256：`7013461b447d04fb7469fc4f1802a4c7aadb6b5c03d8defadc9b9dcc91d81f55`；
- 元数据 SHA-256：`f3bf8d97a9eccc88478fa6b4f0cfd39202efa449c8e00ba31829d60fc48caa22`。

## 6. 数据隔离

本节点没有训练新基线，没有读取C-4、D-5或未来Stage E SCREENING/FINAL。E-4负结果保持不变。

## 7. 下一节点

下一节点为E-5.2第一批低成本基线。运行前需冻结模型集合、每个模型的特征视图、训练协议、三种子策略、独立加载入口和失败条件。不得根据单种子结果删除模型。

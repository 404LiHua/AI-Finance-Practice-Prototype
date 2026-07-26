# Stage E E-5统一接口详细设计草案

草案版本：`stage_e_e5_unified_interface_v1_draft`  
制定日期：2026-07-25  
状态：`IMPLEMENTED AND ACCEPTED / NO NEW BASELINE TRAINING STARTED`

## 1. 目标

所有E-5模型必须读取相同冻结样本键、rolling-origin折、数值窗口、文本视图和掩码，并通过同一评价器输出可复算预测、指标、分组诊断、工程成本和SHA-256清单。

## 2. 数据输入契约

统一输入批次至少包含：

- `sample_row_id`；
- `fold_id`、`split`、`trade_date`、`target_date`；
- `stock_code`及冻结股票顺序；
- 数值窗口 `[batch,time,stock,feature]`；
- `target_raw`、`target_scaled`、TRAIN均值与标准差；
- `sample_mask`、`node_available`；
- 无文本、TF-IDF/SVD、BGE三种视图；
- `text_available`、`text_count`；
- 固定行业图、滚动相关图和单位图引用。

任何模型不得因文本缺失、停牌或目标缺失改变冻结样本行集合；只允许通过掩码排除无效损失和指标行。

## 3. 模型适配器接口

每个适配器必须实现等价能力：

```text
build(config, data_contract) -> model
fit(train_view, validation_view, seed, artifact_dir) -> training_receipt
predict(model_or_checkpoint, evaluation_view) -> frozen_key_predictions
load(checkpoint, config) -> independent_model
describe() -> model_family, parameter_count, license, source_sha256
```

`predict`输出至少包含：

- model_id、seed、fold_id；
- sample_row_id、trade_date、stock_code；
- target_return、prediction、sample_valid；
- text_available；
- checkpoint_sha256、config_sha256。

## 4. 统一训练管理

- 种子由统一种子管理器设置Python、NumPy和PyTorch；
- 配置必须在首次运行前生成SHA-256；
- TRAIN/VALIDATION分离，禁止用VALIDATION正文重新拟合文本变换；
- checkpoint选择规则必须预登记；
- 日志记录epoch、损失、学习率、梯度范数、运行时间和峰值内存；
- 所有模型必须提供独立加载推理入口；
- 单种子工程回执通过后才允许三种子批量运行。

## 5. 统一评价器

回归指标：MSE、MAE、RMSE、MAPE。  
方向指标：Accuracy、Precision、Recall、F1、MCC。  
稳健指标：三种子均值/标准差、MAE变异系数、两两预测相关、最差折。  
分组指标：逐股票、行业、市值、收益十分位、文本可用/缺失。  
工程指标：参数量、训练时间、推理时间、峰值内存、checkpoint大小。

所有指标从落盘预测文件独立复算，不允许直接信任训练脚本内存中的汇总值。

## 6. 首批模型登记

第一批低成本基线：Naive、Stage D冻结FreTS L4、价格版Minimalist Transformer、RF、SVM、行业或因子VAR。  
第二批模型：LSTM、TCN、Time-GNN稳定分支、一个固定实现的MTGNN或Graph WaveNet类模型。

外部实现必须登记仓库版本、许可证、源码SHA-256、依赖版本和本项目适配改动。不得根据VALIDATION结果删除表现不佳的已登记基线。

## 7. 产物目录契约

```text
outputs/stage_e/e5/<model_id>/<run_id>/
  config.json
  environment.json
  training_log.csv
  checkpoint/
  predictions.csv.gz
  fold_metrics.csv
  seed_metrics.csv
  diagnostics_per_stock.csv
  diagnostics_industry.csv
  diagnostics_market_cap.csv
  diagnostics_return_decile.csv
  engineering_cost.json
  metadata.json
```

`metadata.json`记录所有输入、配置、checkpoint、预测和诊断文件的SHA-256，以及未来数据读取标记。

## 8. 验收门槛

- 所有模型预测行集合与冻结评价样本键完全一致；
- 独立加载预测与原预测在预登记容差内一致；
- 指标可从预测文件独立复算；
- 三折和冻结种子齐全；
- 未来路径守卫通过；
- 日志、环境、配置、checkpoint和预测哈希完整；
- 模型失败必须保留失败回执，不得静默删除。

## 9. 与E-4S的关系

E-5统一接口建设可以与E-4S并行，但E-4开发最优结构只有通过E-4S三级门槛后才可进入E-5候选比较。E-5模型运行不授权未来SCREENING，也不修改E-4S训练协议。

下一步是在不训练模型的情况下，将本草案转换为JSON schema、基础适配器抽象类、统一预测文件校验器和评价器单元测试。

## 10. 实现回执

上述草案已经转换为：

- `stage_e/schemas/e5_prediction_contract_v1.json`；
- `stage_e/e5/interface.py`；
- `stage_e/e5/evaluation.py`；
- `stage_e/run_e5_interface_acceptance.py`；
- `stage_e/accept_e5_interface.py`。

机器验收 `outputs/stage_e/e5_interface_acceptance_v1.json` 为 `PASS`。下一步进入E-5.2，不再修改预测必填字段和冻结样本键规则。

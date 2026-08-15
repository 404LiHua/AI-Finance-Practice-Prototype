# 生产 T2 对齐资料交付单（V1）

用途：恢复或重建 `RG_OBGNET_CONFIRMED_SAFE_V1_1` 的四周市场相对 T2 目标，使后续候选能够在同一目标、同一宇宙和同一时间边界上比较。

当前状态：`BLOCKED_UNTIL_EXACT_PRODUCTION_ARTIFACTS_OR_EQUIVALENT_PIT_PACKAGE`。

## 首选获取方式：原始归档恢复

请优先从原项目归档、备份磁盘、Git LFS/对象存储或交付方只读备份中恢复下列原始制品。恢复时不得重算、压缩替换或改写原文件：

- `train.npz` 本体；
- `rev8_ro01_train_target.csv.gz`；
- `rg3_features.csv.gz`；
- `STOCK_REGISTRY.csv`；
- `ORIGIN_REGISTRY.csv`；
- 生产目标构造脚本、配置和版本锁；
- 训练、开发、FRESH、模拟盘的起点清单与过滤规则。

每个文件必须提供原始路径、复制路径、文件大小、SHA-256、导出/恢复时间、来源和许可证或使用授权。

## 备选获取方式：等价 PIT 重建包

只有原始归档确实无法恢复时，才允许提交等价重建包。数据管理员必须提供：

1. 全部生产候选起点的周度交易日历和时区规则；
2. 每个起点的完整股票宇宙、上市状态、停牌、涨跌停、ST/退市过滤及缺失处理；
3. 覆盖训练至最后独立评估起点之后的日线 OHLCV、复权因子和市场基准来源；
4. 四周目标构造实现：起止点、未复权/复权口径、`raw_return - target_week_tradable_cross_sectional_median` 的精确定义；
5. 固定 `-0.01 / +0.01` 三分类阈值及其冻结证明；
6. 与目标逐键对应的特征面板、标签面板、训练/开发/FRESH/模拟盘切分和 purge/embargo 规则；
7. 可在独立环境重放的构建脚本、配置、样例键和逐阶段收据。

## 交付目录和清单

建议目录：

```text
production_t2_alignment_delivery_YYYYMMDD/
  raw_or_reconstructed/
  manifests/FILE_SHA256.csv
  manifests/ORIGIN_REGISTRY.csv
  manifests/STOCK_REGISTRY.csv
  governance/T2_TARGET_CONSTRUCTION_PROTOCOL.json
  governance/DELIVERY_ATTESTATION.json
  scripts/build_t2_alignment_package_v1.py
  receipts/MATERIALIZATION_RECEIPT.json
```

`DELIVERY_ATTESTATION.json` 至少应包含：`status`、`source_type`（archive_restore 或 equivalent_reconstruction）、`exported_at`、`timezone`、`file_sha256`、`origin_registry_sha256`、`stock_registry_sha256`、`target_protocol_id`、`point_in_time_verified`、`future_rows_count`、`universe_rules_verified`、`labels_sealed`。

## 验收门槛

- 所有文件 SHA-256 可复核，路径和版本不可变；
- 目标协议明确为四周市场相对 T2，而非四交易日绝对 H4；
- 训练、开发、FRESH 和模拟盘键域及过滤规则可逐行审计；
- 在预测封存前，标签值、行数、收益汇总和指标对研究者保持关闭；
- 任何缺失、语义不确定或宇宙不一致都必须 `FAIL_CLOSED`，不得用当前 300 股票临时数据补齐。

资料通过上述验收前，保持生产内核不变，不签发一次性评分授权，不启动 V4 晋级。

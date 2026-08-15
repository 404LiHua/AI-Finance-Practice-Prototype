# Stage G：AA-GFMNET T2 研究与单机交付

Stage G 是当前公开快照，面向 T2 市场相对预测的研究复核和单机运行。`project_snapshot_20260814` 保存稳定基线，`project_snapshot_20260815` 保存最新增量交接与治理修复。

## 目录

- `project_snapshot_20260814/aagfm_next_stage_v1/`：稳定规范快照，包含工作包、治理合同、审计、脚本、单机服务和结果证据；
- `project_snapshot_20260815/aagfm_next_stage_v1/`：2026-08-15 增量包，包含 WP22/WP23 修复、单机复验和候选交接材料；
- `project_snapshot_20260814/aagfm_next_stage_v1/local_service/`：单机 loopback 服务及验收说明；
- `project_snapshot_20260814/aagfm_next_stage_v1/audits/wp12_rapid_historical_metrics_v1/`：WP12 快速历史指标结果；
- `project_snapshot_20260814/aagfm_next_stage_v1/EXPORT_SCOPE_20260814.md`：公开导出范围和排除项。

根目录不再维护与快照重复的脚本或结果副本；后续引用请使用上述规范路径。

## 2026-08-15 增量

WP23 定位并修复了市场尺度与 RG2 物化器的 positional `usecols` 绑定错误：源列顺序固定为 `[1,2,11,15]`，分别绑定 `trade_date`、`raw_close`、`adjust_factor`、`qfq_close`。开发回放在 256 个 warm-up 后周达到机器精度一致；五个未来窗口已重新生成 RG2、market-scale 与 C1 label-free shadow。未来标签未读取，指标未计算，生产注册表未修改。

审计入口：`project_snapshot_20260815/aagfm_next_stage_v1/governance/WP23_COLUMN_BINDING_REPAIR_AUDIT_20260815.md`。

## 研究边界

当前生产内核是 `RG_OBGNET_CONFIRMED_SAFE_V1_1`。WP12/C0 的结果支持研究和 shadow，不构成生产替换授权；V4 未来独立窗口在数据冻结和一次性评分前不可读取标签或宣称增益。

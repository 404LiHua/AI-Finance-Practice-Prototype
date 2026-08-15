# WP23：未来 T2 输入预消费审计闭环（2026-08-15）

## 完成内容

在未来标签到达前，已冻结并实现 CPU-only 的 T2 输入合同。数据交付后先运行该审计，任何失败都会在读取标签、训练、生成预测或启动 GPU 前关闭。

审计固定检查：

- 至少 12 个严格晚于 2026-06-26 的周五 origin，以及精确的 Asia/Shanghai cutoff；
- 每个 origin 的逐时点有效股票池，最低 300 支，禁止当前存活股票回填；
- RG3 14 项特征与 `source_trade_date` PIT 约束；
- C1 必需的 18 项 RG2 状态/图摘要及三项 PIT 日期；
- 市场波动率尺度、键覆盖、重复键、NaN/Inf；
- 所有输入列不得含 target/label/return/ordinal/FRESH 等字段。

合成测试已通过 2/2：12 个周五起点×300 股票的合法包通过；加入 `target_label` 列的包按预期 `FAIL_CLOSED`。

## 当前状态

该工作包验证的是“未来数据到达后的可执行性”，不是未来数据已交付，也不是模型指标或生产晋级证据。当前生产内核保持不变。

证据：

- 协议：`research_tracks/aagfm_next_stage_v1/governance/WP23_FUTURE_T2_INPUT_PRECONSUMPTION_CONTRACT_FREEZE_20260815.json`
- 审计器：`research_tracks/aagfm_next_stage_v1/local_service/audit_wp23_future_t2_input_contract_v1.py`
- 合成测试：`research_tracks/aagfm_next_stage_v1/local_service/test_audit_wp23_future_t2_input_contract_v1.py`



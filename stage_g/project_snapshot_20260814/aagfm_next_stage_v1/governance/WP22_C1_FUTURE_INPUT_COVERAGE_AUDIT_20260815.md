# WP22 C1 future-input coverage audit（2026-08-15）

## 结论

`market_volatility_4w` 的逐键尺度面板已在不读取标签的条件下补齐，可用于 2026-07-17 的 300 个 C1 shadow 键。生产注册表、生产模型和 T2 生产内核未修改。

物化收据：

`research_tracks/aagfm_next_stage_v1/outputs/wp22_future_scale_panel_20260815_v1/WP22_C1_FUTURE_MARKET_SCALE_PANEL_RECEIPT.json`

面板：

`research_tracks/aagfm_next_stage_v1/outputs/wp22_future_scale_panel_20260815_v1/WP22_C1_FUTURE_MARKET_SCALE_PANEL.parquet`

## 已证明内容

- 目标日期：`2026-07-17`；300 行、300 个唯一 `stock_code`。
- shadow 键与冻结 300 股票池完全一致。
- 计算公式与归档 RG 口径一致：冻结股票池的周收益均值，再做四周总体标准差（`ddof=0`）。
- 使用 12 个历史退市键完成开发期回放；除归档协议允许的首个 warm-up 异常外，261 个开发日期中 256 个日期机器精度完全一致。
- 2026-07-17 市场状态值：`0.03637829621393865`。
- 面板 SHA256：`d4ce5e0173b66cd3bc1e7fd86dd531f1f65d172498ecc5a6c6a9730d156de838`。
- 未读取 FRESH、SCREENING、FINAL 或任何目标标签；未使用 GPU；未修改生产注册表。

## 仍未通过的硬门

C1 正式 shadow 还需要同一日期、同一 300 键的 18 项 RG2 PIT 状态/图摘要特征。当前可见 `rg2_state_features.csv.gz` 仅覆盖 2018-06-08 至 2023-05-05 的开发期，不能广播、回填或从历史值伪造 2026-07-17 的 RG2 特征。

因此目前只能接受该尺度面板，C1 runner 仍应对缺少未来 RG2 输入保持 `FAIL_CLOSED`。补齐 RG2 前，不生成 C1 未来预测、不读取标签、不做候选选择，也不改变生产锚点。

## 下一步所需唯一补件

一份冻结、可审计的未来 RG2 面板，至少包含：

`trade_date, stock_code, sample_key_sha256` 以及 WP17 V2 规定的 18 个状态/图摘要列；覆盖 `2026-07-17` 的同一 300 个股票键，并附每个来源的 PIT 日期、输入清单 SHA256 和物化收据。

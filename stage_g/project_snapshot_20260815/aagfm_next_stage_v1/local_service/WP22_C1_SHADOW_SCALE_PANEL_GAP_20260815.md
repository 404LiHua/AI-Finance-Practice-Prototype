# C1 无标签 shadow scale panel 缺口（2026-08-15）

当前可见的 `WP11_LABEL_FREE_MARKET_VOLATILITY_4W.parquet` 只有市场级单行记录，没有 `stock_code`，不能作为 C1 所需的逐键 `(trade_date, stock_code, market_volatility_4w)` 面板。

runner 已支持 CSV/CSV.GZ/Parquet，并对缺少 `stock_code` 的面板明确 fail-closed；禁止广播、未来填补或伪造逐键值。补齐与 RG3/RG2 同日期同股票键的 scale panel（或冻结、可审计的广播规则）前，不重算、不宣称新的 shadow 结果；此前封存的 300 行 WP22 收据保持有效。

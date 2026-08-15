# T7 授权范围修正请求（2026-08-15）

当前模板 `T7_TRAIN_AUTHORIZATION_REQUEST_V1.json` 把范围写成 `ONE_CANDIDATE_TRAIN_ONLY_H4_ONCE`，与生产目标 `T2_MARKET_RELATIVE_FIXED` 不一致。

必须改为明确的 T2 TRAIN-only 范围，例如：

```text
ONE_CANDIDATE_TRAIN_ONLY_T2_ONCE
```

授权正文还必须明确：四周周度 origin、`origin_index + 4`、市场相对收益、±1% 三分类、5,513 键全集、同键生产锚点比较；不得引用 H4 四交易日绝对收益。

独立保管者签名、训练源及其 SHA-256、输入清单 SHA-256 仍缺失。修正前不得读标签、训练或宣称 T7 增益。本请求只修正授权语义，不授权任何标签读取或训练。

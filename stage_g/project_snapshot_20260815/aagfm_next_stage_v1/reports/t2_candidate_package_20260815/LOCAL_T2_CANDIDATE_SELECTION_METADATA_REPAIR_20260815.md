# Active T2 候选选择元数据修复

生产服务此前会在 active confirmed 模型响应中附带历史 V24 formal candidate 字段，容易被误读为当前候选。现已改为 `formal_candidate=null`，并明确保持生产锚点、无候选激活。候选选择继续由独立治理收据决定，服务不执行候选选择或生产替换。

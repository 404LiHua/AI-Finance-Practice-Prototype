# WP22：C1 单机影子制品与未来比较协议闭环（2026-08-15）

## 结果

已在不读取任何 FRESH/WP10 输出或未来标签的前提下，完成 C1 全开发期重拟合制品与无标签加载检查。

| 项目 | 状态 |
|---|---|
| C1 制品 | `NON_PRODUCTION_LABEL_FREE_SHADOW_ARTIFACT` |
| 模型 | `REV8_C1_RG2_STATE_AUGMENTED_HETEROSKEDASTIC_ORDINAL` |
| 模型 SHA-256 | `563ce8a19fd6a9aea20171abc34707fc866fa5ead3b34f0b8f395f5b0032b53c` |
| 训练有效行 / 周起点 | 67,706 / 220 |
| 加载冒烟 | 300 行、三类概率有限且逐行和为 1，`PASS_NON_PRODUCTION_C1_ARTIFACT_LOAD_AND_PROBABILITY_SMOKE` |
| 影子推理冒烟 | 300 行、输出 T2 类别/三类概率/可靠性，`PASS_NON_PRODUCTION_C1_LABEL_FREE_SHADOW_INFERENCE` |
| GPU | 未使用；CPU 固定 2 线程 |
| 生产注册表 | 未修改 |

该制品不是生产模型、不能预测缺少 RG2 状态特征的当前 shadow 输入，更不能直接替换 `RG_OBGNET_CONFIRMED_SAFE_V1_1`。

## 已冻结的未来选择

开发门合格的 C0 与 C1 同时保留为唯一正式比较对象。WP22 在任何未来输入或标签读取前固定：

- 至少 12 个晚于 2026-06-26、四周结算完成的同口径 T2 周起点；
- 对每个起点分别审计 RG3、C1 所需的 18 项 RG2 特征、市场波动率、PIT 日期与键覆盖；
- C0、C1、incumbent 先无标签预测并封存，再由新的单次授权读取标签；
- 同时报告 MCC、Brier、ECE、周度 IC、移动块置信区间与纸面成本结果；
- 不允许将 C2（最差折门失败）或 C3（事后探索）带入正式选择。

若两者均不满足所有硬门，或无法按预注册规则区分胜者，保持 incumbent；平均指标不抵消任一 PIT、最差窗口、校准或成本失败。

## 证据

- 协议：`research_tracks/aagfm_next_stage_v1/governance/WP22_C0_C1_FUTURE_T2_COMPARISON_AND_C1_SHADOW_ARTIFACT_PROTOCOL_FREEZE_20260815.json`
- 制品：`research_tracks/aagfm_next_stage_v1/local_service/models/wp22_c1_full_development_artifact_v1/WP22_C1_FULL_DEVELOPMENT_ARTIFACT.json`
- 加载审计：`research_tracks/aagfm_next_stage_v1/local_service/models/wp22_c1_full_development_artifact_v1/WP22_C1_ARTIFACT_LOAD_AUDIT.json`
- 影子收据：`research_tracks/aagfm_next_stage_v1/local_service/shadow/wp22_c1_smoke_20260815/C1_SHADOW_RECEIPT.json`



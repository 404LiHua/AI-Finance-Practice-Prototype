# V4 未来独立 FRESH 晋级工作说明（独立运行版）

> 文档编号：`CSN_V4_FUTURE_ANCHOR_ELIGIBLE_FRESH_STANDALONE_RUNBOOK_V1`  
> 状态：`PRE_REGISTERED / NOT YET MATERIALIZABLE`  
> 适用日期基准：2026-08-14  
> 用途：让未阅读任何历史对话的研究人员、数据管理员或后续代理，能够理解并安全继续 V4 的数据、预测、评分与晋级流程。

## 1. 这项工作要解决什么

> **2026-08-15 语义纠偏：本运行手册的生产晋级路径已撤销。** 冻结候选是四交易日绝对 H4，生产锚点是四周市场相对 T2；两者不可直接比较。WP13/WP14/WP15 仅保留为标签无关工程链路演练，禁止按本手册物化真实标签、评分、纸面交易或晋级。权威撤销收据：`WORK_PACKAGE_16_PRODUCTION_T2_ALIGNMENT_AND_V4_PROMOTION_REVOCATION_V1.md` 与 `V4_PRODUCTION_T2_TARGET_SEMANTICS_CONFLICT_FREEZE_V1.json`。

项目有一个已经冻结、但尚未完成正式晋级的候选模型：

`AA_GFMNET_CROSS_SECTIONAL_NEUTRALIZED_RESIDUAL_TCN_V1`

它拟与当前生产内核比较：

`RG_OBGNET_CONFIRMED_SAFE_V1_1`

V4 的唯一目标是：在一个候选和生产锚点都没有用过、且标签在预测封存前保持关闭的 8 周窗口上，比较两者的预测质量和固定成本下的纸面交易表现。只有两份独立收据都通过，候选才允许以原子操作替换生产内核。V4 不是重训练、调参或挑选新模型的阶段。

当前日期早于 V4 最后标签结算日，因此当前不得构建、预测、评分或晋级；本文件是预注册与运行手册，不是提前使用未来数据的授权。

## 2. 当前模型与不可变制品

### 2.1 冻结候选

候选目录：

`C:\Users\27793\Documents\project1\AI_Finance_Prototype\research_tracks\aagfm_pit_multisource_candidate_v1\candidate_freeze\csn_residual_full_development_v1`

候选结构：8 周、6 维数值因果 TCN 主干，加上严格 PIT 的横截面 rank 技术/基本面有界残差适配器；输出 H4 连续收益预测及 T2 三分类概率。

不可变身份：

| 项目 | 固定值 |
| --- | --- |
| 候选 ID | `AA_GFMNET_CROSS_SECTIONAL_NEUTRALIZED_RESIDUAL_TCN_V1` |
| 模型清单 SHA-256 | `c7ee9368bf0e71eb21efc3f4de05b86f0a68335008891afc401e8a7b0fa6908e` |
| 模型规格 SHA-256 | `49b4e6d6c441bb44e94949da35a86d4bc2fb2324c8952394b99bb4c5a6786741` |
| 三个固定随机种子 | 20260813、20260814、20260815 |
| 开发 origin 范围 | 2018-01-01 至 2024-06-10，共 337 个 |

候选已通过开发期冻结制品审计，但那只证明它“可进入独立检验”，不证明它已优于生产内核。

### 2.2 生产锚点

生产模型文件：

`C:\Users\27793\Documents\project1\deliverables\RG_OBGNet_source_v1\models\rg_obgnet_confirmed_safe_v1_1\MODEL.json`

锚点 ID 是 `RG_OBGNET_CONFIRMED_SAFE_V1_1`，当前文件 SHA-256 为：

`d8e4316d0fab70d3785b775c695a1f3a31225edf441a3603e3830f7351c4e2e8`

锚点的 `fit_receipt.last_origin_date` 为 **2026-06-26**。因此任何用于当前锚点相对比较的独立窗口，首个 origin 必须严格晚于 2026-06-26。

生产注册表位于：

`C:\Users\27793\Documents\project1\AI_Finance_Prototype\research_tracks\pit_information_incremental_v1\governance\KERNEL_REGISTRY.json`

在全部 V4 门槛通过前，禁止修改它。

## 3. 为什么是 V4，而不是复用旧窗口

以下窗口已关闭，不能拿来重新评分或晋级：

| 事项 | 结论 | 原因 |
| --- | --- | --- |
| 2026-05-22 至 2026-07-10 的 V1 FRESH | 已消费，禁止复用 | 一次性授权曾错误绑定优化后且不合格的制品；两份收据失败，标签已消费 |
| 2024-07-12 至 2024-08-30 的历史 V3 | 预测前关闭 | 当前生产锚点训练到 2026-06-26，可能已学习 2024 窗口标签；比较会泄漏 |
| 1990–2017 数据 | 可作另行研究，不可自动作为本次晋级证据 | 当前 300 股票池存在上市覆盖/幸存者偏差；还需独立 PIT 和锚点训练边界审计 |

V4 必须同时满足三条边界：

1. 晚于候选开发截止日 2024-06-10；
2. 晚于当前锚点训练截止日 2026-06-26；
3. 不与已消费 FRESH 标签集（截至 2026-07-10）重叠。

## 4. V4 预注册窗口与标签协议

### 4.1 8 个固定 origin

V4 的 origin 已经预注册，不能因结果好坏增删或替换：

| 序号 | origin（周一 09:30） | 正常情况下标签目标交易日 |
| ---: | --- | --- |
| 1 | 2026-07-20 | 2026-07-24 |
| 2 | 2026-07-27 | 2026-07-31 |
| 3 | 2026-08-03 | 2026-08-07 |
| 4 | 2026-08-10 | 2026-08-14 |
| 5 | 2026-08-17 | 2026-08-21 |
| 6 | 2026-08-24 | 2026-08-28 |
| 7 | 2026-08-31 | 2026-09-04 |
| 8 | 2026-09-07 | 2026-09-11 |

最早可物化日期为 2026-09-11 的收盘数据、版本和哈希都已冻结之后。

### 4.2 H4 的唯一有效定义

V4 必须复刻冻结候选的训练标签协议 `TEXTCU_V2_H4_MONDAY_0930_V1`，而不是使用旧 FRESH 脚本的近似实现。

对任一 `(origin_date, stock_code)`：

1. `origin_date` 是周一 09:30，预测时点不包含该周一收盘信息；
2. `anchor_trade_date` 是严格早于 origin 的最近交易日，正常为上周五；
3. `first_trade_on_or_after_origin` 是 origin 当日或之后的第一个交易日，正常为周一；
4. `target_trade_date` 是第 4 个“origin 当日或之后”的交易日，正常为周五；
5. `h4_return = adjusted_close(target_trade_date) / adjusted_close(anchor_trade_date) - 1`；
6. 目标价格的复权口径必须与开发标签一致（`close_times_adjust_factor` / 可审计等价复权收盘）；
7. 数据不足时保留该键，写入 `label_valid=false` 与具体 `invalid_reason`，不得悄悄丢弃。

密封标签文件至少应有：

`origin_date, stock_code, h4_return, target_horizon_trading_days, anchor_trade_date, first_trade_on_or_after_origin, target_trade_date, label_realized_at, label_valid, invalid_reason, target_price_basis, source_file_sha256, label_protocol_id`。

旧 `build_csn_fresh_inputs_v1.py` 和 V3 包装器使用周四/周五 origin、并以不同的 `<= origin` 收盘锚点计算标签；**禁止用于 V4**。

## 5. 数据要求与 PIT 规则

权威需求文件：

`governance\data_requests\20260814_FUTURE_ANCHOR_ELIGIBLE_FRESH_V4_DATA_REQUEST.md`

数据管理员在 2026-09-11 之后提供或冻结数据时，必须提供：

1. 每股日线 CSV：交易日期、前复权 OHLC、成交量、成交额、RSI、MACD、BOLL 等计算字段；覆盖技术特征前置期至 2026-09-11。
2. PIT 基本面事件：仅保留 `available_at <= origin_date` 的记录。
3. 正式 V4 股票池。
4. 密封 H4 标签：只能在预测哈希绑定完成后由评分器读取。
5. 导出日期、版本、来源、逐文件 SHA-256 与缺失说明。

300 池中的 `600355.SH` 已停牌早止。V4 预期使用经签名的 **299 股票池**（剔除该股），但这不是单侧删行：数值张量、技术面板、基本面面板、股票池、候选预测、锚点预测和标签必须使用完全相同的 `(origin_date, stock_code)` 键域。

所有特征必须严格 PIT：数值周频序列和技术特征截止到周一 09:30 前最后可得信息；基本面由 `available_at` 约束；训练与预测期间不得联网刷新数据。

## 6. V4 执行顺序（历史计划，当前已撤销）

本节保留原始预注册顺序，仅用于审计历史。由于目标语义冲突，任何真实数据到齐后的阶段 A–D 均不得启动；必须先完成 WP16 的生产 T2 目标恢复并新建同口径候选。

### 阶段 A：数据交付和标签无关预检

1. 先在 V4 数据需求文件的“交付核验记录”补充实际文件、版本、导出时间、SHA-256、逐股覆盖和有效股票池。
2. 运行覆盖审计：确认所有有效股票具备足够技术前置历史及完整 H4 结算日期。
3. 使用**新建且已审计的 V4 专用构建器**生成：
   - `FRESH_NUMERIC.npz`，形状 `8 × 8 × N × 6`；
   - `FRESH_TECHNICAL.parquet`；
   - `FRESH_FUNDAMENTALS.parquet`；
   - `FRESH_UNIVERSE.parquet`；
   - 冻结日线清单；
   - `SEALED_FRESH_H4_LABELS.parquet`。
4. 构建器必须不打印标签值、标签行数、收益汇总或任何评分指标；收据只能披露标签文件 SHA-256 与“未读取”状态。
5. 运行 V4 标签无关输入契约审计。它必须额外断言：恰有 8 个、均为周一、周度唯一、与预注册日期完全一致的 origin。旧 V1 输入审计只检查 origin 个数，不能单独作为 V4 语义证明。

### 阶段 B：封存预测

1. 使用上述不可变候选清单在 GPU 上执行一次批量推断；建议一个 GPU 作业，批大小 1024，CPU 线程上限 4。
2. 用当前生产锚点在同一日线快照和同一 299 键域上推断；锚点为 CPU/I/O 工作，建议 1–2 个 worker，且不可与候选 GPU 作业并发。
3. 两份预测都必须含：`origin_date, stock_code, h4_prediction, p_down, p_neutral, p_up`，概率和为 1，键唯一。
4. 输出候选/锚点预测各自 SHA-256、候选清单/规格 SHA-256、锚点模型 SHA-256、输入 SHA-256 与设备/线程收据。

### 阶段 C：预测绑定与一次性评分

1. 预测封存后，运行预消费绑定审计：候选 ID、候选清单 SHA、候选规格 SHA、锚点 ID/模型 SHA、两份预测键域、V4 输入哈希和密封标签哈希必须一致。
2. 只有绑定通过，才由保管流程签发新的、未使用的一次性授权。旧 V1 授权及旧标签不能复用。
3. 一次性评分器读取密封标签，生成：
   - `FRESH_SCORING_RECEIPT.json`；
   - `PAPER_TRADING_RECEIPT.json`；
   - 授权消费收据。
4. 评分器不得训练、调参、重跑预测或选择交易规则。

### 阶段 D：晋级或关闭

只有下列条件同时成立，才调用原子晋级脚本：

- 候选平均周度 IC 严格高于锚点；
- 候选 Brier 不高于锚点；
- 20bps 固定成本下候选年化净收益不低于锚点；
- 20bps 下候选最大回撤不差于锚点；
- 两份收据均为 `PASS`；
- 所有绑定哈希匹配，且生产注册表仍指向原锚点。

任一项失败：写入关闭收据，候选不得晋级，V4 标签被视为已消费，禁止以同一窗口重跑或调参后重评。

## 7. 已知失败模式与禁止事项

1. **制品错绑**：此前 FRESH 评分使用了 optimized 制品而非冻结候选。V4 必须将清单 SHA `c7ee...6908e` 写入预测、授权和评分绑定。
2. **锚点时间泄漏**：历史 V3 因当前锚点训练时间晚于测试窗口而关闭。V4 必须每次读取当前锚点 `fit_receipt.last_origin_date` 并断言 `< 2026-07-20`；若生产锚点在 V4 前更新过，必须重新做该审计。
3. **标签协议漂移**：不得用周五特征/同日收盘标签替代候选的周一 09:30 H4 协议。
4. **股票池单侧删除**：不得仅在日线或标签中剔除停牌股。
5. **提前读取**：构建密封标签不等于允许研究者查看其内容；预测和绑定前不得读取标签值、行数、回报或指标。
6. **日期挑选**：预注册 origin 不能因缺失、收益或结果而改动；覆盖不足时只能 fail-closed。
7. **硬件失衡**：不并发运行候选 GPU 推断和锚点多线程 I/O；审计/CSV 处理限定 2–4 CPU 线程。

## 8. 当前状态与恢复条件

截至 2026-08-14：V4 数据尚未完整结算，现有日线末端不足以覆盖至 2026-09-11；因此没有 V4 输入、预测、授权或评分产物。

恢复执行所需最小条件：

1. 日期已到 2026-09-11 之后；
2. 数据管理员按 V4 请求冻结日线与 PIT 基本面材料，并补充版本/SHA-256；
3. 覆盖和键域审计通过；
4. 当前生产锚点仍满足训练截止早于 V4 首日，或对更新后的锚点重新完成时间资格审计。

达到条件后，不训练候选，直接从“阶段 A”开始。若 V4 成功，原子晋级脚本才会修改注册表；若失败，保留现有生产内核并关闭 V4。

## 9. 关键路径索引

| 用途 | 路径 |
| --- | --- |
| V4 数据请求 | `governance\\data_requests\\20260814_FUTURE_ANCHOR_ELIGIBLE_FRESH_V4_DATA_REQUEST.md` |
| 数据更新治理规则 | `governance\\data_requests\\DATA_UPDATE_DOCUMENTATION_POLICY_V1.md` |
| 候选冻结根 | `candidate_freeze\\csn_residual_full_development_v1` |
| 候选全开发制品审计 | `audits\\csn_full_development_candidate_audit_v1\\AUDIT_DECISION.json` |
| 历史 V3 关闭审计 | `audits\\csn_historical_fresh_v3_scope_closure_v1\\DECISION.json` |
| 当前生产模型 | `C:\\Users\\27793\\Documents\\project1\\deliverables\\RG_OBGNet_source_v1\\models\\rg_obgnet_confirmed_safe_v1_1\\MODEL.json` |
| 生产注册表 | `research_tracks\\pit_information_incremental_v1\\governance\\KERNEL_REGISTRY.json` |
| 候选批量无标签预测器 | `scripts\\predict_csn_candidate_fresh_label_free_batch_v1.py` |
| 锚点无标签预测器 | `scripts\\predict_confirmed_anchor_fresh_label_free_v1.py` |
| 一次性评分器 | `scripts\\score_csn_fresh_once_v1.py` |
| 原子晋级器 | `scripts\\promote_csn_candidate_atomically_v1.py` |

本手册优先于旧 V1/V3 FRESH 构建脚本的日期和标签约定；任何新 V4 实现都必须以本手册的周一 H4 协议和预注册日期为准。

# Stage E E-1：开发协议、数据封存与 SHA-256 冻结报告

冻结日期：2026-07-24

## 1. 冻结结论

Stage E 开发协议 `stage_e_rolling_origin_v1` 已建立并在 30 股票 `panel_v2` 批次上完成两次一致复算。

- 开发截止日期：`2023-06-02`
- 证据类别：`STAGE_E_SELECTION_EXPOSED_DEVELOPMENT_ONLY`
- C-4 行级数据读取：否
- D-5 行级数据读取：否
- 未来 Stage E SCREENING/FINAL 读取：否
- 数据批次：`panel_v2_30stocks_20220603_20230602_v1`
- 处理批次 SHA-256：`bc027addda7e75d3310af7a234ed35f33f65584a378906d9b72c14a18f899a2b`

## 2. 数据拒绝规则

读取任何开发输入前，`stage_e/custody.py` 会执行路径和日期双重检查。

禁止路径：

- `data/screening/`
- `outputs/screening/`
- `outputs/stage_d/d5_screening_20240614_20250613/`
- 任何文件名或路径中包含已登记 C-4、D-5、SCREENING 预测或 Stage E FINAL 标识符的路径

封存时间区间：

- C-4：2023-06-09 至 2024-06-07
- D-5：2024-06-14 至 2025-06-13
- 未来 Stage E 数据：最早可能从 2025-06-20 开始，当前状态为未获取、未读取、未授权

任何 `trade_date` 或 `target_date` 超过 2023-06-02 都会在处理前触发 `StageEDataCustodyViolation`。

## 3. 冻结 rolling-origin 折

| 折 | TRAIN 截止 | Purge | VALIDATION | TRAIN 样本 | VALIDATION 样本 |
|---|---|---|---|---:|---:|
| E_RO_01 | 2023-01-13 | 2023-01-20 | 2023-02-03 至 2023-03-10 | 568 | 150 |
| E_RO_02 | 2023-03-03 | 2023-03-10 | 2023-03-17 至 2023-04-21 | 748 | 150 |
| E_RO_03 | 2023-04-14 | 2023-04-21 | 2023-04-28 至 2023-06-02 | 928 | 150 |

每折均覆盖 30 只股票。TRAIN 的实际首个合格样本日期为 2022-08-26；后续增加更早历史时，允许扩展 TRAIN 起点，但不得移动 TRAIN 截止、Purge 和 VALIDATION 边界。

折协议 SHA-256：`91dd42f05bc8661c3910399e65d6c33ba9657acf68f62eb04bb7adcce915c82f`

## 4. 文件与批次哈希

本批次逐个计算并登记 5,091 个文件：

| 文件类别 | 数量 |
|---|---:|
| 九章原始周线 | 5,000 |
| 九章股票基础信息 | 1 |
| CSMAR 原始 ZIP | 4 |
| BaoStock 原始/清单文件 | 61 |
| CSMAR 处理文件 | 7 |
| Stage A 处理批次 | 12 |
| panel_v2 处理批次 | 6 |

关键根哈希：

- 源文件清单根 SHA-256：`ce05f5c05779fc47c0c4fa5519d14d113b860e67c846db789b2bc16aae9c6a70`
- `panel_v2.csv.gz` SHA-256：`909dab06300db13b1027d457444bb8e949706958608f4083dbddd78878eb6c11`
- panel 行集合 SHA-256：`f4905b46d2e35478d9897f20be341481f1bbfb8c85935e8702eadd0a95d8852b`
- fold assignments SHA-256：`38ba950f07e009b8beb91e145ddf6d9a5fbfa3078f46fd0281d9477aa56a6979`
- fold summary SHA-256：`5f2264b57a2d0815f832743e3e244dc593b833bc923d8ca17cb51cc0a8abc8f7`

完整逐文件清单保存在本地忽略目录：

`outputs/stage_e/development_protocol_v1/panel_v2_30stocks_20220603_20230602_v1/source_file_manifest.jsonl`

受许可限制的绝对路径、原始内容和逐样本数据不会提交到 Git；Git 仅保存冻结快照、聚合哈希、代码、配置、测试和报告。

## 5. 各折样本哈希

| 折 | TRAIN 行集合 | VALIDATION 行集合 |
|---|---|---|
| E_RO_01 | `0f9bac407ee4ce7cf1cdb3e7e31494ef467165102286cfaa9c66d69ff67bc150` | `5c51ea7ce5ecbf8f0c0fffe8c72052afb415da794033aa4f2443d541d78c3cdb` |
| E_RO_02 | `8c65725095ff045acba296cb8e6fec899e45e9b3b2f517bd87421a4a42ce5089` | `f0c745b8d0c878f78bba476a4a346f51d25dbc6f9ddcd1a7db39043a81cb8a60` |
| E_RO_03 | `83cce0461a83525de1fed571a4af196748ae9da2784a571595c4f9c7de148409` | `737289d56d14acb330dd6b89f065fc87c7d9f773b9576d9f1f38694e7430248d` |

每折还同时保存包含目标收益内容的 SHA-256，防止样本主键不变但标签被修改。

## 6. 一致复算

同一批次连续运行两次后，下列内容保持完全一致：

- 处理批次 SHA-256；
- 源文件清单根 SHA-256；
- panel 文件和行集合 SHA-256；
- 折协议与每折样本 SHA-256；
- 确定性 gzip 的 fold assignments SHA-256；
- fold summary SHA-256。

独立验证入口 `stage_e/verify_development_protocol.py` 的 8 项检查全部通过，最终判定：`PASS`。

## 7. 后续数据扩大规则

从 30 股票扩大到 100/300 股票时必须保持以下项目不变：

- 开发截止日期；
- VALIDATION 和 Purge 边界；
- 12 周 lookback；
- 点时可交易与样本资格逻辑；
- C-4、D-5、未来 SCREENING/FINAL 拒绝规则；
- 文件、行集合和内容哈希算法。

允许变化：增加历史时点可投资股票、增加截止日前历史长度、增加在样本时点已经公开且获得授权的特征源。

每次扩大必须使用新的 `data_batch_id`，重新生成完整源文件清单、处理批次哈希、panel 行集合哈希和全部折样本哈希。不得在看到模型结果后移动折边界。

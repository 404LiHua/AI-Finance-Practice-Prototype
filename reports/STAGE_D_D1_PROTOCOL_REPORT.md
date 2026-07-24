# 阶段 D-1：滚动验证、数据封存与跨折汇总框架报告

完成日期：2026-07-24

## 1. 完成结论

D-1 已建立三项可复用工程能力：

- 严格按时间前推的 rolling-origin 折生成器。
- 在打开文件前执行的 C-4 路径拒绝和日期上限守卫。
- 面向“模型 × 折 × 种子”的统一跨折指标汇总器。

本次只读取日期不晚于 2023-06-02 的阶段 A 开发面板，没有读取 C-4 逐样本数据，也没有获取暂定 D-SCREENING 数据。

## 2. Rolling-origin 协议

协议参数：3 折、每折 6 个验证周、折间前推 6 周、1 个 purge 周、至少 24 个训练周。由于下一周收益目标必须仍位于本折验证区间，每折实际得到 5 个可评分周，即 30 股票 × 5 周 = 150 个验证样本。

| 折 | 训练截止 | Purge | 验证区间 | 训练样本 | 验证样本 |
|---|---|---|---|---:|---:|
| D_RO_01 | 2023-01-13 | 2023-01-20 | 2023-02-03 至 2023-03-10 | 568 | 150 |
| D_RO_02 | 2023-03-03 | 2023-03-10 | 2023-03-17 至 2023-04-21 | 748 | 150 |
| D_RO_03 | 2023-04-14 | 2023-04-21 | 2023-04-28 至 2023-06-02 | 928 | 150 |

三折均覆盖固定 30 只股票，训练目标日期严格早于验证观察日期。协议摘要为：

`5dcd3c7e28b5577472743c171bd98ba17a514920e8088c405c4ecca68f102cb6`

## 3. C-4 数据封存拒绝

封存守卫执行两层检查：

1. 路径层：拒绝 `data/screening`、C-4 输出目录以及包含 C-4 标识符的其他路径，拒绝发生在任何文件读取之前。
2. 数据层：拒绝 `trade_date` 或 `target_date` 超过 2023-06-02，或与 2023-06-09 至 2024-06-07 封存区间重叠的数据帧。

暂定 2024-06-14 至 2025-06-13 的 D-SCREENING 仍保持 `RESERVED_NOT_ACQUIRED_NOT_READ`，未授权、未读取。

## 4. 跨折统一汇总器

输入必须包含：`model/fold_id/seed/samples/mae/rmse/direction_accuracy/direction_f1`。汇总器会拒绝重复行、非有限数值、缺失 Naive 或不完整的模型—折—种子网格。

统一输出包括：

- 每折三种子均值和标准差。
- 跨折总体 MAE、RMSE、方向 Accuracy/F1 和 MAE 变异系数。
- 相对 Naive 的平均 MAE/RMSE 改善比例。
- 击败 Naive 的折数和折胜率。
- 最差折 MAE 以及最差折相对 Naive 的劣化比例。

所有结果仍属于阶段 D 选择暴露开发证据，不是新的独立证据。

## 5. 产物与验证

- 配置：`stage_d/configs/d1_protocol.json`
- 封存策略：`stage_d/configs/data_custody.json`
- 正式协议：`stage_d/protocols/rolling_origin_v1.json`
- 折生成器：`stage_d/rolling_origin.py`
- 数据守卫：`stage_d/custody.py`
- 汇总器：`stage_d/aggregation.py`
- 协议入口：`stage_d/run_d1_protocol.py`
- 汇总入口：`stage_d/run_cross_fold_summary.py`
- 生成折分配：`outputs/stage_d/d1_rolling_origin_v1/fold_assignments.csv.gz`

阶段 D-1 单元测试 8/8 通过。

## 6. 下一步

进入 D-2：严格使用已登记的三折协议运行 Naive、FreTS L4、价格版 Minimalist Transformer、仅时域、固定时间图及预登记固定收缩候选。不得修改折定义，不得读取 C-4 或未来 D-SCREENING 数据。

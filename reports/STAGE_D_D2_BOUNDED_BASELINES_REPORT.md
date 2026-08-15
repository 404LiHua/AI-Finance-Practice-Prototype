# 阶段 D-2：受限稳健基线三折实验报告

报告日期：2026-07-24

实验 ID：`stage_d_d2_bounded_baselines_v1`

协议：`stage_d_rolling_origin_v1`
协议 SHA-256：`5dcd3c7e28b5577472743c171bd98ba17a514920e8088c405c4ecca68f102cb6`

## 1. 执行边界

- 严格使用已经登记的三折边界，未重新生成或修改折定义。
- 每折均从同一 30 股票开发面板动态生成 TRAIN/VALIDATION；未使用原静态 `train/validation/test` 标签。
- 每折 TRAIN 与 VALIDATION 的样本数、股票数和行集合 SHA-256 均与 D-1 登记结果一致。
- 仅使用 `2023-06-02` 及以前的开发数据。
- C-4 逐样本数据读取数为 0；未来 D-SCREENING 读取数为 0。
- 本报告属于滚动开发期、选择暴露证据，不是独立 SCREENING 结论。

## 2. 冻结实验清单

基础模型：Naive、FreTS return-only L4、价格版 Minimalist Transformer L8、仅时域控制组、固定时间图控制组。

冻结种子：`20260723`、`20260724`、`20260725`。

固定收缩只作用于四个非 Naive 学习模型，公式为：

`prediction_shrunk = alpha × prediction + (1-alpha) × 0`

其中 `alpha` 仅允许 `0.25`、`0.50`、`0.75`。未在看到结果后增加系数或模型族。

完整网格包含 17 个模型结果、3 折、3 种子，共 153 行统一指标；其中基础运行 45 次（含 9 次无训练 Naive），学习模型训练 36 次。

## 3. 跨折结果

| 排名 | 模型 | 平均 MAE | MAE 标准差 | 相对 Naive 平均 MAE 改善 | 优于 Naive 的折数 | 最差折相对 Naive |
|---:|---|---:|---:|---:|---:|---:|
| 1 | FreTS L4 + 固定收缩 0.75 | 0.031242 | 0.002612 | 1.780% | 3/3 | -1.110% |
| 2 | FreTS L4 + 固定收缩 0.50 | 0.031274 | 0.002635 | 1.684% | 3/3 | -1.055% |
| 3 | FreTS L4 原始预测 | 0.031316 | 0.002583 | 1.538% | 3/3 | -0.862% |
| 4 | FreTS L4 + 固定收缩 0.25 | 0.031461 | 0.002683 | 1.102% | 3/3 | -0.672% |
| — | Naive | 0.031808 | 0.002667 | 0.000% | 0/3 | 0.000% |

`FreTS L4 + 固定收缩 0.75` 的平均 RMSE 为 `0.044065`，相对 Naive 平均改善 `0.869%`；方向准确率均值为 `0.554815`，方向 F1 均值为 `0.529751`。

分折 MAE：

| 折 | Naive | FreTS L4 | FreTS L4 + 收缩 0.75 |
|---|---:|---:|---:|
| D_RO_01 | 0.028688 | 0.028198 | 0.028089 |
| D_RO_02 | 0.034845 | 0.034132 | 0.034099 |
| D_RO_03 | 0.031892 | 0.031617 | 0.031538 |

## 4. 其他控制组结论

- 价格版 Minimalist Transformer 原始预测平均 MAE 为 `0.035091`，比 Naive 差 `10.837%`；0.25 收缩后为 `0.031712`，仅 2/3 折优于 Naive，最差折仍高 `0.311%`。
- 仅时域控制组原始预测平均 MAE 为 `0.036492`，比 Naive 差 `15.660%`，种子 MAE 变异系数为 `0.147919`，是基础模型中最不稳定者。
- 固定时间图原始预测平均 MAE 为 `0.035790`，比 Naive 差 `12.751%`；其所有冻结收缩版本均未达到三折一致优于 Naive。
- 本轮证据支持“短收益序列 + 明确收缩”的简单结构，不支持在当前 30 股票开发数据上以价格 Transformer 或固定时间图替代 FreTS L4。

## 5. D-2 验收结论

D-2 已完成。完整模型—折—种子网格通过统一汇总器检查；折定义未变，原静态 split 未使用，C-4 与未来 D-SCREENING 均未读取。

当前最强的已登记开发期结果是 `frets_return_l4__fixed_shrink_a075`。该结果只能进入 D-3 跨折诊断，不能直接宣称获得独立外推能力，也不能据此访问未来 D-SCREENING。

## 6. 证据文件

- 冻结配置：`stage_d/configs/d2_baselines.json`
- 运行器：`stage_d/run_d2_baselines.py`
- 逐折逐种子指标：`outputs/stage_d/d2_bounded_baselines_v1/metrics_by_fold_seed.csv`
- 分折汇总：`outputs/stage_d/d2_bounded_baselines_v1/per_fold_summary.csv`
- 跨折汇总：`outputs/stage_d/d2_bounded_baselines_v1/cross_fold_model_summary.csv`
- 封存与完整性证据：`outputs/stage_d/d2_bounded_baselines_v1/evidence_manifest.json`
- 输出 SHA-256：`outputs/stage_d/d2_bounded_baselines_v1/sha256_manifest.json`

# Minimalist Transformer TRAIN 结果

## 证据边界

- QRG 谱系节点：`minimalist-transformer-v1`
- 证据类别：`SELECTION_EXPOSED_NON_INDEPENDENT`
- 原 Stage A train/validation/test 均已在 QRG 建立前查看，因此本报告全部属于 TRAIN 探索结果。
- 未来 SCREENING `future_weekly_screening_20230609_20240607_v1` 未获取、未读取，访问次数为 0。
- 本结果不构成部署批准、投资建议或独立样本外确认。

## 冻结结构

- 输入：与 LSTM 相同的 8 周序列、30 个训练期标准化特征。
- 线性输入投影：30 → 32。
- 学习式位置编码：8 × 32。
- Transformer Encoder：1 层、4 个注意力头、前馈维度 64、GELU、dropout 0.1、pre-norm。
- 聚合：最后一个时间点表示。
- 输出：LayerNorm 加单线性收益率预测头。
- 参数量：9,889。
- 随机种子：20260723、20260724、20260725。

## 三随机种子结果

| 原切分标签 | MAE | RMSE | 方向 Accuracy | 方向 F1 |
|---|---:|---:|---:|---:|
| validation | 0.038305 ± 0.001612 | 0.055097 ± 0.001258 | 54.44% | 0.3168 |
| test | 0.039334 ± 0.002094 | 0.057612 ± 0.003424 | 52.54% | 0.5222 |

## 与现有基线比较

- validation MAE：优于 LSTM（0.042190）和移动平均（0.039316），与 ARIMA（0.038308）基本相同，但弱于 Naive（0.036745）。
- 原 test 标签 MAE：优于 LSTM（0.041684），弱于 Naive（0.034914）、ARIMA（0.035983）和移动平均（0.038482）。
- 方向 F1：原 test 标签为 0.5222，接近 ARIMA 的 0.5278，优于移动平均和 Naive。
- 三随机种子存在可见波动，说明当前一年期小样本不足以支持“Transformer 优于简单基线”的结论。

## 节点结论

`INCONCLUSIVE`：Minimalist Transformer 相对 LSTM 有明确工程改进，但没有超过 Naive，且稳定性仍有限。
保留其作为正式数值时序 Transformer 基线，不将其锁定为唯一晋级候选。

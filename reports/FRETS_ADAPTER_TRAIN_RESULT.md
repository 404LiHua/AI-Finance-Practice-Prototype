# FreTS 金融适配 TRAIN 结果

## 证据边界

- QRG 节点：`frets-adapter-v1`
- 证据类别：`SELECTION_EXPOSED_NON_INDEPENDENT`
- 统一样本：train 688、validation 120、test 210；原 validation/test 标签均属于已查看 TRAIN。
- 未来 SCREENING 和 FINAL 未读取，本结果不构成独立样本外确认或部署批准。

## 适配方式

- 上游实现：`D:/项目/源文件/deploy/FreTS-main/models/FreTS.py`
- 不使用上游 COVID/比例切分 loader。
- 每只股票使用截至预测时点的 8 周 `return_1w` 序列。
- 标准化参数只使用训练期数据拟合，预测后还原到原始收益率尺度。
- 输出交由统一 `prediction_frame` 和 `evaluate_predictions` 评估。
- 上游 FreTS 参数量：328,833。

## 三随机种子结果

| 原切分标签 | MAE | RMSE | 方向 Accuracy | 方向 F1 |
|---|---:|---:|---:|---:|
| validation | 0.037213 ± 0.000642 | 0.054206 ± 0.000751 | 56.67% | 0.4153 |
| test | 0.037504 ± 0.001122 | 0.055121 ± 0.001959 | 55.40% | 0.5725 |

## 结论

FreTS 的 validation MAE 优于 ARIMA、移动平均、LSTM 和 Minimalist Transformer，接近但未超过 Naive；
原 test 标签的方向 F1 为当前基线中较高水平，但 MAE 仍弱于 Naive 和 ARIMA。

节点结论为 `INCONCLUSIVE`：保留为正式频域基线，不锁定为唯一晋级候选。

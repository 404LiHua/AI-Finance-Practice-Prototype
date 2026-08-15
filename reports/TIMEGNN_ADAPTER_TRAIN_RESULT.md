# Time-GNN 金融适配 TRAIN 结果

## 证据边界

- QRG 节点：`timegnn-adapter-v1`
- 证据类别：`SELECTION_EXPOSED_NON_INDEPENDENT`
- 统一样本：train 688、validation 120、test 210；原 validation/test 标签均属于已查看 TRAIN。
- 未来 SCREENING 和 FINAL 未读取，本结果不构成独立样本外确认或部署批准。

## 适配方式

- 上游实现：`D:/项目/源文件/deploy/Time-GNN-main/models/TimeGNN.py`
- 不使用上游 weather 数据及其比例切分 loader。
- 输入为与 LSTM/Transformer 相同的 8 周、30 个训练期标准化特征。
- 使用上游 Gumbel-Softmax 动态时间图与 GraphSAGE，输出下一周单一收益率。
- 输出交由统一 `prediction_frame` 和 `evaluate_predictions` 评估。
- 参数量：36,742。

## 三随机种子结果

| 原切分标签 | MAE | RMSE | 方向 Accuracy | 方向 F1 |
|---|---:|---:|---:|---:|
| validation | 0.041959 ± 0.005138 | 0.057107 ± 0.001904 | 48.89% | 0.3866 |
| test | 0.042412 ± 0.005020 | 0.058338 ± 0.005204 | 49.05% | 0.5188 |

## 结论

Time-GNN 的平均误差没有超过 FreTS、Minimalist Transformer 或简单基线，并且三随机种子 MAE 波动明显。
这与预先登记的“小样本动态图可能过拟合”的反向预测一致。

节点结论为 `BRANCH_REJECTED / STABILITY_FAILURE`：保留代码和结果用于复现与后续分析，
当前配置不作为晋级候选。若后续继续研究，应建立新的有界结构节点，而不能把本次结果事后修复为独立证据。

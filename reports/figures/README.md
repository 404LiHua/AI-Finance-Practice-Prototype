# 阶段 B 图表

- `prediction_comparison_30stocks.png`：30 股票横截面平均真实收益与七个模型平均预测路径。
- `training_curves_30stocks.png`：LSTM、Minimalist Transformer、FreTS、Time-GNN 三随机种子训练曲线。
- `figure_manifest.json`：输入结果表哈希、图像哈希和证据边界说明。

所有图表均来自已选择暴露的 TRAIN 结果；原 validation/test 只是历史文件标签，不是独立 SCREENING/FINAL 证据。

重建命令：

```powershell
D:\项目\源文件\deploy\.venv-timegnn\Scripts\python.exe experiments\plot_baseline_results.py
```

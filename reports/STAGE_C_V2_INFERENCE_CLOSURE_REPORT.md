# 阶段 C 推荐 v2 独立加载与推理闭环报告

完成日期：2026-07-24  
模型：`fixed_control_ensemble_v2`

## 1. 完成内容

- 从模型清单独立解析仅时域和固定时间图两个组件。
- 从两个 `model.pt` 恢复网络参数、训练配置、特征顺序、缺失值中位数、均值、标准差和序列长度。
- 从 Stage A 处理数据重新构造逐股票 8 周输入序列。
- 校验组件种子一致、特征一致、固定权重非负且合计为 1。
- 使用 CPU 推理分别生成组件预测，再执行 0.5/0.5 固定集成。
- 保存预测、指标、环境、模型清单哈希和组件权重 SHA-256。
- 增加缺失权重、错误特征顺序、异常缩放器形状和错误模型清单测试。

推理过程不读取历史组件预测 CSV。历史推荐 v2 预测仅在推理完成后用于数值核对。

## 2. 三种子复算结果

| 种子 | 对齐样本 | 最大绝对差 | 平均绝对差 | 门槛 | 结果 |
|---:|---:|---:|---:|---:|---|
| 20260723 | 120 | 1.58e-16 | 6.50e-17 | 1e-7 | 通过 |
| 20260724 | 120 | 1.87e-16 | 5.59e-17 | 1e-7 | 通过 |
| 20260725 | 120 | 1.73e-16 | 7.17e-17 | 1e-7 | 通过 |

三个种子的独立推理指标与历史结果一致。差异仅为浮点 CSV 往返造成的机器精度级误差。

## 3. 推理入口

```powershell
D:\项目\源文件\deploy\.venv-timegnn\Scripts\python.exe .\stage_c\run_recommended_v2_inference.py --seed 20260723
```

验证历史结果：

```powershell
D:\项目\源文件\deploy\.venv-timegnn\Scripts\python.exe .\stage_c\run_recommended_v2_inference.py `
  --seed 20260723 `
  --verify-reference .\outputs\experiments\stage_c_30stocks_recommended_v2\fixed_control_ensemble_v2_seed20260723\predictions.csv
```

## 4. 输出

每次推理保存到：

```text
outputs/inference/stage_c_recommended_v2/fixed_control_ensemble_v2_seed<seed>_<split>/
```

包含：

- `predictions.csv`
- `metrics.json`
- `inference_provenance.json`
- `verification.json`

## 5. 结论

推荐 v2 的独立模型加载和推理闭环已经完成。模型可以在不依赖历史预测文件的情况下，仅使用 Stage A 处理数据、相对路径模型清单和两个组件权重重建相同结果。

阶段 C 的下一工作包更新为 C-2：工程成本与误差诊断。


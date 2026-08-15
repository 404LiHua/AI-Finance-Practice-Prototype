# 阶段 B 正式验收报告

版本：`v0.2.0-stage-b`
归档日期：2026-07-23
实验范围：固定 30 只 A 股、一年期周频数据

## 1. 验收结论

阶段 B“统一基线实验”已完成，满足项目计划规定的工程验收条件：建立统一配置、训练、
评估、日志和随机种子框架；完成首批时序基线；使用一致的数据切分和评估器；保存可复现
产物；形成结果总表、实验日志、训练曲线和预测对比图。

本次归档不包含受许可限制的 CSMAR 原始数据、处理后数据、模型权重或完整运行输出。
这些内容由 `.gitignore` 管理，可通过本地数据和归档代码重新生成。

## 2. 已完成内容

- 基线模型：Naive、移动平均、ARIMA(1,0,0)、LSTM、Minimalist Transformer、FreTS、Time-GNN。
- 基线实验：7 个模型 × 3 个随机种子，共 21 次运行。
- 有界消融：6 个固定配置 × 3 个随机种子，共 18 次运行。
- 统一指标：MSE、MAE、RMSE、零值安全 MAPE、方向 Accuracy、方向 F1、逐股票指标。
- 统一产物：解析后配置、随机种子、环境信息、耗时、日志、权重、预测文件和训练历史。
- 图表：30 股票预测对比图、四类神经模型训练曲线图。
- 测试：阶段 B 框架及适配器共 12 项单元测试通过。

## 3. 关键实验结论

- Naive 在首轮基线的误差指标上仍是最强简单基准。
- FreTS 是首轮表现最好的复杂模型，但尚不能据此认定其获得独立验证。
- FreTS 4 周收益单通道在有界消融中优于原 8 周配置；更长窗口和 OHLC 多通道未改善验证误差。
- Minimalist Transformer 的仅价格视图优于当前价格+文本视图；当前文本事件覆盖不足。
- 原 Time-GNN 分支存在明显随机种子不稳定，后续稳定化工作移交阶段 C。

## 4. 可复现入口

```powershell
# 全部首轮基线
powershell -ExecutionPolicy Bypass -File .\experiments\run_stage_b.ps1

# 有界消融
D:\项目\源文件\deploy\.venv-timegnn\Scripts\python.exe .\experiments\run_bounded_ablations.py

# 图表
D:\项目\源文件\deploy\.venv-timegnn\Scripts\python.exe .\experiments\plot_baseline_results.py

# 单元测试
D:\项目\源文件\deploy\.venv-timegnn\Scripts\python.exe -m unittest `
  experiments.tests.test_framework `
  experiments.tests.test_external_adapters `
  experiments.tests.test_bounded_ablations
```

## 5. 归档文件

- `experiments/`：统一框架、模型、外部适配器、消融运行器、配置和测试。
- `reports/STAGE_B_BASELINES_INITIAL_REPORT.md`：首轮基线及后续更新。
- `reports/BOUNDED_ABLATIONS_TRAIN_RESULT.md`：FreTS 和 Transformer 有界消融。
- `reports/figures/`：预测对比图、训练曲线及生成清单。
- `plans/STAGE_C_IMPLEMENTATION_PLAN.md`：阶段 C 实施安排。

## 6. 移交阶段 C 的事项

阶段 B 本身不存在阻塞项。以下工作属于阶段 C：实现自研动态图频融合模型、完成稳定化
Time-GNN 思路、开展核心模块消融和邻接矩阵解释。仅在候选与规则冻结、独立复算完成并
获得授权后，才获取未来 SCREENING 数据。

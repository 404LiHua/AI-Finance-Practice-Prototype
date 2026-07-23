# Stage B 统一基线框架

该目录统一管理配置、随机种子、训练、评估、日志和实验产物。当前已直接实现
Naive、移动平均、ARIMA、LSTM、Minimalist Transformer、FreTS 和 Time-GNN。FreTS 与 Time-GNN
通过仓库内适配器绕过上游自带的比例切分，直接使用统一的金融样本、训练期标准化和评估器；
不能把原项目在 COVID/weather 等数据上的结果当作本项目结果。

## 环境部署

```powershell
py -3.12 -m venv .venv-baselines
.\.venv-baselines\Scripts\python.exe -m pip install -r experiments\requirements-stage-b.txt
```

如果网络环境无法下载 PyTorch，可直接复用已经部署的 Time-GNN 解释器：

```powershell
D:\项目\源文件\deploy\.venv-timegnn\Scripts\python.exe experiments\run_all_baselines.py
```

ARIMA 默认优先使用 `statsmodels`；未安装时，配置中的 ARIMA(1,0,0) 会使用数学上等价的
带截距 AR(1) 最小二乘后端，因此离线环境仍可复现实验。

## 单模型运行

```powershell
.\.venv-baselines\Scripts\python.exe experiments\run_experiment.py --model naive --seed 20260723
.\.venv-baselines\Scripts\python.exe experiments\run_experiment.py --model moving_average --seed 20260723
.\.venv-baselines\Scripts\python.exe experiments\run_experiment.py --model arima --seed 20260723
.\.venv-baselines\Scripts\python.exe experiments\run_experiment.py --model lstm --seed 20260723
.\.venv-baselines\Scripts\python.exe experiments\run_experiment.py --model minimalist_transformer --seed 20260723
D:\项目\源文件\deploy\.venv-timegnn\Scripts\python.exe experiments\run_external_baselines.py --models frets --seeds 20260723
D:\项目\源文件\deploy\.venv-timegnn\Scripts\python.exe experiments\run_external_baselines.py --models timegnn --seeds 20260723
```

## 全量运行

```powershell
powershell -ExecutionPolicy Bypass -File .\experiments\run_stage_b.ps1
```

该脚本优先使用项目内 `.venv-baselines`；若本地环境尚未装完，会自动检测并复用已部署的
`D:\项目\源文件\deploy\.venv-timegnn` PyTorch 环境。默认先运行仓库内基线，再运行 FreTS/Time-GNN
外部适配器；使用 `-SkipExternal` 可只运行仓库内基线。

默认运行三个随机种子。输出位于 `outputs/experiments/stage_b_30stocks_baselines/`，包括：

- `resolved_config.json`、`environment.json`、`seeds.json`
- `run.log`、`metrics.json`、`predictions.csv`
- LSTM 的 `model.pt` 和 `training_history.json`
- 跨模型的 `baseline_results.csv` 与 `baseline_summary.csv`

所有模型共享 Stage A 的 train/validation/test 样本和指标实现。模型选择只能查看验证集；测试集只用于最终报告。

## 图表

```powershell
D:\项目\源文件\deploy\.venv-timegnn\Scripts\python.exe experiments\plot_baseline_results.py
```

生成的预测对比图、训练曲线和哈希清单位于 `reports/figures/`。在 QRG 治理下，当前原 validation/test
标签均属于已查看的选择暴露 TRAIN，图表不表示独立 SCREENING/FINAL 证据。

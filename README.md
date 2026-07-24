# 基于多模态融合与自适应图学习的金融时序预测系统

本仓库服务于省级大学生创新项目“基于多模态融合与自适应图学习的金融时序预测模型研究”，目标是形成具备可审计数据链路、稳健模型验证、独立推理、风险治理和生产部署能力的金融智能系统。

当前阶段已完成 30 股票研究闭环和一次性未来独立验证，但这代表“研究证据通过”，不等同于已经达到商业生产环境要求。后续建设以可商用水准为最终验收目标，具体门槛见 `plans/COMMERCIALIZATION_ROADMAP.md`。

## 快速预览

1. 双击打开 `dashboard.html`。
2. 用页面中的“真实值 vs 预测值 / 预测误差 / 风险等级”切换按钮查看当前研究界面。
3. 项目汇报可配合 `docs/project_presentation_script.md` 和 `docs/project_technical_note.md`。

## 从 GitHub 部署

```powershell
git clone https://github.com/404LiHua/AI-Finance-Practice-Prototype.git
cd AI-Finance-Practice-Prototype
powershell -ExecutionPolicy Bypass -File .\portable_scripts\setup_all_windows.ps1
```

仅查看离线研究界面时无需安装依赖，直接打开 `dashboard.html` 即可。训练、评估和生产化验证必须使用仓库中的正式流水线。

## 阶段 A：真实数据 Pipeline

仓库已完成 30 只股票的一年期数据准备与验收流程，覆盖：

- 九章量化未复权周线和股票基础信息；
- BaoStock 前复权周线及复权因子；
- CSMAR 特殊处理、上市状态和股本变动事件；
- 交易周历、逐行溯源、时间顺序切分和防泄漏标签；
- 训练集拟合的 TF-IDF、SVD 文本降维和 KMeans 聚类；
- 自动质量门槛和随机森林数据链路基线。

安装并执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\data_pipeline\setup_stage_a.ps1
.\.venv-baostock\Scripts\python.exe .\data_pipeline\run_stage_a.py
```

详细说明见 `data_pipeline/README.md`，公开验收结果见 `reports/STAGE_A_30_STOCKS_REPORT.md`。受许可约束的原始 CSMAR 数据、本地处理数据和模型权重不会提交到 Git。

## 阶段 B：统一基线框架

仓库已建立统一配置、随机种子、训练、评估和日志框架，并在相同的 30 股票切分上实现
Naive、移动平均、ARIMA(1,0,0)、LSTM、单层 Minimalist Transformer、FreTS 与 Time-GNN。
FreTS/Time-GNN 通过独立适配器使用相同样本和评估器。默认运行三个随机种子，统一输出 MSE、MAE、
RMSE、零值安全 MAPE、方向 Accuracy/F1、逐股票指标、预测文件、环境信息和模型权重。

```powershell
D:\项目\源文件\deploy\.venv-timegnn\Scripts\python.exe .\experiments\run_all_baselines.py
```

详细用法见 `experiments/README.md`，正式验收见 `reports/STAGE_B_ACCEPTANCE_REPORT.md`，
完整实验结果见 `reports/STAGE_B_BASELINES_INITIAL_REPORT.md` 与
`reports/BOUNDED_ABLATIONS_TRAIN_RESULT.md`。

## 阶段 C：图频模型、冻结与独立 SCREENING

阶段 C 已完成动态图学习、频域传播、结构消融、稳定化、固定控制组集成、独立加载推理、工程诊断、候选冻结和一次性独立 SCREENING。
开发期推荐模型固定为：

```text
fixed_control_ensemble_v2
= 0.5 × temporal_only_control
+ 0.5 × fixed_temporal_graph_control
```

该模型在开发期 validation 上取得最低平均 MAE，但在 2023-06-09 至 2024-06-07 的 30 股票独立 SCREENING 中，MAE 为 0.042212，高于 Naive 的 0.039879，按预冻结规则判定为 `FAIL`。因此阶段 C 的准确结论是“工程实现完成，独立性能验收未通过”，不得把开发期结果解释为已验证的泛化优势。

完整结论见 `reports/STAGE_C_FINAL_REPORT.md` 和 `reports/STAGE_C_C4_INDEPENDENT_SCREENING_REPORT.md`。

## 阶段 D：新候选与新独立证据

阶段 D 将封存阶段 C 的 1,500 条 SCREENING 样本，不基于逐样本误差继续调参；在 2023-06-02 及以前的数据上使用滚动起点验证研究稳健的新候选，在候选与规则再次冻结后，才申请读取另一段从未使用的未来数据。

D-1 至 D-4 已完成滚动协议、受限比较、唯一候选诊断和冻结。D-5 已在明确授权后对 `2024-06-14—2025-06-13` 的 1,560 条未来样本完成唯一一次独立评估，冻结候选相对 Naive 的 MAE 改善 0.501%，最终判定为 `PASS`；该授权已经消耗，禁止在同一区间重跑或调参。

实施安排见 `plans/STAGE_D_IMPLEMENTATION_PLAN.md`。

## 商用化目标与当前差距

项目最终交付必须达到可商用水准，包括合法数据授权、生产级数据质量、跨市场和跨周期稳健验证、包含交易成本的业务评估、服务可用性与性能 SLA、权限与密钥管理、模型版本治理、漂移监控、回滚和审计、隐私与网络安全以及合规评审。

Stage D 的 `PASS` 只证明冻结候选在固定 30 股票和一个独立年度区间内通过预登记门槛。当前尚未完成大规模股票池验证、实时增量数据服务、成本后业务价值验证、高可用推理服务、安全合规和生产监控，因此不得将当前版本描述为已可直接商用。

## 仓库内容

- `dashboard.html`：离线金融预测与风险解释展示页面。
- `data/`：研究样例、数据契约和本地处理结果目录。
- `docs/`：技术说明、汇报材料和风险解释样例。
- `scripts/`：已经验证的模型运行命令。
- `portable_scripts/`：跨终端部署脚本。
- `source_zips/`：FreTS、Time-GNN、SEP 的开源源码归档，仅用于本项目学习、复现和部署。
- `data_pipeline/`：阶段 A 数据获取、清洗、文本处理、切分、训练与验收代码。
- `experiments/`：阶段 B 统一基线配置、训练、评估、日志和随机种子框架。
- `reports/`：不包含受限原始数据的阶段验收汇总。
- `plans/`：后续阶段的任务拆解、验收标准和时间安排。

## 项目定位

本项目是面向省级创新项目和后续产品化的金融时序智能研发工程，聚焦数据治理、预测建模、独立验证、风险解释和可审计部署。未经完整商用验收、合规审批和业务授权，当前研究版本不用于自动交易，也不构成投资建议。

完整运行 SEP 需要单独配置 API Key 和相应算力。任何密钥均不得提交到本仓库。

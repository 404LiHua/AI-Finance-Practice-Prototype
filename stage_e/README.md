# Stage E：扩展面板与冻结开发协议

本目录实现 `panel_v2 = trade_date × stock_code × feature`、点时股票池、交易日历、复权、公司行动、市值分组、文本事件、数据封存和 rolling-origin 三折协议。

## 冻结边界

- 开发截止日期：`2023-06-02`
- C-4 已使用区间：`2023-06-09` 至 `2024-06-07`
- D-5 已使用区间：`2024-06-14` 至 `2025-06-13`
- 未来 Stage E SCREENING/FINAL：未获取、未读取、未授权
- rolling-origin：固定 `E_RO_01` 至 `E_RO_03`，扩大数据时不移动验证与 purge 边界

## 已构建批次

| 批次 | 日期 | 股票数 | 面板行数 | 状态 |
|---|---:|---:|---:|---|
| `panel_v2_100stocks_3y_20200605_20230602_v1` | 2020-06-05 至 2023-06-02 | 100 | 15,700 | PASS |
| `panel_v2_300stocks_5y_20180608_20230602_v1` | 2018-06-08 至 2023-06-02 | 300 | 78,300 | PASS |

模型价格由 `D:/项目/data/stk_factor` 日频前复权 OHLC 聚合至周频；九章量化未复权周线保留用于审计，未复权收盘价与点时股本用于市值计算。整周休市保留在稠密日历中并标记为 `is_market_open_week=false`，预测目标指向下一实际开市周。

## 复算命令

```powershell
python stage_e/build_expanded_panel.py --config stage_e/configs/panel_v2_100stocks_3y.json
python stage_e/run_development_protocol.py --config stage_e/configs/development_protocol_100stocks_3y_v1.json
python stage_e/verify_development_protocol.py --config stage_e/configs/development_protocol_100stocks_3y_v1.json

python stage_e/build_expanded_panel.py --config stage_e/configs/panel_v2_300stocks_5y.json
python stage_e/run_development_protocol.py --config stage_e/configs/development_protocol_300stocks_5y_v1.json
python stage_e/verify_development_protocol.py --config stage_e/configs/development_protocol_300stocks_5y_v1.json
```

每个批次生成面板、股票池、公司行动账本、原始文件 SHA-256 清单、验证报告和元数据。协议层另外生成三折样本分配、逐折行集合与目标内容哈希、处理批次回执及冻结快照。

## 数据限制

- CSMAR 导出文件标注“仅供嘉兴大学使用”，不得把当前数据许可表述为商业授权，也不得对外分发受限原始数据。
- `stock_basic` 行业是快照分类，不是完整的历史行业变更序列；面板已明确记录该限制。
- E-2 文本模态已接入巨潮正式公告正文；特殊处理和股本事件仍作为结构化事件保留，不替代公告正文。
- 股本表没有公告时间字段，当前按生效日可用；后续若获得公告时间，应升级为 `max(公告日, 生效日)`。

## E-2 文本视图

`build_e2_text_views.py` 构建三种同样本视图：无文本、逐折 TRAIN 拟合的 TF-IDF/SVD、许可登记的本地预训练语义编码。正文必须满足 `data_pipeline/schemas/licensed_text_input_template.csv`，许可登记必须满足 `data_pipeline/schemas/license_registry_template.csv`。

E-2 已完成100股票×3年和300股票×5年两级验收。两级批次均已生成无文本、逐折 TRAIN 拟合的 TF-IDF/SVD、BGE 预训练语义编码三种视图；机器验收结果见 `outputs/stage_e/e2_acceptance_v1.json`，状态为 `PASS / E-2 ACCEPTED`。项目用途为学术研究，不把商业授权作为准入条件，但仍保留来源、使用依据、许可证据和禁止原始 PDF 再分发约束。

## E-3 横截面图结构

E-3 已完成。100/300股票均已生成单位图、固定行业图和26周滚动相关 Top-k 图，邻接矩阵为真实股票节点 `[date,stock,stock]`。自适应图学习器默认使用确定性 Top-k，已通过排列等变、最小过拟合、三个冻结种子和300股票前反向规模检查；Gumbel仅保留为训练消融且评估阶段确定性。统一验收为 `outputs/stage_e/e3_acceptance_v1.json`，完成报告见 `reports/STAGE_E_E3_COMPLETION_REPORT.md`。

## E-4 图频与多模态融合

E-4协议已关闭，但阶段完成门槛未通过。架构、适配、首层10变体30次运行、固定图第二层6变体18次运行、唯一固定图结构9次三种子复核、两个控制组18次复核和分组诊断均已完成。固定图+BGE mid、无图时序和固定行业时域控制的最低预测相关分别为0.44539、0.39088和0.25557，均低于0.90门槛，因此没有稳定降级模型。统一回执为 `outputs/stage_e/e4_closure_acceptance_100stocks_v1.json`，状态 `NOT_COMPLETED_STABILITY_GATE_FAILURE`。300股票图频/文本融合扩展和未来SCREENING均未启动。

控制组关闭复算：

```powershell
python stage_e/run_e4_control_closure.py --config stage_e/configs/e4_control_closure_100stocks_v1.json
python stage_e/accept_e4_closure.py --config stage_e/configs/e4_control_closure_100stocks_v1.json
```

## 后续计划V2

E阶段已按E-4稳定性失败重新安排。后续不直接运行300股票图频/文本融合，而是先执行E-4S稳定性指标审计和单一训练协议V2；同时建设E-5统一基线与评价接口。详细门槛、八周安排和SCREENING前置条件见 `plans/STAGE_E_REVISED_EXECUTION_PLAN_V2.md`。

E-4S.1已完成。独立审计确认样本键、目标、反归一化和原指标正确，预测非恒定；失败来源为训练收敛与VALIDATION最佳checkpoint选择不稳定。审计回执为 `outputs/stage_e/e4_stability_audit_acceptance_v1.json`。训练协议V2已冻结为 `stage_e/configs/e4_training_protocol_v2.json`，在E-4S.2运行前不得修改。

E-4S.2已完成。两个控制组使用全部TRAIN截面、固定40 epoch和最终EMA后，最低预测相关提高到0.76792和0.73864，但仍低于0.90门槛；门槛A失败。机器回执为 `outputs/stage_e/e4s2_control_stability_acceptance_v2.json`。E-4已关闭为负结果，不运行E-4S.3或300股票图频/文本融合；下一阶段进入E-5统一接口与基线评价。

## E-5统一接口

E-5.1已完成。统一折数据视图、模型适配抽象接口、预测文件契约、独立评价器以及逐股票/行业/市值/收益/文本诊断均已实现。无训练机器回执为 `outputs/stage_e/e5_interface_acceptance_v1.json`，结果 `PASS`。下一节点为E-5.2第一批低成本基线。

## E-5.2 第一批低成本基线

六个冻结基线已完成三折单种子工程回执：Naive、Stage D FreTS L4固定收缩、价格版Minimalist Transformer、RF、RBF-SVM和行业VAR(1)+ridge。18次运行全部通过，所有模型均可从落盘checkpoint独立加载并复现预测，最大差为0；机器验收见 `outputs/stage_e/e5_low_cost_single_seed_acceptance_v1.json`。

复算命令：

```powershell
.venv-text\Scripts\python.exe -B stage_e\run_e5_low_cost_single_seed.py --config stage_e\configs\e5_low_cost_baselines_v1.json
.venv-text\Scripts\python.exe -B stage_e\accept_e5_low_cost_single_seed.py --config stage_e\configs\e5_low_cost_baselines_v1.json
```

该节点只允许单种子工程暴露，不允许排名、删模、候选晋级或SCREENING访问。详细报告见 `reports/STAGE_E_E5_LOW_COST_SINGLE_SEED_REPORT.md`。

## E-5.2 三种子统一复核

已在不修改第一批冻结协议的条件下补运行 `20260723/20260724`，并复用原种子 `20260725`。三折、六模型、三种子共54份回执全部通过；机器验收见 `outputs/stage_e/e5_low_cost_three_seed_acceptance_v1.json`，详细报告见 `reports/STAGE_E_E5_LOW_COST_THREE_SEED_REPORT.md`。

复算与验收：

```powershell
.venv-text\Scripts\python.exe -B stage_e\run_e5_low_cost_three_seed_review.py --config stage_e\configs\e5_low_cost_three_seed_review_v1.json
.venv-text\Scripts\python.exe -B stage_e\accept_e5_low_cost_three_seed_review.py --config stage_e\configs\e5_low_cost_three_seed_review_v1.json
```

默认复算会校验并复用两个已完成的新种子批次，不会重新训练。下一节点为E-5.3第二批神经网络与真实股票节点图基线的运行前冻结。

## E-5.3 神经网络与真实股票节点图基线

E-5.3已冻结LSTM、TCN、确定性Top-k Time-GNN和固定行业图Graph-WaveNet类基线。Time-GNN登记MIT许可和上游SHA，并移除随机Gumbel采样；股票图固定使用100个真实 `stock_code` 节点。三折单种子共12次运行全部通过，机器验收见 `outputs/stage_e/e5_neural_graph_single_seed_acceptance_v1.json`。

运行与验收：

```powershell
.venv-text\Scripts\python.exe -B stage_e\run_e5_neural_graph_single_seed.py --config stage_e\configs\e5_neural_graph_baselines_v1.json
.venv-text\Scripts\python.exe -B stage_e\accept_e5_neural_graph_single_seed.py --config stage_e\configs\e5_neural_graph_baselines_v1.json
```

详细报告见 `reports/STAGE_E_E5_NEURAL_GRAPH_SINGLE_SEED_REPORT.md`。下一节点为保持协议不变补运行另外两个种子。

## E-5.3 三种子统一复核

已保持四模型协议不变补运行 `20260723/20260724`，并复用 `20260725`。三折×四模型×三种子共36份回执全部通过；Time-GNN确定性Top-k和真实股票节点固定行业图在所有种子与折中保持冻结。

```powershell
.venv-text\Scripts\python.exe -B stage_e\run_e5_neural_graph_three_seed_review.py --config stage_e\configs\e5_neural_graph_three_seed_review_v1.json
.venv-text\Scripts\python.exe -B stage_e\accept_e5_neural_graph_three_seed_review.py --config stage_e\configs\e5_neural_graph_three_seed_review_v1.json
```

机器验收见 `outputs/stage_e/e5_neural_graph_three_seed_acceptance_v1.json`，详细报告见 `reports/STAGE_E_E5_NEURAL_GRAPH_THREE_SEED_REPORT.md`。下一节点为E-5.4统一跨批次诊断。

## E-5.4 十模型统一诊断

E-5.4已冻结并执行只读统一诊断，合并E-5.2与E-5.3共10个模型、三折、三种子的45,000行预测。已生成工程成本、最差折、逐股票、行业、市值、收益十分位、种子离散度和45组模型/组件分歧；没有重新训练或写checkpoint。

```powershell
.venv-text\Scripts\python.exe -B stage_e\run_e5_unified_diagnostics.py --config stage_e\configs\e5_unified_diagnostics_v1.json
.venv-text\Scripts\python.exe -B stage_e\accept_e5_unified_diagnostics.py --config stage_e\configs\e5_unified_diagnostics_v1.json
```

机器验收见 `outputs/stage_e/e5_unified_diagnostics_acceptance_v1.json`，详细报告见 `reports/STAGE_E_E5_UNIFIED_DIAGNOSTICS_REPORT.md`。下一节点为E-6.1候选门槛冻结，门槛冻结前不得生成候选排序。

## E-6.1 候选门槛冻结

E-6.1已在候选指标读取和排序前冻结总体改善、最差折、逐股票、行业、市值、收益双尾、三种子稳定性、工程成本和三种子等权聚合门槛。冻结过程只读取Naive/FreTS基线锚点，没有计算任何候选模型资格。

```powershell
.venv-text\Scripts\python.exe -B stage_e\freeze_e6_candidate_gate.py --config stage_e\configs\e6_candidate_gate_v1.json
```

冻结回执见 `outputs/stage_e/e6_candidate_gate_freeze_receipt_v1.json`，详细报告见 `reports/STAGE_E_E6_CANDIDATE_GATE_FREEZE_REPORT.md`。下一节点为按冻结门槛一次性审查现有8个候选模型。

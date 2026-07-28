# 基于多模态融合与自适应图学习的金融时序预测系统

本仓库服务于省级大学生创新项目“基于多模态融合与自适应图学习的金融时序预测模型研究”。项目已形成从可追溯数据、点时股票池、多模态特征、rolling-origin训练，到统一诊断、候选冻结和独立推理的完整研究工程。

> 当前版本是高标准研究与产品化原型，不是自动交易系统，不承诺收益，也不构成投资建议。阶段F已正式关闭且没有产生新的可晋级模型；保留阶段E模型不等同于生产部署批准，新的SCREENING和FINAL仍未授权。

## 当前结论

E阶段已正式收尾。预先冻结34项硬门槛后，仅一次读取8个非基线模型既有诊断，最终形成唯一候选：

```text
stock_node_gwnet_fixed_industry_l8
```

该模型使用100只真实股票作为横截面节点、固定同行业图、8周价格数值窗口和Graph-WaveNet类传播结构。

| 指标 | 冻结结果 |
|---|---:|
| 总体 MAE | 0.0316216251 |
| 相对 Naive MAE 改善 | 1.963% |
| 相对 FreTS L4 MAE 改善 | 0.895% |
| 总体 RMSE | 0.0462010679 |
| 最差折 MAE | 0.0353079508 |
| 优于 Naive 的股票数 | 78 / 100 |
| 三种子最小 Pearson | 0.844276 |
| 参数量 | 15,777 |
| 硬门槛 | 34 / 34 PASS |

三种子推理固定使用 `20260723/20260724/20260725` 算术平均，不允许事后选择最佳种子。

## 快速查看

直接打开 [dashboard.html](dashboard.html) 可离线查看E阶段进度、8个候选模型对比、门槛结论和产品化差距，无需安装依赖。

![E阶段研究看板](docs/assets/e_stage_dashboard_preview.png)

最佳模型独立包位于 [releases/e_stage_best_model_v1](releases/e_stage_best_model_v1)，包含三折×三种子共9个检查点、独立推理入口、股票顺序、特征契约、来源证明及SHA-256清单。

```powershell
cd releases\e_stage_best_model_v1
python verify_package.py
python inference.py --input adapter_input.npz --fold E_RO_03 --output predictions.npz
```

## 阶段进展

- 阶段A：30股票真实数据流水线、交易日历、文本处理、切分与验收；
- 阶段B：统一训练/评估框架，Minimalist Transformer、FreTS、Time-GNN等基线与消融；
- 阶段C：图频模型、稳定化、独立加载和首次SCREENING，泛化门槛未通过；
- 阶段D：固定三折rolling-origin和新候选，一次性D-SCREENING通过；
- 阶段E：`panel_v2`、100/300股票扩展、新闻/公告三文本视图、图频文本消融、10模型统一诊断和唯一候选审查，已关闭。
- 阶段F：已正式关闭。四个鲁棒性候选分别通过12/20、13/20、13/20和11/20项硬门槛，均不可晋级；GAN四项稳定性失败不可补偿，正式保留阶段E固定行业图模型并完成源码、证据和SHA-256归档。

详细总结见 [项目截至目前工作总结](reports/PROJECT_WORK_SUMMARY_TO_DATE.md)，当前唯一权威总计划见 [项目总体实施计划V3](plans/PROJECT_MASTER_PLAN_V3.md)，E阶段结论见 [阶段E最终报告](reports/STAGE_E_FINAL_REPORT.md) 和 [最佳模型结论](reports/STAGE_E_BEST_MODEL_CONCLUSION.md)。

## 核心目录

- `data_pipeline/`：点时面板、交易日历、公司行动、文本和自动化数据流水线；
- `experiments/`：阶段B统一基线、适配器、训练与评估框架；
- `stage_c/`、`stage_d/`、`stage_e/`、`stage_f/`：分阶段协议、模型、验收器与冻结配置；
- `reports/`：阶段报告、诊断结论和工作总结；
- `plans/`：项目总体实施计划、阶段计划和商业化差距路线；
- `releases/e_stage_best_model_v1/`：E阶段唯一候选独立交付包；
- `releases/stage_f_closure_v1/`：F阶段正式负面结论、必要源码、紧凑证据与SHA-256冻结包；
- `dashboard.html`：离线图形化研究看板。

原始受限数据、处理后大文件和本地运行输出默认不提交Git；公开仓库保留可复现代码、配置、报告、必要小型模型包与哈希证据。

## F-2 GAN附录状态

F-2.0至F-3已完成唯一GAN候选`stock_node_gwnet_bounded_cwgan_gp_l8`的协议冻结、训练健康、真实三折、三种子复核、统一鲁棒性诊断和正式归档。9次工程运行全部通过，但最低Pearson 0.3212、Spearman 0.1840及两项预测方差硬失败；统一诊断仅通过11/20项门槛。因此GAN正式不可晋级，阶段F继续保留`stock_node_gwnet_fixed_industry_l8`。详见[阶段F最终报告](reports/STAGE_F_FINAL_REPORT.md)。

## 从GitHub部署

```powershell
git clone https://github.com/404LiHua/AI-Finance-Practice-Prototype.git
cd AI-Finance-Practice-Prototype
powershell -ExecutionPolicy Bypass -File .\portable_scripts\setup_all_windows.ps1
```

完整生产化仍需跨市场/跨周期独立验证、交易成本与容量评估、实时数据SLA、在线推理服务、漂移监控、权限审计、安全合规、灰度发布和回滚机制。

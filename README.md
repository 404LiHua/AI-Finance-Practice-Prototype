# AI Finance Practice Prototype

基于多模态融合与自适应图学习的金融时序预测研究与单机工程原型。

本仓库同时保存早期阶段 A–F 的可追溯研究材料，以及当前 Stage G 的 T2 生产重建、点时（PIT）治理、快速历史指标和单机服务交付。它是研究与工程验证仓库，不是自动交易系统，不构成投资建议，也不承诺收益。

## 当前状态（Stage G）

- 生产内核仍为 `RG_OBGNET_CONFIRMED_SAFE_V1_1`，目标为 `T2_MARKET_RELATIVE_FIXED`。
- T2 单机 loopback 服务已经完成本地验收；生产边界、标签保管和输入质量合同仍然有效。
- WP12 已提供 6 个历史滚动 OOS 折、分类/概率/校准/排序指标和逐周审计。C0 候选只保留为研究或 shadow：MCC 略有改善，但 IC、Brier、宏 F1、校准和 top-bottom spread 未全面优于 incumbent，不能替换生产内核。
- 2026-08-15 增量快照已补充 WP22/WP23 治理、单机复验和候选交接材料；WP23 修复了市场尺度物化器的 positional `usecols` 绑定错误，并重新封存五个未来窗口的 label-free shadow 输入/预测。修复版脚本已同步到 Stage G 规范快照。
- 同一增量快照现补充 WP24/WP25：WP24 共享主干路线因 H4/T2 联合门失败而关闭；WP25 生产锚定低幅残差路线通过开发联合门并完成 CPU 回放，但仍为研究 shadow，未替换生产 T2 内核。
- V4 独立 FRESH 窗口已预注册，但在最后一个 H4 标签于 2026-09-11 结算并完成数据冻结前，不得物化、评分或晋级。

## 从哪里开始

### Stage G 当前交付

从 [stage_g/README.md](stage_g/README.md) 开始。公开快照由稳定基线和增量交接包组成：
`stage_g/project_snapshot_20260814/aagfm_next_stage_v1/` 与
`stage_g/project_snapshot_20260815/aagfm_next_stage_v1/`。其中包含：

- 工作包、数据与 PIT 合同、治理收据；
- WP12 快速历史指标脚本与审计结果；
- T2 单机推理服务、loopback/soak 验收脚本；
- 经过筛选的证据和哈希清单；
- WP23 列绑定修复审计及其对应的五窗口 label-free shadow 封存记录。
- WP24/WP25 协议、开发评价、CPU 回放决定和公开导出边界说明。

### 本地单机运行

单机服务说明见 [Stage G local service README](stage_g/project_snapshot_20260814/aagfm_next_stage_v1/local_service/README.md)。旧阶段的独立部署脚本仍保留在 [portable_scripts](portable_scripts/)，但不能据此推断 Stage G 已获线上部署批准。

### 历史研究阶段

- `data_pipeline/`：数据流水线与输入合同；
- `experiments/`：统一基线、消融和评估框架；
- `stage_c/`、`stage_d/`、`stage_e/`、`stage_f/`：历史研究阶段及其关闭证据；
- `reports/`、`plans/`：历史报告、计划和商业化差距材料；
- `releases/stage_f_closure_v1/`：阶段 F 的正式负面结论与冻结证据。

这些目录是研究历史，不代表当前生产模型或新的晋级授权。

## 可复现性与数据边界

仓库不提交原始行情、受限基本面/文本数据、FRESH/SCREENING/FINAL 标签、密封预测、shadow 表、模型权重或临时运行输出。公开快照只保留可审计的代码、协议、报告、必要的小型示例和哈希证据；完整范围见 [Stage G export scope](stage_g/project_snapshot_20260814/aagfm_next_stage_v1/EXPORT_SCOPE_20260814.md)。

任何候选只有在独立窗口、预测绑定、一次性评分和晋级门槛全部通过后，才可能进入生产讨论；连续收益头未经严格验证不能替代当前 T2 生产内核。

## 克隆

```powershell
git clone https://github.com/404LiHua/AI-Finance-Practice-Prototype.git
cd AI-Finance-Practice-Prototype
```

安装和运行具体阶段前，请先阅读对应目录的 README、数据许可说明和治理合同。联网数据、模型训练和线上部署不由本仓库自动触发。

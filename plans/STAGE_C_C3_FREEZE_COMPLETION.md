# 阶段 C-3 冻结完成与 C-4 准入状态

更新日期：2026-07-24

## 当前状态

阶段 C-3 已完成，冻结编号为 `stage_c_recommended_v2_c3_20260724`。

- 推荐模型结构、0.5/0.5 权重、30 股票范围、价格特征、8 周序列和三个种子已冻结。
- Naive、FreTS L4、价格版 Minimalist Transformer、仅时域控制、固定时间图控制和图频 v1 已冻结为比较基线。
- 指标优先级、通过门槛、性能失败条件、完整性硬失败条件和不确定区间已冻结。
- SHA-256 清单覆盖 174 个工件，根摘要为 `4e0af90936d06f0a16999e86d5c7ff1e70466502175b6c0d52f789a028041c47`。
- 只读复核已通过，未来 SCREENING 数据仍为未获取、未读取、未使用。

## C-4 准入门

C-4 尚未启动。只有获得用户明确授权后，才可获取未来 SCREENING 数据并一次性执行候选与冻结基线。运行后只允许记录 `PASS`、`FAIL`、`INCONCLUSIVE` 或 `INVALID_INTEGRITY_FAILURE`，不得在同一 SCREENING 数据上重新训练、调权、选种子、改指标或改门槛。

在授权前允许的操作仅限于：

- 只读执行 SHA-256 冻结复核。
- 整理版本、说明文档和归档包。
- 检查 SCREENING 执行脚本是否严格读取冻结配置，但不得连接或读取未来数据。

冻结规则详见 `stage_c/configs/recommended_v2_freeze_c3.json`，完整说明详见 `reports/STAGE_C_CANDIDATE_AND_RULE_FREEZE.md`。

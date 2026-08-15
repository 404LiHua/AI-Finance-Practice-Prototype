# WP13：V4 未来锚点合格 FRESH 预检

状态：`IMPLEMENTED_AND_SYNTHETICALLY_VERIFIED / NOT_YET_MATERIALIZABLE`

WP13 为预注册 V4 提供独立于旧 V1/V3 的输入构建和无标签合同审计。它不训练候选、不运行推断、不打开真实标签，也不修改生产注册表。

## 新增工具

- `scripts/build_csn_future_anchor_eligible_fresh_v4_inputs_v1.py`：唯一允许的 V4 输入物化器；固定 8 个周一 09:30 origin、严格使用前一交易日锚点和 origin 后第 4 个交易日（正常周五）目标。要求数据交付证明、逐日线 SHA-256、完整覆盖以及 2026-09-11 后的物化日期。
- `scripts/audit_csn_future_anchor_eligible_fresh_v4_input_contract_v1.py`：没有标签路径参数；断言 8 个预注册周一、8×8×N×6、数值/技术/基本面/股票池等键域、候选不可变身份和生产锚点训练边界。
- `scripts/test_csn_future_anchor_eligible_fresh_v4_contract_v1.py`：CPU-only 合成回归测试；以历史特征侧夹具改名，不打开标签文件；额外证明 H4 从周一到周五，并验证 2026-08-15 会被构建器 fail-closed 拒绝。
- `scripts/test_csn_future_anchor_eligible_fresh_v4_materializer_v1.py`：全链路临时合成数据测试；实际生成 `8×8×200×6`、密封合成标签和日线快照，再运行无标签审计；测试不打开生成的标签文件。

## 执行限制

真实 V4 物化仍须等到 2026-09-11 收盘、数据交付冻结及证明文件填写完毕。届时先运行构建器和无标签审计；候选 GPU 推断与生产锚点 CPU/I/O 推断不得并发。该工作包不授予读取密封标签、评分或晋级的权限。

## 证据

最终合成测试收据：

- `audits/csn_future_anchor_eligible_fresh_v4_synthetic_contract_test_v3/SYNTHETIC_CONTRACT_TEST_SUMMARY.json`；
- `audits/csn_future_anchor_eligible_fresh_v4_synthetic_materializer_test_v1/SYNTHETIC_MATERIALIZER_TEST_SUMMARY.json`。


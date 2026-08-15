# 单机 T2 推理冒烟验收（2026-08-14）

**状态**：`PASS_LOCAL_CPU_INFERENCE`  
**active 模型**：`RG_OBGNET_CONFIRMED_SAFE_V1_1`  
**候选 C0**：开发门通过、FRESH 前不可注册。

## 实测

- 输入：81,432 行已恢复 RG3 特征，仅含历史技术/结构特征，不含 FRESH。
- 输出：81,432 行，字段包括 T2 类别、DOWN/NEUTRAL/UP 三类概率、confidence、distribution_support、reliability、gated_ordinal_score。
- 最大概率和误差：`2.22e-16`。
- 概率与可靠性相关输出无非有限值。
- reliability 区间：`[0.01337, 0.28376]`。
- CPU-only；无 GPU、无 FRESH、无生产模型/注册表修改。
- 另以 840 行等间隔 RG3 样本将本地 JSON 实现与受控源码
  `ConfirmedSafeRGOBGNet` 逐项对照：三类概率、confidence、distribution support、reliability、gated ordinal score 的最大绝对差均为 `0.0`。

## 固化证据

- `PRODUCTION_T2_LOCAL_MODEL_REGISTRY_V1.json`
- `run_t2_local_inference_v1.py`
- `smoke/production_t2_inference_rg3_reconstruction.parquet`
- `smoke/production_t2_inference_receipt.json`

本工具为个人单机批量运行层，不启动网络 listener。线上服务、自动交易和注册表切换必须在 FRESH 结论、纸面交易与独立审批后另行进行。



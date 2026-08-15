# WP15：V4 标签无关端到端运行时演练

状态：`IMPLEMENTED_AND_VERIFIED_ON_SYNTHETIC_INPUTS / NOT_A_REAL_V4_RESULT / PRODUCTION_PROMOTION_REVOKED_BY_WP16`

WP15 在不创建、读取或评分任何标签的前提下，实际加载冻结候选和生产锚点，验证 V4 未来运行链的设备、接口、键域和资源收据。

## 验证内容

- 8 个预注册 origin；基础运行时演练使用 2 个合成股票，完整绑定演练使用 200 个合成股票；
- 候选预测器真实加载冻结候选清单中的模型，在 CUDA 上以 batch 1024、最多 1 个 GPU 作业运行；
- 候选完成后，锚点预测器以 CPU `workers=1` 串行运行，不与候选 GPU 推理并发；
- 两份输出键域唯一，三类概率和为 1；
- 完整绑定演练把两份**真实预测器**输出送入 WP14 绑定审计，验证输入哈希、模型身份、键域和资源收据可共同通过；
- 两份收据明确 `labels_read=false`，测试临时目录不产生标签文件；
- 生产注册表和生产内核不被修改。

## 运行方式

```powershell
& C:\Users\27793\Documents\deep\cuda-venv\Scripts\python.exe `
  scripts\test_csn_v4_label_free_e2e_runtime_v1.py `
  --candidate-root candidate_freeze\csn_residual_full_development_v1 `
  --anchor-model C:\Users\27793\Documents\project1\deliverables\RG_OBGNet_source_v1\models\rg_obgnet_confirmed_safe_v1_1\MODEL.json `
  --candidate-predictor scripts\predict_csn_candidate_fresh_label_free_batch_v1.py `
  --anchor-predictor scripts\predict_confirmed_anchor_fresh_label_free_v1.py `
  --binding-audit scripts\audit_csn_future_anchor_eligible_fresh_v4_preconsumption_binding_v1.py `
  --stock-count 200 `
  --output-root audits\csn_v4_label_free_e2e_runtime_smoke_test_v1
```

绑定模式强制股票数为 200–300，以匹配 WP14 的键域要求；它使用合成的“密封标签 SHA”占位符，但不创建该文件。该演练只证明运行时链路可执行，不证明候选优于锚点，也不授权真实 V4 物化、评分或晋级。真实运行仍必须遵守 WP13/WP14 的日期、PIT、标签封存和一次性授权门控。

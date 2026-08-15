# WP14：V4 预测封存前绑定

状态：`IMPLEMENTED_AND_SYNTHETICALLY_VERIFIED / PRODUCTION_PROMOTION_REVOKED_BY_WP16`

注意：WP14 只证明无标签绑定器的接口和资源收据逻辑；它不证明当前四交易日绝对 H4 候选与四周市场相对 T2 生产锚点可比。未完成 WP16 语义恢复前，不得对真实 V4 读取标签、评分或晋级。

WP14 连接 V4 的标签无关输入审计与未来的一次性评分授权。它不是评分器：接口不接收标签文件，不能读取标签值、收益或指标，不能训练模型或修改生产内核。

## 不可变绑定对象

- 8 个精确预注册周一 origin；
- 完全相同的 `origin_date, stock_code` 键域；
- 冻结候选清单 SHA `c7ee...6908e` 与规格 SHA `49b4...6741`；
- 当前锚点 `RG_OBGNET_CONFIRMED_SAFE_V1_1` 及模型 SHA；
- V2 输入审计、密封物化收据、候选/锚点预测和两份预测收据；
- 只从物化收据取得密封标签 SHA，不打开标签文件。

## 新增工具

- `scripts/audit_csn_future_anchor_eligible_fresh_v4_input_contract_v2.py`：在 V1 不可变基础上补充输入与物化收据哈希交叉绑定；
- `scripts/audit_csn_future_anchor_eligible_fresh_v4_preconsumption_binding_v1.py`：精确 V4 预测绑定器，附带 GPU 候选（最多 1 作业）与 CPU 锚点（1–2 worker）资源收据检查；
- `scripts/test_csn_future_anchor_eligible_fresh_v4_binding_v1.py`：200 股、8 周的 CPU-only 合成预测绑定测试，不创建标签文件。

## 运行顺序

真实数据到齐后，仅可按如下顺序运行：V4 构建器 → V2 输入审计 → 候选 GPU 无标签预测 → 锚点 CPU 无标签预测（不得并发）→ WP14 绑定审计 → 独立保管人签发新的一次性评分授权。任何检查失败时，不得读取标签或晋级。

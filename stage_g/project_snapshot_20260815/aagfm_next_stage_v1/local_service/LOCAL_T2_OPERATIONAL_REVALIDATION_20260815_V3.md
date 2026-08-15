# 本机 T2 服务重验证 V3（2026-08-15）

V3 修复并锁定 active confirmed 服务的身份语义：`health.research_only=false`、`health.active_production=true`；模型和预测输出显式标记 `target_id=T2_MARKET_RELATIVE_FIXED`、`target_horizon=T2_4W`。研究候选仍保持 research-only、不可晋级。

回环验收通过健康、模型、预测、概率和、缓存 MISS/HIT、非法股票 400、未来日期 400、目标契约、服务关闭和 GPU 禁用检查。未读取标签/FRESH/WP10，未训练，未修改 active 模型或注册表。

# E阶段最佳模型独立交付包

模型：`stock_node_gwnet_fixed_industry_l8`

这是E-6一次性冻结硬门槛审查形成的唯一候选。模型使用真实股票代码作为节点、固定同行业邻接图、8周价格数值窗口和Graph-WaveNet类时域图传播结构。它通过全部34项硬门槛；该结论是研究开发候选结论，不等同于生产部署批准或投资建议。

目录包含三折、三种子的9个检查点。每折推理固定使用种子 `20260723/20260724/20260725` 的算术平均，不允许事后选择最佳种子。

输入NPZ必须包含 `values`，形状为 `[batch, 8, 100, 6]`，特征与股票顺序分别见 `feature_schema.json` 和 `stock_order.json`。可选的 `stock_code` 字段存在时会被严格校验。

```powershell
python verify_package.py
python inference.py --input adapter_input.npz --fold E_RO_03 --output predictions.npz
```

`manifest.json`记录全部文件的SHA-256；`provenance.json`记录E-6结论、检查点来源与独立加载证据；`model_metrics.json`只包含已封存的最终指标。

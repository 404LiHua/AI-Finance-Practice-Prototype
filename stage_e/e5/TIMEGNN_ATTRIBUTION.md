# Time-GNN attribution for the E-5.3 stable adapter

The deterministic E-5.3 `timegnn_stable` adapter is an independent pure-PyTorch adaptation of architectural ideas from:

> Xu, Nancy; Kosma, Chrysoula; Vazirgiannis, Michalis. “TimeGNN: Temporal Dynamic Graph Learning for Time Series Forecasting”, Complex Networks, 2023.

Upstream repository snapshot: `source_zips/Time-GNN-main.zip`  
Upstream model file: `models/TimeGNN.py`  
Upstream license: MIT, copyright (c) 2023 xun468.

Frozen upstream hashes:

- ZIP SHA-256: `6ae4188f39e71df584f90f10c8a4d59e60ac59b7f14acf94ffc1f5f742bff7a2`
- Upstream `TimeGNN.py` SHA-256: `f09717e7c37dfb8865b7f08ab45bff08d83c354c13bc002de68f32980468d65b`
- Upstream `LICENSE` SHA-256: `b3322d79391ac2d968d97c63c3a244cd013b074cca8a213608bc954c702a79a2`

Stability adaptation differences are fixed before training: hard Gumbel sampling is removed; graph edges use deterministic temperature-scaled Top-k weights; graph propagation is implemented with dense PyTorch operations so PyG is not required. This adapter remains a temporal-position graph baseline. The separate `stock_node_gwnet_fixed_industry` model is the real-stock-node graph control.

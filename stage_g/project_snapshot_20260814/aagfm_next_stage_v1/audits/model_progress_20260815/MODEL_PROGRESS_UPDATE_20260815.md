# AA-GFMNET 模型进度更新（2026-08-15）

## 当前结论

- 生产内核仍为 `RG_OBGNET_CONFIRMED_SAFE_V1_1`，生产 T2 仍输出三类概率、分类、可靠性和排序分数。
- WP10 一次性 FRESH 已消费，但只允许得出“保留 incumbent”；没有生产替换授权。
- C1（RG2 状态/图摘要增强）是当前形式上通过开发门的最佳正式候选；C2 因最差折 MCC 门失败关闭；C3 仅为事后探索，不具备正式选择资格。
- T7 的动量变换和模型元数据与生产 T2 对齐，但授权模板误写为 H4 范围，已被审计 fail-closed；训练源和独立一次性授权也仍缺失，因此没有训练结果或增益声明。

## T7 语义审计

已核对生产模型 SHA、目标 `T2_MARKET_RELATIVE_FIXED`、`origin_index + 4`、±1% 阈值、14 项特征、5,513 键映射和 300 股正式面板。唯一新增操作是对前三个动量特征应用 `sign(x) * log1p(abs(x))`；但授权范围仍是 `ONE_CANDIDATE_TRAIN_ONLY_H4_ONCE`，与生产 T2 不一致。此节点不读取标签、不训练、不改生产资产。

## 计算环境

共享环境 `C:\Users\27793\Documents\deep\cuda-venv\Scripts\python.exe` 中 PyTorch CUDA 与 XGBoost CUDA 小型冒烟均通过；CuPy 包存在但找不到 CUDA headers，因此不宣称全程 GPU 推理等价。调度继续采用 CPU 做 I/O/哈希/PIT/指标，GPU 一次只跑一个重任务。

## 下一硬门

先把授权范围改为明确的 `ONE_CANDIDATE_TRAIN_ONLY_T2_ONCE`，再获得独立保管者签发的一次性授权，并交付与 5,513 键集绑定的训练源及输入清单 SHA。未满足前不得训练、读标签、调参或修改生产内核。

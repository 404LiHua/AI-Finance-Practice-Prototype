# 阶段F：F-1.1有界非GAN候选实现与最小测试报告

日期：2026-07-28  
状态：`PASS / NEXT F-1.2 SINGLE-SEED ENGINEERING`

## 1. 实现范围

本节点严格复用阶段E的`stock_node_gwnet_fixed_industry_l8`网络、固定行业邻接和L8数值窗口，没有改变模型层数、图传播、输入特征或推理行为。新增实现位于`stage_f/robustness.py`，只包含F-0允许的三个训练期变化：

1. `stock_node_gwnet_tail_weighted_l8`：仅使用TRAIN有效原始目标绝对值拟合Q90；尾部样本在masked Huber中权重固定为2；
2. `stock_node_gwnet_noise_aug_l8`：仅在TRAIN数值张量施加按运行种子确定的高斯噪声，sigma固定为0.03；
3. `stock_node_gwnet_feature_mask_l8`：仅在TRAIN数值张量施加按运行种子确定的逐元素掩码，概率固定为0.05。

VALIDATION和推理输入不施加噪声或掩码，尾部阈值禁止使用VALIDATION重新拟合。

## 2. 最小测试结果

| 检查 | 结果 | 验收内容 |
|---|---|---|
| 确定性 | PASS | 同种子连续增强序列逐元素一致；推理输入逐位不变 |
| 张量形状 | PASS | 输入保持`batch × time × stock × feature`，输出保持`batch × stock` |
| 最小过拟合 | PASS | 三个候选均把合成全批次Huber损失降至初始值35%以下 |
| 数据边界 | PASS | 拒绝VALIDATION阈值拟合、C-4、D-5、SCREENING、FINAL及截止日后日期 |

测试共4/4通过。测试数据为4股票、8周窗口的纯合成小样本，只验证工程连通性，不构成正式折训练或候选性能证据。

## 3. 哈希证据

- F-0主协议：`a25cb0c9074623cd6acfbed79d65bdfa508572f8bd74ee1721aa6ade046a566e`
- F-1.1实现：`8f196a4deb3d709fe81dcb85d91c363dd138bf9f208de61430fff28e7fb30dd4`
- F-1.1测试：`806200981bba4716ef75902b09c54ff1af95e197f6bb5c9f877b24535722763d`
- F-1.1实施契约：`b0cf9e8ed7ab0134ccdea446913df5bdb327a2c81c986c24cd72d69daba478b5`
- F-1.1验收回执：`85df91b0d9457117b58b53992de478b6fc3ae1fdf38663e2b4e8d0f2b302a6b1`

## 4. 数据与训练声明

- 正式三折训练：未执行；
- 候选指标读取和排序：未执行；
- 失败模型或种子删除：未执行；
- GAN训练：未执行且未授权；
- SCREENING/FINAL：未访问且未授权。

## 5. 下一节点

进入`F-1.2 单种子工程回执`。只允许使用种子`20260725`在冻结三折上运行上述三个候选，检查损失有限性、冻结样本键、检查点独立加载和正常/压力推理入口。本节点不得排名、删模或形成晋级建议；通过后才能申请补运行`20260723/20260724`。

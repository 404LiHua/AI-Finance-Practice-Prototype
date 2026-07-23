# 阶段 A：30 只股票周线数据 Pipeline

本目录完成项目申报书第一阶段的数据准备工作：真实数据溯源、交易周历处理、复权价格、股票基础信息、CSMAR 事件文本、历史特殊处理状态、时间顺序切分、文本降维聚类和自动质量验收。

## 当前实验范围

- 股票池：30 只当前仍上市、代码排序靠前且具有完整历史的深市 A 股。
- 日期：2022-06-03 至 2023-06-02。
- 输入窗口：12 周。
- 预测目标：下一周前复权收盘价、收益率和涨跌方向。
- 切分：70% 训练、10% 验证、20% 测试，集合间清除 1 周。

股票池、路径和全部参数统一存放在 `configs/weekly_a_share.json`。

## 数据来源

1. 九章量化：未复权周线行情和股票基础信息。
2. BaoStock：前复权周线、换手率、收益率和复权因子。
3. CSMAR：公司信息、特殊处理变动、上市状态变动和股本结构变动。

原始数据不会被修改。所有源文件均记录 SHA-256；处理后的行保留源文件、哈希和原始行号。CSMAR 文件标注“仅供嘉兴大学使用”，因此原始文件和生成数据均被 Git 忽略。

## 环境安装

```powershell
powershell -ExecutionPolicy Bypass -File .\data_pipeline\setup_stage_a.ps1
```

依赖安装在项目独立环境 `.venv-baostock` 中。

## 一键执行阶段 A

```powershell
.\.venv-baostock\Scripts\python.exe .\data_pipeline\run_stage_a.py
```

执行顺序：

1. 运行单元测试；
2. 读取并标准化 CSMAR ZIP；
3. 从 BaoStock 下载 30 只股票前复权数据；
4. 构建周历、特征、标签及训练/验证/测试集；
5. 仅使用训练集文本拟合 TF-IDF、SVD 和 KMeans；
6. 训练随机森林数据链路基线；
7. 执行质量门槛并生成阶段 A 报告。

若 BaoStock 数据已经下载，可增加 `--skip-download`。

## 核心输出

本地生成目录：`data/processed/weekly_30stocks_stage_a_v1/`。

- `source_manifest.csv.gz`：5000 个九章源文件的哈希和日期范围。
- `stock_basic.csv.gz`：标准化股票基础信息。
- `weekly_calendar.csv.gz`：标准交易周历和横截面覆盖率。
- `panel.csv.gz`：完整长表和逐行溯源字段。
- `train.csv.gz`、`validation.csv.gz`、`test.csv.gz`：无标签跨界的样本。
- `text_features.csv.gz`：训练集拟合的文本 SVD 和聚类结果。
- `metadata.json`、`text_features_metadata.json`：构建与文本特征元数据。

可提交的验收结果位于 `reports/`。

## 防止数据泄漏

- 时间切分不随机打乱。
- 移动平均和波动率只使用当期及历史数据。
- 预测标签不得跨越 train/validation/test 边界。
- 验证和测试可读取边界前的历史输入，但不能读取未来数据。
- TF-IDF、SVD 和文本聚类仅在训练期的非空文本上拟合。

## 已知限制

- 当前 CSMAR 导出属于特殊处理类，而不是覆盖全部股票的日常财经新闻库。
- 正常股票在一年内通常没有特殊处理文本，文本模态较稀疏。
- 随机森林仅验证数据链路，不代表申报书的动态图频融合模型已完成。
- 在扩大股票池前，应进入阶段 B，完成 LSTM、Transformer、FreTS、Time-GNN 等统一基线和消融设计。

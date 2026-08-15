# 阶段 D 工程入口

## D-1 rolling-origin 协议

生成固定三折开发协议：

```powershell
python -m stage_d.run_d1_protocol
```

输出位于 `outputs/stage_d/d1_rolling_origin_v1/`，包括折分配、折摘要和协议清单。正式边界及哈希记录在 `stage_d/protocols/rolling_origin_v1.json`。

## 数据封存

所有阶段 D 入口必须先加载 `stage_d/configs/data_custody.json`。守卫会在文件打开前拒绝：

- `data/screening` 下的任何路径。
- 阶段 C-4 SCREENING 输出目录。
- 名称中包含 C-4 标识符的其他文件。
- `trade_date` 或 `target_date` 晚于 2023-06-02 的开发数据。

暂定未来 D-SCREENING 区间仍未授权，不得获取或读取。

## 跨折统一汇总

输入 CSV 每行对应一个“模型 × 折 × 种子”结果，必须包含：

```text
model,fold_id,seed,samples,mae,rmse,direction_accuracy,direction_f1
```

运行：

```powershell
python -m stage_d.run_cross_fold_summary `
  --input outputs/stage_d/example/metrics_by_fold_seed.csv `
  --output outputs/stage_d/example/summary
```

汇总器要求所有模型覆盖完全相同的折—种子网格，并以 Naive 为默认比较基线。

## 测试

```powershell
python -m unittest discover -s stage_d/tests -p "test_*.py"
```

## D-2 受限稳健基线

使用已冻结配置在登记三折上运行完整模型—折—种子网格：

```powershell
python -m stage_d.run_d2_baselines
```

冻结配置位于 `stage_d/configs/d2_baselines.json`，输出位于
`outputs/stage_d/d2_bounded_baselines_v1/`。运行器只构造动态 TRAIN/VALIDATION，
逐折核验登记行哈希，并在任何数据读取前应用 C-4/未来数据封存守卫。

## D-3 稳健诊断与唯一建议

```powershell
python -m stage_d.run_d3_diagnostics
```

该入口只读取 D-2 已登记的开发期输出，复核来源哈希后生成逐股票、收益分组、
最差折、种子稳定性和组件分歧诊断。固定准入门槛与唯一排序规则位于
`stage_d/configs/d3_diagnostics.json`，不允许新增候选或修改收缩系数。

## D-4 候选冻结与独立复算

```powershell
python -m stage_d.freeze_stage_d_candidate --verify
python -m stage_d.run_d4_independent_recalc
```

冻结目录为 `stage_d/frozen/frets_l4_shrink_a075_d4/`。其中包含三个种子检查点、
FreTS 上游源码副本、独立推理清单、31 项 SHA-256 和冻结回执。未来筛选必须复用
冻结推理及判定代码，未经授权不得获取 D-SCREENING。

## D-5 一次性独立 SCREENING

D-5 已在明确授权后使用冻结候选和 Naive 完成一次性执行，结果为 `PASS`。
授权已消耗，禁止在同一区间重新运行、修改门槛或继续调参。正式证据位于
`outputs/stage_d/d5_screening_20240614_20250613/`，授权区间原始提取已封存于
`data/screening/stage_d_d5_20240614_20250613/`。

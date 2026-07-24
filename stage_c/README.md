# 阶段 C 第一版：动态图频域融合原型

本目录实现一个可训练、可评估的最小动态图频域模型。第一版固定使用阶段 A 的 30 只股票和阶段 B 的统一数据切分、日志、随机种子与指标接口。

模型把单只股票最近 8 个周窗口视为 8 个节点，依次执行：价格特征投影、轻量 Transformer 时域编码、Gumbel-Softmax 边概率、逐行 Top-k 稀疏化、FFT 复数图传播、IFFT 重构、门控时频融合和下一周收益预测。

默认配置只评估 validation，test 分段继续保留，不参与第一版结构迭代。

运行：

```powershell
D:\项目\源文件\deploy\.venv-timegnn\Scripts\python.exe .\stage_c\run_prototype.py
```

可用 `--seed 20260724` 覆盖配置中的冻结种子。三个种子完成后运行统一比较：

```powershell
D:\项目\源文件\deploy\.venv-timegnn\Scripts\python.exe .\stage_c\compare_baselines.py
```

运行阶段 C 预先限定的三种子结构消融：

```powershell
D:\项目\源文件\deploy\.venv-timegnn\Scripts\python.exe .\stage_c\run_ablations.py
```

消融矩阵包含完整模型、单位图、固定时间图、仅时域、Top-k=4 和无门控等权融合。默认仍只评估 validation，test 不参与训练、早停、指标计算或结构选择。

运行动态图稳定化和横截面节点准入判断：

```powershell
D:\项目\源文件\deploy\.venv-timegnn\Scripts\python.exe .\stage_c\run_stabilization.py
```

该节点比较仅时域、固定时间图、确定性 Top-k 和线性温度退火，并额外计算跨种子预测相关性与逐样本预测离散度。

构建当前推荐的固定控制组集成 v2：

```powershell
D:\项目\源文件\deploy\.venv-timegnn\Scripts\python.exe .\stage_c\build_recommended_v2.py
```

推荐 v2 对仅时域控制和固定时间图的预测做 0.5/0.5 固定平均，不增加可学习融合参数。其结果仍属于开发期 validation 选择暴露，不是独立测试结论。

从两个组件权重独立加载并执行推荐 v2 推理：

```powershell
D:\项目\源文件\deploy\.venv-timegnn\Scripts\python.exe .\stage_c\run_recommended_v2_inference.py --seed 20260723
```

入口会从模型清单解析两个 `model.pt`，恢复各自训练期标准化参数和 8 周序列构造，校验特征顺序与种子，输出组件预测、集成预测、指标、权重 SHA-256 和环境信息。可通过 `--verify-reference <predictions.csv>` 执行逐行复算核对。

测试：

```powershell
D:\项目\源文件\deploy\.venv-timegnn\Scripts\python.exe -m unittest discover -s stage_c\tests -p "test_*.py"
```

输出位于 `outputs/experiments/stage_c_graph_frequency_v1/`。本版仅用于工程连通性、数值稳定性和初步可训练性验证，不作为最终模型结论。

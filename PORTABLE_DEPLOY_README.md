# AI金融时序预测与风险解释原型系统 - 可迁移部署版

这个文件夹现在可以整体复制到其他 Windows 电脑或其他终端使用。

## 里面有什么

- `dashboard.html`：离线原型展示页面，双击即可打开。
- `data/`：样例金融时序数据。
- `docs/`：技术说明、汇报讲稿、PPT结构、风险解释样例。
- `source_zips/`：FreTS、Time-GNN、SEP 三个源码 zip，已经打包进来。
- `portable_scripts/setup_all_windows.ps1`：一键部署脚本。

## 新电脑上怎么部署

前提：电脑需要有 Python 3.10+ 或 Python 3.12，并且能联网安装 pip 包。

打开 PowerShell，进入本文件夹：

```powershell
cd "复制后的\AI_Finance_Prototype"
```

执行部署：

```powershell
powershell -ExecutionPolicy Bypass -File .\portable_scripts\setup_all_windows.ps1
```

部署完成后会生成：

- `runtime/sources/`：解压后的源码
- `runtime/venvs/`：三个独立虚拟环境
- `run_frets.ps1`
- `run_timegnn.ps1`
- `run_sep_check.ps1`

## 部署后怎么运行

FreTS：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_frets.ps1
```

Time-GNN：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_timegnn.ps1
```

SEP 环境检查，不调用 API、不花钱：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_sep_check.ps1
```

## 注意

1. `dashboard.html` 不需要部署，复制后直接双击可用。
2. FreTS 和 Time-GNN 会真实跑模型测试，可以截图作为“模型演示照”。
3. SEP 完整运行需要 OpenAI API Key 和大模型资源，本包默认只做环境导入检查。
4. 如果网络慢，可以把脚本里的清华源换成默认 PyPI 或其他镜像。
5. 本项目仅用于社会实践、技术学习和风险识别科普，不构成投资建议。

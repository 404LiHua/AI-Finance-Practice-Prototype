# 单机 T2 推理工具

该目录提供当前生产 incumbent 的离线批量推理，输出 T2 类别、三类概率和可靠性。它不启动网络服务、不访问 FRESH、不写模型注册表。

输入必须提供模型登记的 14 个 RG3 日技术特征；可选 `trade_date`、`stock_code` 将原样带入输出。默认使用 CPU，适合个人电脑批量运行。

```powershell
& 'C:\Users\27793\Documents\deep\cuda-venv\Scripts\python.exe' `
  '...\local_service\run_t2_local_inference_v1.py' `
  --registry '...\local_service\PRODUCTION_T2_LOCAL_MODEL_REGISTRY_V1.json' `
  --input '<features.parquet-or-csv>' `
  --output '<new-output.parquet>' `
  --receipt '<new-receipt.json>'
```

注册表中的 C0 仅保留为开发结果：WP10 FRESH 证据已永久污染关闭，当前仅允许 WP11 无标签 shadow；不得手动改为 active。

## 本机网页服务（仅回环地址）

通过下列入口可启动个人电脑上的只读网页服务。它强制绑定 `127.0.0.1`、禁用 GPU、限制 CPU 线程，并不包含自动交易能力。

```powershell
& 'C:\Users\27793\Documents\deep\cuda-venv\Scripts\python.exe' `
  '...\local_service\serve_t2_local_loopback_v1.py' `
  --package-root 'C:\Users\27793\Documents\project1\deliverables\RG_OBGNet_source_v1' `
  --registry '...\local_service\PRODUCTION_T2_LOCAL_MODEL_REGISTRY_V1.json' `
  --daily-root 'D:\项目\data\rixian\stk_factor' `
  --port 8765
```

浏览器打开 `http://127.0.0.1:8765/`。停止服务请在该 PowerShell 窗口按 `Ctrl+C`。

运行前或环境迁移后，先运行 `audit_local_t2_http_operational_acceptance_v1.py`；只有其回环验收通过，才可将本机服务视为可运行。线上部署、外网监听和自动交易不在此工具范围内。



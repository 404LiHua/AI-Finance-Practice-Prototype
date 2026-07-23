# AI金融时序预测与风险解释原型系统

这是用于“AI金融与新质生产力”暑期社会实践的可直接展示原型包。

## 直接使用

1. 双击打开 `dashboard.html`。
2. 用页面中的“真实值 vs 预测值 / 预测误差 / 风险等级”切换按钮做展示。
3. 汇报时配合 `docs/project_presentation_script.md` 和 `docs/project_technical_note.md`。

## 从 GitHub 部署

```powershell
git clone https://github.com/404LiHua/AI-Finance-Practice-Prototype.git
cd AI-Finance-Practice-Prototype
powershell -ExecutionPolicy Bypass -File .\portable_scripts\setup_all_windows.ps1
```

仅展示原型时无需安装依赖，直接打开 `dashboard.html` 即可。

## 仓库内容

- `dashboard.html`：离线金融预测与风险解释展示页面。
- `data/`：原型使用的样例数据和摘要。
- `docs/`：技术说明、汇报材料和风险解释样例。
- `scripts/`：已经验证的模型运行命令。
- `portable_scripts/`：跨终端部署脚本。
- `source_zips/`：FreTS、Time-GNN、SEP 的开源源码归档，仅用于本项目学习、复现和部署。

## 项目定位

本项目不做真实投资建议，而是展示人工智能如何用于金融时间序列数据整理、趋势预测、结果可视化和风险解释。

完整运行 SEP 需要单独配置 API Key 和相应算力。任何密钥均不得提交到本仓库。

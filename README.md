# ToxHERB

> **Private research software / 未发表研究软件**
>
> 本仓库当前仅供未发表研究使用，保留所有权利，暂不授予开源许可。请勿在未经作者明确许可的情况下复制、公开或再分发。

ToxHERB 2.0 是一个面向方剂、单味中药和中药来源化学成分的本地肝毒性预测与知识关联分析平台。系统由单页 Web 前端、FastAPI 后端、预测脚本、机器学习模型、本地数据资源和任务缓存模块组成。

## 重要声明

- 本软件仅用于科研、方法开发和结果探索，不构成医疗建议、临床诊断或用药决策依据。
- 预测结果必须由具备相关专业背景的研究人员结合实验与临床证据审慎解释。
- `data/` 中的数据清单标记为本地研究数据；在任何公开发布或第三方共享前，必须逐项核验上游来源、许可和引用要求。
- `models/` 中的模型文件及相关训练资源在公开发布前也必须完成权属和许可审查。

## 环境要求

- Windows 10/11
- Python 3.13（当前开发环境为 Python 3.13.3）
- Git 与 [Git LFS](https://git-lfs.com/)
- Node.js 20+ 和 npm（仅运行 Playwright 端到端测试时需要）

仓库包含约 1.17 GB 数据及 126 MB 模型。首次克隆前请确保 Git LFS 已安装，并预留足够的磁盘空间。

## 获取与运行

```powershell
git lfs install
git clone https://github.com/Super-dong94/ToxHERB.git
Set-Location ToxHERB

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

.\run.bat
```

程序启动后会选择一个可用的本地端口并自动打开浏览器。默认仅监听 `127.0.0.1`。

## 项目结构

```text
backend/              FastAPI 服务、任务管理、安全配置和预测封装
frontend/             单页 Web 界面及本地 ECharts 资源
prediction_scripts/   方剂、中药和化学成分预测流程
models/               训练后的模型与校准状态清单（Git LFS）
data/                 本地数据库、索引及关联数据（Git LFS）
scripts/              数据清单维护与本地冒烟测试脚本
tests/                Python 单元/集成测试与 Playwright 测试
main.py               本地应用入口
run.bat               Windows 启动脚本
build_exe.bat         Windows 可执行程序构建脚本
```

## 测试

运行 Python 测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

运行端到端测试时，先在一个终端启动服务并固定端口：

```powershell
$env:TOXHERB_PORT = "7860"
.\.venv\Scripts\python.exe main.py
```

再在另一个终端执行：

```powershell
npm install --no-package-lock
npm run test:e2e
```

测试和日常运行生成的用户数据库、日志及预测结果均由 `.gitignore` 排除。

## 数据与模型完整性

`data_manifest.json` 记录数据文件的名称、大小、校验信息和许可提醒；`models/calibration/calibration_manifest.json` 记录模型校准状态。发布、归档或迁移仓库时应同时保留这些清单。

## 开源与引用计划

当前版本没有开源许可证。论文发表并完成数据、模型及第三方组件许可审计后，再补充正式许可证、论文引用、数据来源和可复现性说明。

---

## English Summary

ToxHERB 2.0 is a local research platform for hepatotoxicity prediction and knowledge-association analysis across herbal formulas, individual herbs, and herb-derived compounds. It combines a FastAPI backend, a single-page web interface, prediction pipelines, trained machine-learning models, and local research datasets.

This repository is currently private and contains unpublished research software. All rights are reserved and no open-source license is granted. The software is intended for research only and must not be used as a substitute for medical, diagnostic, or prescribing decisions. Dataset and model provenance, redistribution rights, and citation requirements must be reviewed before any public release.

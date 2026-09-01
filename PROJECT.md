# GISclaw Desktop — 项目状态

> 本文件记录桌面版的**当前状态**：阶段、工作单元、待决事项、下一步。历史在 `git log`，技术决定在 `docs/04_decisions/`。

| | |
|---|---|
| **目标** | 把 GISclaw 从「需要 Docker 的本地 Web 应用」做成「下载即用、母语界面」的桌面 GIS 分析工具 |
| **周期** | 2026-09 ～ 2027-01（v1.0.0 → v2.0.0，中间只发 beta） |
| **当前阶段** | **v2.0.0 已发布（安装包 + 三语界面 + 本地模型 + 底图）** |

---

## 1. 现在的状态

| 阶段 | 状态 | 进入下一阶段的条件 |
|---|---|---|
| v1.0.0 · Docker 本地 Web 应用，三语 README 与 PDF 手册，云端与本地模型 | ✅ 2026-09-01 发布 | — |
| v2.0.0-beta · 三语界面 + uv 安装脚本 + Win/mac 安装包（未签名），在 `v2` 分支开发 | ⬜ | Win / mac 各一台干净机器从零跑通示例；三语界面自动化测试通过 |
| v2.0.0 · 正式版 | ⬜ | beta 反馈处理完；三语 README / 手册 / 截图更新 |
| 之后可选 · 代码签名与公证 | ⬜ 按需 | 视用户需求决定 |

**约定**：v1.1 / v1.2 不单独发版 —— 每次发版都要更新三语 README、三份 PDF 手册与截图；三语界面与安装包相互独立，可并行；uv 安装脚本本身就是安装包的构建步骤。
**签名**：先发未签名版。macOS 15+ 需在「系统设置 → 隐私与安全性」点「仍要打开」，手册写清；通过终端脚本安装的文件不带 quarantine 属性，uv 路线在 mac 上无此提示。

**形态与边界**
- 浏览器 UI ↔ FastAPI ↔ ReAct 单 agent ↔ Python GIS 沙箱；云端 provider + 本地 Ollama / LM Studio / vLLM。
- 28 个确定性算子（Toolbox）、项目 / run 溯源、Skills 目录包、prompt caching、chat / JOURNAL / LOG / MEMORY 持久化。
- `paper-v2` 分支 + tag `v2-gsis-submission` 是论文引用版本，保持不变。

## 2. 工作单元 (Epic)

| ID | 内容 | 状态 |
|---|---|---|
| E0 | 发版基础：tag `v1.0.0`、GitHub Release、`CHANGELOG.md`、`/api/version` 与 About 显示版本 | ✅ 2026-09-01 |
| E1 | 三语界面（en / zh / ko）：静态字典 + `data-i18n`、语言设置持久化、后端消息「码 + 参数」、agent 收尾 / LOG 跟随界面语言 | ✅ 2026-09-01（ADR-002 按提案实施） |
| E2 | 无 Docker 安装：`install.sh` / `install.ps1` 用 uv 建 Python 3.11 环境（PyPI 二进制 wheel），`desktop/launcher.py` 开原生窗口，macOS 生成 `GISclaw.app` | 🟡 Linux 上验证通过（全测试套件）；mac / Windows 待实机 |
| E3 | **原生安装包**：uv 管理的可搬迁 Python + 全部 wheel 装进包内 + pywebview 窗口；macOS `.app` + `.dmg`（ad-hoc 签名），Windows Inno Setup `.exe`；`.github/workflows/desktop.yml` 出包并发 Release | 🟡 脚本在 Linux 上 dry-run 通过；等 Actions 实跑 + 实机 |
| E4 | 原生模式的安全边界：DISCLAIMER 与 README 说明「原生模式没有容器隔离」；沙箱工作目录限定、UI 提示 | 🟡 文档已加（英文）；限定与提示待做 |
| E5 | 开发环境统一：以本仓库为唯一开发树 | ⬜ B1 |
| E6 | 其余待办：run 历史浏览、报告导出（HTML / PDF）、大矢量切片、更多算子、`code.py` 收录失败片段、discipline 补随机种子 / 空值纪律 | ⬜ 按需 |
| E7 | **代码审视与加固**（v2 开工前）：Stop 真正停止、刷新后重连、单 run 互斥、沙箱超时/取消、只绑本机 + 同页校验、Toolbox 进记录、缓存计费、上下文上限随模型、Windows junction、前端修补、测试套件 | ✅ 2026-09-01（`v2` 分支） |

## 3. 🔴 待决事项

> 第一列以 `B1`、`B2` … 开头（brief.sh 按此模式抓取）。

| # | 内容 | 何时要 |
|---|---|---|
| B1 | 开发环境统一：以本仓库为唯一开发树，本地容器改挂此处 | v2 开工前 |
| B2 | Windows 测试机：没有实机则 Windows 侧靠 GitHub Actions 与 beta 用户验证 | E2 开工时 |
| B3 | Intel Mac 是否支持：PyPI 上 rasterio 以 `macosx_10_13_x86_64` 标签查不到 wheel（可能只是标签太旧）。倾向不支持，Intel Mac 走 Docker | E3 出包前 |

## 4. 决策记录 (ADR)

| ID | 标题 | 状态 |
|---|---|---|
| ADR-002 | 界面多语言：静态字典 + `data-i18n`，后端消息「码 + 参数」，agent 文本跟随界面语言 | 决定 |
| ADR-003 | 原生分发：完整 Python 环境随包 + 只冻结启动器 | 提案 |

## 5. 下一步

> `- [ ]` / `- [x]`（brief.sh 只抓未完成项）。

- [x] E0：tag `v1.0.0` + GitHub Release + `CHANGELOG.md` + 版本号（2026-09-01）
- [x] 开 `v2` 分支；E7 代码审视与加固（2026-09-01）
- [ ] 决定 B1；`release` 只接 v1.0.x 修复
- [ ] 确认 ADR-002 / ADR-003，状态改「决定」
- [x] E1：三语界面，字典在 `app/web/i18n.js`（2026-09-01）
- [x] E4：DISCLAIMER 加「原生模式没有容器隔离」（README 英文段已加；中 / 韩段随 v2 文档更新）
- [x] 验证 wheel 路线：Linux 上 `install.sh` → 715 MB venv，测试套件全过（2026-09-01）
- [ ] mac 实机：装 Release 里的 `.dmg`；看窗口是否出现、CJK 图标签、示例项目跑通、Local models 面板连本机 Ollama
- [ ] Windows 实机或 Actions：`install.ps1` 未测
- [ ] 三语 PDF 手册补「本地模型」一节

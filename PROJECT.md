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
| v1.0.0 · Docker 本地 Web 应用 | ✅ 2026-09-01 | — |
| v2.0.0 · macOS `.dmg` / Windows `.exe`、三语界面、本地模型面板、底图选择与缓存、测试套件 | ✅ 2026-09-01 发布（beta.1–7 在 Mac / Windows 实机验证后） | — |
| 2.0.x · 收集用户反馈，修小问题 | 🟡 进行中 | 反馈稳定后决定 2.1 内容 |
| 之后可选 · 代码签名与公证（Apple Developer） | ⬜ 按需 | 出现 MDM 用户或下载量说明值得 |

**约定**：每次发版 = 提交到 `v2` → 推私有库打 beta tag 出包实测 → 确认后 `v2` 推公开 `main` + 正式 tag（Actions 出包并发 Release）。文档只维护 README（三语），PDF 手册已废弃。
**签名**：未签名。macOS 15+ 首次打开走「仍要打开」，README 已写明。

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
| B1 | 开发环境统一：以本仓库为唯一开发树，本地开发容器改挂此处（目前容器仍跑旧树） | 下次改动 agent 内核前 |
| B2 | Intel Mac 是否支持：PyPI 的 rasterio 以 `macosx_10_13_x86_64` 标签查不到 wheel（可能只是标签太旧，用新标签再查）。倾向不支持，Intel Mac 走 Docker | 有人问起时 |

## 4. 决策记录 (ADR)

| ID | 标题 | 状态 |
|---|---|---|
| ADR-002 | 界面多语言：静态字典 + `data-i18n`，后端消息「码 + 参数」，agent 文本跟随界面语言 | 决定 |
| ADR-003 | 原生分发：完整 Python 环境随包 + 只冻结启动器 | 决定 |

## 5. 下一步

> `- [ ]` / `- [x]`（brief.sh 只抓未完成项）。

- [x] v2.0.0 发布：安装包、三语界面、本地模型、底图、测试套件、README 重写（2026-09-01）
- [ ] 截图换成 2.0 界面（现在是 1.0 的图加窗口框；缺语言按钮、Local models、Map 面板）
- [ ] Help → About 与 DISCLAIMER 的中 / 韩文本（目前仅英文）
- [ ] 决定 B1，改本地开发容器的挂载
- [ ] 收集 2.0 反馈：Windows 上 WebView2 缺失时的回退是否顺畅、Ollama 上下文提示是否管用

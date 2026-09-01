# GISclaw Desktop — 项目状态

> 本文件记录桌面版的**当前状态**：阶段、工作单元、待决事项、下一步。历史在 `git log`，技术决定在 `docs/04_decisions/`。

| | |
|---|---|
| **目标** | 把 GISclaw 从「需要 Docker 的本地 Web 应用」做成「下载即用、母语界面」的桌面 GIS 分析工具 |
| **周期** | 2026-09 ～ 2027-01（v1.0.0 → v2.0.0，中间只发 beta） |
| **当前阶段** | **v1.0.0 已发布 → v2.0 开发中（三语界面 + 无 Docker 安装 + Win/mac 安装包，一次发布）** |

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
| E1 | 三语界面（en / zh / ko）：静态字典 + `data-i18n`、语言设置持久化、后端消息「码 + 参数」、agent 收尾 / LOG 跟随界面语言 | ⬜ ADR-002 |
| E2 | 无 Docker 安装：`install.sh` / `install.ps1` 用 uv 建 Python 3.11 环境（PyPI 二进制 wheel），`gisclaw` 启动器打开浏览器 | ⬜ |
| E3 | 原生安装包：python-build-standalone 环境 + pywebview 窗口（同一份 Web 前端，Win 用 WebView2 / mac 用 WKWebView）+ Inno Setup / create-dmg；GitHub Actions 矩阵出包 | ⬜ ADR-003 |
| E4 | 原生模式的安全边界：沙箱工作目录限定、UI 提示、DISCLAIMER 补「原生模式没有容器隔离」 | ⬜ E2 前置 |
| E5 | 开发环境统一：以本仓库为唯一开发树 | ⬜ B1 |
| E6 | 其余待办：run 历史浏览、停止运行、报告导出（HTML / PDF）、大矢量切片、更多算子、`code.py` 收录失败片段、discipline 补随机种子 / 空值纪律 | ⬜ 按需 |

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
| ADR-002 | 界面多语言：静态字典 + `data-i18n`，后端消息「码 + 参数」，agent 文本跟随界面语言 | 提案 |
| ADR-003 | 原生分发：完整 Python 环境随包 + 只冻结启动器 | 提案 |

## 5. 下一步

> `- [ ]` / `- [x]`（brief.sh 只抓未完成项）。

- [x] E0：tag `v1.0.0` + GitHub Release + `CHANGELOG.md` + 版本号（2026-09-01）
- [ ] 决定 B1；开 `v2` 分支，`release` 只接 v1.0.x 修复
- [ ] 确认 ADR-002 / ADR-003，状态改「决定」
- [ ] E1 第一步：抽取 `index.html` 与 `app.js` 的界面字符串为 `app/web/i18n/en.json`，再生成 zh / ko
- [ ] E4：DISCLAIMER 与 README 三语补「原生模式没有容器隔离」
- [ ] 验证 wheel 路线：干净 mac 上 `uv venv` + `uv pip install geopandas rasterio fiona pyproj shapely rtree`
- [ ] 三语 PDF 手册补「本地模型」一节

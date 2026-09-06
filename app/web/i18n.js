/*
 * GISclaw — an LLM agent for geospatial analysis.
 * Copyright (C) 2026 Han Jinzhen
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 *
 * This file is part of GISclaw. GISclaw is free software: you can redistribute
 * it and/or modify it under the terms of the GNU Affero General Public License
 * as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version. It is distributed in the hope
 * that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Affero General Public License in the LICENSE file, or
 * <https://www.gnu.org/licenses/>, for more details.
 */

/* ==========================================================================
   Interface languages.

   English is the source: every string in the page and in app.js is written in
   English and looked up here. Static markup carries `data-i18n` (the element's
   original content is the key) or `data-i18n-attr="placeholder,title"`;
   runtime strings go through t(). A missing entry falls back to English, so
   an untranslated string is never an error — only English.

   To add a language: one more object in DICT (its keys are the English
   strings), one more entry in LANGUAGES.
   ========================================================================== */
window.I18N = (() => {
  'use strict';
  const LANGUAGES = [['en', 'English', 'EN'], ['zh', '中文', '中'], ['ko', '한국어', '한']];
  const DICT = { en: {} };
  let lang = 'en';

  const norm = s => String(s == null ? '' : s).replace(/\s+/g, ' ').trim();

  function t(source, vars) {
    const raw = String(source == null ? '' : source);
    const key = norm(raw);
    const d = DICT[lang] || {};
    const lead = raw.match(/^\s*/)[0], trail = raw.match(/\s*$/)[0];
    let out = Object.prototype.hasOwnProperty.call(d, key) ? lead + d[key] + trail : raw.replace(/\s+/g, ' ');
    if (vars) Object.keys(vars).forEach(k => { out = out.split('{' + k + '}').join(String(vars[k])); });
    return out;
  }

  function apply(root) {
    root = root || document;
    root.querySelectorAll('[data-i18n]').forEach(el => {
      if (el.dataset.i18nSrc === undefined) el.dataset.i18nSrc = norm(el.innerHTML);
      const out = t(el.dataset.i18nSrc);
      if (out !== norm(el.innerHTML)) el.innerHTML = out;
    });
    root.querySelectorAll('[data-i18n-attr]').forEach(el => {
      el.dataset.i18nAttr.split(',').forEach(a => {
        a = a.trim();
        const store = 'i18nSrc' + a.charAt(0).toUpperCase() + a.slice(1);
        if (el.dataset[store] === undefined) el.dataset[store] = el.getAttribute(a) || '';
        el.setAttribute(a, t(el.dataset[store]));
      });
    });
    document.documentElement.lang = lang === 'zh' ? 'zh-CN' : lang;
    const code = document.getElementById('langCode');
    if (code) code.textContent = (LANGUAGES.find(l => l[0] === lang) || LANGUAGES[0])[2];
  }

  function set(code, opts) {
    lang = DICT[code] ? code : 'en';
    try { localStorage.setItem('gisclaw.lang', lang); } catch (e) {}
    apply();
    if (!(opts && opts.silent)) document.dispatchEvent(new CustomEvent('gisclaw:lang', { detail: lang }));
  }

  function detect(serverLang) {
    let stored = '';
    try { stored = localStorage.getItem('gisclaw.lang') || ''; } catch (e) {}
    const nav = (navigator.language || '').toLowerCase();
    const guess = nav.startsWith('zh') ? 'zh' : nav.startsWith('ko') ? 'ko' : 'en';
    return DICT[serverLang] ? serverLang : (DICT[stored] ? stored : guess);
  }

  function register(code, dict) { DICT[code] = Object.assign(DICT[code] || {}, dict); }

  return { t, apply, set, detect, register, LANGUAGES, get lang() { return lang; } };
})();

/* ------------------------------------------------------------------ 中文 -- */
I18N.register('zh', {
  'GIS Analyst Agent': 'GIS 分析助手',
  'Idle': '空闲', 'Running': '运行中', 'Stopping…': '正在停止…', 'Done': '完成', 'Stopped': '已停止', 'Failed': '失败',
  'Stop after the current step': '在当前这一步结束后停止',
  'Stopping — finishing the current step…': '正在停止 —— 等当前这一步结束…',
  'Already stopping — press again to stop watching this run': '已在停止 —— 再按一次就不再跟随这次运行',
  'Stopped watching this run. It ends on the server after the current step, and the result is kept in the project.':
    '已不再跟随这次运行。它会在当前这一步结束后停止，结果仍会记入项目。',
  'The run had already ended.': '这次运行已经结束了。',
  'Could not reach the server to stop the run.': '无法连接服务器来停止这次运行。',
  'Running in <b>{p}</b>': '<b>{p}</b> 正在运行', 'Open it': '打开',
  'Project': '项目', 'Tools': '工具', 'Settings': '设置', 'Help': '帮助',
  '＋ New project…': '＋ 新建项目…', '＋ Add data…': '＋ 添加数据…', 'Log (compacted)…': '日志（摘要）…',
  'Journal (full record)…': '实验记录（完整）…', 'Add note to journal…': '向记录添加备注…',
  'Start a new conversation': '开始新对话', 'Archived projects…': '已归档项目…', 'Toolbox…': '工具箱…',
  'API keys…': 'API 密钥…', 'Models…': '模型…', 'Local models…': '本地模型…', 'Map…': '地图…', 'Skills…': 'Skills…', 'Memory…': '记忆…',
  'About GISclaw &amp; source code…': '关于 GISclaw 与源代码…',
  'Projects': '项目', '＋ New': '＋ 新建',
  'Use the <b>Project</b> menu above to create a project and add data.': '用上方的<b>项目</b>菜单新建项目并添加数据。',
  'Map': '地图', 'Code': '代码', 'Table': '表格', 'Result': '结果', 'Map layers': '图层',
  'Click a dataset on the left to show it here.': '点击左侧的数据集即可显示在这里。',
  'Raw text': '原始文本', 'Ready': '就绪', 'Executed': '已执行', 'Error': '出错',
  'Execution output': '执行输出', 'Output appears here as the agent runs code.': '助手运行代码时，输出会显示在这里。',
  'Steps · 0': '步骤 · 0', 'Self-corrections · 0': '自我纠错 · 0', 'Elapsed · 00:00': '用时 · 00:00',
  'Steps · {n}': '步骤 · {n}', 'Self-corrections · {n}': '自我纠错 · {n}', 'Elapsed · {t}': '用时 · {t}',
  'No project selected': '未选择项目', 'AI analyst': 'AI 分析员',
  'Describe the analysis, pick a model, then press Run.': '描述你要做的分析，选择模型，然后按「运行」。',
  'Info': '信息', 'You': '你', 'Thought': '思考', 'Action': '动作', 'Observation': '观察', 'Issue': '问题', 'Answer': '回答',
  'Create a project on the left, add your GIS data, then describe an analysis and press <b>Run</b>.': '在左侧新建项目，添加你的 GIS 数据，然后描述分析需求并按<b>运行</b>。',
  'Clear': '清空', 'Run': '运行', 'Stop': '停止',
  'Add data': '添加数据',
  "Pick files or folders from the workspace. They are copied into the project's <code>data/</code>.": '从工作区选择文件或文件夹，它们会被复制到项目的 <code>data/</code> 目录。',
  'Upload files…': '上传文件…', 'Upload a folder…': '上传文件夹…',
  '…or drop them here. Use this for data anywhere on your computer — the list below only reaches the mounted workspace.': '…或直接拖到这里。电脑上任何位置的数据都可以这样上传——下方列表只能看到挂载的工作区。',
  '0 selected': '已选 0 项', '{n} selected': '已选 {n} 项', 'Cancel': '取消', 'Attach selected': '添加所选',
  'Archived projects': '已归档项目',
  'Archiving moves a project out of the workspace but keeps every file. Restore brings it back.': '归档会把项目移出工作区但保留全部文件；恢复即可放回。',
  'Close': '关闭', 'Rename project': '重命名项目',
  'The folder on disk is renamed too, unless that name is already taken.': '磁盘上的文件夹也会一起改名，除非该名称已被占用。',
  'Rename': '重命名', 'Are you sure?': '确定吗？', 'Delete': '删除', 'New project': '新建项目',
  'A project is a working folder holding your data, outputs, and run history.': '项目是一个工作文件夹，存放你的数据、产出和运行历史。',
  'Create': '创建', 'Toolbox': '工具箱',
  'Run a GIS operation directly (deterministic, no AI cost) — or insert it as a request for the agent.': '直接运行一个 GIS 算子（确定性，不花 AI 费用），或把它作为请求插入对话交给助手。',
  'Select an operation on the left.': '在左侧选择一个算子。', 'Attribute table': '属性表',
  'API keys': 'API 密钥', 'Models': '模型', 'Local models': '本地模型', 'Skills': 'Skills', 'Memory': '记忆',
  'Keys are stored on the server in <code id="setPath">…</code> (file mode 600) and survive container rebuilds. They are never sent back to this page — only a mask is.': '密钥保存在本机的 <code id="setPath">…</code>（文件权限 600），重建容器也不会丢失；永远不会回传到本页面，只回传掩码。',
  'Only enabled models with a configured key appear in the run selector. Any OpenAI-compatible endpoint works — pick the <b>Custom</b> provider and give its base URL.': '只有已启用且配置了密钥的模型才会出现在运行下拉框中。任何 OpenAI 兼容端点都可以用——选择 <b>Custom</b> 提供商并填写其 base URL。',
  'What your providers are serving right now': '各提供商当前在服的模型', 'Fetch model list': '获取模型列表',
  'Add / edit a model': '添加 / 编辑模型', 'Id': '标识', 'Display name': '显示名称', 'Provider': '提供商',
  'API model name': 'API 模型名', 'Base URL': 'Base URL', 'Max rounds': '最大轮数', 'Max tokens': '最大 token 数',
  'Cost in / out per 1M': '每百万 token 价格（输入 / 输出）', 'Clear form': '清空表单', 'Save model': '保存模型',
  "A model served on this computer — Ollama, LM Studio, vLLM — keeps the whole analysis here: nothing is sent to a provider, nothing is billed. Small models plan multi-step work less reliably than the hosted flagships; the ones recommended below have completed this project's benchmark tasks. <b>Context length matters most:</b> a run needs at least <span id=\"localMinCtx\">8,192</span> tokens, or the server silently drops the instructions off the front of the prompt.": '在本机运行的模型（Ollama、LM Studio、vLLM）让整个分析留在本地：不向任何服务商发送数据，也不产生费用。小模型规划多步分析的可靠性不如云端旗舰模型；下面推荐的模型都完成过本项目的基准任务。<b>上下文长度最关键：</b>一次运行至少需要 <span id="localMinCtx">8,192</span> 个 token，否则服务器会悄悄把提示词前面的指令截掉。',
  'Server': '服务器', 'Connect': '连接', 'install ↗': '安装 ↗', 'Recommended models': '推荐模型',
  'The map behind your data. Tiles are fetched by GISclaw, not by this page — a key stays on this computer, and every tile you have looked at is kept in the data folder, so the area stays available offline. Continents, borders and lakes are always drawn from a built-in offline layer, whatever else loads.': '数据背后的底图。瓦片由 GISclaw 而不是本页面获取——密钥留在本机，看过的每块瓦片都保存在数据目录里，离线时该区域仍可显示。无论底图能否加载，内置的离线图层始终绘制大陆、国界和湖泊。',
  'Source': '来源', 'Key': '密钥', 'Tile template': '瓦片模板', 'Attribution': '署名', 'MBTiles file': 'MBTiles 文件',
  'get a key ↗': '获取密钥 ↗', 'Keep the tiles I have looked at, for offline use': '保留看过的瓦片，供离线使用',
  'Clear tile cache': '清空瓦片缓存', 'Apply': '应用',
  'A skill is a <b>directory bundle</b> — <code>SKILL.md</code> (a short router) plus <code>references/</code>, <code>assets/</code>, <code>manifest.yaml</code>. Same shape as Claude Code skills, so bundles are interchangeable. Only the one-line description is always in context; the router and its references load on demand.': 'Skill 是一个<b>目录包</b>：<code>SKILL.md</code>（简短的路由说明）加上 <code>references/</code>、<code>assets/</code>、<code>manifest.yaml</code>。与 Claude Code 的 skill 结构相同，可以互换使用。只有一行描述常驻上下文，路由与参考资料按需加载。',
  'Auto-load the best-matching skill for each task <i>(models rarely open one themselves)</i>': '为每个任务自动加载最匹配的 skill <i>（模型很少会自己打开）</i>',
  'New': '新建', 'Import .zip': '导入 .zip', 'Import folder': '导入文件夹', 'Save': '保存',
  'Standing preferences applied to <b>every</b> project — cartography, deliverable conventions, house CRS. Injected into the agent\'s system prompt at the start of each run. Stored in <code id="memPath">…</code>.': '适用于<b>所有</b>项目的常驻偏好——制图风格、交付规范、常用坐标系。每次运行开始时注入助手的系统提示词。保存在 <code id="memPath">…</code>。',
  'Apply memory to runs': '在运行中应用记忆',
  'Follow the agent between Map, Code and Result while it works <i>(off keeps the tab you chose)</i>': '运行时视图跟随助手在地图、代码、结果之间切换 <i>（关闭则停留在你选的标签页）</i>',
  'Save memory': '保存记忆', 'Journal': '实验记录',
  'Append-only record of this project: every run, what was asked, what it produced. Lives at <code id="journalPath">…</code> — readable without this app.': '本项目的追加式记录：每次运行、提问内容、产出结果。位于 <code id="journalPath">…</code>，无需本应用也能阅读。',
  'Add a note': '添加备注',
  'Create a new project': '新建项目', 'Add data to the selected project': '为所选项目添加数据',
  'Run GIS operations directly': '直接运行 GIS 算子', 'Drag to resize · double-click to reset': '拖动调整宽度 · 双击复位',
  'Zoom in': '放大', 'Zoom out': '缩小', 'Fit to data': '缩放至数据', 'Filter rows…': '筛选行…',
  'e.g. Compute building density per block and map it.': '例如：计算每个街区的建筑密度并制图。',
  'Model': '模型', 'Clear the view': '清空视图', 'Project name': '项目名称',
  'Query — filter rows by any field…': '查询——按任意字段筛选行…', "paste the provider's key": '粘贴该提供商的密钥',
  '© whoever made the map': '© 地图制作者', '…or a folder path under the workspace, e.g. _incoming/my-skill': '…或工作区内的文件夹路径，例如 _incoming/my-skill',
  // runtime
  'Basemap': '底图', 'offline reference': '离线参考图', 'vector': '矢量', 'raster': '栅格', 'Toggle visibility': '显示 / 隐藏',
  'Right-click a layer for symbology, attribute table…': '右键图层可设置符号、查看属性表…',
  'Show layer': '显示图层', 'Hide layer': '隐藏图层', 'Zoom to layer': '缩放至图层', 'Fit to all layers': '缩放至全部图层',
  'Symbology…': '符号设置…', 'Remove layer': '移除图层', 'Opacity': '不透明度', 'Color': '颜色', 'Fill opacity': '填充不透明度',
  '{name} — {n} features': '{name} — {n} 个要素', '(showing first {n})': '（仅显示前 {n} 个）', '{n} rows': '{n} 行',
  'Could not load {name}': '无法加载 {name}', '{name}: not a valid GeoJSON layer.': '{name}：不是有效的 GeoJSON 图层。', 'overlay failed': '叠加失败',
  'Rename…': '重命名…', 'Export as zip': '导出为 zip', 'Archive…': '归档…', 'Collapse': '折叠', 'Open': '打开', 'Journal…': '实验记录…',
  'Delete project…': '删除项目…', 'Delete file…': '删除文件…',
  'data': '数据', 'records': '记录', 'outputs': '产出', '(empty — add data)': '（空——请添加数据）',
  'Journal (full record)': '实验记录（完整）', 'Log (compacted)': '日志（摘要）', 'Conversation': '对话',
  'No projects yet.<br/>Press <b>＋ New</b> to create one.': '还没有项目。<br/>按<b>＋ 新建</b>创建一个。',
  'Describe an analysis and press Run.': '描述分析需求，然后按「运行」。', 'Add data to this project to begin.': '先为项目添加数据。',
  'Working…': '处理中…', 'Reasoning · {n} steps': '推理 · {n} 步', 'Reasoning · {n} steps · {t}s': '推理 · {n} 步 · {t} 秒',
  'Agent output': '助手产出', '{n} lines': '{n} 行', '{a} of {b} rows × {c} cols': '{a} / {b} 行 × {c} 列', '· showing first {n}': '· 仅显示前 {n} 行',
  'Started a new conversation. The previous one is archived in the project folder, and JOURNAL.md still holds every run.': '已开始新对话。之前的对话已归档到项目文件夹，JOURNAL.md 仍保留每次运行。',
  'Please describe the analysis first.': '请先描述要做的分析。', 'This project has no data yet. Add data first.': '该项目还没有数据，请先添加。',
  'No model configured — open <b>Settings → API keys</b> and add one first.': '尚未配置模型——请先打开<b>设置 → API 密钥</b>添加一个。',
  'Rejoined the run in progress ({id}).': '已重新接上正在进行的运行（{id}）。',
  'Could not start ({code}).': '无法启动（{code}）。', 'The connection to the server was lost.': '与服务器的连接已断开。',
  'Could not rejoin the run ({code}).': '无法重新接上该运行（{code}）。', 'Network error': '网络错误', 'Stopped.': '已停止。',
  '_No closing summary was written for this run._': '_本次运行没有写收尾总结。_', 'Produced': '产出',
  'Finished in <b>{t}s</b> · {n} rounds · {k} output(s)': '用时 <b>{t} 秒</b> · {n} 轮 · {k} 个产出',
  'Log entry written to LOG.md': '日志条目已写入 LOG.md',
  'uploading {i}/{n} — {name}': '正在上传 {i}/{n} — {name}', '{n} uploaded': '已上传 {n} 个', ', {n} failed': '，{n} 个失败',
  'Uploaded {n} file(s) to {p}.': '已上传 {n} 个文件到 {p}。', '{n} failed.': ' {n} 个失败。',
  'Delete file': '删除文件',
  '<b>{f}</b> will be deleted from <code>{w}/</code>.<br/>This cannot be undone. Earlier runs keep their own copies under <code>runs/</code>.': '<b>{f}</b> 将从 <code>{w}/</code> 中删除。<br/>此操作无法撤销。之前的运行在 <code>runs/</code> 下仍保留各自的副本。',
  '<br/><br/>Its <code>.shx</code>/<code>.dbf</code>/<code>.prj</code> siblings go with it — a lone <code>.shp</code> is unreadable.': '<br/><br/>它的 <code>.shx</code>/<code>.dbf</code>/<code>.prj</code> 伴随文件会一起删除——单独的 <code>.shp</code> 无法读取。',
  'Deleted {list} from {w}/.': '已从 {w}/ 删除 {list}。',
  'Delete project': '删除项目',
  '{n} data file(s), {m} output(s), {r} run(s)': '{n} 个数据文件、{m} 个产出、{r} 次运行', '{n} data file(s) and its whole run history': '{n} 个数据文件及全部运行历史',
  '<b>{p}</b> and everything in it — {counts} — will be deleted from disk. This cannot be undone.<br/><br/>Prefer <b>Archive</b> if you might want it back, or <b>Export as zip</b> first.': '<b>{p}</b> 及其全部内容——{counts}——将从磁盘删除。此操作无法撤销。<br/><br/>如果以后可能还要用，建议先<b>归档</b>或<b>导出为 zip</b>。',
  'Deleted project "{p}".': '已删除项目“{p}”。',
  'Archived "{p}". Nothing was deleted — bring it back from Project → Archived projects.': '已归档“{p}”。没有删除任何内容——可从 项目 → 已归档项目 恢复。',
  'Loading…': '加载中…', 'Nothing archived yet.': '还没有归档的项目。', '{n} file(s) · {m} run(s)': '{n} 个文件 · {m} 次运行',
  'Restore': '恢复', 'Restored "{p}".': '已恢复“{p}”。', 'Delete archived project': '删除已归档项目',
  '<b>{p}</b> — {n} file(s), {m} run(s) — will be deleted from disk. This cannot be undone.': '<b>{p}</b>（{n} 个文件，{m} 次运行）将从磁盘删除。此操作无法撤销。',
  'Deleted archived project "{p}".': '已删除已归档项目“{p}”。', 'workspace': '工作区',
  'Added {n} file(s) to {p}.': '已向 {p} 添加 {n} 个文件。',
  'output name': '输出名称', 'Insert into chat': '插入对话', 'Running…': '运行中…', 'Failed — check inputs.': '失败——请检查输入。', 'Done → {files}': '完成 → {files}',
  '(no {kind} layers — add data)': '（没有{kind}图层——请添加数据）',
  'no key needed': '无需密钥', 'key saved': '已保存密钥', 'from {env}': '来自 {env}', 'no key': '无密钥', 'set up →': '去设置 →',
  '{mask} (stored — type to replace)': '{mask}（已保存——输入即可替换）', 'paste your API key': '粘贴你的 API 密钥',
  'Test': '测试', 'Remove': '移除', 'Nothing to save — fill in the address first.': '没有可保存的内容——请先填写地址。',
  'Nothing to save — paste a key first.': '没有可保存的内容——请先粘贴密钥。', 'Saving…': '保存中…', 'Saved.': '已保存。',
  'Calling the API…': '正在调用 API…', 'Works — {model} replied "{reply}".': '可用——{model} 回复了“{reply}”。', 'Failed.': '失败。',
  'Show in the run selector': '显示在运行下拉框中', 'ready': '可用', 'no endpoint': '无端点', 'custom': '自定义', 'Edit': '编辑', 'Disable': '停用',
  'Asking the provider…': '正在询问提供商…', 'Could not list models.': '无法列出模型。',
  '{n} chat model(s)': '{n} 个对话模型', '· {n} non-chat hidden': ' · 已隐藏 {n} 个非对话模型', 'Nothing returned.': '没有返回结果。', 'added': '已添加', 'Add this model': '添加该模型',
  'Added {m} — it is now in the run selector.': '已添加 {m}——现在可以在运行下拉框中选择。',
  'Press <b>Connect</b> to see what the server is serving.': '按<b>连接</b>查看服务器正在提供的模型。',
  'added · context {n} chars per round': '已添加 · 每轮上下文 {n} 字符', 'context {n}': '上下文 {n}', '(loaded)': '（已加载）',
  'context up to {n} · server default applies': '上下文最高 {n} · 按服务器默认值', 'context unknown': '上下文未知',
  'The server answered but has no models. Pull one first — see below.': '服务器有响应但没有模型。请先拉取一个——见下方。',
  'Copy': '复制', 'Copied': '已复制', 'Enter the server address first.': '请先填写服务器地址。', 'Connecting…': '连接中…', 'No answer.': '没有响应。',
  'an OpenAI-compatible server': 'OpenAI 兼容服务器', 'Connected to {kind} — {n} model(s).': '已连接到 {kind}——{n} 个模型。',
  '{m} is loaded with a {ctx}-token context. Raise it in the Ollama app (Settings → Context length) or start the server with OLLAMA_CONTEXT_LENGTH={rec}.': '{m} 当前以 {ctx} token 的上下文加载。请在 Ollama 应用的 设置 → Context length 中调大，或用 OLLAMA_CONTEXT_LENGTH={rec} 启动服务。',
  'Added {m} — it is in the run selector. Press Test to load it once.': '已添加 {m}——已在运行下拉框中。按「测试」加载一次。',
  'Calling {m}… (the first call loads the model; this can take a minute)': '正在调用 {m}…（首次调用需要加载模型，可能要一分钟）',
  'Works — {m} replied "{reply}".': '可用——{m} 回复了“{reply}”。', 'Loaded with a {ctx}-token context.': ' 以 {ctx} token 的上下文加载。',
  'Cache: {mb} MB.': '缓存：{mb} MB。', 'Source reachable ({ms} ms).': '来源可访问（{ms} 毫秒）。', 'Source failed: {detail}': '来源获取失败：{detail}', 'Applied — {name}.': '已应用——{name}。', 'Tile cache cleared.': '瓦片缓存已清空。',
  'Importing {f}…': '正在导入 {f}…', 'Installed “{n}”.': '已安装“{n}”。',
  'Make available to the agent': '提供给助手使用', 'always on': '常驻', 'user': '用户', 'builtin': '内置', '{n} files': '{n} 个文件',
  'Router ~{n} tok, loaded on demand': '路由约 {n} token，按需加载', '{n} tok always': '常驻 {n} token', 'View': '查看', 'Export': '导出', 'Fork': '复制到工作区',
  'Copied into your workspace — your version now wins.': '已复制到你的工作区——现在以你的版本为准。',
  'Enabled skills cost ~{n} tokens on every call (routers and references load on demand, on top).': '已启用的 skill 每次调用约占 {n} token（路由与参考资料按需另计）。',
  'No skills found.': '没有找到 skill。',
  'Your bundle at {p}. Changes apply on the next run — no restart.': '你的 skill 包位于 {p}。改动在下次运行生效，无需重启。',
  'Read from {p} ({s}). Saving forks the whole bundle into your workspace, where your copy wins.': '读取自 {p}（{s}）。保存会把整个包复制到你的工作区，之后以你的副本为准。',
  'bundle:': '包内文件：', '+{n} more': '还有 {n} 个', 'no other files in this bundle': '该包没有其他文件', 'Saved to {p}': '已保存到 {p}',
  'Editing “{n}”.': '正在编辑“{n}”。', 'Id and API model name are both required.': '标识和 API 模型名都必须填写。', 'Saved “{n}”.': '已保存“{n}”。',
  'Saved — this applies from the next run onwards.': '已保存——从下次运行开始生效。',
  'Remembered — added to your global memory: <i>{t}</i>': '已记住——已加入你的全局记忆：<i>{t}</i>', 'Remember': '记住',
  'Save this as a standing preference in your global memory': '把这条保存为全局记忆中的常驻偏好',
  '{p} — running log': '{p} — 运行日志', '{p} — journal': '{p} — 实验记录',
  'No compacted entries yet — one is written after each analysis run.': '还没有摘要条目——每次分析运行后会写入一条。', '(Compaction is currently switched off.)': '（摘要功能当前已关闭。）',
  'Nothing recorded yet — the journal is written when a run finishes.': '还没有记录——运行结束时会写入实验记录。',
  'A decision, a client requirement, why an approach was dropped…': '一个决定、一条客户要求、放弃某个方法的原因…', 'Save note': '保存备注',
  'Note ·': '备注 ·', 'earlier · {n} runs': '之前 · {n} 次运行', 'now': '现在', 'replay · {id}': '回放 · {id}', 'end of replay': '回放结束',
  '{n} rounds · {c} self-corr · {t}s': '{n} 轮 · {c} 次自我纠错 · {t} 秒', 'stopped after {n} rounds · {t}s': '{n} 轮后停止 · {t} 秒', 'run failed': '运行失败', 'failed': '失败',
  'Project <b>{p}</b> is open. Describe an analysis and press <b>Run</b>.': '项目 <b>{p}</b> 已打开。描述分析需求并按<b>运行</b>。',
  'No trace stored for {id}.': '没有 {id} 的轨迹记录。',
  'No model is available yet. Open <b>Settings → API keys</b> and paste an API key (the key is stored on the server, not in this browser), or point the <b>Local model</b> entry at a server of your own and fetch what it is serving.': '还没有可用的模型。打开<b>设置 → API 密钥</b>粘贴一个密钥（密钥保存在本机，不在浏览器里），或在<b>本地模型</b>中指向你自己的服务器并获取它提供的模型。',
  'No model configured': '未配置模型',
  'Initializing {model}...': '正在初始化 {model}…', 'Agent running...': '助手运行中…', 'Loaded skill: {name}': '已加载 skill：{name}',
  'Writing the closing note…': '正在写收尾说明…', 'Writing project log…': '正在写项目日志…',
  'Stopped by request after {n} round(s).': '按要求在第 {n} 轮后停止。', '{k} file(s) had been produced by then.': '此前已产出 {k} 个文件。',
  'A run is already in progress. Stop it or wait for it to finish.': '已有一次运行在进行中。请先停止它或等它结束。',
  // server-provided names and hints
  'Esri Light Gray (no key · no buildings, zoom to 16)': 'Esri 浅灰底图（免 key · 无建筑，最高 16 级）', 'Esri Street Map (no key)': 'Esri 街道图（免 key）',
  'Esri Topographic (no key)': 'Esri 地形图（免 key）', 'Esri World Imagery (no key)': 'Esri 卫星影像（免 key）', 'OpenTopoMap (no key)': 'OpenTopoMap（免 key）',
  'OpenStreetMap (no key · light use only)': 'OpenStreetMap（免 key · 仅限轻量使用）', 'MapTiler (key)': 'MapTiler（需 key）', 'Mapbox (key)': 'Mapbox（需 key）',
  'Thunderforest (key)': 'Thunderforest（需 key）', 'Custom XYZ template': '自定义 XYZ 模板', 'MBTiles file (offline)': 'MBTiles 文件（离线）', 'No basemap (data only)': '无底图（只显示数据）',
  "The OpenStreetMap Foundation's own tile servers are for light, occasional use; for regular work use another source or an MBTiles file.": 'OpenStreetMap 基金会的瓦片服务器只供轻量、偶尔使用；日常工作请换其他来源或 MBTiles 文件。',
  'Any {z}/{x}/{y} service — a national portal, a company server, a tileserver of your own, or a keyed CARTO basemap. Use {key} in the template for a token, {s} for a subdomain, {r} for @2x tiles.': '任何 {z}/{x}/{y} 服务——国家地理信息平台、单位内网、自建 tileserver，或带 key 的 CARTO 底图。模板中用 {key} 代表令牌，{s} 代表子域，{r} 代表 @2x 瓦片。',
  'A raster .mbtiles on this computer. QGIS makes one from any layers: Processing → Raster tools → Generate XYZ tiles (MBTiles).': '本机上的栅格 .mbtiles 文件。QGIS 可以从任意图层生成：处理 → 栅格工具 → 生成 XYZ 瓦片（MBTiles）。',
  'Ollama': 'Ollama', 'LM Studio': 'LM Studio', 'vLLM': 'vLLM', 'Other (OpenAI-compatible)': '其他（OpenAI 兼容）',
  'Local model (Ollama / LM Studio / vLLM)': '本地模型（Ollama / LM Studio / vLLM）', 'Anthropic (Claude)': 'Anthropic（Claude）', 'Google Gemini': 'Google Gemini', 'Custom (OpenAI-compatible)': '自定义（OpenAI 兼容）',
  'Serve a model first (e.g. `ollama pull qwen2.5-coder:14b`), then press Fetch below to list what it is serving. Nothing leaves your machine with these — see Help → About.': '先在本机启动一个模型服务（例如 `ollama pull qwen2.5-coder:14b`），然后在下方点「获取」列出它提供的模型。使用本地模型时数据不会离开本机——见 帮助 → 关于。',
  '24 GB': '24 GB', '16 GB (not a 16 GB Mac)': '16 GB（16 GB 的 Mac 不够）', '16 GB': '16 GB',
  'Completed the full urban-heat workflow in 12 rounds with no help.': '无需干预，12 轮完成了完整的城市热岛分析流程。',
  'Reliable on 3–5 step tasks; the usual choice on a desktop GPU.': '3–5 步的任务表现稳定；台式机显卡上的常用选择。',
  'Mixture-of-experts; fast, adequate planning.': '混合专家模型；速度快，规划能力够用。',
});

/* ---------------------------------------------------------------- 한국어 -- */
I18N.register('ko', {
  'GIS Analyst Agent': 'GIS 분석 에이전트',
  'Idle': '대기', 'Running': '실행 중', 'Stopping…': '중지하는 중…', 'Done': '완료', 'Stopped': '중지됨', 'Failed': '실패',
  'Stop after the current step': '현재 단계가 끝나면 중지',
  'Stopping — finishing the current step…': '중지하는 중 — 현재 단계를 마치는 중…',
  'Already stopping — press again to stop watching this run': '이미 중지하는 중 — 다시 누르면 이 실행을 더 이상 보지 않습니다',
  'Stopped watching this run. It ends on the server after the current step, and the result is kept in the project.':
    '이 실행을 더 이상 보지 않습니다. 현재 단계가 끝나면 중지되며, 결과는 프로젝트에 기록됩니다.',
  'The run had already ended.': '이 실행은 이미 끝났습니다.',
  'Could not reach the server to stop the run.': '실행을 중지하기 위해 서버에 연결하지 못했습니다.',
  'Running in <b>{p}</b>': '<b>{p}</b>에서 실행 중', 'Open it': '열기',
  'Project': '프로젝트', 'Tools': '도구', 'Settings': '설정', 'Help': '도움말',
  '＋ New project…': '＋ 새 프로젝트…', '＋ Add data…': '＋ 데이터 추가…', 'Log (compacted)…': '로그(요약)…',
  'Journal (full record)…': '저널(전체 기록)…', 'Add note to journal…': '저널에 메모 추가…',
  'Start a new conversation': '새 대화 시작', 'Archived projects…': '보관된 프로젝트…', 'Toolbox…': '도구 상자…',
  'API keys…': 'API 키…', 'Models…': '모델…', 'Local models…': '로컬 모델…', 'Map…': '지도…', 'Skills…': '스킬…', 'Memory…': '메모리…',
  'About GISclaw &amp; source code…': 'GISclaw 정보 및 소스 코드…',
  'Projects': '프로젝트', '＋ New': '＋ 새로 만들기',
  'Use the <b>Project</b> menu above to create a project and add data.': '위의 <b>프로젝트</b> 메뉴에서 프로젝트를 만들고 데이터를 추가하세요.',
  'Map': '지도', 'Code': '코드', 'Table': '표', 'Result': '결과', 'Map layers': '지도 레이어',
  'Click a dataset on the left to show it here.': '왼쪽의 데이터셋을 클릭하면 여기에 표시됩니다.',
  'Raw text': '원본 텍스트', 'Ready': '준비됨', 'Executed': '실행됨', 'Error': '오류',
  'Execution output': '실행 출력', 'Output appears here as the agent runs code.': '에이전트가 코드를 실행하면 출력이 여기에 표시됩니다.',
  'Steps · 0': '단계 · 0', 'Self-corrections · 0': '자가 수정 · 0', 'Elapsed · 00:00': '경과 · 00:00',
  'Steps · {n}': '단계 · {n}', 'Self-corrections · {n}': '자가 수정 · {n}', 'Elapsed · {t}': '경과 · {t}',
  'No project selected': '선택된 프로젝트 없음', 'AI analyst': 'AI 분석가',
  'Describe the analysis, pick a model, then press Run.': '분석 내용을 설명하고 모델을 고른 뒤 실행을 누르세요.',
  'Info': '안내', 'You': '나', 'Thought': '생각', 'Action': '동작', 'Observation': '관찰', 'Issue': '문제', 'Answer': '답변',
  'Create a project on the left, add your GIS data, then describe an analysis and press <b>Run</b>.': '왼쪽에서 프로젝트를 만들고 GIS 데이터를 추가한 뒤, 분석 내용을 설명하고 <b>실행</b>을 누르세요.',
  'Clear': '지우기', 'Run': '실행', 'Stop': '중지',
  'Add data': '데이터 추가',
  "Pick files or folders from the workspace. They are copied into the project's <code>data/</code>.": '작업 공간에서 파일이나 폴더를 고르세요. 프로젝트의 <code>data/</code>로 복사됩니다.',
  'Upload files…': '파일 업로드…', 'Upload a folder…': '폴더 업로드…',
  '…or drop them here. Use this for data anywhere on your computer — the list below only reaches the mounted workspace.': '…또는 여기에 끌어다 놓으세요. 컴퓨터 어디에 있는 데이터든 이렇게 올릴 수 있습니다. 아래 목록은 마운트된 작업 공간만 보여 줍니다.',
  '0 selected': '0개 선택됨', '{n} selected': '{n}개 선택됨', 'Cancel': '취소', 'Attach selected': '선택 항목 추가',
  'Archived projects': '보관된 프로젝트',
  'Archiving moves a project out of the workspace but keeps every file. Restore brings it back.': '보관하면 프로젝트가 작업 공간에서 빠지지만 파일은 모두 유지됩니다. 복원하면 되돌아옵니다.',
  'Close': '닫기', 'Rename project': '프로젝트 이름 바꾸기',
  'The folder on disk is renamed too, unless that name is already taken.': '디스크의 폴더 이름도 함께 바뀝니다(같은 이름이 이미 있으면 제외).',
  'Rename': '이름 바꾸기', 'Are you sure?': '정말 진행할까요?', 'Delete': '삭제', 'New project': '새 프로젝트',
  'A project is a working folder holding your data, outputs, and run history.': '프로젝트는 데이터, 산출물, 실행 기록을 담는 작업 폴더입니다.',
  'Create': '만들기', 'Toolbox': '도구 상자',
  'Run a GIS operation directly (deterministic, no AI cost) — or insert it as a request for the agent.': 'GIS 연산을 직접 실행하거나(결정적, AI 비용 없음) 에이전트에게 요청으로 넣습니다.',
  'Select an operation on the left.': '왼쪽에서 연산을 선택하세요.', 'Attribute table': '속성 테이블',
  'API keys': 'API 키', 'Models': '모델', 'Local models': '로컬 모델', 'Skills': '스킬', 'Memory': '메모리',
  'Keys are stored on the server in <code id="setPath">…</code> (file mode 600) and survive container rebuilds. They are never sent back to this page — only a mask is.': '키는 이 컴퓨터의 <code id="setPath">…</code>에 저장되며(파일 권한 600) 컨테이너를 다시 만들어도 유지됩니다. 이 페이지로는 마스킹된 값만 돌아옵니다.',
  'Only enabled models with a configured key appear in the run selector. Any OpenAI-compatible endpoint works — pick the <b>Custom</b> provider and give its base URL.': '활성화되고 키가 설정된 모델만 실행 선택 목록에 나타납니다. OpenAI 호환 엔드포인트라면 무엇이든 됩니다. <b>Custom</b> 제공자를 고르고 base URL을 입력하세요.',
  'What your providers are serving right now': '제공자가 현재 서비스 중인 모델', 'Fetch model list': '모델 목록 가져오기',
  'Add / edit a model': '모델 추가 / 편집', 'Id': '식별자', 'Display name': '표시 이름', 'Provider': '제공자',
  'API model name': 'API 모델 이름', 'Base URL': 'Base URL', 'Max rounds': '최대 라운드', 'Max tokens': '최대 토큰',
  'Cost in / out per 1M': '100만 토큰당 비용(입력 / 출력)', 'Clear form': '양식 지우기', 'Save model': '모델 저장',
  "A model served on this computer — Ollama, LM Studio, vLLM — keeps the whole analysis here: nothing is sent to a provider, nothing is billed. Small models plan multi-step work less reliably than the hosted flagships; the ones recommended below have completed this project's benchmark tasks. <b>Context length matters most:</b> a run needs at least <span id=\"localMinCtx\">8,192</span> tokens, or the server silently drops the instructions off the front of the prompt.": '이 컴퓨터에서 실행되는 모델(Ollama, LM Studio, vLLM)은 분석 전체를 로컬에 둡니다. 제공자에게 아무것도 보내지 않고 요금도 없습니다. 작은 모델은 다단계 분석 계획이 클라우드 플래그십보다 덜 안정적입니다. 아래 추천 모델은 이 프로젝트의 벤치마크 과제를 완료한 것들입니다. <b>컨텍스트 길이가 가장 중요합니다.</b> 한 번의 실행에 최소 <span id="localMinCtx">8,192</span> 토큰이 필요하며, 부족하면 서버가 프롬프트 앞부분의 지시를 조용히 잘라냅니다.',
  'Server': '서버', 'Connect': '연결', 'install ↗': '설치 ↗', 'Recommended models': '추천 모델',
  'The map behind your data. Tiles are fetched by GISclaw, not by this page — a key stays on this computer, and every tile you have looked at is kept in the data folder, so the area stays available offline. Continents, borders and lakes are always drawn from a built-in offline layer, whatever else loads.': '데이터 뒤에 깔리는 지도입니다. 타일은 이 페이지가 아니라 GISclaw가 가져오므로 키는 이 컴퓨터에 남고, 한 번 본 타일은 데이터 폴더에 보관되어 오프라인에서도 그 지역을 볼 수 있습니다. 대륙, 국경, 호수는 무엇이 로드되든 내장 오프라인 레이어로 항상 그려집니다.',
  'Source': '출처', 'Key': '키', 'Tile template': '타일 템플릿', 'Attribution': '저작자 표시', 'MBTiles file': 'MBTiles 파일',
  'get a key ↗': '키 받기 ↗', 'Keep the tiles I have looked at, for offline use': '본 타일을 오프라인용으로 보관',
  'Clear tile cache': '타일 캐시 비우기', 'Apply': '적용',
  'A skill is a <b>directory bundle</b> — <code>SKILL.md</code> (a short router) plus <code>references/</code>, <code>assets/</code>, <code>manifest.yaml</code>. Same shape as Claude Code skills, so bundles are interchangeable. Only the one-line description is always in context; the router and its references load on demand.': '스킬은 <b>디렉터리 번들</b>입니다. <code>SKILL.md</code>(짧은 라우터)와 <code>references/</code>, <code>assets/</code>, <code>manifest.yaml</code>로 이루어집니다. Claude Code 스킬과 같은 구조라 서로 바꿔 쓸 수 있습니다. 한 줄 설명만 항상 컨텍스트에 있고, 라우터와 참고 자료는 필요할 때 로드됩니다.',
  'Auto-load the best-matching skill for each task <i>(models rarely open one themselves)</i>': '작업마다 가장 잘 맞는 스킬을 자동 로드 <i>(모델이 스스로 여는 일은 드뭅니다)</i>',
  'New': '새로 만들기', 'Import .zip': '.zip 가져오기', 'Import folder': '폴더 가져오기', 'Save': '저장',
  'Standing preferences applied to <b>every</b> project — cartography, deliverable conventions, house CRS. Injected into the agent\'s system prompt at the start of each run. Stored in <code id="memPath">…</code>.': '<b>모든</b> 프로젝트에 적용되는 상시 선호 사항입니다. 지도 표현, 산출물 규칙, 기본 좌표계 등. 실행이 시작될 때마다 에이전트의 시스템 프롬프트에 들어갑니다. 저장 위치: <code id="memPath">…</code>.',
  'Apply memory to runs': '실행에 메모리 적용',
  'Follow the agent between Map, Code and Result while it works <i>(off keeps the tab you chose)</i>': '실행 중 지도·코드·결과 탭을 에이전트를 따라 전환 <i>(끄면 선택한 탭에 머뭅니다)</i>',
  'Save memory': '메모리 저장', 'Journal': '저널',
  'Append-only record of this project: every run, what was asked, what it produced. Lives at <code id="journalPath">…</code> — readable without this app.': '이 프로젝트의 추가 전용 기록입니다. 모든 실행, 요청 내용, 산출물. 위치: <code id="journalPath">…</code>. 이 앱 없이도 읽을 수 있습니다.',
  'Add a note': '메모 추가',
  'Create a new project': '새 프로젝트 만들기', 'Add data to the selected project': '선택한 프로젝트에 데이터 추가',
  'Run GIS operations directly': 'GIS 연산 직접 실행', 'Drag to resize · double-click to reset': '끌어서 크기 조절 · 더블클릭으로 초기화',
  'Zoom in': '확대', 'Zoom out': '축소', 'Fit to data': '데이터에 맞추기', 'Filter rows…': '행 필터…',
  'e.g. Compute building density per block and map it.': '예: 블록별 건물 밀도를 계산해 지도로 그려 줘.',
  'Model': '모델', 'Clear the view': '화면 지우기', 'Project name': '프로젝트 이름',
  'Query — filter rows by any field…': '쿼리 — 아무 필드로나 행 필터…', "paste the provider's key": '제공자의 키를 붙여넣기',
  '© whoever made the map': '© 지도 제작자', '…or a folder path under the workspace, e.g. _incoming/my-skill': '…또는 작업 공간 안의 폴더 경로(예: _incoming/my-skill)',
  // runtime
  'Basemap': '배경 지도', 'offline reference': '오프라인 참고 지도', 'vector': '벡터', 'raster': '래스터', 'Toggle visibility': '표시 / 숨기기',
  'Right-click a layer for symbology, attribute table…': '레이어를 오른쪽 클릭하면 심볼, 속성 테이블 등을 열 수 있습니다.',
  'Show layer': '레이어 표시', 'Hide layer': '레이어 숨기기', 'Zoom to layer': '레이어로 이동', 'Fit to all layers': '모든 레이어에 맞추기',
  'Symbology…': '심볼 설정…', 'Remove layer': '레이어 제거', 'Opacity': '불투명도', 'Color': '색상', 'Fill opacity': '채우기 불투명도',
  '{name} — {n} features': '{name} — 피처 {n}개', '(showing first {n})': '(처음 {n}개만 표시)', '{n} rows': '{n}행',
  'Could not load {name}': '{name}을(를) 불러오지 못했습니다', '{name}: not a valid GeoJSON layer.': '{name}: 올바른 GeoJSON 레이어가 아닙니다.', 'overlay failed': '오버레이 실패',
  'Rename…': '이름 바꾸기…', 'Export as zip': 'zip으로 내보내기', 'Archive…': '보관…', 'Collapse': '접기', 'Open': '열기', 'Journal…': '저널…',
  'Delete project…': '프로젝트 삭제…', 'Delete file…': '파일 삭제…',
  'data': '데이터', 'records': '기록', 'outputs': '산출물', '(empty — add data)': '(비어 있음 — 데이터를 추가하세요)',
  'Journal (full record)': '저널(전체 기록)', 'Log (compacted)': '로그(요약)', 'Conversation': '대화',
  'No projects yet.<br/>Press <b>＋ New</b> to create one.': '아직 프로젝트가 없습니다.<br/><b>＋ 새로 만들기</b>를 누르세요.',
  'Describe an analysis and press Run.': '분석 내용을 설명하고 실행을 누르세요.', 'Add data to this project to begin.': '먼저 이 프로젝트에 데이터를 추가하세요.',
  'Working…': '작업 중…', 'Reasoning · {n} steps': '추론 · {n}단계', 'Reasoning · {n} steps · {t}s': '추론 · {n}단계 · {t}초',
  'Agent output': '에이전트 산출물', '{n} lines': '{n}줄', '{a} of {b} rows × {c} cols': '{b}행 중 {a}행 × {c}열', '· showing first {n}': '· 처음 {n}행만 표시',
  'Started a new conversation. The previous one is archived in the project folder, and JOURNAL.md still holds every run.': '새 대화를 시작했습니다. 이전 대화는 프로젝트 폴더에 보관되었고, JOURNAL.md에는 모든 실행이 남아 있습니다.',
  'Please describe the analysis first.': '먼저 분석 내용을 설명해 주세요.', 'This project has no data yet. Add data first.': '이 프로젝트에는 아직 데이터가 없습니다. 먼저 추가하세요.',
  'No model configured — open <b>Settings → API keys</b> and add one first.': '설정된 모델이 없습니다. <b>설정 → API 키</b>에서 먼저 추가하세요.',
  'Rejoined the run in progress ({id}).': '진행 중인 실행에 다시 연결했습니다({id}).',
  'Could not start ({code}).': '시작할 수 없습니다({code}).', 'The connection to the server was lost.': '서버와의 연결이 끊어졌습니다.',
  'Could not rejoin the run ({code}).': '실행에 다시 연결할 수 없습니다({code}).', 'Network error': '네트워크 오류', 'Stopped.': '중지되었습니다.',
  '_No closing summary was written for this run._': '_이 실행에는 마무리 요약이 작성되지 않았습니다._', 'Produced': '산출물',
  'Finished in <b>{t}s</b> · {n} rounds · {k} output(s)': '<b>{t}초</b> 만에 완료 · {n}라운드 · 산출물 {k}개',
  'Log entry written to LOG.md': 'LOG.md에 로그 항목을 기록했습니다',
  'uploading {i}/{n} — {name}': '업로드 중 {i}/{n} — {name}', '{n} uploaded': '{n}개 업로드됨', ', {n} failed': ', {n}개 실패',
  'Uploaded {n} file(s) to {p}.': '{p}에 파일 {n}개를 업로드했습니다.', '{n} failed.': ' {n}개 실패.',
  'Delete file': '파일 삭제',
  '<b>{f}</b> will be deleted from <code>{w}/</code>.<br/>This cannot be undone. Earlier runs keep their own copies under <code>runs/</code>.': '<b>{f}</b>이(가) <code>{w}/</code>에서 삭제됩니다.<br/>되돌릴 수 없습니다. 이전 실행은 <code>runs/</code>에 자체 복사본을 유지합니다.',
  '<br/><br/>Its <code>.shx</code>/<code>.dbf</code>/<code>.prj</code> siblings go with it — a lone <code>.shp</code> is unreadable.': '<br/><br/>함께 있는 <code>.shx</code>/<code>.dbf</code>/<code>.prj</code>도 같이 삭제됩니다. <code>.shp</code> 하나만으로는 읽을 수 없습니다.',
  'Deleted {list} from {w}/.': '{w}/에서 {list}을(를) 삭제했습니다.',
  'Delete project': '프로젝트 삭제',
  '{n} data file(s), {m} output(s), {r} run(s)': '데이터 파일 {n}개, 산출물 {m}개, 실행 {r}회', '{n} data file(s) and its whole run history': '데이터 파일 {n}개와 전체 실행 기록',
  '<b>{p}</b> and everything in it — {counts} — will be deleted from disk. This cannot be undone.<br/><br/>Prefer <b>Archive</b> if you might want it back, or <b>Export as zip</b> first.': '<b>{p}</b>와 그 안의 모든 것({counts})이 디스크에서 삭제됩니다. 되돌릴 수 없습니다.<br/><br/>나중에 필요할 수 있다면 <b>보관</b>하거나 먼저 <b>zip으로 내보내기</b>를 하세요.',
  'Deleted project "{p}".': '프로젝트 "{p}"을(를) 삭제했습니다.',
  'Archived "{p}". Nothing was deleted — bring it back from Project → Archived projects.': '"{p}"을(를) 보관했습니다. 삭제된 것은 없으며 프로젝트 → 보관된 프로젝트에서 되돌릴 수 있습니다.',
  'Loading…': '불러오는 중…', 'Nothing archived yet.': '보관된 프로젝트가 없습니다.', '{n} file(s) · {m} run(s)': '파일 {n}개 · 실행 {m}회',
  'Restore': '복원', 'Restored "{p}".': '"{p}"을(를) 복원했습니다.', 'Delete archived project': '보관된 프로젝트 삭제',
  '<b>{p}</b> — {n} file(s), {m} run(s) — will be deleted from disk. This cannot be undone.': '<b>{p}</b>(파일 {n}개, 실행 {m}회)이(가) 디스크에서 삭제됩니다. 되돌릴 수 없습니다.',
  'Deleted archived project "{p}".': '보관된 프로젝트 "{p}"을(를) 삭제했습니다.', 'workspace': '작업 공간',
  'Added {n} file(s) to {p}.': '{p}에 파일 {n}개를 추가했습니다.',
  'output name': '출력 이름', 'Insert into chat': '대화에 넣기', 'Running…': '실행 중…', 'Failed — check inputs.': '실패 — 입력을 확인하세요.', 'Done → {files}': '완료 → {files}',
  '(no {kind} layers — add data)': '({kind} 레이어 없음 — 데이터를 추가하세요)',
  'no key needed': '키 불필요', 'key saved': '키 저장됨', 'from {env}': '{env}에서', 'no key': '키 없음', 'set up →': '설정하기 →',
  '{mask} (stored — type to replace)': '{mask}  (저장됨 — 입력하면 교체)', 'paste your API key': 'API 키를 붙여넣기',
  'Test': '테스트', 'Remove': '제거', 'Nothing to save — fill in the address first.': '저장할 내용이 없습니다. 먼저 주소를 입력하세요.',
  'Nothing to save — paste a key first.': '저장할 내용이 없습니다. 먼저 키를 붙여넣으세요.', 'Saving…': '저장 중…', 'Saved.': '저장했습니다.',
  'Calling the API…': 'API 호출 중…', 'Works — {model} replied "{reply}".': '정상 — {model}이(가) "{reply}"라고 답했습니다.', 'Failed.': '실패했습니다.',
  'Show in the run selector': '실행 선택 목록에 표시', 'ready': '준비됨', 'no endpoint': '엔드포인트 없음', 'custom': '사용자 정의', 'Edit': '편집', 'Disable': '비활성화',
  'Asking the provider…': '제공자에 문의 중…', 'Could not list models.': '모델 목록을 가져오지 못했습니다.',
  '{n} chat model(s)': '대화 모델 {n}개', '· {n} non-chat hidden': ' · 대화용이 아닌 {n}개 숨김', 'Nothing returned.': '돌아온 결과가 없습니다.', 'added': '추가됨', 'Add this model': '이 모델 추가',
  'Added {m} — it is now in the run selector.': '{m}을(를) 추가했습니다. 이제 실행 선택 목록에 있습니다.',
  'Press <b>Connect</b> to see what the server is serving.': '<b>연결</b>을 누르면 서버가 제공하는 모델을 볼 수 있습니다.',
  'added · context {n} chars per round': '추가됨 · 라운드당 컨텍스트 {n}자', 'context {n}': '컨텍스트 {n}', '(loaded)': ' (로드됨)',
  'context up to {n} · server default applies': '컨텍스트 최대 {n} · 서버 기본값 적용', 'context unknown': '컨텍스트 알 수 없음',
  'The server answered but has no models. Pull one first — see below.': '서버는 응답했지만 모델이 없습니다. 먼저 하나 받으세요(아래 참고).',
  'Copy': '복사', 'Copied': '복사됨', 'Enter the server address first.': '먼저 서버 주소를 입력하세요.', 'Connecting…': '연결 중…', 'No answer.': '응답이 없습니다.',
  'an OpenAI-compatible server': 'OpenAI 호환 서버', 'Connected to {kind} — {n} model(s).': '{kind}에 연결했습니다. 모델 {n}개.',
  '{m} is loaded with a {ctx}-token context. Raise it in the Ollama app (Settings → Context length) or start the server with OLLAMA_CONTEXT_LENGTH={rec}.': '{m}이(가) {ctx} 토큰 컨텍스트로 로드되어 있습니다. Ollama 앱의 설정 → Context length에서 늘리거나 OLLAMA_CONTEXT_LENGTH={rec}로 서버를 시작하세요.',
  'Added {m} — it is in the run selector. Press Test to load it once.': '{m}을(를) 추가했습니다. 실행 선택 목록에 있습니다. 테스트를 눌러 한 번 로드하세요.',
  'Calling {m}… (the first call loads the model; this can take a minute)': '{m} 호출 중… (첫 호출은 모델을 로드하므로 1분쯤 걸릴 수 있습니다)',
  'Works — {m} replied "{reply}".': '정상 — {m}이(가) "{reply}"라고 답했습니다.', 'Loaded with a {ctx}-token context.': ' {ctx} 토큰 컨텍스트로 로드되었습니다.',
  'Cache: {mb} MB.': '캐시: {mb} MB.', 'Source reachable ({ms} ms).': '출처에 연결됨({ms}ms).', 'Source failed: {detail}': '출처 실패: {detail}', 'Applied — {name}.': '적용됨 — {name}.', 'Tile cache cleared.': '타일 캐시를 비웠습니다.',
  'Importing {f}…': '{f} 가져오는 중…', 'Installed “{n}”.': '“{n}”을(를) 설치했습니다.',
  'Make available to the agent': '에이전트가 쓸 수 있게 하기', 'always on': '항상 켜짐', 'user': '사용자', 'builtin': '내장', '{n} files': '파일 {n}개',
  'Router ~{n} tok, loaded on demand': '라우터 약 {n}토큰, 필요할 때 로드', '{n} tok always': '항상 {n}토큰', 'View': '보기', 'Export': '내보내기', 'Fork': '작업 공간으로 복사',
  'Copied into your workspace — your version now wins.': '작업 공간으로 복사했습니다. 이제 내 버전이 우선합니다.',
  'Enabled skills cost ~{n} tokens on every call (routers and references load on demand, on top).': '활성화된 스킬은 호출마다 약 {n}토큰을 씁니다(라우터와 참고 자료는 필요할 때 추가로 로드).',
  'No skills found.': '스킬이 없습니다.',
  'Your bundle at {p}. Changes apply on the next run — no restart.': '내 번들 위치: {p}. 변경 사항은 다음 실행부터 적용되며 재시작은 필요 없습니다.',
  'Read from {p} ({s}). Saving forks the whole bundle into your workspace, where your copy wins.': '{p}({s})에서 읽었습니다. 저장하면 번들 전체가 작업 공간으로 복사되고 내 복사본이 우선합니다.',
  'bundle:': '번들 파일:', '+{n} more': '외 {n}개', 'no other files in this bundle': '이 번들에는 다른 파일이 없습니다', 'Saved to {p}': '{p}에 저장했습니다',
  'Editing “{n}”.': '“{n}” 편집 중.', 'Id and API model name are both required.': '식별자와 API 모델 이름은 모두 필요합니다.', 'Saved “{n}”.': '“{n}”을(를) 저장했습니다.',
  'Saved — this applies from the next run onwards.': '저장했습니다. 다음 실행부터 적용됩니다.',
  'Remembered — added to your global memory: <i>{t}</i>': '기억했습니다. 전역 메모리에 추가됨: <i>{t}</i>', 'Remember': '기억하기',
  'Save this as a standing preference in your global memory': '이 내용을 전역 메모리의 상시 선호로 저장',
  '{p} — running log': '{p} — 실행 로그', '{p} — journal': '{p} — 저널',
  'No compacted entries yet — one is written after each analysis run.': '아직 요약 항목이 없습니다. 분석 실행이 끝날 때마다 하나씩 기록됩니다.', '(Compaction is currently switched off.)': ' (요약 기능이 꺼져 있습니다.)',
  'Nothing recorded yet — the journal is written when a run finishes.': '아직 기록이 없습니다. 실행이 끝나면 저널에 기록됩니다.',
  'A decision, a client requirement, why an approach was dropped…': '결정 사항, 고객 요구, 어떤 방법을 접은 이유…', 'Save note': '메모 저장',
  'Note ·': '메모 ·', 'earlier · {n} runs': '이전 · 실행 {n}회', 'now': '지금', 'replay · {id}': '재생 · {id}', 'end of replay': '재생 끝',
  '{n} rounds · {c} self-corr · {t}s': '{n}라운드 · 자가 수정 {c}회 · {t}초', 'stopped after {n} rounds · {t}s': '{n}라운드 후 중지 · {t}초', 'run failed': '실행 실패', 'failed': '실패',
  'Project <b>{p}</b> is open. Describe an analysis and press <b>Run</b>.': '프로젝트 <b>{p}</b>이(가) 열렸습니다. 분석 내용을 설명하고 <b>실행</b>을 누르세요.',
  'No trace stored for {id}.': '{id}의 추적 기록이 없습니다.',
  'No model is available yet. Open <b>Settings → API keys</b> and paste an API key (the key is stored on the server, not in this browser), or point the <b>Local model</b> entry at a server of your own and fetch what it is serving.': '아직 쓸 수 있는 모델이 없습니다. <b>설정 → API 키</b>에서 키를 붙여넣거나(키는 브라우저가 아니라 이 컴퓨터에 저장됩니다) <b>로컬 모델</b>에서 내 서버를 지정해 제공 중인 모델을 가져오세요.',
  'No model configured': '설정된 모델 없음',
  'Initializing {model}...': '{model} 초기화 중…', 'Agent running...': '에이전트 실행 중…', 'Loaded skill: {name}': '스킬 로드됨: {name}',
  'Writing the closing note…': '마무리 메모 작성 중…', 'Writing project log…': '프로젝트 로그 작성 중…',
  'Stopped by request after {n} round(s).': '요청에 따라 {n}라운드 후 중지했습니다.', '{k} file(s) had been produced by then.': '그때까지 파일 {k}개가 생성되었습니다.',
  'A run is already in progress. Stop it or wait for it to finish.': '이미 실행이 진행 중입니다. 중지하거나 끝날 때까지 기다리세요.',
  // server-provided names and hints
  'Esri Light Gray (no key · no buildings, zoom to 16)': 'Esri 라이트 그레이(키 불필요 · 건물 없음, 16레벨까지)', 'Esri Street Map (no key)': 'Esri 도로 지도(키 불필요)',
  'Esri Topographic (no key)': 'Esri 지형도(키 불필요)', 'Esri World Imagery (no key)': 'Esri 위성 영상(키 불필요)', 'OpenTopoMap (no key)': 'OpenTopoMap(키 불필요)',
  'OpenStreetMap (no key · light use only)': 'OpenStreetMap(키 불필요 · 가벼운 사용만)', 'MapTiler (key)': 'MapTiler(키 필요)', 'Mapbox (key)': 'Mapbox(키 필요)',
  'Thunderforest (key)': 'Thunderforest(키 필요)', 'Custom XYZ template': '사용자 정의 XYZ 템플릿', 'MBTiles file (offline)': 'MBTiles 파일(오프라인)', 'No basemap (data only)': '배경 지도 없음(데이터만)',
  "The OpenStreetMap Foundation's own tile servers are for light, occasional use; for regular work use another source or an MBTiles file.": 'OpenStreetMap 재단의 타일 서버는 가볍고 가끔 쓰는 용도입니다. 일상 작업에는 다른 출처나 MBTiles 파일을 쓰세요.',
  'Any {z}/{x}/{y} service — a national portal, a company server, a tileserver of your own, or a keyed CARTO basemap. Use {key} in the template for a token, {s} for a subdomain, {r} for @2x tiles.': '{z}/{x}/{y} 형식의 어떤 서비스든 됩니다. 국가 포털, 회사 서버, 직접 띄운 타일 서버, 키가 있는 CARTO 지도 등. 템플릿에서 {key}는 토큰, {s}는 서브도메인, {r}는 @2x 타일입니다.',
  'A raster .mbtiles on this computer. QGIS makes one from any layers: Processing → Raster tools → Generate XYZ tiles (MBTiles).': '이 컴퓨터의 래스터 .mbtiles 파일. QGIS에서 어떤 레이어로든 만들 수 있습니다: 공간 처리 → 래스터 도구 → XYZ 타일 생성(MBTiles).',
  'Ollama': 'Ollama', 'LM Studio': 'LM Studio', 'vLLM': 'vLLM', 'Other (OpenAI-compatible)': '기타(OpenAI 호환)',
  'Local model (Ollama / LM Studio / vLLM)': '로컬 모델(Ollama / LM Studio / vLLM)', 'Anthropic (Claude)': 'Anthropic(Claude)', 'Google Gemini': 'Google Gemini', 'Custom (OpenAI-compatible)': '사용자 정의(OpenAI 호환)',
  'Serve a model first (e.g. `ollama pull qwen2.5-coder:14b`), then press Fetch below to list what it is serving. Nothing leaves your machine with these — see Help → About.': '먼저 모델을 띄우고(예: `ollama pull qwen2.5-coder:14b`) 아래의 가져오기를 눌러 제공 중인 모델을 확인하세요. 로컬 모델은 데이터가 컴퓨터 밖으로 나가지 않습니다(도움말 → 정보 참고).',
  '24 GB': '24 GB', '16 GB (not a 16 GB Mac)': '16 GB(16 GB Mac은 부족)', '16 GB': '16 GB',
  'Completed the full urban-heat workflow in 12 rounds with no help.': '도움 없이 12라운드 만에 도시 열섬 분석 전체 흐름을 완료했습니다.',
  'Reliable on 3–5 step tasks; the usual choice on a desktop GPU.': '3~5단계 과제에서 안정적. 데스크톱 GPU에서 흔히 쓰는 선택.',
  'Mixture-of-experts; fast, adequate planning.': '전문가 혼합 모델. 빠르고 계획 능력도 무난합니다.',
});

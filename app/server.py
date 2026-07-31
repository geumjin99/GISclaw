#!/usr/bin/env python3
# GISclaw — an LLM agent for geospatial analysis.
# Copyright (C) 2026 Han Jinzhen
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This file is part of GISclaw. GISclaw is free software: you can redistribute
# it and/or modify it under the terms of the GNU Affero General Public License
# as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed in the hope
# that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Affero General Public License in the LICENSE file, or
# <https://www.gnu.org/licenses/>, for more details.

"""
GISclaw — product backend (single-agent ReAct, cloud LLMs).

This is the *product* server, separate from the benchmark-oriented ui/server.py.
It has no notion of benchmark tasks; instead it manages user "projects" (working
folders) under a workspace root, runs the single ReAct agent live over a
project's data, and streams Thought/Action/Observation to the browser via SSE.

Run (dev):   uvicorn app.server:app --host 0.0.0.0 --port 8765 --reload
Run (docker): handled by Dockerfile / docker-compose.yml

API keys are managed in the UI (Settings) and stored under
<WORKSPACE>/.gisclaw/settings.json; the environment variables OPENAI_API_KEY,
CLAUDE_API_KEY, DEEPSEEK_API_KEY, GEMINI_API_KEY still work as a fallback, so
existing .env setups keep running unchanged.
Workspace root comes from GISCLAW_WORKSPACE (default: <repo>/projects for dev,
/workspace in Docker).

Durability: every run appends to the project's chat.jsonl (conversation, rebuilt
by the UI on load) and JOURNAL.md (human-readable lab notebook), and the recent
history plus the global MEMORY.md are injected back into the system prompt so a
months-long project keeps its context. See app/journal.py, app/settings_store.py.
"""
import asyncio
import json
import os
import queue
import re
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from app import journal, reflect
from app.logging_setup import RunRecorder, get_app_logger
from app.settings_store import PROVIDERS, SettingsStore, mask_key
from app.skills_store import SkillsStore

# ============================================================
# Config
# ============================================================
WEB_DIR = os.path.join(PROJECT_ROOT, "app", "web")
APP_LOG = os.path.join(PROJECT_ROOT, "app", "server.log")
log = get_app_logger(APP_LOG)


def _resolve_workspace() -> str:
    """Workspace root: env override, else <repo>/projects for local dev."""
    ws = os.environ.get("GISCLAW_WORKSPACE")
    if not ws:
        ws = os.path.join(PROJECT_ROOT, "projects")
    os.makedirs(ws, exist_ok=True)
    return os.path.abspath(ws)


WORKSPACE = _resolve_workspace()
log.info(f"Workspace root: {WORKSPACE}")

# Settings live on the mounted volume so they survive container rebuilds.
# The curated model list now comes from settings_store.BUILTIN_MODELS, which the
# user can disable, edit, or extend with their own OpenAI-compatible endpoints.
# Per-project record files, surfaced in the UI tree (read-only).
RECORD_FILES = ("JOURNAL.md", "LOG.md", "chat.jsonl")

STORE = SettingsStore(WORKSPACE)
SKILLS = SkillsStore(WORKSPACE, os.path.join(PROJECT_ROOT, "app", "skills"))
log.info(f"Settings: {STORE.path}")

# System prompt (product free-form mode) — same rules as the v4 ReAct prompt.
SYSTEM_PROMPT = """You are an expert GIS analyst agent. You solve geospatial analysis tasks by thinking step-by-step and using tools to interact with data in a persistent Python sandbox.

## Available Tools

{tool_descriptions}

## Response Format

You MUST respond in this EXACT format every time:

Thought: <your reasoning about what to do next>
Action: <tool_name>
Args: <arguments as JSON object>

## Core Analysis Rules
{skill_rule}
1. Start by listing files (list_files), then load and inspect ALL datasets.
2. Plan your approach in your first Thought.
3. **Prefer built-in geoprocess operations** for standard GIS steps (reproject,
   buffer, clip, overlay, spatial/attribute join, dissolve, zonal statistics,
   slope/aspect/hillshade, rasterize, etc.). They are deterministic and CRS-aware
   — do NOT hand-write code for anything a geoprocess op already covers.
4. Use execute() (write code) ONLY for what no geoprocess op covers: custom
   formulas, bespoke modelling, and visualization/plots.
5. CRS first — geoprocess handles this for you; in execute() use a projected CRS
   for distance/area operations.
6. Schema-driven — read actual column names from data, don't hardcode.
7. Visualization — plt.savefig() in the SAME execute() call. Never plt.show().
8. Save ALL final outputs to pred_results/ before calling finish().
9. After saving, verify: print(os.listdir('pred_results/'))

## Available Packages
geopandas, rasterio, shapely, fiona, pyproj, numpy, pandas, scipy, matplotlib,
sklearn, libpysal, esda, mgwr, xarray, rasterstats, networkx, osmnx, seaborn,
mapclassify, h3, momepy, pointpats, spaghetti, openpyxl, rtree, geoplot, cartopy
{skills_block}{catalog_block}{memory_block}{context_block}"""

# Level 0 — bodies of `always: true` skills. Standing rules, injected every run.
SKILLS_BLOCK = """
## Operating skills

{skills}
"""

# Level 1 — the catalog. One line per on-demand skill; the model opens what it
# needs with the `skill` tool. This is the only part every call pays for.
SKILL_CATALOG_BLOCK = """
## Available skills (load on demand)

Each entry below is a bundle you have NOT read yet: a short router plus reference
files holding the detailed procedure.

{catalog}

**Before your first Action, decide whether one of them covers this task.** If one
does, your first Action must be `skill` — reading it after you have already
started is too late to change the plan, which is the whole point of loading it.
If none applies, proceed normally and do not mention them again.

- `skill(name)` — read that skill's router and list its bundled files.
- `skill(name, path)` — read one bundled file, e.g. `references/api.md`.

Follow the router: it tells you which files to read and in what order. Read them
one at a time, only when a step needs them — never all of them up front.
"""

# Numbered rules are what the model actually obeys — prose further down the
# prompt loses to them. So the skill step has to live inside the rule list.
SKILL_RULE = """
0. **Skills first.** Check the "Available skills" section at the end of this
   prompt. If one covers this task, your VERY FIRST action must be
   `skill(name="…")`, before list_files — its router decides the analysis path,
   so loading it after you have started is pointless. If none fits, skip this.
"""

# Level 2, promoted server-side when the task clearly matches a skill. The model
# keeps the option of going deeper into the bundle on its own.
SKILL_AUTOLOAD_BLOCK = """
## Loaded skill: {name}

This skill was matched to the task and loaded for you. Follow it — it defines the
analysis path. Its reference files are NOT loaded; read them one at a time, as
its router instructs, with `skill(name="{name}", path="…")`.

{body}
"""

SKILL_TOOL_DESC = """

skill: Open an available skill, or read one file from its bundle. Use when a listed
  skill matches the task — its router tells you which of its files to read next.
  Args: {"name": "<skill name>"} or {"name": "<skill name>", "path": "references/api.md"}
  Returns: the router text plus a file listing, or the requested file's contents."""


def _skill_tool(name: str = "", path: str = "") -> str:
    """The agent-facing side of progressive disclosure (level 2 and 3)."""
    overrides = STORE.skill_overrides()
    name = (name or "").strip()
    if not name:
        cat = SKILLS.build_catalog(overrides)
        return f"Available skills:\n{cat}" if cat else "No skills are enabled."
    sk = SKILLS.get(name, overrides)
    if not sk or not sk.get("enabled", False):
        avail = [s["name"] for s in SKILLS.discover(overrides) if s["enabled"]]
        return f"❌ No enabled skill named '{name}'. Available: {', '.join(avail) or 'none'}"

    if path:
        res = SKILLS.read_resource(name, path, overrides)
        if res.get("error"):
            files = [r["path"] for r in SKILLS.list_resources(name, overrides) if r["readable"]]
            return f"❌ {res['error']}\nReadable files in '{name}': {', '.join(files[:40]) or 'none'}"
        return f"📄 {name}/{path}\n\n{res['text']}"

    files = SKILLS.list_resources(name, overrides)
    listing = "\n".join(
        f"  - {f['path']}  ({f['size']} bytes)" + ("" if f["readable"] else "  [not text]")
        for f in files[:60]) or "  (no bundled files)"
    more = f"\n  … and {len(files) - 60} more" if len(files) > 60 else ""
    return (f"📘 Skill: {name}\n\n{sk['body']}\n\n"
            f"--- Bundled files (read with skill(name=\"{name}\", path=\"…\")) ---\n"
            f"{listing}{more}")


def _install_skill_tool():
    """Give every GISToolkit a `skill` tool — product process only.

    The agent builds its own toolkit inside `react_agent.run()`, so the only way
    to add a tool without editing research code is to patch the class here. This
    process never runs the paper experiments, so `tools.py` keeps its 6-tool
    surface everywhere that matters for reproduction. Same trick the product
    already uses for `build_system_prompt`.
    """
    from src.agent.tools import GISToolkit
    if getattr(GISToolkit, "_gisclaw_skill_patched", False):
        return
    orig_init = GISToolkit.__init__

    def patched_init(self, *a, **kw):
        orig_init(self, *a, **kw)
        if SKILLS.build_catalog(STORE.skill_overrides()):
            self.tools["skill"] = _skill_tool
            self.tool_descriptions = self.tool_descriptions + SKILL_TOOL_DESC
        _make_finish_tolerant(self)

    GISToolkit.__init__ = patched_init
    GISToolkit._gisclaw_skill_patched = True


def _salvage_summary(extra: dict) -> str:
    """Recover a summary from mis-parsed finish arguments.

    Long markdown summaries come back as `Args: {"summary": "…"}` with embedded
    newlines; when that fails to parse, the whole JSON string arrives as a single
    kwarg *name*. The run then ends with an argument error and the agent's final
    write-up — often the only place a caveat like "170 blocks were imputed" is
    stated — is lost. Salvage it instead of discarding it.
    """
    for key, value in extra.items():
        for candidate in (key, value):
            if not isinstance(candidate, str):
                continue
            try:
                parsed = json.loads(candidate)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(parsed, dict) and parsed.get("summary"):
                return str(parsed["summary"])
        if isinstance(key, str) and len(key) > 20:
            return key          # last resort: the raw text is better than nothing
    return ""


def _make_finish_tolerant(toolkit):
    orig_finish = toolkit.tools.get("finish")
    if orig_finish is None:
        return

    def finish(summary: str = "", **extra):
        if not summary and extra:
            summary = _salvage_summary(extra)
            if summary:
                log.warning("finish args failed to parse; summary salvaged")
        return orig_finish(summary=summary)

    toolkit.tools["finish"] = finish


_install_skill_tool()

# Injected only when non-empty, so a first run in a fresh install pays no tokens.
MEMORY_BLOCK = """
## Standing user preferences

These apply to every project. Follow them unless this task says otherwise —
especially for cartography, symbology and deliverable conventions.

{memory}
"""

CONTEXT_BLOCK = """
## This project so far

{context}
"""

# ============================================================
# Path safety
# ============================================================
def _safe_join(root: str, *parts: str) -> str:
    """Join under root and refuse anything that escapes it (path traversal)."""
    root = os.path.abspath(root)
    target = os.path.abspath(os.path.join(root, *parts))
    if target != root and not target.startswith(root + os.sep):
        raise ValueError(f"Path escapes root: {target}")
    return target


def _slug(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip()).strip("_")
    return s or "project"


# ============================================================
# Project helpers
# ============================================================
def _project_dir(pid: str) -> str:
    return _safe_join(WORKSPACE, _slug(pid))


def _project_layout(pdir: str):
    for sub in ("data", "outputs", "runs"):
        os.makedirs(os.path.join(pdir, sub), exist_ok=True)


def _read_manifest(pdir: str) -> dict:
    mpath = os.path.join(pdir, "project.json")
    if os.path.exists(mpath):
        try:
            with open(mpath, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _write_manifest(pdir: str, manifest: dict):
    with open(os.path.join(pdir, "project.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def _list_projects() -> list:
    out = []
    for name in sorted(os.listdir(WORKSPACE)):
        pdir = os.path.join(WORKSPACE, name)
        if not os.path.isdir(pdir):
            continue
        if not os.path.exists(os.path.join(pdir, "project.json")):
            continue  # a project is a folder we created with a manifest
        m = _read_manifest(pdir)
        data_dir = os.path.join(pdir, "data")
        n_data = len([f for f in os.listdir(data_dir)]) if os.path.isdir(data_dir) else 0
        out.append({
            "id": name,
            "name": m.get("name", name),
            "created_at": m.get("created_at", ""),
            "notes": m.get("notes", ""),
            "data_count": n_data,
        })
    return out


def _dir_tree(base: str) -> list:
    """Flat list of files under base (relative paths), skipping dotfiles."""
    items = []
    if not os.path.isdir(base):
        return items
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fn in sorted(files):
            if fn.startswith("."):
                continue
            rel = os.path.relpath(os.path.join(root, fn), base)
            items.append(rel)
    return sorted(items)


# ============================================================
# LLM init — cfg comes fully resolved from SettingsStore.model_config()
# ============================================================
def init_llm(cfg: dict):
    """Build an engine from a resolved model config (engine/api_key/base_url)."""
    engine = cfg["engine"]
    key = cfg.get("api_key", "")
    if not key:
        raise RuntimeError(
            f"No API key configured for provider '{cfg.get('provider', '?')}'. "
            "Open Settings → API keys and paste one (or set the matching "
            "environment variable), then try again."
        )
    if engine == "claude":
        from src.agent.llm_engine import ClaudeEngine
        llm = ClaudeEngine(model=cfg["model_name"], api_key=key, temperature=0.1,
                           max_tokens=cfg["max_tokens"],
                           cost_per_m=cfg.get("cost_per_m", (3.0, 15.0)))
    elif engine == "openai":
        from src.agent.llm_engine import OpenAIEngine
        llm = OpenAIEngine(model=cfg["model_name"], api_key=key, temperature=0.1,
                           max_tokens=cfg["max_tokens"],
                           base_url=cfg.get("base_url") or None,
                           cost_per_m=cfg.get("cost_per_m", (2.5, 10.0)))
    else:
        raise ValueError(f"Unknown engine: {engine}")
    llm.load_model()
    return llm


# ============================================================
# Agent run in a thread, streamed via queue + on_step callback
# ============================================================
def run_agent_in_thread(pid: str, model_key: str, instruction: str, msg_queue: queue.Queue):
    recorder = None
    try:
        cfg = STORE.model_config(model_key)
        if not cfg:
            msg_queue.put({"type": "error", "content": f"Unknown model: {model_key}"})
            return
        pdir = _project_dir(pid)
        if not os.path.isdir(pdir):
            msg_queue.put({"type": "error", "content": f"Project not found: {pid}"})
            return
        _project_layout(pdir)
        data_dir = os.path.join(pdir, "data")
        manifest = _read_manifest(pdir)
        project_name = manifest.get("name", pid)

        # The question itself is part of the record, logged before we spend a token.
        journal.append_chat(pdir, {"role": "user", "text": instruction, "model": model_key})

        run_id = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(pdir, "runs", run_id)
        os.makedirs(run_dir, exist_ok=True)
        pred_dir = os.path.join(run_dir, "pred_results")
        os.makedirs(pred_dir, exist_ok=True)

        recorder = RunRecorder(run_dir)
        recorder.log(f"model={model_key} instruction={instruction!r}")
        log.info(f"[{pid}] run {run_id} model={model_key}")

        msg_queue.put({"type": "status", "content": f"Initializing {cfg['display']}...", "run_id": run_id})

        llm = init_llm(cfg)

        # Intent gate: a question about the project shouldn't become an analysis
        # run. The agent's finish-guard demands output files, so without this a
        # "what did you do?" forces it to redo work and write a placeholder just
        # to be allowed to stop.
        manifest = _read_manifest(pdir)
        gate_context = reflect.recent_log(pdir) or journal.build_context(pdir, manifest)
        verdict = reflect.classify_request(llm, instruction, gate_context)
        if verdict["mode"] != "analysis":
            answer = verdict.get("answer", "")
            recorder.log(f"intent={verdict['mode']} answered without running the agent")
            journal.append_chat(pdir, {
                "role": "agent", "run_id": run_id, "model": model_key,
                "model_display": cfg.get("display", model_key), "ask": instruction,
                "success": True, "rounds": 0, "self_corrections": 0, "elapsed_s": 0,
                "outputs": [], "answer": answer, "kind": verdict["mode"],
            })
            msg_queue.put({"type": "answer", "mode": verdict["mode"], "content": answer})
            msg_queue.put({"type": "done", "run_id": run_id, "success": True,
                           "output_files": [], "rounds": 0, "self_corrections": 0,
                           "elapsed_s": 0, "cost": {}, "answered": True})
            return

        from src.agent.error_memory import ErrorMemory
        from src.agent.react_agent import GISReActAgent
        from src.agent.sandbox import PythonSandbox
        from src.agent.tools import GISToolkit
        import src.agent.prompts as prompts_module

        agent = GISReActAgent(llm_engine=llm, timeout=cfg["timeout"],
                              max_rounds=cfg["max_rounds"], verbose=True,
                              rag=None, error_memory=ErrorMemory())

        # Inject product system prompt (build_system_prompt override, same trick as ui/server.py)
        _sandbox = PythonSandbox(work_dir=run_dir, timeout=cfg["timeout"])
        _toolkit = GISToolkit(_sandbox, data_dir="dataset")

        # Continuity layer: standing user preferences + what this project already did.
        memory_text = STORE.memory_for_prompt() if STORE.memory_enabled() else ""
        memory_block = MEMORY_BLOCK.format(memory=memory_text) if memory_text else ""
        # Prefer the compacted log over the raw ask list — same continuity, far
        # fewer tokens, and it carries caveats the ask list never had.
        digest = reflect.recent_log(pdir)
        context_text = journal.build_context(pdir, manifest, digest=digest)
        context_block = CONTEXT_BLOCK.format(context=context_text) if context_text else ""
        overrides = STORE.skill_overrides()
        skills_text = SKILLS.build_always_block(overrides)
        skills_block = SKILLS_BLOCK.format(skills=skills_text) if skills_text else ""

        # Pre-load the one skill this task is about, if any (see SkillsStore.match).
        matched = SKILLS.match(instruction, overrides) if STORE.skills_auto() else None
        if matched:
            body = matched["body"]
            # Honour the bundle's own manifest.yaml `always_load` list, the
            # ecosystem convention for fragments a router cannot do without.
            for frag in SKILLS.manifest_always_load(matched["name"], overrides):
                body += f"\n\n--- {matched['name']}/{frag['path']} ---\n\n{frag['text']}"
            skills_block += SKILL_AUTOLOAD_BLOCK.format(name=matched["name"], body=body)
            log.info(f"[{pid}] skill auto-loaded: {matched['name']} (score {matched['match_score']})")
            msg_queue.put({"type": "status",
                           "content": f"Loaded skill: {matched['name']}"})

        catalog_text = SKILLS.build_catalog(overrides, exclude=matched["name"] if matched else "")
        catalog_block = SKILL_CATALOG_BLOCK.format(catalog=catalog_text) if catalog_text else ""

        prompt_text = SYSTEM_PROMPT.format(
            tool_descriptions=_toolkit.tool_descriptions,
            skill_rule=SKILL_RULE if catalog_text else "",
            skills_block=skills_block,
            catalog_block=catalog_block,
            memory_block=memory_block,
            context_block=context_block,
        )
        recorder.log(f"prompt: always_skills={len(skills_text)}c catalog={len(catalog_text)}c "
                     f"memory={len(memory_text)}c context={len(context_text)}c")
        del _sandbox, _toolkit
        _orig_bsp = prompts_module.build_system_prompt
        prompts_module.build_system_prompt = lambda **kw: prompt_text

        # Track which output files we've already announced
        seen_outputs = set()
        steps_log = []          # condensed step list for JOURNAL.md

        def on_step(ev: dict):
            """Called by the agent after each round — push a live SSE event."""
            recorder.event(ev)
            step = {
                "round": ev.get("round", 0),
                "action": ev.get("action", ""),
                "thought": ev.get("thought", ""),
                "success": ev.get("success", True),
            }
            if ev.get("action") == "finish":
                # The agent's closing write-up — the one place caveats usually land.
                step["observation"] = ev.get("observation_full") or ev.get("observation") or ""
            steps_log.append(step)
            step_msg = {
                "type": "step",
                "round": ev.get("round", 0),
                "thought": ev.get("thought", ""),
                "action": ev.get("action", ""),
                "code": ev.get("code", ""),
                "observation": ev.get("observation_full", ev.get("observation", "")),
                "success": ev.get("success", True),
            }
            msg_queue.put(step_msg)
            # Announce any new result files as they appear
            if os.path.isdir(pred_dir):
                for fn in sorted(os.listdir(pred_dir)):
                    if fn.startswith(".") or fn in seen_outputs:
                        continue
                    seen_outputs.add(fn)
                    msg_queue.put({"type": "result", "run_id": run_id,
                                   "filename": fn,
                                   "url": f"/api/projects/{pid}/runfile?run={run_id}&path={fn}"})

        msg_queue.put({"type": "status", "content": "Agent running...", "run_id": run_id})
        t0 = time.time()
        result = agent.run(
            task_id=0, instruction=instruction, workflow="",
            data_dir=data_dir, work_dir=run_dir,
            domain_knowledge="", dataset_description="", rag_context="", skill_text="",
            on_step=on_step,
        )
        elapsed = time.time() - t0
        prompts_module.build_system_prompt = _orig_bsp

        # Persist generated code as code.py, copy outputs to project/outputs/
        code_src = os.path.join(run_dir, "_react_t0.py")
        if os.path.exists(code_src):
            shutil.copy2(code_src, os.path.join(run_dir, "code.py"))
        out_root = os.path.join(pdir, "outputs")
        os.makedirs(out_root, exist_ok=True)
        output_files = []
        if os.path.isdir(pred_dir):
            for fn in sorted(os.listdir(pred_dir)):
                if fn.startswith("."):
                    continue
                output_files.append(fn)
                shutil.copy2(os.path.join(pred_dir, fn), os.path.join(out_root, fn))

        cost_info = {}
        if hasattr(llm, "get_stats"):
            s = llm.get_stats()
            cost_info = {
                "api_calls": s.get("total_calls", 0),
                "input_tokens": s.get("total_input_tokens", 0),
                "output_tokens": s.get("total_output_tokens", 0),
                "cost_usd": s.get("estimated_cost_usd", 0),
            }

        recorder.log(f"done success={result.success} rounds={result.total_rounds} "
                     f"self_corrections={result.self_corrections} outputs={output_files}")

        # Durable record: one conversation entry + one journal section per run.
        summary = {
            "role": "agent",
            "run_id": run_id,
            "model": model_key,
            "model_display": cfg.get("display", model_key),
            "ask": instruction,
            "success": bool(result.success),
            "rounds": result.total_rounds,
            "self_corrections": result.self_corrections,
            "elapsed_s": round(elapsed, 1),
            "outputs": output_files,
            "cost": cost_info,
        }
        summary["final_summary"] = next(
            (s.get("observation", "") for s in reversed(steps_log)
             if s.get("action") == "finish"), "")
        entry = journal.append_chat(pdir, summary)
        try:
            journal.append_run(pdir, project_name, dict(entry, steps=steps_log))
        except Exception as e:
            log.warning(f"journal write failed: {e}")

        # One extra API call to compact this run into LOG.md — the digest a later
        # session reads instead of the whole transcript.
        if STORE.compact_log():
            try:
                msg_queue.put({"type": "status", "content": "Writing project log…"})
                digest = reflect.compact_run(llm, pdir, project_name,
                                             dict(entry, steps=steps_log), steps_log)
                if digest:
                    msg_queue.put({"type": "log", "content": digest})
            except Exception as e:
                log.warning(f"log compaction failed: {e}")

        msg_queue.put({
            "type": "done",
            "run_id": run_id,
            "success": bool(result.success),
            "output_files": output_files,
            "rounds": result.total_rounds,
            "self_corrections": result.self_corrections,
            "elapsed_s": round(elapsed, 1),
            "cost": cost_info,
        })
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log.error(f"run failed: {e}\n{tb[-800:]}")
        if recorder:
            recorder.log(f"ERROR {e}\n{tb[-800:]}")
        # A failed run is still history — record it so the conversation is honest.
        try:
            pdir_err = _project_dir(pid)
            if os.path.isdir(pdir_err):
                journal.append_chat(pdir_err, {
                    "role": "agent", "model": model_key, "ask": instruction,
                    "success": False, "error": str(e), "outputs": [],
                    "rounds": 0, "self_corrections": 0, "elapsed_s": 0,
                })
        except Exception:
            pass
        msg_queue.put({"type": "error", "content": f"{e}\n{tb[-500:]}"})
    finally:
        if recorder:
            recorder.close()
        msg_queue.put(None)


# ============================================================
# FastAPI app
# ============================================================
app = FastAPI(title="GISclaw", description="GIS Analyst Agent (product)")


@app.get("/api/models")
async def api_models(all: int = 0):
    """Models for the run selector. Default: only enabled ones with a key."""
    models = STORE.models_public()
    if not all:
        models = [m for m in models if m["enabled"] and m["ready"]]
    return models


# ------------------------------ settings: keys, models, memory --------------
@app.get("/api/settings")
async def api_settings():
    """Everything the Settings panel needs. Keys come back masked, never raw."""
    return {
        "providers": STORE.providers_public(),
        "models": STORE.models_public(),
        "memory_enabled": STORE.memory_enabled(),
        "settings_path": STORE.path,
        "memory_path": STORE.memory_path,
    }


@app.post("/api/settings/providers/{provider_id}")
async def api_set_provider(provider_id: str, request: Request):
    """Store an API key (and base_url for custom endpoints) for one provider."""
    if provider_id not in PROVIDERS:
        return JSONResponse({"error": f"unknown provider '{provider_id}'"}, status_code=400)
    body = await request.json()
    key = body.get("api_key")
    base_url = body.get("base_url")
    if body.get("clear"):
        STORE.clear_provider_key(provider_id)
        log.info(f"cleared API key for {provider_id}")
    else:
        STORE.set_provider(provider_id, api_key=key, base_url=base_url)
        log.info(f"stored API key for {provider_id} ({mask_key(STORE.provider_key(provider_id))})")
    return {"providers": STORE.providers_public(), "models": STORE.models_public()}


@app.post("/api/settings/providers/{provider_id}/test")
async def api_test_provider(provider_id: str, request: Request):
    """Smallest possible live call, so a bad key fails here instead of mid-run."""
    if provider_id not in PROVIDERS:
        return JSONResponse({"error": f"unknown provider '{provider_id}'"}, status_code=400)
    body = await request.json() if await request.body() else {}
    model_name = (body.get("model_name") or "").strip()
    if not model_name:
        for m in STORE.models().values():
            if m.get("provider") == provider_id and m.get("enabled", True):
                model_name = m.get("model_name", "")
                break
    if not model_name:
        return JSONResponse({"error": "no model to test with — add one first"}, status_code=400)

    def _probe():
        cfg = {
            "engine": PROVIDERS[provider_id]["engine"],
            "model_name": model_name,
            # Thinking models burn the budget before emitting text — see CLAUDE.md.
            "max_tokens": 512,
            "api_key": STORE.provider_key(provider_id),
            "base_url": STORE.provider_base_url(provider_id),
            "provider": provider_id,
            "cost_per_m": (0.0, 0.0),
        }
        llm = init_llm(cfg)
        return llm.generate("Reply with the single word: ok")

    try:
        res = await asyncio.get_event_loop().run_in_executor(None, _probe)
        text = (res.get("text") if isinstance(res, dict) else str(res)) or ""
        # The engines return API failures as text rather than raising.
        if text.startswith("Error"):
            return {"ok": False, "model_name": model_name, "error": text[:400]}
        return {"ok": True, "model_name": model_name, "reply": text.strip()[:120]}
    except Exception as e:
        return {"ok": False, "model_name": model_name, "error": str(e)[:400]}


@app.post("/api/settings/models")
async def api_upsert_model(request: Request):
    """Add or edit a model. Built-ins accept overrides; new ids become custom."""
    body = await request.json()
    mid = _slug(body.get("id") or body.get("model_name") or "")
    if not mid:
        return JSONResponse({"error": "id required"}, status_code=400)
    provider = body.get("provider", "openai")
    if provider not in PROVIDERS:
        return JSONResponse({"error": f"unknown provider '{provider}'"}, status_code=400)
    spec = {
        "provider": provider,
        "model_name": (body.get("model_name") or "").strip(),
        "display": (body.get("display") or mid).strip(),
        "tier": body.get("tier", "Custom"),
        "timeout": int(body.get("timeout", 300)),
        "max_rounds": int(body.get("max_rounds", 50)),
        "max_tokens": int(body.get("max_tokens", 4096)),
        "cost_per_m": [float(body.get("cost_in", 0) or 0), float(body.get("cost_out", 0) or 0)],
        "enabled": bool(body.get("enabled", True)),
    }
    if body.get("base_url"):
        spec["base_url"] = body["base_url"].strip()
    if not spec["model_name"]:
        return JSONResponse({"error": "model_name required (the API's own id)"}, status_code=400)
    STORE.upsert_model(mid, spec)
    log.info(f"model saved: {mid} -> {provider}/{spec['model_name']}")
    return {"id": mid, "models": STORE.models_public()}


@app.post("/api/settings/models/{mid}/toggle")
async def api_toggle_model(mid: str, request: Request):
    body = await request.json()
    STORE.upsert_model(mid, {"enabled": bool(body.get("enabled", True))})
    return {"models": STORE.models_public()}


@app.delete("/api/settings/models/{mid}")
async def api_delete_model(mid: str):
    ok = STORE.delete_model(mid)
    return {"ok": ok, "models": STORE.models_public()}


def _discover_models(provider_id: str) -> dict:
    """Ask the provider what it is actually serving right now.

    Every OpenAI-compatible endpoint answers GET /models; Anthropic has its own
    models.list(). Both SDKs are already dependencies, so no raw HTTP here.
    """
    key = STORE.provider_key(provider_id)
    if not key:
        return {"ok": False, "error": "No API key for this provider yet."}
    meta = PROVIDERS.get(provider_id, {})
    try:
        if meta.get("engine") == "claude":
            import anthropic
            client = anthropic.Anthropic(api_key=key)
            ids = [m.id for m in client.models.list(limit=100).data]
        else:
            from openai import OpenAI
            base = STORE.provider_base_url(provider_id)
            client = OpenAI(api_key=key, **({"base_url": base} if base else {}))
            ids = [m.id for m in client.models.list().data]
    except Exception as e:
        return {"ok": False, "error": str(e)[:300]}

    # Chat models only — these endpoints also serve embeddings, audio, images.
    skip = ("embed", "whisper", "tts", "dall-e", "moderation", "image",
            "audio", "realtime", "rerank", "aqa", "veo", "imagen")
    chat = [i for i in ids if not any(s in i.lower() for s in skip)]
    known = {m["model_name"] for m in STORE.models().values()}
    return {
        "ok": True,
        "models": [{"id": i, "already_added": i in known} for i in sorted(chat)],
        "total": len(ids),
        "filtered_out": len(ids) - len(chat),
    }


@app.get("/api/settings/providers/{provider_id}/available")
async def api_available_models(provider_id: str):
    """Live list of the models this provider is serving to your key."""
    if provider_id not in PROVIDERS:
        return JSONResponse({"error": f"unknown provider '{provider_id}'"}, status_code=400)
    return await asyncio.get_event_loop().run_in_executor(
        None, lambda: _discover_models(provider_id))


# ------------------------------------------------- skills (prompt injection) --
@app.get("/api/skills")
async def api_skills():
    skills = SKILLS.discover(STORE.skill_overrides())
    return {
        "skills": skills,
        "user_dir": SKILLS.user_dir,
        "auto": STORE.skills_auto(),
        "roots": [{"source": s, "path": p, "exists": os.path.isdir(p)}
                  for s, p in SKILLS.roots()],
        # What the prompt pays unconditionally: always-bodies + catalog lines.
        "enabled_tokens_est": sum(s["always_tokens_est"] for s in skills if s["enabled"]),
    }


@app.get("/api/skills/{name}")
async def api_skill(name: str):
    sk = SKILLS.get(name, STORE.skill_overrides())
    if not sk:
        return JSONResponse({"error": "not found"}, status_code=404)
    with open(sk["path"], encoding="utf-8") as f:
        raw = f.read()
    return dict(sk, raw=raw, files=SKILLS.list_resources(name, STORE.skill_overrides()))


@app.get("/api/skills/{name}/file")
async def api_skill_file(name: str, path: str):
    """Read one file from a bundle — the same view the agent gets."""
    res = SKILLS.read_resource(name, path, STORE.skill_overrides())
    if res.get("error"):
        return JSONResponse(res, status_code=400)
    return res


@app.post("/api/skills/import")
async def api_import_skill(request: Request):
    """Install a skill bundle.

    Two transports, both aligned with how skills are shared in the Claude Code
    ecosystem: a raw .zip body, or a path to a directory the server can see
    (anything under the mounted workspace).
    """
    ctype = request.headers.get("content-type", "")
    try:
        if "application/json" in ctype:
            body = await request.json()
            src = (body.get("path") or "").strip()
            if not src:
                return JSONResponse({"error": "path required"}, status_code=400)
            if not os.path.isabs(src):
                src = _safe_join(WORKSPACE, src)
            name = SKILLS.import_path(src)
        else:
            blob = await request.body()
            if not blob:
                return JSONResponse({"error": "empty upload"}, status_code=400)
            name = SKILLS.import_zip(blob)
    except (ValueError, FileNotFoundError) as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    log.info(f"skill imported: {name}")
    return {"ok": True, "name": name, **(await api_skills())}


@app.get("/api/skills/{name}/export")
async def api_export_skill(name: str):
    """Zip the bundle back up — drop it into ~/.claude/skills and it just works."""
    try:
        blob = SKILLS.export_zip(name)
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    return Response(content=blob, media_type="application/zip", headers={
        "Content-Disposition": f'attachment; filename="{_slug(name)}.zip"'})


@app.post("/api/skills/auto")
async def api_skills_auto(request: Request):
    body = await request.json()
    STORE.set_skills_auto(bool(body.get("enabled", True)))
    return await api_skills()


@app.post("/api/skills/{name}/toggle")
async def api_toggle_skill(name: str, request: Request):
    body = await request.json()
    STORE.set_skill_enabled(name, bool(body.get("enabled", True)))
    log.info(f"skill {name} enabled={body.get('enabled')}")
    return await api_skills()


@app.put("/api/skills/{name}")
async def api_write_skill(name: str, request: Request):
    """Save a skill into the workspace (editing a builtin forks it there)."""
    body = await request.json()
    text = body.get("raw")
    if not text:
        return JSONResponse({"error": "raw (the SKILL.md text) required"}, status_code=400)
    path = SKILLS.write_user_skill(name, text)
    log.info(f"skill saved: {path}")
    return {"ok": True, "path": path, **(await api_skills())}


@app.post("/api/skills")
async def api_new_skill(request: Request):
    body = await request.json()
    name = _slug(body.get("name") or "")
    if not name:
        return JSONResponse({"error": "name required"}, status_code=400)
    if SKILLS.get(name):
        return JSONResponse({"error": f"a skill named '{name}' already exists"}, status_code=409)
    path = SKILLS.write_user_skill(name, SKILLS.new_skill_template(name))
    return {"ok": True, "name": name, "path": path, **(await api_skills())}


@app.post("/api/skills/{name}/fork")
async def api_fork_skill(name: str):
    """Copy the whole bundle (references included) into the workspace to edit."""
    try:
        path = SKILLS.fork(name)
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    return {"ok": True, "path": path, **(await api_skills())}


@app.delete("/api/skills/{name}")
async def api_delete_skill(name: str):
    """Removes the workspace copy; a shadowed builtin reappears underneath."""
    ok = SKILLS.delete_user_skill(name)
    if ok and not SKILLS.get(name):
        STORE.forget_skill(name)
    return {"ok": ok, **(await api_skills())}


@app.get("/api/memory")
async def api_get_memory():
    return {"text": STORE.read_memory(), "enabled": STORE.memory_enabled(),
            "path": STORE.memory_path}


@app.put("/api/memory")
async def api_put_memory(request: Request):
    body = await request.json()
    if "text" in body:
        STORE.write_memory(body["text"])
    if "enabled" in body:
        STORE.set_memory_enabled(bool(body["enabled"]))
    return {"text": STORE.read_memory(), "enabled": STORE.memory_enabled()}


@app.post("/api/memory/append")
async def api_append_memory(request: Request):
    """The 'Remember this' button: one fact, filed under a heading."""
    body = await request.json()
    line = (body.get("text") or "").strip()
    if not line:
        return JSONResponse({"error": "text required"}, status_code=400)
    section = (body.get("section") or "Notes").strip() or "Notes"
    text = STORE.append_memory(line, section=section)
    log.info(f"memory += [{section}] {line[:80]}")
    return {"text": text}


@app.get("/api/tools")
async def api_tools():
    """Structured geoprocess catalog for the UI Toolbox (grouped ops + param specs)."""
    from src.agent import geo_ops
    return geo_ops.catalog()


@app.get("/api/projects")
async def api_projects():
    return _list_projects()


@app.post("/api/projects")
async def api_create_project(request: Request):
    body = await request.json()
    name = (body.get("name") or "").strip()
    if not name:
        return JSONResponse({"error": "name required"}, status_code=400)
    pid = _slug(name)
    pdir = _project_dir(pid)
    if os.path.exists(os.path.join(pdir, "project.json")):
        return JSONResponse({"error": "project already exists", "id": pid}, status_code=409)
    _project_layout(pdir)
    _write_manifest(pdir, {
        "name": name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "notes": body.get("notes", ""),
        "crs_hint": body.get("crs_hint", ""),
    })
    log.info(f"created project {pid}")
    return {"id": pid, "name": name}


@app.get("/api/projects/{pid}/tree")
async def api_project_tree(pid: str):
    pdir = _project_dir(pid)
    if not os.path.isdir(pdir):
        return JSONResponse({"error": "not found"}, status_code=404)
    return {
        "id": pid,
        "manifest": _read_manifest(pdir),
        "data": _dir_tree(os.path.join(pdir, "data")),
        "outputs": _dir_tree(os.path.join(pdir, "outputs")),
        # The project's own record files, so they are visible in the tree rather
        # than only behind menu items.
        "records": [f for f in RECORD_FILES if os.path.isfile(os.path.join(pdir, f))],
        "runs": sorted([d for d in os.listdir(os.path.join(pdir, "runs"))
                        if os.path.isdir(os.path.join(pdir, "runs", d))], reverse=True)
        if os.path.isdir(os.path.join(pdir, "runs")) else [],
    }


@app.get("/api/browse")
async def api_browse(path: str = ""):
    """Server-side file browser rooted at WORKSPACE (the mounted volume)."""
    try:
        target = _safe_join(WORKSPACE, path) if path else WORKSPACE
    except ValueError:
        return JSONResponse({"error": "invalid path"}, status_code=400)
    if not os.path.isdir(target):
        return JSONResponse({"error": "not a directory"}, status_code=404)
    entries = []
    for name in sorted(os.listdir(target)):
        if name.startswith("."):
            continue
        full = os.path.join(target, name)
        entries.append({
            "name": name,
            "is_dir": os.path.isdir(full),
            "rel": os.path.relpath(full, WORKSPACE),
            "size": os.path.getsize(full) if os.path.isfile(full) else 0,
        })
    rel_here = os.path.relpath(target, WORKSPACE)
    return {
        "root": WORKSPACE,
        "here": "" if rel_here == "." else rel_here,
        "entries": entries,
    }


# Formats that are really a set of files sharing one stem. Attaching only the
# .shp yields a layer nothing can open, and a missing .prj silently drops the
# CRS — so when one member is picked, its siblings travel with it.
SIDECAR_GROUPS = {
    ".shp": [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qpj", ".qix", ".sbn",
             ".sbx", ".fbn", ".fbx", ".ain", ".aih", ".atx", ".shp.xml"],
    ".tab": [".tab", ".dat", ".map", ".id", ".ind"],
    ".mif": [".mif", ".mid"],
}
_SIDECAR_OF = {ext: grp for grp in SIDECAR_GROUPS.values() for ext in grp}


def _companion_files(src: str) -> list:
    """Every file that has to travel with `src` for it to stay readable."""
    low = src.lower()
    ext = ".shp.xml" if low.endswith(".shp.xml") else os.path.splitext(low)[1]
    group = _SIDECAR_OF.get(ext)
    if not group:
        return [src]
    stem = src[: len(src) - len(ext)]
    out = []
    for e in group:
        for cand in (stem + e, stem + e.upper()):
            if os.path.isfile(cand) and cand not in out:
                out.append(cand)
    return out or [src]


@app.post("/api/projects/{pid}/attach")
async def api_attach(pid: str, request: Request):
    """Copy selected files/dirs (relative to WORKSPACE) into project data/."""
    pdir = _project_dir(pid)
    if not os.path.isdir(pdir):
        return JSONResponse({"error": "project not found"}, status_code=404)
    _project_layout(pdir)
    body = await request.json()
    rels = body.get("paths", [])
    data_dir = os.path.join(pdir, "data")
    attached = []
    notices = []
    for rel in rels:
        try:
            src = _safe_join(WORKSPACE, rel)
        except ValueError:
            continue
        if not os.path.exists(src):
            continue
        base = os.path.basename(src.rstrip("/"))
        dst = os.path.join(data_dir, base)
        try:
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                attached.append(base)
            else:
                members = _companion_files(src)
                for m in members:
                    shutil.copy2(m, os.path.join(data_dir, os.path.basename(m)))
                    attached.append(os.path.basename(m))
                extras = len(members) - 1
                if extras > 0:
                    notices.append(
                        f"{base}: brought {extras} companion file(s) along "
                        "— a shapefile is unreadable without them.")
                if base.lower().endswith(".shp") and not any(
                        m.lower().endswith(".prj") for m in members):
                    notices.append(
                        f"{base}: no .prj alongside it, so the layer declares no "
                        "coordinate system. It may sit in the wrong place on the "
                        "map, and the agent will see CRS = None.")
        except Exception as e:
            log.error(f"attach failed {rel}: {e}")
    log.info(f"[{pid}] attached {attached}")
    return {"attached": attached, "notices": notices, "data": _dir_tree(data_dir)}


MAX_DISPLAY_FEATURES = 50_000   # above this, thin geometry for the map only
MAX_OVERLAY_PX = 2048           # longest side of a raster overlay PNG

def _serve_geo_or_image(fpath: str):
    """Serve a file for the viewer; convert shp/tif to web-friendly forms."""
    ext = fpath.lower().rsplit(".", 1)[-1] if "." in fpath else ""
    if ext in ("png", "jpg", "jpeg"):
        return FileResponse(fpath, media_type="image/png" if ext == "png" else "image/jpeg")
    # Vector formats -> reproject to WGS84 (EPSG:4326) so Leaflet places them on OSM.
    # Leaflet expects lon/lat; a shp/geojson in any projected CRS must be reprojected
    # first, otherwise the geometry lands in the ocean.
    if ext in ("shp", "geojson", "json", "gpkg", "gml", "kml"):
        try:
            import geopandas as gpd
            gdf = gpd.read_file(fpath)
            if gdf.crs is None:
                # No .prj / no CRS declared — cannot know the projection.
                log.warning(f"{fpath}: no CRS defined; assuming EPSG:4326 (may be misplaced on map)")
            else:
                try:
                    gdf = gdf.to_crs(epsg=4326)  # no-op if already 4326
                except Exception as e:
                    log.warning(f"{fpath}: reproject to 4326 failed ({e}); serving in native CRS")
            # Browsers choke long before GeoPandas does. Above this many
            # features the geometry is thinned *for display only* — the file on
            # disk and everything the agent reads stay untouched.
            notice = ""
            if len(gdf) > MAX_DISPLAY_FEATURES:
                minx, miny, maxx, maxy = gdf.total_bounds
                tol = max(maxx - minx, maxy - miny) / 4000.0
                if tol > 0:
                    gdf = gdf.copy()
                    gdf["geometry"] = gdf.geometry.simplify(
                        tol, preserve_topology=True)
                notice = (f"{len(gdf):,} features — a large layer. Line and "
                          "polygon geometry is thinned for the map (points "
                          "cannot be), so drawing may still be slow. The data "
                          "itself is unchanged.")
            payload = gdf.to_json()
            if notice:
                payload = '{"_notice":' + json.dumps(notice) + "," + payload[1:]
            return Response(content=payload, media_type="application/json")
        except Exception as e:
            # Fall back to raw bytes for plain JSON that geopandas can't parse as vector
            if ext in ("geojson", "json"):
                return FileResponse(fpath, media_type="application/json")
            return JSONResponse({"error": f"vector preview failed: {e}"}, status_code=500)
    if ext in ("tif", "tiff"):
        try:
            import io
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import rasterio
            from rasterio.plot import show as rioshow
            buf = io.BytesIO()
            with rasterio.open(fpath) as ds:
                fig, ax = plt.subplots(figsize=(6, 6))
                rioshow(ds, ax=ax)
                ax.axis("off")
                fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
                plt.close(fig)
            return Response(content=buf.getvalue(), media_type="image/png")
        except Exception as e:
            return JSONResponse({"error": f"raster preview failed: {e}"}, status_code=500)
    # csv / txt / others -> raw text
    try:
        with open(fpath, "r", errors="replace") as f:
            return JSONResponse({"content": f.read()[:20000]})
    except Exception:
        return FileResponse(fpath)


@app.get("/api/projects/{pid}/file")
async def api_project_file(pid: str, path: str, where: str = "data"):
    """Serve a data/ or outputs/ file from a project (for map / image tabs)."""
    pdir = _project_dir(pid)
    if where == "records":
        # Only the project's own record files, by exact name — never arbitrary
        # paths in the project root (settings and manifests stay out of reach).
        if path not in RECORD_FILES:
            return JSONResponse({"error": "not a record file"}, status_code=400)
        fpath = os.path.join(pdir, path)
        if not os.path.isfile(fpath):
            return JSONResponse({"error": "not found"}, status_code=404)
        with open(fpath, encoding="utf-8", errors="replace") as f:
            return {"content": f.read(), "name": path}
    sub = "outputs" if where == "outputs" else "data"
    try:
        fpath = _safe_join(os.path.join(pdir, sub), path)
    except ValueError:
        return JSONResponse({"error": "invalid path"}, status_code=400)
    if not os.path.isfile(fpath):
        return JSONResponse({"error": "not found"}, status_code=404)
    return _serve_geo_or_image(fpath)


def _raster_overlay_payload(fpath: str) -> dict:
    """Reproject a raster to WGS84, render to a colored PNG, return image + bounds
    so the frontend can place it on the OSM basemap via L.imageOverlay."""
    import base64
    import io
    import numpy as np
    import rasterio
    from rasterio.warp import Resampling, calculate_default_transform, reproject
    import matplotlib
    matplotlib.use("Agg")

    with rasterio.open(fpath) as src:
        if src.crs is None:
            raise ValueError("raster has no CRS; cannot place on map")
        dst_crs = "EPSG:4326"
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        # A full-resolution reprojection gets base64'd into the page, so a large
        # scene would otherwise hang the browser. Displaying it smaller costs
        # nothing: the analysis never touches this path.
        if max(width, height) > MAX_OVERLAY_PX:
            f = MAX_OVERLAY_PX / max(width, height)
            width, height = max(1, int(width * f)), max(1, int(height * f))
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds,
                dst_width=width, dst_height=height)
        # Three or more bands are treated as RGB; anything else is a single
        # measured band and gets a colour ramp.
        bands = [1, 2, 3] if src.count >= 3 else [1]
        stack = []
        for b in bands:
            buf_b = np.full((height, width), np.nan, dtype="float32")
            reproject(
                source=rasterio.band(src, b), destination=buf_b,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=transform, dst_crs=dst_crs,
                resampling=Resampling.bilinear, dst_nodata=np.nan,
                src_nodata=src.nodata)
            stack.append(buf_b)
        data = stack[0]
        band_count = src.count
        west, north = transform * (0, 0)
        east, south = transform * (width, height)

    def _stretch(a):
        """2-98 percentile stretch to 0..1, ignoring nodata."""
        ok = np.isfinite(a)
        if not ok.any():
            return np.zeros_like(a), 0.0, 1.0
        lo, hi = np.nanpercentile(a[ok], 2), np.nanpercentile(a[ok], 98)
        if hi <= lo:
            hi = lo + 1
        return np.clip((a - lo) / (hi - lo), 0, 1), float(lo), float(hi)

    valid = np.isfinite(data)
    if len(stack) == 3:
        rgb = np.dstack([_stretch(b)[0] for b in stack])
        rgba = np.dstack([np.nan_to_num(rgb),
                          np.where(valid, 0.95, 0.0)[..., None]])
        vmin, vmax = 0.0, 1.0
        mode = "rgb"
    else:
        norm, vmin, vmax = _stretch(data)
        rgba = matplotlib.colormaps["viridis"](np.nan_to_num(norm))
        rgba[..., 3] = np.where(valid, 0.85, 0.0)  # transparent where nodata
        mode = "ramp"
    buf = io.BytesIO()
    import matplotlib.pyplot as plt
    plt.imsave(buf, rgba, format="png")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {
        "bounds": {"south": float(south), "west": float(west),
                   "north": float(north), "east": float(east)},
        "image": "data:image/png;base64," + b64,
        "vmin": float(vmin), "vmax": float(vmax),
        "mode": mode, "bands": int(band_count), "width": int(width),
        "height": int(height),
    }


@app.get("/api/projects/{pid}/overlay")
async def api_overlay(pid: str, path: str, where: str = "outputs", run: str = ""):
    """Georeferenced raster overlay (PNG + WGS84 bounds) for a .tif on the map."""
    pdir = _project_dir(pid)
    try:
        if run:
            fpath = _safe_join(os.path.join(pdir, "runs", _slug(run), "pred_results"), path)
        else:
            sub = "outputs" if where == "outputs" else "data"
            fpath = _safe_join(os.path.join(pdir, sub), path)
    except ValueError:
        return JSONResponse({"error": "invalid path"}, status_code=400)
    if not os.path.isfile(fpath):
        return JSONResponse({"error": "not found"}, status_code=404)
    try:
        return _raster_overlay_payload(fpath)
    except Exception as e:
        return JSONResponse({"error": f"overlay failed: {e}"}, status_code=500)


@app.get("/api/projects/{pid}/runfile")
async def api_run_file(pid: str, run: str, path: str):
    """Serve a file produced inside a specific run's pred_results/."""
    pdir = _project_dir(pid)
    try:
        fpath = _safe_join(os.path.join(pdir, "runs", _slug(run), "pred_results"), path)
    except ValueError:
        return JSONResponse({"error": "invalid path"}, status_code=400)
    if not os.path.isfile(fpath):
        return JSONResponse({"error": "not found"}, status_code=404)
    return _serve_geo_or_image(fpath)


@app.get("/api/projects/{pid}/chat")
async def api_chat(pid: str):
    """The durable conversation — the UI rebuilds the thread from this on load."""
    pdir = _project_dir(pid)
    if not os.path.isdir(pdir):
        return JSONResponse({"error": "not found"}, status_code=404)
    return {"entries": journal.read_chat(pdir)}


@app.delete("/api/projects/{pid}/chat")
async def api_clear_chat(pid: str):
    """Start a fresh thread. The old one is archived, never deleted."""
    pdir = _project_dir(pid)
    if not os.path.isdir(pdir):
        return JSONResponse({"error": "not found"}, status_code=404)
    journal.clear_chat(pdir)
    log.info(f"[{pid}] conversation archived")
    return {"ok": True}


@app.get("/api/projects/{pid}/log")
async def api_project_log(pid: str):
    """The compacted running log — one model-written digest per run."""
    pdir = _project_dir(pid)
    if not os.path.isdir(pdir):
        return JSONResponse({"error": "not found"}, status_code=404)
    return {"markdown": reflect.read_log(pdir), "path": reflect.log_path(pdir),
            "enabled": STORE.compact_log()}


@app.post("/api/settings/compact-log")
async def api_set_compact_log(request: Request):
    body = await request.json()
    STORE.set_compact_log(bool(body.get("enabled", True)))
    return {"enabled": STORE.compact_log()}


@app.get("/api/projects/{pid}/journal")
async def api_journal(pid: str):
    pdir = _project_dir(pid)
    if not os.path.isdir(pdir):
        return JSONResponse({"error": "not found"}, status_code=404)
    return {"markdown": journal.read_journal(pdir),
            "path": journal.journal_path(pdir)}


@app.post("/api/projects/{pid}/journal/note")
async def api_journal_note(pid: str, request: Request):
    """Pin a human note into the project journal (decisions, meeting outcomes…)."""
    pdir = _project_dir(pid)
    if not os.path.isdir(pdir):
        return JSONResponse({"error": "not found"}, status_code=404)
    body = await request.json()
    text = (body.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "text required"}, status_code=400)
    manifest = _read_manifest(pdir)
    journal.append_note(pdir, manifest.get("name", pid), text)
    journal.append_chat(pdir, {"role": "note", "text": text})
    return {"ok": True, "markdown": journal.read_journal(pdir)}


@app.get("/api/projects/{pid}/trace")
async def api_trace(pid: str, run: str):
    """Replay one past run's rounds (thought/action/observation)."""
    pdir = _project_dir(pid)
    try:
        tpath = _safe_join(os.path.join(pdir, "runs", _slug(run)), "trace.jsonl")
    except ValueError:
        return JSONResponse({"error": "invalid path"}, status_code=400)
    if not os.path.isfile(tpath):
        return JSONResponse({"error": "not found"}, status_code=404)
    events = []
    with open(tpath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    code = ""
    cpath = os.path.join(pdir, "runs", _slug(run), "code.py")
    if os.path.isfile(cpath):
        with open(cpath, encoding="utf-8") as f:
            code = f.read()
    return {"run_id": run, "events": events, "code": code}


@app.get("/api/projects/{pid}/runs")
async def api_runs(pid: str):
    pdir = _project_dir(pid)
    runs_dir = os.path.join(pdir, "runs")
    if not os.path.isdir(runs_dir):
        return []
    out = []
    for rid in sorted(os.listdir(runs_dir), reverse=True):
        rdir = os.path.join(runs_dir, rid)
        if not os.path.isdir(rdir):
            continue
        pred = os.path.join(rdir, "pred_results")
        out.append({
            "run_id": rid,
            "has_code": os.path.exists(os.path.join(rdir, "code.py")),
            "outputs": _dir_tree(pred) if os.path.isdir(pred) else [],
        })
    return out


def _run_geoprocess_direct(pid: str, op: str, inputs: dict, params: dict, output: str) -> dict:
    """Run ONE geoprocess op deterministically (no LLM), for the UI Toolbox.

    Loads the selected project files into a fresh sandbox, runs the op, copies
    outputs to the project's outputs/, and reports them for auto-render.
    """
    from src.agent import geo_ops
    from src.agent.sandbox import PythonSandbox
    from src.agent.tools import GISToolkit

    if op not in geo_ops.REGISTRY:
        return {"error": f"unknown op '{op}'"}
    if not output or not output.isidentifier():
        return {"error": "invalid output name (must be a valid identifier)"}
    pdir = _project_dir(pid)
    if not os.path.isdir(pdir):
        return {"error": "project not found"}
    _project_layout(pdir)

    # Resolve each input filename to an absolute path in data/ or outputs/.
    resolved = {}
    for role, fname in (inputs or {}).items():
        cand = None
        for sub in ("data", "outputs"):
            try:
                p = _safe_join(os.path.join(pdir, sub), fname)
            except ValueError:
                continue
            if os.path.isfile(p):
                cand = p
                break
        if cand is None:
            return {"error": f"input '{role}' file not found: {fname}"}
        resolved[role] = cand

    kinds = {i["role"]: i["kind"] for i in geo_ops.SPECS.get(op, {}).get("in", [])}
    tool_dir = os.path.join(pdir, "runs", "tool_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(os.path.join(tool_dir, "pred_results"), exist_ok=True)

    _cwd = os.getcwd()
    try:
        sandbox = PythonSandbox(work_dir=tool_dir, timeout=180)
        tk = GISToolkit(sandbox, data_dir="dataset")
        for role, path in resolved.items():
            if kinds.get(role) == "raster":
                tk.load_raster(path, role)
            else:
                tk.load_vector(path, role)
        obs = tk.geoprocess(op, inputs={r: r for r in resolved},
                            params=params or {}, output=output)
    finally:
        os.chdir(_cwd)

    # Collect outputs and copy into the project outputs/.
    pred = os.path.join(tool_dir, "pred_results")
    out_root = os.path.join(pdir, "outputs")
    os.makedirs(out_root, exist_ok=True)
    produced = []
    if os.path.isdir(pred):
        for fn in sorted(os.listdir(pred)):
            if fn.startswith("."):
                continue
            shutil.copy2(os.path.join(pred, fn), os.path.join(out_root, fn))
            ext = fn.lower().rsplit(".", 1)[-1]
            produced.append({"filename": fn, "kind": "raster" if ext in ("tif", "tiff") else "vector"})
    ok = "❌" not in obs
    return {"ok": ok, "observation": obs, "outputs": produced}


@app.post("/api/projects/{pid}/geoprocess")
async def api_geoprocess(pid: str, request: Request):
    body = await request.json()
    op = body.get("op", "")
    inputs = body.get("inputs", {})
    params = body.get("params", {})
    output = body.get("output", "") or (op + "_out")
    result = await asyncio.get_event_loop().run_in_executor(
        None, lambda: _run_geoprocess_direct(pid, op, inputs, params, output))
    status = 200 if not result.get("error") else 400
    return JSONResponse(result, status_code=status)


@app.post("/api/run")
async def api_run(request: Request):
    body = await request.json()
    model_key = body.get("model", "gpt-5.4")
    pid = body.get("project_id", "")
    instruction = (body.get("instruction") or "").strip()

    cfg = STORE.model_config(model_key)
    if not cfg:
        return JSONResponse({"error": f"Unknown model: {model_key}"}, status_code=400)
    if not cfg.get("api_key"):
        return JSONResponse(
            {"error": f"No API key for '{cfg.get('provider')}'. Open Settings → API keys."},
            status_code=400)
    if not pid:
        return JSONResponse({"error": "project_id required"}, status_code=400)
    if not instruction:
        return JSONResponse({"error": "instruction required"}, status_code=400)

    msg_queue: queue.Queue = queue.Queue()
    thread = threading.Thread(target=run_agent_in_thread,
                              args=(pid, model_key, instruction, msg_queue), daemon=True)
    thread.start()

    async def event_generator():
        while True:
            try:
                msg = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: msg_queue.get(timeout=120))
                if msg is None:
                    break
                yield {"event": msg["type"], "data": json.dumps(msg, ensure_ascii=False)}
            except queue.Empty:
                yield {"event": "heartbeat", "data": "{}"}
            except Exception:
                break

    return EventSourceResponse(event_generator())


# ============================================================
# Static frontend (mounted last so /api/* wins)
# ============================================================
class NoCacheStaticFiles(StaticFiles):
    """Serve the frontend with no-cache so edits land on a normal refresh.
    The assets are tiny and local; correctness beats caching here."""

    def is_not_modified(self, response_headers, request_headers) -> bool:
        return False  # always re-send; never answer 304 from a stale copy

    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return resp


@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse(
        os.path.join(WEB_DIR, "index.html"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


if os.path.isdir(WEB_DIR):
    app.mount("/", NoCacheStaticFiles(directory=WEB_DIR, html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")

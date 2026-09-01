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

"""The life of an analysis run.

A run is started here, executed on its own thread, and streamed to whoever is
listening. Three things this module guarantees that the routes should not have
to think about:

- **One at a time.** The sandbox executes code in a shared process with a
  process-wide working directory and global plotting state, so two runs at
  once would corrupt each other. A second request while one is active is
  refused, not queued.
- **A run can be stopped.** The stop flag is checked before every round and
  polled while a snippet executes, so a Stop reaches code that is running.
- **A run outlives its browser tab.** Every event is kept, so a page that
  reloads mid-run can subscribe again and see the whole run from the start.
"""
import os
import queue
import re
import shutil
import threading
import time
import traceback

from app import data_profile, journal, paths, reflect
from app.logging_setup import RunRecorder

STORE = None       # SettingsStore, set by configure()
SKILLS = None      # SkillsStore
log = None


def configure(store, skills, logger):
    global STORE, SKILLS, log
    STORE, SKILLS, log = store, skills, logger


# ============================================================
# Prompt assembly
# ============================================================
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

## Available Packages (these are installed; nothing else is)
geopandas, rasterio, shapely, pyogrio, pyproj, rtree, numpy, pandas, scipy,
matplotlib, seaborn, sklearn, libpysal, esda, momepy, rasterstats, mapclassify,
networkx, osmnx, h3, openpyxl
{skills_block}{catalog_block}{data_block}{memory_block}{context_block}"""

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

# Injected only when non-empty, so a first run in a fresh install pays no tokens.
MEMORY_BLOCK = """
## Standing user preferences

These apply to every project. Follow them unless this task says otherwise —
especially for cartography, symbology and deliverable conventions.

{memory}
"""

DATA_BLOCK = """
## Data already in this project

This was read from the files themselves and cached, so relying on it is not
the same as assuming — it satisfies the rule about taking schema from the data
rather than from memory. Use these names, coordinate systems and extents to
plan with; you do not need a discovery round to rediscover them.

This listing is complete for data/, so list_files adds nothing for these — go
straight to loading what you need. You do still call load_vector / load_raster:
that puts the data in the sandbox, which reading this cannot do.

It covers structure only. Before you compute on a column you still have to look
at its values — nulls, ranges, units, and what a join actually matched.

{data}
"""

CONTEXT_BLOCK = """
## This project so far

{context}
"""


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


def _skill_tools() -> dict:
    """The `skill` tool, offered to the agent only when there is one to open."""
    if not SKILLS.build_catalog(STORE.skill_overrides()):
        return {}
    return {"skill": (_skill_tool, SKILL_TOOL_DESC)}


# ============================================================
# LLM
# ============================================================
class ApiCallFailed(RuntimeError):
    """The model provider refused or could not be reached."""


# Every engine reports transport and auth failures by *returning* them as the
# model's text. The ReAct loop would treat "Error code: 401 ..." as a badly
# formatted reply and retry until it ran out of rounds, producing nothing and
# explaining nothing. One failed call is enough to know the run cannot proceed.
_ENGINE_ERROR_PREFIXES = (
    "Error during Claude API call:", "Error during API call:",
    "Error: API client not initialized",
)


def init_llm(cfg: dict):
    """Build an engine from a resolved model config (engine/api_key/base_url)."""
    engine = cfg["engine"]
    key = cfg.get("api_key", "")
    if not key and cfg.get("key_optional"):
        # A model served from your own machine authenticates nobody, but the
        # OpenAI client refuses to start without *something* in the field.
        key = "local"
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


def guard_engine(llm):
    """Turn an engine's returned error text into an exception."""
    inner = llm.generate

    def generate(*a, **kw):
        r = inner(*a, **kw)
        text = (r or {}).get("text", "")
        if isinstance(text, str) and text.startswith(_ENGINE_ERROR_PREFIXES):
            raise ApiCallFailed(text)
        return r

    llm.generate = generate
    return llm


def clean_summary(observation: str) -> str:
    """The agent's own closing words, without the finish tool's packaging.

    `finish` returns "Task complete / Summary: … / Output files (n): …". Only the
    middle part was written for a person to read; the file list is shown in the
    interface anyway, and the first line says nothing.
    """
    text = (observation or "").strip()
    if not text:
        return ""
    text = re.sub(r"^(?:\U0001F4CB\s*)?Task complete\s*\n?", "", text, flags=re.I)
    m = re.match(r"\s*Summary:\s*", text)
    if m:
        text = text[m.end():]
    text = re.split(r"\n(?:Output files \(\d+\):|⚠️ No output files)", text)[0]
    return text.strip()


def _cost_info(llm) -> dict:
    if not hasattr(llm, "get_stats"):
        return {}
    s = llm.get_stats()
    return {
        "api_calls": s.get("total_calls", 0),
        # input_tokens is the uncached remainder only — recording it alone
        # would make a cached run look far cheaper in tokens than it was.
        # prompt_tokens is the honest total.
        "input_tokens": s.get("total_input_tokens", 0),
        "cache_read_tokens": s.get("cache_read_tokens", 0),
        "cache_write_tokens": s.get("cache_write_tokens", 0),
        "prompt_tokens": s.get("prompt_tokens", s.get("total_input_tokens", 0)),
        "cache_hit_rate": s.get("cache_hit_rate", 0),
        "output_tokens": s.get("total_output_tokens", 0),
        "cost_usd": s.get("estimated_cost_usd", 0),
    }


# ============================================================
# Runs
# ============================================================
class RunBusy(RuntimeError):
    """Another run (or a Toolbox operation) is already using the sandbox."""


class Run:
    """One analysis run: its identity, its stop flag, and everything it emitted."""

    def __init__(self, run_id: str, pid: str, model_key: str, instruction: str, run_dir: str):
        self.id = run_id
        self.pid = pid
        self.model = model_key
        self.instruction = instruction
        self.dir = run_dir
        self.started = time.time()
        self.done = False
        self.success = None
        self.cancel = threading.Event()
        self.events = []
        self._lock = threading.Lock()
        self._listeners = []

    def stop_requested(self) -> bool:
        return self.cancel.is_set()

    def emit(self, msg: dict):
        with self._lock:
            self.events.append(msg)
            for q in self._listeners:
                q.put(msg)

    def finish(self):
        with self._lock:
            self.done = True
            for q in self._listeners:
                q.put(None)

    def subscribe(self):
        """(events so far, queue for the rest). Atomic, so nothing is missed."""
        q = queue.Queue()
        with self._lock:
            snapshot = list(self.events)
            self._listeners.append(q)
            if self.done:
                q.put(None)
        return snapshot, q

    def unsubscribe(self, q):
        with self._lock:
            if q in self._listeners:
                self._listeners.remove(q)

    def public(self) -> dict:
        return {"run_id": self.id, "project_id": self.pid, "model": self.model,
                "instruction": self.instruction, "started": self.started,
                "done": self.done, "success": self.success,
                "stopping": self.cancel.is_set() and not self.done}


_RUNS = {}
_registry_lock = threading.Lock()
_active = None
# Held by whoever is using the sandbox — an agent run or a Toolbox operation.
SANDBOX_LOCK = threading.Lock()
_KEEP_RUNS = 6


def get_run(run_id: str):
    return _RUNS.get(run_id)


def active_run(pid: str = ""):
    """The run currently executing (for this project, if given), or None."""
    r = _active
    if r is None or r.done:
        return None
    if pid and r.pid != pid:
        return None
    return r


def cancel_run(run_id: str) -> bool:
    r = _RUNS.get(run_id)
    if not r or r.done:
        return False
    r.cancel.set()
    log.info(f"[{r.pid}] run {r.id} stop requested")
    return True


def start_run(pid: str, model_key: str, instruction: str) -> Run:
    """Validate, register, and start a run on its own thread."""
    global _active
    cfg = STORE.model_config(model_key)
    if not cfg:
        raise ValueError(f"Unknown model: {model_key}")
    if not cfg.get("api_key") and not cfg.get("key_optional"):
        raise ValueError(f"No API key for '{cfg.get('provider')}'. Open Settings → API keys.")
    pdir = paths.project_dir(pid)
    if not os.path.isdir(pdir):
        raise FileNotFoundError(f"Project not found: {pid}")
    paths.project_layout(pdir)

    with _registry_lock:
        if _active is not None and not _active.done:
            raise RunBusy("A run is already in progress. Stop it or wait for it to finish.")
        run_dir = paths.new_run_dir(pdir)
        run = Run(os.path.basename(run_dir), pid, model_key, instruction, run_dir)
        _RUNS[run.id] = run
        _active = run
        # Keep a few finished runs around so a reloaded page can still replay
        # the one that just ended; older ones are on disk anyway.
        for old in sorted(_RUNS.values(), key=lambda r: r.started)[:-_KEEP_RUNS]:
            if old.done:
                _RUNS.pop(old.id, None)

    threading.Thread(target=_run_agent, args=(run, cfg), daemon=True,
                     name=f"run-{run.id}").start()
    return run


def _run_agent(run: Run, cfg: dict):
    pid, model_key, instruction = run.pid, run.model, run.instruction
    recorder = None
    llm = None
    SANDBOX_LOCK.acquire()
    try:
        pdir = paths.project_dir(pid)
        data_dir = os.path.join(pdir, "data")
        manifest = paths.read_manifest(pdir)
        project_name = manifest.get("name", pid)
        run_dir = run.dir
        pred_dir = os.path.join(run_dir, "pred_results")
        os.makedirs(pred_dir, exist_ok=True)

        run.emit({"type": "run", "run_id": run.id, "project_id": pid, "model": model_key})

        # The question itself is part of the record, logged before we spend a token.
        journal.append_chat(pdir, {"role": "user", "text": instruction, "model": model_key,
                                   "run_id": run.id})

        recorder = RunRecorder(run_dir)
        recorder.log(f"model={model_key} instruction={instruction!r}")
        log.info(f"[{pid}] run {run.id} model={model_key}")

        run.emit({"type": "status", "content": f"Initializing {cfg['display']}...", "run_id": run.id})

        llm = guard_engine(init_llm(cfg))

        # Intent gate: a question about the project shouldn't become an analysis
        # run. The agent's finish-guard demands output files, so without this a
        # "what did you do?" forces it to redo work and write a placeholder just
        # to be allowed to stop.
        gate_context = reflect.recent_log(pdir) or journal.build_context(pdir, manifest)
        verdict = reflect.classify_request(llm, instruction, gate_context)
        if verdict["mode"] != "analysis":
            answer = verdict.get("answer", "")
            recorder.log(f"intent={verdict['mode']} answered without running the agent")
            journal.append_chat(pdir, {
                "role": "agent", "run_id": run.id, "model": model_key,
                "model_display": cfg.get("display", model_key), "ask": instruction,
                "success": True, "rounds": 0, "self_corrections": 0, "elapsed_s": 0,
                "outputs": [], "answer": answer, "kind": verdict["mode"],
            })
            run.emit({"type": "answer", "mode": verdict["mode"], "content": answer})
            run.success = True
            run.emit({"type": "done", "run_id": run.id, "success": True,
                      "output_files": [], "rounds": 0, "self_corrections": 0,
                      "elapsed_s": 0, "cost": _cost_info(llm), "answered": True})
            return

        if run.stop_requested():
            _record_stopped(run, pdir, cfg, [], 0, 0.0, _cost_info(llm))
            return

        from src.agent.error_memory import ErrorMemory
        from src.agent.react_agent import GISReActAgent

        # Continuity layer: standing user preferences + what this project already did.
        memory_text = STORE.memory_for_prompt() if STORE.memory_enabled() else ""
        memory_block = MEMORY_BLOCK.format(memory=memory_text) if memory_text else ""
        try:
            data_text = data_profile.build_block(data_profile.profile_project(pdir))
        except Exception as e:
            log.warning(f"data profile failed: {e}")
            data_text = ""
        data_block = DATA_BLOCK.format(data=data_text) if data_text else ""
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
            run.emit({"type": "status", "content": f"Loaded skill: {matched['name']}"})

        catalog_text = SKILLS.build_catalog(overrides, exclude=matched["name"] if matched else "")
        catalog_block = SKILL_CATALOG_BLOCK.format(catalog=catalog_text) if catalog_text else ""

        # The tool descriptions are only known once the agent has built its
        # toolkit, so hand it a builder rather than a finished prompt.
        def build_prompt(tool_descriptions: str) -> str:
            return SYSTEM_PROMPT.format(
                tool_descriptions=tool_descriptions,
                skill_rule=SKILL_RULE if catalog_text else "",
                skills_block=skills_block,
                catalog_block=catalog_block,
                memory_block=memory_block,
                context_block=context_block,
                data_block=data_block,
            )

        recorder.log(f"prompt: always_skills={len(skills_text)}c catalog={len(catalog_text)}c "
                     f"memory={len(memory_text)}c context={len(context_text)}c "
                     f"data={len(data_text)}c")

        agent = GISReActAgent(llm_engine=llm, timeout=cfg["timeout"],
                              max_rounds=cfg["max_rounds"], verbose=True,
                              error_memory=ErrorMemory(),
                              system_prompt_builder=build_prompt,
                              extra_tools=_skill_tools(),
                              context_chars=cfg.get("context_chars") or 100_000)

        seen_outputs = set()
        steps_log = []          # condensed step list for JOURNAL.md

        def on_step(ev: dict):
            """Called by the agent after each round — push a live event."""
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
            elif ev.get("success") is False:
                # Keep failures too: they are what a closing note has to explain.
                step["observation"] = (ev.get("observation_full")
                                       or ev.get("observation") or "")[:800]
            steps_log.append(step)
            run.emit({
                "type": "step",
                "round": ev.get("round", 0),
                "thought": ev.get("thought", ""),
                "action": ev.get("action", ""),
                "code": ev.get("code", ""),
                "observation": ev.get("observation_full", ev.get("observation", "")),
                "success": ev.get("success", True),
            })
            # Announce any new result files as they appear
            if os.path.isdir(pred_dir):
                for fn in sorted(os.listdir(pred_dir)):
                    if fn.startswith(".") or fn in seen_outputs:
                        continue
                    seen_outputs.add(fn)
                    run.emit({"type": "result", "run_id": run.id, "filename": fn,
                              "url": f"/api/projects/{pid}/runfile?run={run.id}&path={fn}"})

        run.emit({"type": "status", "content": "Agent running...", "run_id": run.id})
        t0 = time.time()
        result = agent.run(instruction=instruction, data_dir=data_dir,
                           work_dir=run_dir, on_step=on_step,
                           should_stop=run.stop_requested)
        elapsed = time.time() - t0

        output_files = _collect_outputs(pdir, pred_dir)
        cost_info = _cost_info(llm)
        recorder.log(f"done success={result.success} rounds={result.total_rounds} "
                     f"self_corrections={result.self_corrections} outputs={output_files}"
                     + (" stopped" if getattr(result, "stopped", False) else ""))

        if getattr(result, "stopped", False):
            _record_stopped(run, pdir, cfg, output_files, result.total_rounds, elapsed,
                            cost_info, steps_log, project_name)
            return

        # Durable record: one conversation entry + one journal section per run.
        summary = {
            "role": "agent",
            "run_id": run.id,
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
        # Every run ends in words. Normally they are the agent's own, from the
        # finish call; when it stopped without writing any — out of rounds, stuck
        # repeating itself — one small call produces the closing note instead, so
        # the reader is never left with just a row of counters.
        final_text = clean_summary(next(
            (s.get("observation", "") for s in reversed(steps_log)
             if s.get("action") == "finish"), ""))
        if not final_text and steps_log:
            run.emit({"type": "status", "content": "Writing the closing note…"})
            try:
                final_text = reflect.closing_note(llm, instruction, summary, steps_log)
            except Exception as e:
                log.warning(f"closing note failed: {e}")
        summary["final_summary"] = final_text
        run.emit({"type": "summary", "run_id": run.id, "content": final_text,
                  "outputs": output_files, "success": bool(result.success)})
        entry = journal.append_chat(pdir, summary)
        try:
            journal.append_run(pdir, project_name, dict(entry, steps=steps_log))
        except Exception as e:
            log.warning(f"journal write failed: {e}")

        # One extra API call to compact this run into LOG.md — the digest a later
        # session reads instead of the whole transcript.
        if STORE.compact_log() and not run.stop_requested():
            try:
                run.emit({"type": "status", "content": "Writing project log…"})
                digest = reflect.compact_run(llm, pdir, project_name,
                                             dict(entry, steps=steps_log), steps_log)
                if digest:
                    run.emit({"type": "log", "content": digest})
            except Exception as e:
                log.warning(f"log compaction failed: {e}")

        run.success = bool(result.success)
        run.emit({
            "type": "done",
            "run_id": run.id,
            "success": bool(result.success),
            "output_files": output_files,
            "rounds": result.total_rounds,
            "self_corrections": result.self_corrections,
            "elapsed_s": round(elapsed, 1),
            "cost": cost_info,
        })
    except ApiCallFailed as e:
        # Say what actually happened. A rejected key used to end as a run that
        # simply produced nothing, which reads like the software is broken.
        detail = str(e)
        hint = ("The provider rejected the key. Open Settings → API keys, "
                "re-paste it and press Test." if "401" in detail
                or "authentication" in detail.lower() or "invalid x-api-key" in detail
                else "Check the key and your connection, then try again.")
        log.error(f"run aborted, provider call failed: {detail}")
        if recorder:
            recorder.log(f"API FAILED {detail}")
        _record_failure(run, model_key, instruction, detail)
        run.success = False
        run.emit({"type": "error",
                  "content": f"{cfg.get('display', model_key)} could not be "
                             f"reached.\n\n{detail}\n\n{hint}"})
        run.emit({"type": "done", "run_id": run.id, "success": False,
                  "output_files": [], "rounds": 0, "self_corrections": 0,
                  "elapsed_s": 0, "cost": {}})
    except Exception as e:
        tb = traceback.format_exc()
        log.error(f"run failed: {e}\n{tb[-800:]}")
        if recorder:
            recorder.log(f"ERROR {e}\n{tb[-800:]}")
        _record_failure(run, model_key, instruction, str(e))
        run.success = False
        run.emit({"type": "error", "content": f"{e}\n{tb[-500:]}"})
        run.emit({"type": "done", "run_id": run.id, "success": False,
                  "output_files": [], "rounds": 0, "self_corrections": 0,
                  "elapsed_s": 0, "cost": {}})
    finally:
        if recorder:
            recorder.close()
        run.finish()
        SANDBOX_LOCK.release()


def _collect_outputs(pdir: str, pred_dir: str) -> list:
    """Copy this run's outputs into the project's own outputs/."""
    out_root = os.path.join(pdir, "outputs")
    os.makedirs(out_root, exist_ok=True)
    output_files = []
    if os.path.isdir(pred_dir):
        for fn in sorted(os.listdir(pred_dir)):
            if fn.startswith("."):
                continue
            output_files.append(fn)
            shutil.copy2(os.path.join(pred_dir, fn), os.path.join(out_root, fn))
    return output_files


def _record_stopped(run: Run, pdir: str, cfg: dict, outputs: list, rounds: int,
                    elapsed: float, cost: dict, steps=None, project_name: str = ""):
    """A run the user stopped is still history — recorded as such, no extra calls."""
    text = (f"Stopped by request after {rounds} round(s)."
            + (f" {len(outputs)} file(s) had been produced by then." if outputs else ""))
    entry = journal.append_chat(pdir, {
        "role": "agent", "run_id": run.id, "model": run.model,
        "model_display": cfg.get("display", run.model), "ask": run.instruction,
        "success": False, "stopped": True, "rounds": rounds, "self_corrections": 0,
        "elapsed_s": round(elapsed, 1), "outputs": outputs, "cost": cost,
        "final_summary": text,
    })
    if steps:
        try:
            journal.append_run(pdir, project_name or run.pid, dict(entry, steps=steps))
        except Exception as e:
            log.warning(f"journal write failed: {e}")
    run.success = False
    run.emit({"type": "summary", "run_id": run.id, "content": text,
              "outputs": outputs, "success": False, "stopped": True})
    run.emit({"type": "done", "run_id": run.id, "success": False, "stopped": True,
              "output_files": outputs, "rounds": rounds, "self_corrections": 0,
              "elapsed_s": round(elapsed, 1), "cost": cost})


def _record_failure(run: Run, model_key: str, instruction: str, detail: str):
    # A failed run is still history — record it so the conversation is honest.
    try:
        pdir = paths.project_dir(run.pid)
        if os.path.isdir(pdir):
            journal.append_chat(pdir, {
                "role": "agent", "run_id": run.id, "model": model_key, "ask": instruction,
                "success": False, "error": detail, "outputs": [],
                "rounds": 0, "self_corrections": 0, "elapsed_s": 0,
            })
    except Exception:
        pass


# ============================================================
# Toolbox: one deterministic operation, no model
# ============================================================
def run_geoprocess_direct(pid: str, op: str, inputs: dict, params: dict, output: str) -> dict:
    """Run ONE geoprocess op deterministically (no LLM), for the UI Toolbox.

    Loads the selected project files into a fresh sandbox, runs the op, copies
    outputs to the project's outputs/, and records it like any other run so
    the journal says where the file came from.
    """
    from src.agent import geo_ops
    from src.agent.sandbox import PythonSandbox
    from src.agent.tools import GISToolkit

    if op not in geo_ops.REGISTRY:
        return {"error": f"unknown op '{op}'"}
    if not output or not output.isidentifier():
        return {"error": "invalid output name (must be a valid identifier)"}
    pdir = paths.project_dir(pid)
    if not os.path.isdir(pdir):
        return {"error": "project not found"}
    paths.project_layout(pdir)

    # Resolve each input filename to an absolute path in data/ or outputs/.
    resolved = {}
    for role, fname in (inputs or {}).items():
        cand = None
        for sub in ("data", "outputs"):
            try:
                p = paths.safe_join(os.path.join(pdir, sub), fname)
            except ValueError:
                continue
            if os.path.isfile(p):
                cand = p
                break
        if cand is None:
            return {"error": f"input '{role}' file not found: {fname}"}
        resolved[role] = cand

    if not SANDBOX_LOCK.acquire(blocking=False):
        raise RunBusy("A run is in progress — the sandbox is busy. Stop it or wait.")
    try:
        kinds = {i["role"]: i["kind"] for i in geo_ops.SPECS.get(op, {}).get("in", [])}
        tool_dir = paths.new_run_dir(pdir, prefix="tool")
        tool_id = os.path.basename(tool_dir)
        os.makedirs(os.path.join(tool_dir, "pred_results"), exist_ok=True)
        t0 = time.time()

        sandbox = PythonSandbox(work_dir=tool_dir, timeout=180)
        tk = GISToolkit(sandbox, data_dir="dataset")
        for role, path in resolved.items():
            if kinds.get(role) == "raster":
                tk.load_raster(path, role)
            else:
                tk.load_vector(path, role)
        obs = tk.geoprocess(op, inputs={r: r for r in resolved},
                            params=params or {}, output=output)
        with open(os.path.join(tool_dir, "code.py"), "w", encoding="utf-8") as f:
            f.write(sandbox.get_full_code())
    finally:
        SANDBOX_LOCK.release()

    produced = []
    for fn in _collect_outputs(pdir, os.path.join(tool_dir, "pred_results")):
        ext = fn.lower().rsplit(".", 1)[-1]
        produced.append({"filename": fn, "kind": "raster" if ext in ("tif", "tiff") else "vector"})
    ok = "❌" not in obs

    # The record: which files came from which operation, with what settings.
    ask = f"Toolbox: {op}(" + ", ".join(
        [f"{r}={os.path.basename(p)}" for r, p in resolved.items()]
        + [f"{k}={v}" for k, v in (params or {}).items()]) + f") → {output}"
    manifest = paths.read_manifest(pdir)
    entry = journal.append_chat(pdir, {
        "role": "agent", "kind": "tool", "run_id": tool_id, "model": "toolbox",
        "model_display": "Toolbox", "ask": ask, "success": ok, "rounds": 1,
        "self_corrections": 0, "elapsed_s": round(time.time() - t0, 1),
        "outputs": [p["filename"] for p in produced],
        "final_summary": obs.strip()[:2000],
    })
    try:
        journal.append_run(pdir, manifest.get("name", pid), dict(
            entry, steps=[{"round": 1, "action": f"geoprocess {op}",
                           "thought": ask, "success": ok}]))
    except Exception as e:
        log.warning(f"journal write failed: {e}")
    log.info(f"[{pid}] toolbox {op} -> {output} ok={ok}")
    return {"ok": ok, "observation": obs, "outputs": produced, "run_id": tool_id}

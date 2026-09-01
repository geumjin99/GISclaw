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
GISclaw — application backend (single ReAct agent, cloud LLMs).

Manages user "projects" — working folders under a workspace root — runs the
agent live over a project's data, and streams Thought/Action/Observation to the
browser over SSE.

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
import shutil
import sys
from datetime import datetime

APP_VERSION = "2.0.0-beta.5"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from app import basemap, data_profile, journal, local_models, paths, reflect, runner
from app.logging_setup import get_app_logger
from app.settings_store import PROVIDERS, SettingsStore, mask_key, in_container
from app.skills_store import SkillsStore

# ============================================================
# Config
# ============================================================
WEB_DIR = os.path.join(PROJECT_ROOT, "app", "web")
APP_LOG = os.environ.get("GISCLAW_LOG") or os.path.join(PROJECT_ROOT, "app", "server.log")
log = get_app_logger(APP_LOG)

# Workspace root: env override, else <repo>/projects for local dev.
WORKSPACE = paths.configure(os.environ.get("GISCLAW_WORKSPACE")
                            or os.path.join(PROJECT_ROOT, "projects"))
log.info(f"Workspace root: {WORKSPACE}")

# Settings live on the mounted volume so they survive container rebuilds.
STORE = SettingsStore(WORKSPACE)
SKILLS = SkillsStore(WORKSPACE, os.path.join(PROJECT_ROOT, "app", "skills"))
log.info(f"Settings: {STORE.path}")
runner.configure(STORE, SKILLS, log)

# Short names for the path helpers the routes use on every request.
_safe_join = paths.safe_join
_slug = paths.slug
_project_dir = paths.project_dir
_project_layout = paths.project_layout
_read_manifest = paths.read_manifest
_write_manifest = paths.write_manifest
_list_projects = paths.list_projects
_dir_tree = paths.dir_tree
_companion_files = paths.companion_files
_archive_root = paths.archive_root
RECORD_FILES = paths.RECORD_FILES


# ============================================================
# FastAPI app
# ============================================================
app = FastAPI(title="GISclaw", description="GIS Analyst Agent (product)", version=APP_VERSION)

# The server listens on localhost, and anything on localhost is reachable from
# any web page the user happens to have open: a page elsewhere could POST to
# /api/run and have the model execute code on this machine with the user's own
# keys. Two checks close that. Every state-changing request must carry a header
# that only this front-end sends — a cross-site page cannot add it without a
# CORS preflight, which is never answered — and, when the browser names an
# origin, it has to be this server.
CLIENT_HEADER = "x-gisclaw"


@app.middleware("http")
async def _same_client_only(request: Request, call_next):
    if request.url.path.startswith("/api/") and request.method not in ("GET", "HEAD", "OPTIONS"):
        if request.headers.get(CLIENT_HEADER) != "1":
            return JSONResponse({"error": "missing client header"}, status_code=403)
        origin = request.headers.get("origin")
        if origin:
            from urllib.parse import urlsplit
            if urlsplit(origin).netloc.lower() != (request.headers.get("host") or "").lower():
                return JSONResponse({"error": "cross-site request refused"}, status_code=403)
    return await call_next(request)


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
        "viewer_follow": STORE.viewer_follow(),
        "language": STORE.language(),
        "settings_path": STORE.path,
        "memory_path": STORE.memory_path,
    }


@app.post("/api/settings/ui")
async def api_set_ui(request: Request):
    """Interface preferences shared by every browser on this machine."""
    body = await request.json()
    if "language" in body:
        STORE.set_language(str(body.get("language") or ""))
    return {"language": STORE.language()}


@app.post("/api/settings/viewer_follow")
async def api_set_viewer_follow(request: Request):
    """Toggle whether the viewer follows the agent between tabs."""
    body = await request.json()
    STORE.set_viewer_follow(bool(body.get("enabled", True)))
    return {"enabled": STORE.viewer_follow()}


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
    if not model_name and STORE.provider_key_optional(provider_id):
        # A local server is normally tested before any model has been added —
        # the point of the test is to learn whether it is up at all. Ask it what
        # it is serving and use the first answer.
        found = await asyncio.get_event_loop().run_in_executor(
            None, lambda: _discover_models(provider_id))
        if not found.get("ok"):
            return {"ok": False, "model_name": "", "error": found.get("error", "")}
        if found.get("models"):
            model_name = found["models"][0]["id"]
    if not model_name:
        return JSONResponse({"error": "no model to test with — add one first"}, status_code=400)

    def _probe():
        cfg = {
            "engine": PROVIDERS[provider_id]["engine"],
            "model_name": model_name,
            # A thinking model can spend its whole budget before any text.
            "max_tokens": 512,
            "api_key": STORE.provider_key(provider_id),
            "key_optional": STORE.provider_key_optional(provider_id),
            "base_url": STORE.provider_base_url(provider_id),
            "provider": provider_id,
            "cost_per_m": (0.0, 0.0),
        }
        llm = runner.init_llm(cfg)
        return llm.generate("Reply with the single word: ok")

    try:
        res = await asyncio.get_event_loop().run_in_executor(None, _probe)
        text = (res.get("text") if isinstance(res, dict) else str(res)) or ""
        # The engines return API failures as text rather than raising.
        if text.startswith("Error"):
            return {"ok": False, "model_name": model_name, "error": text[:400]}
        out = {"ok": True, "model_name": model_name, "reply": text.strip()[:120]}
        if STORE.provider_key_optional(provider_id):
            # The call just loaded the model, so the server can now say what
            # context it gave it — the figure that decides whether runs work.
            base = STORE.provider_base_url(provider_id) or ""
            ctx = await asyncio.get_event_loop().run_in_executor(
                None, lambda: local_models.loaded_context(base, model_name))
            out["context_length"] = ctx
            out["context_advice"] = local_models.context_advice(ctx, "ollama" if ctx is not None else "")
        return out
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
    if body.get("context_chars"):
        spec["context_chars"] = max(4000, int(body["context_chars"]))
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


def _endpoint_error(provider_id: str, base: str, exc: Exception) -> str:
    """Turn a connection failure into something the user can act on.

    "Connection error." is all the SDK says when nothing is listening, which is
    exactly the case a local server hits — and inside a container the reason is
    usually that the address is right but reaches the wrong machine.
    """
    msg = str(exc)[:300]
    if not STORE.provider_key_optional(provider_id):
        return msg
    low = msg.lower()
    if not any(s in low for s in ("connect", "refus", "timed out", "timeout",
                                  "unreachable", "name or service")):
        return msg
    extra = f" Tried {base}." if base else ""
    if in_container():
        extra += (" Running in Docker, so localhost is the container: the address"
                  " is rewritten to reach your machine, which needs"
                  " `extra_hosts: [\"host.docker.internal:host-gateway\"]` in"
                  " docker-compose.yml (it is there by default). Also make sure"
                  " the server listens on all interfaces —"
                  " `OLLAMA_HOST=0.0.0.0 ollama serve`.")
    else:
        extra += " Is the server running? For Ollama: `ollama serve`."
    return msg + extra


def _discover_models(provider_id: str) -> dict:
    """Ask the provider what it is actually serving right now.

    Every OpenAI-compatible endpoint answers GET /models; Anthropic has its own
    models.list(). Both SDKs are already dependencies, so no raw HTTP here.
    """
    key = STORE.provider_key(provider_id)
    optional = STORE.provider_key_optional(provider_id)
    if not key and not optional:
        return {"ok": False, "error": "No API key for this provider yet."}
    meta = PROVIDERS.get(provider_id, {})
    base = STORE.provider_base_url(provider_id)
    try:
        if meta.get("engine") == "claude":
            import anthropic
            client = anthropic.Anthropic(api_key=key)
            ids = [m.id for m in client.models.list(limit=100).data]
        else:
            from openai import OpenAI
            # A server that is not running should say so in seconds; the
            # client's default is a ten-minute wait with retries on top.
            local = STORE.provider_key_optional(provider_id)
            client = OpenAI(api_key=key or "local",
                            **({"base_url": base} if base else {}),
                            **({"timeout": 8.0, "max_retries": 0} if local else {}))
            ids = [m.id for m in client.models.list().data]
    except Exception as e:
        return {"ok": False, "error": _endpoint_error(provider_id, base, e)}

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


# ---------------------------------------------------------------- basemap --
@app.get("/api/settings/basemap")
async def api_basemap_settings():
    out = basemap.public(STORE)
    out["cache_bytes"] = basemap.cache_size(STORE)
    return out


@app.post("/api/settings/basemap")
async def api_set_basemap(request: Request):
    body = await request.json()
    try:
        basemap.save(STORE, body)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    log.info(f"basemap: {basemap.settings(STORE)['provider']}")
    out = basemap.public(STORE)
    out["cache_bytes"] = basemap.cache_size(STORE)
    return out


@app.post("/api/settings/basemap/clear_cache")
async def api_clear_tile_cache():
    basemap.clear_cache(STORE)
    return {"ok": True, "cache_bytes": 0}


@app.get("/api/basemap/tile/{z}/{x}/{y}")
async def api_basemap_tile(z: int, x: int, y: int, r: str = ""):
    """One map tile, from the cache, the provider, or the MBTiles file."""
    res, err = await asyncio.get_event_loop().run_in_executor(
        None, lambda: basemap.tile(STORE, z, x, y, r))
    if res is None:
        # A transparent answer, not an error page: the map shows the offline
        # reference layer underneath and moves on.
        return Response(status_code=204, headers={"X-Tile-Error": err[:120]})
    data, ctype = res
    return Response(content=data, media_type=ctype,
                    headers={"Cache-Control": "private, max-age=86400"})


@app.get("/api/settings/local")
async def api_local_settings():
    """Everything the Local models pane needs to start from."""
    return {
        "presets": local_models.PRESETS,
        "recommended": local_models.RECOMMENDED,
        "min_context": local_models.MIN_CONTEXT_TOKENS,
        "recommended_context": local_models.RECOMMENDED_CONTEXT_TOKENS,
        "base_url": STORE.provider_base_url("local", raw=True) or "",
        "models": [m for m in STORE.models_public() if m["provider"] == "local"],
    }


@app.get("/api/settings/local/probe")
async def api_local_probe(base_url: str = ""):
    """Ask the server at base_url what it is and what it serves.

    Saves the address as the Local provider's endpoint when it answers, so
    the models added from the listing run against it.
    """
    base = (base_url or "").strip() or STORE.provider_base_url("local", raw=True) or ""
    if not base:
        return JSONResponse({"ok": False, "error": "address required"}, status_code=400)
    # Inside a container localhost is the container; probe where the request would go.
    from app.settings_store import reachable_base_url
    res = await asyncio.get_event_loop().run_in_executor(
        None, lambda: local_models.probe(reachable_base_url(base) or base))
    if res.get("ok"):
        STORE.set_provider("local", base_url=base)
        known = {m["model_name"] for m in STORE.models().values() if m.get("provider") == "local"}
        for m in res["models"]:
            m["already_added"] = m["id"] in known
            m["context_chars"] = local_models.context_chars_for(m.get("context_set") or m.get("context_max"))
            m["advice"] = local_models.context_advice(m.get("context_set"), res["kind"]) if m.get("context_set") else ""
        res["base_url"] = base
    return res


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


@app.get("/api/version")
async def api_version():
    return {"version": APP_VERSION}


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
    data_profile.invalidate(pdir)
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


@app.post("/api/projects/{pid}/rename")
async def api_rename_project(pid: str, request: Request):
    """Change a project's display name, and its folder when that is safe.

    The folder is what you see in Finder or Explorer, so leaving it on the old
    slug after a rename is quietly confusing. It moves too, unless the new name
    is already taken.
    """
    pdir = _project_dir(pid)
    if not os.path.isdir(pdir):
        return JSONResponse({"error": "project not found"}, status_code=404)
    body = await request.json()
    name = (body.get("name") or "").strip()
    if not name:
        return JSONResponse({"error": "name required"}, status_code=400)

    m = _read_manifest(pdir)
    m["name"] = name
    _write_manifest(pdir, m)

    want = _slug(name)
    if want == pid:
        return {"id": pid, "name": name, "folder_renamed": False}
    try:
        target = _safe_join(WORKSPACE, want)
    except ValueError:
        return {"id": pid, "name": name, "folder_renamed": False,
                "notice": "Renamed, but the folder kept its old name."}
    if os.path.exists(target):
        return {"id": pid, "name": name, "folder_renamed": False,
                "notice": f"Renamed, but the folder stayed as '{pid}': "
                          f"'{want}' is already in use."}
    try:
        os.rename(pdir, target)
    except OSError as e:
        log.warning(f"rename {pid} -> {want} failed: {e}")
        return {"id": pid, "name": name, "folder_renamed": False,
                "notice": f"Renamed, but the folder stayed as '{pid}' ({e})."}
    log.info(f"renamed project {pid} -> {want} ({name})")
    return {"id": want, "name": name, "folder_renamed": True}


@app.post("/api/projects/{pid}/upload")
async def api_upload(pid: str, request: Request, name: str = "", rel: str = ""):
    """Take one file straight from the browser into the project's data/.

    The container can only see the mounted workspace, so data living anywhere
    else on your computer has to arrive this way. The body is the raw file
    rather than a multipart form: that keeps the dependency list unchanged
    (python-multipart is not installed) and lets a large raster stream to disk
    instead of being held in memory.
    """
    pdir = _project_dir(pid)
    if not os.path.isdir(pdir):
        return JSONResponse({"error": "project not found"}, status_code=404)
    _project_layout(pdir)
    data_dir = os.path.join(pdir, "data")

    # A folder upload supplies a relative path; keep its shape, drop anything odd.
    parts = [p for p in (rel or name).replace("\\", "/").split("/")
             if p not in ("", ".", "..")]
    if not parts:
        return JSONResponse({"error": "no filename"}, status_code=400)
    try:
        dst = _safe_join(data_dir, "/".join(parts))
    except ValueError:
        return JSONResponse({"error": "invalid path"}, status_code=400)
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    written = 0
    try:
        with open(dst, "wb") as f:
            async for chunk in request.stream():
                f.write(chunk)
                written += len(chunk)
    except Exception as e:
        log.error(f"upload {parts} failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
    data_profile.invalidate(pdir)
    log.info(f"[{pid}] uploaded {'/'.join(parts)} ({written} bytes)")
    return {"saved": "/".join(parts), "bytes": written}


@app.get("/api/projects/{pid}/data_check")
async def api_data_check(pid: str):
    """Report shapefiles in data/ that are missing the files they need.

    A file dialog makes it easy to pick only the .shp and leave .shx/.dbf/.prj
    behind, which yields a layer nothing can open. Checked after the fact so the
    order files arrive in does not matter.
    """
    pdir = _project_dir(pid)
    data_dir = os.path.join(pdir, "data")
    notices = []
    if os.path.isdir(data_dir):
        for root, _dirs, files in os.walk(data_dir):
            lower = {f.lower() for f in files}
            for f in sorted(files):
                if not f.lower().endswith(".shp"):
                    continue
                stem = f[:-4].lower()
                missing = [e for e in (".shx", ".dbf") if stem + e not in lower]
                if missing:
                    notices.append(
                        f"{f}: missing {' and '.join(missing)}. A shapefile cannot "
                        "be opened without them \u2014 select every file of the set, "
                        "or the folder holding them.")
                elif stem + ".prj" not in lower:
                    notices.append(
                        f"{f}: no .prj, so the layer declares no coordinate system. "
                        "It may sit in the wrong place on the map, and the agent "
                        "will see CRS = None.")
    return {"notices": notices}


@app.post("/api/projects/{pid}/archive")
async def api_archive_project(pid: str):
    """Move a project out of the workspace, keeping it on disk.

    Not a delete: the folder is moved under _archived/ so it stops appearing in
    the project list and stops being reachable by the agent, while everything —
    data, outputs, run history, journal — stays exactly where it can be brought
    back from.
    """
    pdir = _project_dir(pid)
    if not os.path.isdir(pdir):
        return JSONResponse({"error": "project not found"}, status_code=404)
    dst = os.path.join(_archive_root(), os.path.basename(pdir))
    if os.path.exists(dst):
        return JSONResponse(
            {"error": f"an archived project named '{os.path.basename(pdir)}' "
                      "already exists — rename one of them first"}, status_code=409)
    try:
        shutil.move(pdir, dst)
    except OSError as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    log.info(f"archived project {pid}")
    return {"ok": True, "id": pid, "projects": _list_projects()}


@app.delete("/api/projects/{pid}")
async def api_delete_project(pid: str, confirm: str = ""):
    """Delete a project and everything in it, for good.

    Archiving is the reversible option and stays the default in the interface;
    this is for the projects you never want to see again. `confirm` has to be
    the project's own id, so a stray request cannot erase anything.
    """
    pdir = _project_dir(pid)
    if not os.path.isdir(pdir):
        return JSONResponse({"error": "project not found"}, status_code=404)
    if confirm != pid:
        return JSONResponse({"error": "confirmation does not match this project"},
                            status_code=400)
    counts = {
        "data": len(_dir_tree(os.path.join(pdir, "data"))),
        "outputs": len(_dir_tree(os.path.join(pdir, "outputs"))),
        "runs": len(os.listdir(os.path.join(pdir, "runs")))
        if os.path.isdir(os.path.join(pdir, "runs")) else 0,
    }
    try:
        shutil.rmtree(pdir)
    except OSError as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    log.info(f"deleted project {pid} ({counts})")
    return {"ok": True, "id": pid, "deleted": counts, "projects": _list_projects()}


@app.delete("/api/archived/{pid}")
async def api_delete_archived(pid: str, confirm: str = ""):
    """Same, for a project already sitting in _archived/."""
    try:
        src = _safe_join(_archive_root(), _slug(pid))
    except ValueError:
        return JSONResponse({"error": "invalid id"}, status_code=400)
    if not os.path.isdir(src):
        return JSONResponse({"error": "not archived"}, status_code=404)
    if confirm != pid:
        return JSONResponse({"error": "confirmation does not match this project"},
                            status_code=400)
    try:
        shutil.rmtree(src)
    except OSError as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    log.info(f"deleted archived project {pid}")
    return {"ok": True, "id": pid}


@app.delete("/api/projects/{pid}/file")
async def api_delete_file(pid: str, where: str = "outputs", path: str = ""):
    """Delete one file from a project's data/ or outputs/.

    Only those two folders: the run history and the project's own records are
    the audit trail and are not deletable from here. Deleting one member of a
    shapefile takes its siblings with it, for the same reason attaching one
    brings them along — the leftovers are unreadable on their own.
    """
    pdir = _project_dir(pid)
    if not os.path.isdir(pdir):
        return JSONResponse({"error": "project not found"}, status_code=404)
    if where not in ("data", "outputs"):
        return JSONResponse({"error": "only data/ and outputs/ files can be deleted"},
                            status_code=400)
    if not path:
        return JSONResponse({"error": "path required"}, status_code=400)
    try:
        target = _safe_join(os.path.join(pdir, where), path)
    except ValueError:
        return JSONResponse({"error": "invalid path"}, status_code=400)
    if not os.path.isfile(target):
        return JSONResponse({"error": "file not found"}, status_code=404)

    removed = []
    for f in _companion_files(target):
        try:
            os.remove(f)
            removed.append(os.path.relpath(f, os.path.join(pdir, where)))
        except OSError as e:
            log.error(f"delete {f} failed: {e}")
    if where == "data":
        data_profile.invalidate(pdir)   # the cached schema described a file that is gone
    log.info(f"[{pid}] deleted from {where}: {removed}")
    return {"ok": True, "removed": removed,
            "data": _dir_tree(os.path.join(pdir, "data")),
            "outputs": _dir_tree(os.path.join(pdir, "outputs"))}


@app.get("/api/archived")
async def api_list_archived():
    """Projects sitting in _archived/, ready to be brought back."""
    root = _archive_root()
    out = []
    for name in sorted(os.listdir(root)):
        pdir = os.path.join(root, name)
        if not os.path.isdir(pdir):
            continue
        if not os.path.exists(os.path.join(pdir, "project.json")):
            continue
        m = _read_manifest(pdir)
        data_dir = os.path.join(pdir, "data")
        runs_dir = os.path.join(pdir, "runs")
        out.append({
            "id": name,
            "name": m.get("name", name),
            "created_at": m.get("created_at", ""),
            "data_count": len(os.listdir(data_dir)) if os.path.isdir(data_dir) else 0,
            "run_count": len(os.listdir(runs_dir)) if os.path.isdir(runs_dir) else 0,
        })
    return out


@app.post("/api/archived/{pid}/restore")
async def api_restore_project(pid: str):
    """Bring an archived project back into the workspace."""
    try:
        src = _safe_join(_archive_root(), _slug(pid))
    except ValueError:
        return JSONResponse({"error": "invalid id"}, status_code=400)
    if not os.path.isdir(src):
        return JSONResponse({"error": "not archived"}, status_code=404)
    dst = _project_dir(pid)
    if os.path.exists(dst):
        return JSONResponse(
            {"error": f"'{os.path.basename(dst)}' already exists in the workspace "
                      "— rename it before restoring"}, status_code=409)
    try:
        shutil.move(src, dst)
    except OSError as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    log.info(f"restored project {pid}")
    return {"ok": True, "id": pid, "projects": _list_projects()}


@app.get("/api/projects/{pid}/export")
async def api_export_project(pid: str):
    """Download the whole project as a zip — for moving it to another machine."""
    import io
    import zipfile
    pdir = _project_dir(pid)
    if not os.path.isdir(pdir):
        return JSONResponse({"error": "project not found"}, status_code=404)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _dirs, files in os.walk(pdir):
            for fn in files:
                full = os.path.join(root, fn)
                z.write(full, os.path.join(
                    os.path.basename(pdir), os.path.relpath(full, pdir)))
    log.info(f"exported project {pid}")
    return Response(content=buf.getvalue(), media_type="application/zip", headers={
        "Content-Disposition": f'attachment; filename="{_slug(pid)}.zip"'})


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
        # Exactly three or four bands is ordinary RGB(A) imagery. Anything
        # else — a single measured band, or a multispectral stack where bands
        # 1-3 are not red/green/blue — gets a colour ramp instead, which is
        # honest about not being a photograph.
        bands = [1, 2, 3] if src.count in (3, 4) else [1]
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


@app.post("/api/projects/{pid}/geoprocess")
async def api_geoprocess(pid: str, request: Request):
    body = await request.json()
    op = body.get("op", "")
    inputs = body.get("inputs", {})
    params = body.get("params", {})
    output = body.get("output", "") or (op + "_out")
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: runner.run_geoprocess_direct(pid, op, inputs, params, output))
    except runner.RunBusy as e:
        return JSONResponse({"error": str(e), "busy": True}, status_code=409)
    status = 200 if not result.get("error") else 400
    return JSONResponse(result, status_code=status)


# ------------------------------------------------------------------- runs --
async def _stream_run(run):
    """Everything the run has emitted so far, then the rest as it happens."""
    snapshot, q = run.subscribe()
    try:
        for msg in snapshot:
            yield {"event": msg["type"], "data": json.dumps(msg, ensure_ascii=False)}
        if run.done:
            return
        while True:
            try:
                msg = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: q.get(timeout=20))
            except queue.Empty:
                yield {"event": "heartbeat", "data": "{}"}
                continue
            except RuntimeError:
                break           # the server is shutting down
            if msg is None:
                break
            yield {"event": msg["type"], "data": json.dumps(msg, ensure_ascii=False)}
    finally:
        run.unsubscribe(q)


@app.post("/api/run")
async def api_run(request: Request):
    """Start a run and stream it. One at a time — a second request gets 409."""
    body = await request.json()
    model_key = body.get("model") or ""
    pid = body.get("project_id", "")
    instruction = (body.get("instruction") or "").strip()
    language = str(body.get("language") or STORE.language() or "en")
    if not pid:
        return JSONResponse({"error": "project_id required"}, status_code=400)
    if not instruction:
        return JSONResponse({"error": "instruction required"}, status_code=400)
    if not model_key:
        ready = [m for m in STORE.models_public() if m["enabled"] and m["ready"]]
        if not ready:
            return JSONResponse({"error": "No model is configured. Open Settings → API keys."},
                                status_code=400)
        model_key = ready[0]["id"]
    try:
        run = runner.start_run(pid, model_key, instruction, language=language)
    except runner.RunBusy as e:
        active = runner.active_run()
        return JSONResponse({"error": str(e), "busy": True,
                             "active": active.public() if active else None}, status_code=409)
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=404)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return EventSourceResponse(_stream_run(run))


@app.get("/api/run/active")
async def api_active_run(project: str = ""):
    """The run in progress (for this project, if given), so a reloaded page can rejoin it."""
    run = runner.active_run(project)
    return {"active": run.public() if run else None}


@app.get("/api/run/{run_id}/stream")
async def api_run_stream(run_id: str):
    """Rejoin a run: replays what was already emitted, then follows it live."""
    run = runner.get_run(run_id)
    if not run:
        return JSONResponse({"error": "no such run in this session"}, status_code=404)
    return EventSourceResponse(_stream_run(run))


@app.post("/api/run/{run_id}/cancel")
async def api_cancel_run(run_id: str):
    """Ask a run to stop. It ends after the current step; running code is interrupted."""
    ok = runner.cancel_run(run_id)
    return {"ok": ok, "run_id": run_id}


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
    # The asset query string is the version, so a new release is never served
    # from a browser's cached copy of the old script.
    with open(os.path.join(WEB_DIR, "index.html"), encoding="utf-8") as f:
        html = f.read().replace("__V__", APP_VERSION)
    return HTMLResponse(html, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


if os.path.isdir(WEB_DIR):
    app.mount("/", NoCacheStaticFiles(directory=WEB_DIR, html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    # Bound to this machine only unless asked otherwise: the sandbox executes
    # whatever code the model writes, and the settings hold API keys. Inside a
    # container the port has to be reachable from the host, so all interfaces.
    host = os.environ.get("GISCLAW_HOST") or ("0.0.0.0" if in_container() else "127.0.0.1")
    port = int(os.environ.get("GISCLAW_PORT") or 8765)
    uvicorn.run(app, host=host, port=port, log_level="info")

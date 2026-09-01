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

"""Talking to a model server on the user's own machine.

The OpenAI-compatible endpoint every local server offers (`/v1/models`,
`/v1/chat/completions`) is enough to run, but not enough to set up well: it
says nothing about how big a model is, how it is quantised, or — the one that
decides whether the agent works at all — how much context the server gives
it. Ollama's own API has all of that, so when the server is Ollama it is asked
directly; anything else falls back to the compatible listing.

Standard library only: this is a few small HTTP calls to localhost.
"""
import json
import re
import urllib.error
import urllib.request
from urllib.parse import urlsplit, urlunsplit

# Where each server listens by default, and what it calls its models.
PRESETS = {
    "ollama":   {"display": "Ollama",    "base_url": "http://localhost:11434/v1",
                 "docs": "https://ollama.com/download"},
    "lmstudio": {"display": "LM Studio", "base_url": "http://localhost:1234/v1",
                 "docs": "https://lmstudio.ai"},
    "vllm":     {"display": "vLLM",      "base_url": "http://localhost:8000/v1",
                 "docs": "https://docs.vllm.ai"},
    "other":    {"display": "Other (OpenAI-compatible)", "base_url": "",
                 "docs": ""},
}

# A ReAct round carries the whole system prompt (tools, operating rules, the
# data profile, the project digest) plus the transcript so far. Below this the
# server silently drops the front of the prompt, which is the system prompt.
MIN_CONTEXT_TOKENS = 8192
RECOMMENDED_CONTEXT_TOKENS = 16384

# Models that have completed the multi-step benchmark tasks in this project's
# own testing, grouped by the memory they need to run.
RECOMMENDED = [
    {"name": "devstral-small-2:24b", "needs": "24 GB",
     "note": "Completed the full urban-heat workflow in 12 rounds with no help."},
    {"name": "qwen2.5-coder:14b", "needs": "16 GB (not a 16 GB Mac)",
     "note": "Reliable on 3–5 step tasks; the usual choice on a desktop GPU."},
    {"name": "gpt-oss:20b", "needs": "16 GB",
     "note": "Mixture-of-experts; fast, adequate planning."},
]


def _get(url: str, timeout: float = 4.0, data: dict = None):
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
    body = json.dumps(data).encode() if data is not None else None
    with urllib.request.urlopen(req, data=body, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8") or "null")


def native_root(base_url: str) -> str:
    """`http://host:11434/v1` -> `http://host:11434` (Ollama's own API lives there)."""
    p = urlsplit(base_url.strip())
    path = re.sub(r"/v1/?$", "", p.path or "")
    return urlunsplit((p.scheme or "http", p.netloc, path.rstrip("/"), "", ""))


def _context_from_show(info: dict):
    """The model's own context limit, from Ollama's /api/show."""
    mi = info.get("model_info") or {}
    for k, v in mi.items():
        if k.endswith(".context_length") and isinstance(v, (int, float)):
            return int(v)
    params = info.get("parameters") or ""
    m = re.search(r"num_ctx\s+(\d+)", params)
    return int(m.group(1)) if m else None


def _ollama_num_ctx(info: dict):
    """A num_ctx set in the Modelfile — the value the server will actually use."""
    m = re.search(r"num_ctx\s+(\d+)", info.get("parameters") or "")
    return int(m.group(1)) if m else None


def probe(base_url: str, detail_cap: int = 16) -> dict:
    """What is running at `base_url`, and what it is serving.

    Returns {ok, kind, version, models:[{id, size_gb, params, quant, family,
    context_max, context_set}], running:[{id, context}], error}.
    """
    base = (base_url or "").strip()
    if not base:
        return {"ok": False, "error": "No address."}
    root = native_root(base)
    out = {"ok": False, "kind": "openai", "version": "", "models": [], "running": [], "base_url": base}

    # Ollama first: its own API tells the whole story.
    try:
        ver = _get(f"{root}/api/version", timeout=3)
        out["kind"] = "ollama"
        out["version"] = str((ver or {}).get("version", ""))
    except Exception:
        ver = None

    if ver is not None:
        try:
            tags = _get(f"{root}/api/tags", timeout=4) or {}
        except Exception as e:
            return dict(out, error=f"Ollama answered, but listing models failed: {e}")
        for m in (tags.get("models") or []):
            d = m.get("details") or {}
            entry = {
                "id": m.get("name") or m.get("model"),
                "size_gb": round((m.get("size") or 0) / 1e9, 1),
                "params": d.get("parameter_size", ""),
                "quant": d.get("quantization_level", ""),
                "family": d.get("family", ""),
                "context_max": None, "context_set": None,
            }
            out["models"].append(entry)
        for entry in out["models"][:detail_cap]:
            try:
                info = _get(f"{root}/api/show", timeout=6, data={"model": entry["id"]}) or {}
                entry["context_max"] = _context_from_show(info)
                entry["context_set"] = _ollama_num_ctx(info)
            except Exception:
                pass
        try:
            ps = _get(f"{root}/api/ps", timeout=3) or {}
            out["running"] = [{"id": r.get("name") or r.get("model"),
                               "context": r.get("context_length")}
                              for r in (ps.get("models") or [])]
        except Exception:
            pass
        out["ok"] = True
        return out

    # Anything else: the compatible listing, names only.
    try:
        listing = _get(f"{base.rstrip('/')}/models", timeout=4) or {}
    except urllib.error.URLError as e:
        return dict(out, error=f"Nothing answered at {base} ({e.reason}). Is the server running?")
    except Exception as e:
        return dict(out, error=f"Could not list models at {base}: {e}")
    for m in (listing.get("data") or []):
        out["models"].append({"id": m.get("id"), "size_gb": None, "params": "", "quant": "",
                              "family": "", "context_max": None, "context_set": None})
    out["ok"] = True
    return out


def loaded_context(base_url: str, model_id: str):
    """After a call: the context length the server actually loaded the model with."""
    try:
        ps = _get(f"{native_root(base_url)}/api/ps", timeout=3) or {}
    except Exception:
        return None
    for r in (ps.get("models") or []):
        if (r.get("name") or r.get("model")) == model_id:
            return r.get("context_length")
    return None


def context_chars_for(context_tokens) -> int:
    """How much transcript to send each round, given the server's window.

    Roughly 3.5 characters per token, minus what the system prompt needs.
    Unknown windows get a conservative figure that fits a 16k context.
    """
    if not context_tokens:
        return 24_000
    usable = max(2000, int(context_tokens) - 5000)
    return int(max(8_000, min(usable * 3, 200_000)))


def context_advice(context_tokens, kind: str) -> str:
    """One paragraph the interface shows when the window is too small."""
    if context_tokens is None or context_tokens >= MIN_CONTEXT_TOKENS:
        return ""
    if kind == "ollama":
        return (f"This model is loaded with a {context_tokens:,}-token context. A run needs at "
                f"least {MIN_CONTEXT_TOKENS:,}; {RECOMMENDED_CONTEXT_TOKENS:,} is comfortable. "
                "Raise it in the Ollama app under Settings → Context length, or start the "
                f"server with OLLAMA_CONTEXT_LENGTH={RECOMMENDED_CONTEXT_TOKENS} — then press Test again.")
    return (f"This model reports a {context_tokens:,}-token context; a run needs at least "
            f"{MIN_CONTEXT_TOKENS:,}. Raise the context length in the server's settings.")

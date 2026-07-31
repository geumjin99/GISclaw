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

"""GISclaw product — persistent settings (API keys, model registry, memory).

Everything the user configures lives in ONE place on the mounted volume:

    <WORKSPACE>/.gisclaw/
        settings.json   API keys + model registry + flags   (mode 0600)
        MEMORY.md       global user preferences, injected into every run

Putting it under WORKSPACE (not the repo) means it survives `docker compose
down`, image rebuilds and container recreation — the same reason project data
lives there. Keys are never returned to the browser in full: the API hands back
`sk-…1234`-style masks and the real value only ever leaves this module towards
the LLM engine.

Environment variables (OPENAI_API_KEY etc.) still work and act as the fallback
when a provider has no key stored here, so existing .env setups keep running.
"""
import json
import os
from typing import Optional

SETTINGS_VERSION = 1

# ---------------------------------------------------------------- providers --
# A "provider" is an API endpoint + credential. Several models can share one.
# `engine` picks the class in src/agent/llm_engine.py; every OpenAI-compatible
# vendor (DeepSeek, Gemini's compat layer, Together, vLLM, Ollama…) reuses
# OpenAIEngine with a different base_url — the same trick the research scripts use.
PROVIDERS = {
    "openai": {
        "display": "OpenAI",
        "engine": "openai",
        "env": "OPENAI_API_KEY",
        "base_url": None,
        "key_hint": "sk-…",
        "docs": "https://platform.openai.com/api-keys",
    },
    "claude": {
        "display": "Anthropic (Claude)",
        "engine": "claude",
        "env": "CLAUDE_API_KEY",
        "base_url": None,
        "key_hint": "sk-ant-…",
        "docs": "https://console.anthropic.com/settings/keys",
    },
    "deepseek": {
        "display": "DeepSeek",
        "engine": "openai",
        "env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "key_hint": "sk-…",
        "docs": "https://platform.deepseek.com/api_keys",
    },
    "gemini": {
        "display": "Google Gemini",
        "engine": "openai",
        "env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_hint": "AIza…",
        "docs": "https://aistudio.google.com/apikey",
    },
    "custom": {
        "display": "Custom (OpenAI-compatible)",
        "engine": "openai",
        "env": "GISCLAW_CUSTOM_API_KEY",
        "base_url": "",          # user supplies it
        "key_hint": "any token your endpoint expects",
        "docs": "",
        "needs_base_url": True,
    },
}

# ------------------------------------------------------------ builtin models --
# Curated defaults. The user can disable any of these and add their own; the
# stored registry only carries overrides, so upgrading this table keeps working.
BUILTIN_MODELS = {
    "gpt-5.4": {
        "provider": "openai", "model_name": "gpt-5.4", "display": "GPT-5.4",
        "tier": "Flagship", "timeout": 300, "max_rounds": 50, "max_tokens": 4096,
        "cost_per_m": (2.0, 8.0),
    },
    "claude-opus-5": {
        "provider": "claude", "model_name": "claude-opus-5",
        "display": "Claude Opus 5", "tier": "Flagship",
        # Thinking is on by default on Opus 5 and counts against max_tokens,
        # so this is deliberately roomier than the other entries.
        "timeout": 600, "max_rounds": 50, "max_tokens": 16000,
        "cost_per_m": (5.0, 25.0),
    },
    "deepseek": {
        "provider": "deepseek", "model_name": "deepseek-chat",
        "display": "DeepSeek V3.2", "tier": "Open-weight",
        "timeout": 180, "max_rounds": 35, "max_tokens": 2048,
        "cost_per_m": (0.28, 0.42),
    },
}

DEFAULT_SETTINGS = {
    "version": SETTINGS_VERSION,
    "providers": {},        # pid -> {"api_key": str, "base_url": str}
    "models": {},           # mid -> override dict (builtin) or full spec (custom)
    "skills": {},           # skill name -> {"enabled": bool}
    "memory_enabled": True,
}

MEMORY_TEMPLATE = """<!--
Standing preferences GISclaw should apply to every analysis, in every project.
Edit freely — this file is read at the start of each run and injected into the
agent's system prompt. Keep it short; it costs tokens on every single call.
Text inside HTML comments is for you, not the model: it is stripped before
injection, so notes-to-self here cost nothing.
-->

## Cartography

- (example) Use colour-blind-safe sequential ramps; avoid rainbow/jet.

## Deliverables

- (example) Every map needs a scale bar, north arrow and the data source.

## Conventions

- (example) Projected CRS for this region: EPSG:5179 (Korea 2000 / Unified CS).
"""


class SettingsStore:
    """Reads/writes the on-disk settings. Cheap enough to re-read every call."""

    def __init__(self, workspace: str):
        self.dir = os.path.join(workspace, ".gisclaw")
        self.path = os.path.join(self.dir, "settings.json")
        self.memory_path = os.path.join(self.dir, "MEMORY.md")
        os.makedirs(self.dir, exist_ok=True)

    # ------------------------------------------------------------- raw I/O --
    def load(self) -> dict:
        data = dict(DEFAULT_SETTINGS)
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as f:
                    stored = json.load(f)
                if isinstance(stored, dict):
                    data.update(stored)
            except Exception:
                pass  # corrupt file must never take the app down
        data.setdefault("providers", {})
        data.setdefault("models", {})
        data.setdefault("skills", {})
        return data

    def save(self, data: dict):
        data["version"] = SETTINGS_VERSION
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)
        try:
            os.chmod(self.path, 0o600)   # it holds plaintext API keys
        except OSError:
            pass

    # ----------------------------------------------------------- providers --
    def provider_key(self, pid: str) -> str:
        """Stored key wins; fall back to the environment so .env keeps working."""
        stored = self.load()["providers"].get(pid, {})
        key = (stored.get("api_key") or "").strip()
        if key:
            return key
        env = PROVIDERS.get(pid, {}).get("env", "")
        return (os.environ.get(env, "") or "").strip() if env else ""

    def provider_base_url(self, pid: str) -> Optional[str]:
        stored = self.load()["providers"].get(pid, {})
        url = (stored.get("base_url") or "").strip()
        if url:
            return url
        return PROVIDERS.get(pid, {}).get("base_url") or None

    def set_provider(self, pid: str, api_key: Optional[str] = None,
                     base_url: Optional[str] = None):
        data = self.load()
        entry = data["providers"].get(pid, {})
        if api_key is not None:
            # An all-mask value means "unchanged" — the UI never holds the real key.
            if not _is_mask(api_key):
                entry["api_key"] = api_key.strip()
        if base_url is not None:
            entry["base_url"] = base_url.strip()
        data["providers"][pid] = entry
        self.save(data)

    def clear_provider_key(self, pid: str):
        data = self.load()
        if pid in data["providers"]:
            data["providers"][pid]["api_key"] = ""
            self.save(data)

    def providers_public(self) -> list:
        """Provider list for the UI — masked keys only."""
        data = self.load()
        out = []
        for pid, meta in PROVIDERS.items():
            stored = data["providers"].get(pid, {})
            stored_key = (stored.get("api_key") or "").strip()
            env_key = (os.environ.get(meta.get("env", ""), "") or "").strip()
            out.append({
                "id": pid,
                "display": meta["display"],
                "key_hint": meta.get("key_hint", ""),
                "docs": meta.get("docs", ""),
                "needs_base_url": bool(meta.get("needs_base_url")),
                "base_url": (stored.get("base_url") or meta.get("base_url") or ""),
                "masked_key": mask_key(stored_key or env_key),
                "configured": bool(stored_key or env_key),
                "from_env": bool(env_key and not stored_key),
                "env_var": meta.get("env", ""),
            })
        return out

    # -------------------------------------------------------------- models --
    def models(self) -> dict:
        """Effective registry: builtins + custom, with user overrides applied."""
        data = self.load()
        overrides = data.get("models", {})
        merged = {}
        for mid, spec in BUILTIN_MODELS.items():
            m = dict(spec)
            m["custom"] = False
            m["enabled"] = True
            ov = overrides.get(mid, {})
            for field in ("display", "model_name", "tier", "timeout", "max_rounds",
                          "max_tokens", "cost_per_m", "enabled", "provider"):
                if field in ov:
                    m[field] = ov[field]
            merged[mid] = m
        for mid, spec in overrides.items():
            if mid in BUILTIN_MODELS or not spec.get("custom"):
                continue
            m = dict(spec)
            m.setdefault("enabled", True)
            m.setdefault("tier", "Custom")
            m.setdefault("timeout", 300)
            m.setdefault("max_rounds", 50)
            m.setdefault("max_tokens", 4096)
            m.setdefault("cost_per_m", (0.0, 0.0))
            merged[mid] = m
        return merged

    def model_config(self, mid: str) -> Optional[dict]:
        """Resolve a model id into everything init_llm() needs, or None."""
        m = self.models().get(mid)
        if not m:
            return None
        provider = m.get("provider", "openai")
        meta = PROVIDERS.get(provider, PROVIDERS["openai"])
        cfg = dict(m)
        cfg["id"] = mid
        cfg["engine"] = meta["engine"]
        cfg["api_key"] = self.provider_key(provider)
        cfg["base_url"] = m.get("base_url") or self.provider_base_url(provider)
        cost = cfg.get("cost_per_m") or (0.0, 0.0)
        cfg["cost_per_m"] = tuple(cost)
        return cfg

    def models_public(self) -> list:
        """Model list for the UI/selector, annotated with readiness."""
        out = []
        for mid, m in self.models().items():
            provider = m.get("provider", "openai")
            has_key = bool(self.provider_key(provider))
            needs_url = bool(PROVIDERS.get(provider, {}).get("needs_base_url"))
            has_url = bool(m.get("base_url") or self.provider_base_url(provider))
            out.append({
                "id": mid,
                "display": m.get("display", mid),
                "tier": m.get("tier", ""),
                "provider": provider,
                "provider_display": PROVIDERS.get(provider, {}).get("display", provider),
                "model_name": m.get("model_name", ""),
                "custom": bool(m.get("custom")),
                "enabled": bool(m.get("enabled", True)),
                "ready": bool(has_key and (has_url or not needs_url)),
                "base_url": m.get("base_url", "") or "",
                "max_rounds": m.get("max_rounds", 50),
                "max_tokens": m.get("max_tokens", 4096),
                "cost_per_m": list(m.get("cost_per_m", (0.0, 0.0))),
            })
        out.sort(key=lambda x: (not x["ready"], x["custom"], x["display"].lower()))
        return out

    def upsert_model(self, mid: str, spec: dict):
        data = self.load()
        cur = data["models"].get(mid, {})
        cur.update(spec)
        if mid not in BUILTIN_MODELS:
            cur["custom"] = True
        data["models"][mid] = cur
        self.save(data)

    def delete_model(self, mid: str) -> bool:
        """Custom models are removed; builtins can only be disabled."""
        data = self.load()
        if mid in BUILTIN_MODELS:
            data["models"].setdefault(mid, {})["enabled"] = False
            self.save(data)
            return True
        if mid in data["models"]:
            del data["models"][mid]
            self.save(data)
            return True
        return False

    # -------------------------------------------------------------- memory --
    def read_memory(self) -> str:
        if not os.path.exists(self.memory_path):
            return MEMORY_TEMPLATE
        try:
            with open(self.memory_path, encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""

    def memory_for_prompt(self) -> str:
        """What the model actually sees: comments stripped, no file title.

        The editor view keeps the user's own scaffolding; the prompt should not
        pay tokens for instructions addressed to the human.
        """
        import re as _re
        text = _re.sub(r"<!--.*?-->", "", self.read_memory(), flags=_re.S)
        lines = text.splitlines()
        while lines and (not lines[0].strip() or _re.match(r"^#\s+\S", lines[0])):
            lines.pop(0)
        return "\n".join(lines).strip()

    def write_memory(self, text: str):
        with open(self.memory_path, "w", encoding="utf-8") as f:
            f.write(text)

    def append_memory(self, line: str, section: str = "Notes"):
        """Append one remembered fact under a `## section` heading, creating it if needed."""
        text = self.read_memory()
        if not os.path.exists(self.memory_path):
            text = MEMORY_TEMPLATE
        line = line.strip()
        if not line:
            return text
        bullet = f"- {line}"
        heading = f"## {section}"
        if heading in text:
            head, _, tail = text.partition(heading)
            # insert at the end of that section (before the next ## or EOF)
            rest_lines = tail.split("\n")
            cut = len(rest_lines)
            for i, ln in enumerate(rest_lines[1:], start=1):
                if ln.startswith("## "):
                    cut = i
                    break
            body = rest_lines[:cut]
            while body and not body[-1].strip():
                body.pop()
            body.append(bullet)
            text = head + heading + "\n".join(body) + "\n\n" + "\n".join(rest_lines[cut:])
        else:
            text = text.rstrip() + f"\n\n{heading}\n\n{bullet}\n"
        self.write_memory(text)
        return text

    # -------------------------------------------------------------- skills --
    def skill_overrides(self) -> dict:
        return self.load().get("skills", {})

    def set_skill_enabled(self, name: str, on: bool):
        data = self.load()
        data.setdefault("skills", {}).setdefault(name, {})["enabled"] = bool(on)
        self.save(data)

    def compact_log(self) -> bool:
        """Spend one extra API call per run to write the compacted LOG.md entry."""
        return bool(self.load().get("compact_log", True))

    def set_compact_log(self, on: bool):
        data = self.load()
        data["compact_log"] = bool(on)
        self.save(data)

    def viewer_follow(self) -> bool:
        """Whether the viewer follows the agent between Map, Code and Result.

        On by default — watching it work is most of the appeal. Off keeps the
        tab you chose, for reading one thing while the run continues.
        """
        return bool(self.load().get("viewer_follow", True))

    def set_viewer_follow(self, on: bool):
        data = self.load()
        data["viewer_follow"] = bool(on)
        self.save(data)

    def skills_auto(self) -> bool:
        """Pre-load a matching skill server-side (models rarely self-invoke)."""
        return bool(self.load().get("skills_auto", True))

    def set_skills_auto(self, on: bool):
        data = self.load()
        data["skills_auto"] = bool(on)
        self.save(data)

    def forget_skill(self, name: str):
        data = self.load()
        if name in data.get("skills", {}):
            del data["skills"][name]
            self.save(data)

    def memory_enabled(self) -> bool:
        return bool(self.load().get("memory_enabled", True))

    def set_memory_enabled(self, on: bool):
        data = self.load()
        data["memory_enabled"] = bool(on)
        self.save(data)


# ------------------------------------------------------------------ helpers --
def mask_key(key: str) -> str:
    """`sk-abcdef…wxyz` — enough to recognise a key, useless to steal."""
    key = (key or "").strip()
    if not key:
        return ""
    if len(key) <= 10:
        return key[:2] + "…" + key[-2:]
    return key[:5] + "…" + key[-4:]


def _is_mask(value: str) -> bool:
    return "…" in (value or "")

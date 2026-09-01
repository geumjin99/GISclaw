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
# vendor (DeepSeek, Gemini's compatibility layer, vLLM, Ollama…) reuses
# OpenAIEngine with a different base_url.
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
    "local": {
        "display": "Local model (Ollama / LM Studio / vLLM)",
        "engine": "openai",
        "env": "GISCLAW_LOCAL_API_KEY",
        # Ollama's OpenAI-compatible port. LM Studio is :1234/v1, vLLM :8000/v1.
        "base_url": "http://localhost:11434/v1",
        "key_hint": "usually none — leave blank",
        "docs": "https://ollama.com/download",
        "needs_base_url": True,
        # A model on your own machine has nobody to bill: these servers accept
        # any token, or none. Requiring a key here would block the one provider
        # that does not have one.
        "key_optional": True,
        "hint": ("Serve a model first (e.g. `ollama pull qwen2.5-coder:14b`), then "
                 "press Fetch below to list what it is serving. Nothing leaves your "
                 "machine with these — see Help → About."),
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

# A local server is reached at localhost from your desktop, but inside the
# container localhost is the container itself, so the address the user typed
# would connect to nothing. Docker publishes the host under this name; on Linux
# it exists only if the compose file asks for it (`extra_hosts: host-gateway`),
# so fall back to the default gateway address when the name does not resolve.
_LOOPBACK = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}
_host_alias_cache: list = []


def in_container() -> bool:
    return os.path.exists("/.dockerenv") or os.environ.get("GISCLAW_IN_DOCKER") == "1"


def _host_alias() -> Optional[str]:
    """Address of the machine running Docker, as seen from inside a container."""
    if _host_alias_cache:
        return _host_alias_cache[0]
    import socket
    alias = None
    try:
        socket.gethostbyname("host.docker.internal")
        alias = "host.docker.internal"
    except OSError:
        try:  # default gateway = the host, on a standard bridge network
            with open("/proc/net/route", encoding="utf-8") as f:
                for line in f.read().splitlines()[1:]:
                    parts = line.split()
                    if len(parts) > 2 and parts[1] == "00000000":
                        raw = int(parts[2], 16)
                        alias = ".".join(str((raw >> s) & 0xFF) for s in (0, 8, 16, 24))
                        break
        except OSError:
            alias = None
    _host_alias_cache.append(alias)
    return alias


def _listening(host: str, port: int, timeout: float = 0.35) -> bool:
    import socket
    try:
        with socket.create_connection((host, port), timeout):
            return True
    except OSError:
        return False


def reachable_base_url(url: Optional[str]) -> Optional[str]:
    """Rewrite a loopback address so it still points at the user's own machine.

    Only applies inside a container, and only when nothing answers at the
    address as written — with `network_mode: host` localhost already is the
    host, and rewriting there would send the request somewhere else entirely.
    The stored value is never rewritten: the interface keeps showing what was
    entered, because that is what is true on the user's own machine.
    """
    if not url or not in_container():
        return url
    from urllib.parse import urlsplit, urlunsplit
    parts = urlsplit(url)
    host = (parts.hostname or "").strip("[]")
    if host.lower() not in _LOOPBACK:
        return url
    port = parts.port or (443 if parts.scheme == "https" else 80)
    if _listening(host, port):
        return url
    alias = _host_alias()
    if not alias:
        return url
    return urlunsplit((parts.scheme, f"{alias}:{port}", parts.path,
                       parts.query, parts.fragment))

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
        # Created private from the first byte — it holds plaintext API keys,
        # and a chmod afterwards would leave a moment where it is not.
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)
        try:
            os.chmod(self.path, 0o600)
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

    def provider_base_url(self, pid: str, raw: bool = False) -> Optional[str]:
        """Endpoint for this provider. `raw=True` returns it exactly as stored."""
        stored = self.load()["providers"].get(pid, {})
        url = (stored.get("base_url") or "").strip()
        if not url:
            url = PROVIDERS.get(pid, {}).get("base_url") or None
        return url if raw else reachable_base_url(url)

    def provider_key_optional(self, pid: str) -> bool:
        """True for endpoints that have nobody to bill — a local server."""
        return bool(PROVIDERS.get(pid, {}).get("key_optional"))

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
            key_optional = bool(meta.get("key_optional"))
            url = (stored.get("base_url") or meta.get("base_url") or "")
            out.append({
                "id": pid,
                "display": meta["display"],
                "key_hint": meta.get("key_hint", ""),
                "docs": meta.get("docs", ""),
                "hint": meta.get("hint", ""),
                "needs_base_url": bool(meta.get("needs_base_url")),
                "key_optional": key_optional,
                "base_url": url,
                "masked_key": mask_key(stored_key or env_key),
                # "Configured" means usable: for a local server that is an
                # endpoint, not a credential.
                "configured": bool(stored_key or env_key or (key_optional and url)),
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
                          "max_tokens", "cost_per_m", "enabled", "provider", "context_chars"):
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
        cfg["key_optional"] = bool(meta.get("key_optional"))
        cfg["base_url"] = reachable_base_url(
            m.get("base_url") or self.provider_base_url(provider, raw=True))
        cost = cfg.get("cost_per_m") or (0.0, 0.0)
        cfg["cost_per_m"] = tuple(cost)
        # How much transcript goes back each round. A hosted model can take
        # the whole run; a server on this machine usually has a small window,
        # and overrunning it drops the system prompt off the front.
        if not cfg.get("context_chars"):
            cfg["context_chars"] = 24_000 if provider == "local" else 100_000
        return cfg

    def models_public(self) -> list:
        """Model list for the UI/selector, annotated with readiness."""
        out = []
        for mid, m in self.models().items():
            provider = m.get("provider", "openai")
            has_key = bool(self.provider_key(provider)) or self.provider_key_optional(provider)
            needs_url = bool(PROVIDERS.get(provider, {}).get("needs_base_url"))
            # `raw` here: this only asks whether an address is configured, and
            # resolving one costs a connection probe inside a container.
            has_url = bool(m.get("base_url") or self.provider_base_url(provider, raw=True))
            ready = has_key and (has_url or not needs_url)
            out.append({
                "id": mid,
                "display": m.get("display", mid),
                "tier": m.get("tier", ""),
                "provider": provider,
                "provider_display": PROVIDERS.get(provider, {}).get("display", provider),
                "model_name": m.get("model_name", ""),
                "custom": bool(m.get("custom")),
                "enabled": bool(m.get("enabled", True)),
                "ready": bool(ready),
                # Why it cannot run, in the words the settings panel should use.
                "blocked": "" if ready else ("no endpoint" if has_key else "no key"),
                "base_url": m.get("base_url", "") or "",
                "max_rounds": m.get("max_rounds", 50),
                "max_tokens": m.get("max_tokens", 4096),
                "timeout": m.get("timeout", 300),
                "context_chars": m.get("context_chars") or (24_000 if provider == "local" else 100_000),
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

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

"""GISclaw product — per-project durable record.

A GIS project can run for months. Three artefacts make that survivable, all of
them inside the project folder so they travel with the data:

    chat.jsonl   machine-readable conversation log — what you asked, what the
                 agent did, which run produced which file. The UI rebuilds the
                 conversation from this on every page load.
    JOURNAL.md   the human-readable lab notebook. One section per run, appended
                 forever, readable in any editor or on GitHub without the app.
    project.json the manifest (name, notes, created_at) — unchanged.

`build_context()` turns the recent history into a compact block that is injected
into the agent's system prompt, so a run in September knows what April decided.
"""
import json
import os
from datetime import datetime

CHAT_FILE = "chat.jsonl"
JOURNAL_FILE = "JOURNAL.md"

# How much history to feed back into the model. Small on purpose — this rides on
# every single API call, and stale detail is worse than none.
CONTEXT_RUNS = 5
CONTEXT_ASK_CHARS = 200


# ------------------------------------------------------------------- chat --
def chat_path(pdir: str) -> str:
    return os.path.join(pdir, CHAT_FILE)


def append_chat(pdir: str, entry: dict) -> dict:
    """Append one conversation entry (user turn, agent turn, or note)."""
    entry = dict(entry)
    entry.setdefault("ts", datetime.now().isoformat(timespec="seconds"))
    os.makedirs(pdir, exist_ok=True)
    with open(chat_path(pdir), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def read_chat(pdir: str, limit: int = 0) -> list:
    """All conversation entries, oldest first. `limit` keeps the last N."""
    path = chat_path(pdir)
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out[-limit:] if limit else out


def clear_chat(pdir: str):
    """Archive the conversation instead of destroying it — this is a lab record."""
    path = chat_path(pdir)
    if os.path.exists(path):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.replace(path, os.path.join(pdir, f".chat_archived_{stamp}.jsonl"))


# ---------------------------------------------------------------- journal --
def journal_path(pdir: str) -> str:
    return os.path.join(pdir, JOURNAL_FILE)


def read_journal(pdir: str) -> str:
    path = journal_path(pdir)
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read()


def ensure_journal_header(pdir: str, project_name: str):
    path = journal_path(pdir)
    if os.path.exists(path):
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            f"# {project_name} — analysis journal\n\n"
            "Written by GISclaw, one section per analysis run: what was asked, what the\n"
            "agent did, what came out, and where the artefacts live. Append-only —\n"
            "safe to read months later, safe to keep in version control.\n"
        )


def append_run(pdir: str, project_name: str, entry: dict):
    """Append one run's section to JOURNAL.md.

    `entry` carries: run_id, model, ask, success, rounds, self_corrections,
    elapsed_s, outputs, cost, steps (list of {round, action, thought}).
    """
    ensure_journal_header(pdir, project_name)
    run_id = entry.get("run_id", "?")
    when = entry.get("ts") or datetime.now().isoformat(timespec="seconds")
    when = when.replace("T", " ")
    ok = entry.get("success")
    verdict = "success" if ok else "failed"
    rounds = entry.get("rounds", 0)
    corr = entry.get("self_corrections", 0)
    secs = entry.get("elapsed_s", 0)
    cost = entry.get("cost") or {}
    outputs = entry.get("outputs") or []
    steps = entry.get("steps") or []

    lines = [
        "",
        "---",
        "",
        f"## {when} · {run_id}",
        "",
        f"- **Model:** {entry.get('model_display') or entry.get('model', '?')}",
        f"- **Result:** {verdict} · {rounds} rounds · {corr} self-correction(s) · {secs}s",
    ]
    if cost.get("cost_usd"):
        lines.append(
            f"- **Cost:** ${cost.get('cost_usd', 0):.4f} "
            f"({cost.get('api_calls', 0)} calls, "
            f"{cost.get('input_tokens', 0)}→{cost.get('output_tokens', 0)} tokens)"
        )
    lines += ["", "**Asked**", "", "> " + (entry.get("ask", "") or "").replace("\n", "\n> "), ""]

    if outputs:
        lines.append("**Produced**")
        lines.append("")
        for fn in outputs:
            lines.append(f"- `outputs/{fn}` (original in `runs/{run_id}/pred_results/{fn}`)")
        lines.append("")

    if steps:
        lines.append("**What the agent did**")
        lines.append("")
        # Only rounds that parsed into a valid action reach the trace, so the list
        # can be shorter than the round count. Say so rather than look complete.
        missing = (rounds or 0) - len(steps)
        if missing > 0:
            lines.append(
                f"*({missing} of {rounds} rounds are not listed — the model's reply "
                f"could not be parsed into an action there; see `runs/{run_id}/run.log`.)*"
            )
            lines.append("")
        for s in steps:
            act = s.get("action", "?")
            th = (s.get("thought", "") or "").strip().replace("\n", " ")
            if len(th) > 160:
                th = th[:157] + "…"
            flag = "" if s.get("success", True) else "  ⟵ failed, then corrected"
            lines.append(f"{s.get('round', '?')}. `{act}` — {th}{flag}")
        lines.append("")

    lines.append(f"**Full trace:** `runs/{run_id}/trace.jsonl` · **code:** `runs/{run_id}/code.py`")
    lines.append("")

    with open(journal_path(pdir), "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def append_note(pdir: str, project_name: str, text: str):
    """A free-text note the user pinned into the journal."""
    ensure_journal_header(pdir, project_name)
    when = datetime.now().isoformat(timespec="seconds").replace("T", " ")
    with open(journal_path(pdir), "a", encoding="utf-8") as f:
        f.write(f"\n---\n\n## {when} · note\n\n{text.strip()}\n")


# ---------------------------------------------------------------- context --
def build_context(pdir: str, manifest: dict, digest: str = "") -> str:
    """Compact project history for the system prompt — the continuity layer.

    Returns "" when there is nothing worth saying, so a first run pays nothing.
    """
    parts = []
    notes = (manifest.get("notes") or "").strip()
    if notes:
        parts.append(f"Project notes: {notes}")

    runs = [e for e in read_chat(pdir) if e.get("role") == "agent"]
    if runs:
        if digest:
            # The compacted log says more per token than the ask list, and it
            # carries the caveats (imputed values, failed checks) the asks don't.
            parts.append("Compacted log of what this project has established so far:")
            parts.append("")
            parts.append(digest)
            parts.append("")
        else:
            parts.append(f"This project has {len(runs)} previous analysis run(s). Most recent first:")
            for e in reversed(runs[-CONTEXT_RUNS:]):
                ask = (e.get("ask", "") or "").strip().replace("\n", " ")
                if len(ask) > CONTEXT_ASK_CHARS:
                    ask = ask[:CONTEXT_ASK_CHARS - 1] + "…"
                state = "ok" if e.get("success") else "failed"
                outs = ", ".join(e.get("outputs", [])[:6]) or "no files"
                when = (e.get("ts", "") or "")[:10]
                parts.append(f'- [{when}, {state}] asked: "{ask}" → produced: {outs}')
        # Naming the actual files makes reuse concrete. Without this the agent
        # was observed re-deriving a quantity a previous run had already computed
        # (and re-deriving it worse), despite the history being in the prompt.
        made_by = {}
        for e in runs:
            for fn in e.get("outputs", []):
                made_by[fn] = e.get("run_id", "")
        out_dir = os.path.join(pdir, "outputs")
        existing = sorted(f for f in os.listdir(out_dir)) if os.path.isdir(out_dir) else []
        if existing:
            parts.append("")
            parts.append("Files already in this project's outputs/ folder:")
            for fn in existing[:25]:
                src = made_by.get(fn)
                parts.append(f"- {fn}" + (f"  (produced by {src})" if src else ""))
            if len(existing) > 25:
                parts.append(f"- … and {len(existing) - 25} more")

        parts.append(
            "\nLoad these like any other dataset — they are in outputs/, alongside "
            "the raw data in data/. If a previous run already computed a quantity "
            "this task needs, REUSE that file; do not recompute it by a different "
            "method, which would make two runs of the same project disagree. Only "
            "redo finished work if the task explicitly asks for it, and say so."
        )

    return "\n".join(parts)

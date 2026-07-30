"""GISclaw product — the two places we spend an extra API call on thinking
*about* the work rather than doing it.

1. `compact_run()` — after each run, turn the raw trace into a short digest and
   append it to the project's LOG.md. JOURNAL.md stays the mechanical record
   (every round, verbatim); LOG.md is the compressed running history a human —
   or the agent, months later — actually reads. This is the same idea as a
   context compaction: the transcript is the source, the digest is what travels.

2. `classify_request()` — decide what an incoming message *is* before spending a
   whole ReAct run on it. The agent's finish-guard requires output files, so a
   plain question ("summarise what you did") previously forced it to redo the
   analysis and write a placeholder file just to be allowed to stop. Questions
   and off-topic requests are answered from the project record instead.
"""
import json
import os
import re

LOG_FILE = "LOG.md"

# Keep the digest short enough that a year of runs still fits in a prompt.
COMPACT_SYSTEM = """You write the running log for a GIS analysis project.

You are given ONE analysis run: what the user asked, what the agent did, and what
it produced. Write a compact entry for the project log.

Rules:
- Only state facts present in the material given. Never infer a number.
- Lead with what the run established or produced, not with what was attempted.
- Record caveats that change how the result may be used — imputed or filled
  values, units with no data, methods chosen for convenience, failed checks.
  If the run filled in missing data, that MUST appear.
- Note anything a later session needs in order to continue: file names produced,
  the CRS worked in, decisions taken.
- No preamble, no headings, no restating these instructions.

Format exactly:
**Result:** one or two sentences.
**Method:** one or two sentences, naming the key operations.
**Numbers:** the specific figures established, or "none reported".
**Caveats:** the honest limitations, or "none noted".
**Carries forward:** files and decisions a later run should reuse, or "nothing".
"""

INTENT_SYSTEM = """You triage incoming messages for a GIS analysis assistant.

Classify the user's message into exactly one of:

- "analysis": asks for geospatial work on this project's data — computing,
  mapping, joining, measuring, modelling, producing files.
- "question": asks about this project, its data, its previous runs, its results,
  or how to use the tool. Answerable from the context provided, with no new
  computation.
- "offtopic": not about GIS, this project, or this tool at all (general trivia,
  weather, cooking, coding help unrelated to the project, chit-chat).

If unsure between "analysis" and "question", choose "analysis" — doing the work
is the safer failure.

Respond with ONLY a JSON object, no other text:
{"mode": "analysis"}
{"mode": "question", "answer": "<direct answer, grounded ONLY in the context; say plainly if the context does not contain it>"}
{"mode": "offtopic", "answer": "<one or two sentences declining, and say what this tool does do>"}

Answer in the same language the user wrote in."""


# ------------------------------------------------------------------ compaction
def log_path(pdir: str) -> str:
    return os.path.join(pdir, LOG_FILE)


def read_log(pdir: str) -> str:
    path = log_path(pdir)
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read()


def _trace_digest(steps: list, cap: int = 24) -> str:
    lines = []
    for s in steps[:cap]:
        thought = re.sub(r"\s+", " ", (s.get("thought") or "")).strip()[:220]
        lines.append(f"{s.get('round')}. {s.get('action')}: {thought}")
    if len(steps) > cap:
        lines.append(f"… {len(steps) - cap} further rounds omitted")
    return "\n".join(lines)


def compact_run(llm, pdir: str, project_name: str, entry: dict, steps: list) -> str:
    """Summarise one run and append it to LOG.md. Returns the digest text."""
    material = (
        f"Project: {project_name}\n"
        f"Asked: {entry.get('ask', '')}\n"
        f"Model: {entry.get('model_display', '')}\n"
        f"Outcome: {'success' if entry.get('success') else 'failed'} · "
        f"{entry.get('rounds', 0)} rounds · {entry.get('self_corrections', 0)} self-corrections\n"
        f"Files produced: {', '.join(entry.get('outputs') or []) or 'none'}\n\n"
        f"What the agent did, round by round:\n{_trace_digest(steps)}\n\n"
        f"Final summary the agent gave:\n{(entry.get('final_summary') or '(none)')[:2000]}\n"
    )
    res = llm.generate(prompt=material, system_prompt=COMPACT_SYSTEM,
                       user_message=material, max_tokens=700, stop=[])
    digest = (res.get("text") if isinstance(res, dict) else str(res)) or ""
    digest = digest.strip()
    if not digest or digest.startswith("Error"):
        return ""

    path = log_path(pdir)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                f"# {project_name} — running log\n\n"
                "One compacted entry per analysis run, written by the model from that\n"
                "run's full trace. This is the digest a later session reads; the verbatim\n"
                "record is in JOURNAL.md and runs/.\n"
            )
    when = (entry.get("ts") or "").replace("T", " ")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n---\n\n## {when} · {entry.get('run_id', '')}\n\n{digest}\n")
    return digest


def recent_log(pdir: str, n_entries: int = 6, char_cap: int = 4000) -> str:
    """The last few digests — what gets injected instead of the raw ask list."""
    text = read_log(pdir)
    if not text:
        return ""
    sections = text.split("\n---\n")
    tail = [s.strip() for s in sections[1:][-n_entries:] if s.strip()]
    out = "\n\n---\n\n".join(tail)
    if len(out) > char_cap:
        out = out[-char_cap:]
        out = out[out.find("## "):] if "## " in out else out
    return out.strip()


# ---------------------------------------------------------------- intent gate
def classify_request(llm, instruction: str, context: str) -> dict:
    """Decide whether this message needs an analysis run. Fails open to analysis."""
    material = (
        f"Context about the current project:\n{context or '(no project history yet)'}\n\n"
        f"User's message:\n{instruction}\n"
    )
    try:
        res = llm.generate(prompt=material, system_prompt=INTENT_SYSTEM,
                           user_message=material, max_tokens=1200, stop=[])
        text = (res.get("text") if isinstance(res, dict) else str(res)) or ""
    except Exception:
        return {"mode": "analysis"}
    if text.startswith("Error"):
        return {"mode": "analysis"}

    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return {"mode": "analysis"}
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"mode": "analysis"}
    mode = str(data.get("mode", "analysis")).lower()
    if mode not in ("analysis", "question", "offtopic"):
        return {"mode": "analysis"}
    if mode != "analysis" and not str(data.get("answer", "")).strip():
        return {"mode": "analysis"}      # no answer to give — just do the work
    return {"mode": mode, "answer": str(data.get("answer", "")).strip()}

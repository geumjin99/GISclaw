# Changelog

The paper's reference implementation stays on the `paper-v2` branch (tag `v2-gsis-submission`).

## Unreleased

- Stop ends the run: the current step is interrupted, the run is recorded as stopped.
- A run continues on the server through a page reload; the page rejoins it.
- One run at a time; a second request, or a Toolbox operation meanwhile, is refused.
- Code executed by the model is interrupted at its time limit instead of hanging the server.
- The interface is published to this machine only, and state-changing requests are accepted from this page only.
- Toolbox operations are recorded in the conversation and journal with their code.
- Cached prompt tokens are counted for OpenAI-compatible providers.
- The transcript sent each round is no longer cut at 16,000 characters; the limit follows the model.
- Windows: the project data link inside a run is a junction when a symlink is not permitted.
- Fixes: a replayed run showed its answer twice; a file name shared by data/ and outputs/ made one layer replace the other; a `.json` that is not GeoJSON opens as text.
- A test suite (`pytest -q tests`).
- Desktop install without Docker (beta): `install.sh` / `install.ps1`, a native window, data in the user data folder.

## 1.0.0 — 2026-09-01

First numbered release of the desktop application.

- Local web application: browser UI, FastAPI server, ReAct agent, persistent Python GIS sandbox; `docker compose up -d`.
- Projects with `data/`, `outputs/`, `runs/`; every run keeps `code.py`, `trace.jsonl`, `run.log`.
- Live streaming of reasoning, code and figures.
- 28 deterministic geoprocessing operators, from the agent or the Toolbox.
- OpenAI, Anthropic, DeepSeek, Gemini, OpenAI-compatible endpoints; local models via Ollama, LM Studio, vLLM.
- Prompt caching; per-project chat, journal and compacted log; global memory; skills as folder bundles.
- Map layers with symbology and attribute tables; table view; data import from any path; project rename, archive, export, delete.
- README and PDF manual in English, Chinese and Korean.
- AGPL-3.0-or-later, commercial licence available; `DISCLAIMER.md`.

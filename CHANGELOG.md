# Changelog

The paper's reference implementation stays on the `paper-v2` branch (tag `v2-gsis-submission`).

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

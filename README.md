# GISclaw

**A comprehensive open-source LLM agent system for realistic, multi-step
geospatial analysis.**

GISclaw runs end-to-end professional GIS pipelines---spatial overlays, raster
algebra, geostatistical interpolation, machine-learning classification, network
analysis, multi-layer cartography---directly through the open-source Python
geospatial stack (GeoPandas, rasterio, scipy, scikit-learn, libpysal, geoplot,
cartopy). It does **not** depend on ArcGIS, QGIS plugins, or any other
proprietary GIS layer.

The system pairs a persistent Jupyter-like Python sandbox with a
backend-agnostic LLM interface (cloud APIs and local 14--70B open-weight models
both supported), three engineered prompt rules (Schema Analysis, Package
Constraint, optional Domain Knowledge Injection), and an Error-Memory module
for self-correction. Two pluggable agent architectures ship in parallel: a
Single-Agent ReAct loop and a Dual-Agent Plan-Execute-Replan pipeline.

This repository is the open-source implementation behind the GISclaw paper
(preprint: <https://arxiv.org/abs/2603.26845>).

---

## Repository layout

```
GISclaw/
|-- src/
|   |-- agent/         # Core agent
|   |   |-- react_agent.py      # ReAct loop (Single-Agent)
|   |   |-- sandbox.py          # Persistent Jupyter-like Python sandbox
|   |   |-- tools.py            # Toolkit exposed to the agent
|   |   |-- llm_engine.py       # Backend-agnostic LLM interface
|   |   |-- error_memory.py     # Cross-task error -> fix memory
|   |   |-- prompts.py          # System-prompt builders
|   |   |-- orchestrator.py     # Dual-Agent Plan-Execute-Replan
|   |   |-- planner.py
|   |   `-- worker.py
|   `-- tools/         # Tool registry used by the Dual-Agent worker
|       |-- registry.py
|       |-- vector_tools.py
|       |-- raster_tools.py
|       |-- analysis_tools.py
|       |-- viz_tools.py
|       |-- conversion_tools.py
|       |-- terrain_tools.py
|       `-- advanced_tools.py
|-- scripts/
|   |-- run_single_task.py      # Single-Agent ReAct runner
|   `-- run_dual_agent_task.py  # Dual-Agent Plan-Execute-Replan runner
|-- evaluation/        # Three-layer evaluation protocol
|   |-- code_eval/eval_code_metrics_full.py     # L1: CodeBLEU + API F1
|   |-- output_eval/eval_output_accuracy.py     # L2: type-specific output check
|   `-- report_eval/eval_log_gpt_embedding.py   # L3: process embedding cosine
|-- configs/default.yaml
|-- requirements.txt
|-- LICENSE
`-- README.md
```

---

## Installation

GISclaw targets Python 3.10+ on Linux or macOS. We recommend a dedicated conda
environment.

```bash
git clone https://github.com/geumjin99/GISclaw.git
cd GISclaw
conda create -n gisclaw python=3.12 -y
conda activate gisclaw
pip install -r requirements.txt
```

API keys are read from the environment. Set the ones you need:

```bash
export OPENAI_API_KEY="..."        # GPT-4.1 / GPT-5.x
export ANTHROPIC_API_KEY="..."     # Claude
export DEEPSEEK_API_KEY="..."      # DeepSeek-V3.2
export GEMINI_API_KEY="..."        # Gemini-3-Flash / 3.1-Pro / 2.5-Pro
export TOGETHER_API_KEY="..."      # Llama-3.3-70B via Together AI
```

For a fully offline deployment, pull a local Qwen2.5-Coder model into Ollama:

```bash
ollama pull qwen2.5-coder:14b
```

No API key is required for the local Ollama backend.

---

## Defining and running a task

A task is a record with the following fields:

| Field                 | Required | Description                                                          |
| --------------------- | -------- | -------------------------------------------------------------------- |
| `id`                  | yes      | Numeric task identifier.                                             |
| `task`                | yes      | Short title of the analysis.                                         |
| `category`            | no       | Free-form category label.                                            |
| `instruction`         | yes      | Full natural-language analysis instruction.                          |
| `dataset_description` | no       | Short description of the input data layout and column meanings.      |
| `domain_knowledge`    | no       | Optional expert hints (calibrated parameters, non-obvious orderings).|
| `workflow`            | no       | Optional high-level analytical decomposition (free text).            |

Place each task's input data under `<task_root>/<task_id>/dataset/` and the
runner will symlink it into a per-task work directory. Outputs are written to
`results/<RESULTS_DIR_NAME>/T<NN>/`. The `results.json` per task is updated
incrementally so the runner can be re-invoked safely under `--skip-existing`.

Example invocations:

```bash
# Single-Agent ReAct loop
python3 scripts/run_single_task.py --model gpt-4.1   --task 1
python3 scripts/run_single_task.py --model deepseek  --task 1-10
python3 scripts/run_single_task.py --model qwen-14b  --task 5

# Dual-Agent Plan-Execute-Replan pipeline
python3 scripts/run_dual_agent_task.py --model deepseek --task 1
python3 scripts/run_dual_agent_task.py --model 14b      --task 1
```

For both runners, `--model` keys are defined in the `MODEL_CONFIGS` /
`MODEL_CONFIG` dict at the top of the script, and `--task` accepts a single
integer or an inclusive range.

Other useful flags:

| Flag | Effect | SA | DA |
|------|--------|----|----|
| `--task-timeout 1200` | Hard wall-clock cap (seconds) per task. | yes | yes |
| `--skip-existing` | Skip tasks that already succeeded under the same model label. | yes | yes |
| `--no-workflow` | Run without the optional high-level workflow hint (ablation). | yes | -- |

---

## Three-layer evaluation protocol

After running tasks, score them along three complementary dimensions:

```bash
# L1: code structure (CodeBLEU + API operation F1)
EVAL_SA_RESULTS_DIR=results/single_agent EVAL_CELL_LABEL=run1_SA \
  python3 evaluation/code_eval/eval_code_metrics_full.py

# L2: type-specific output verification (vision + raster/vector/tabular)
EVAL_SA_RESULTS_DIR=results/single_agent EVAL_CELL_LABEL=run1_SA \
  python3 evaluation/output_eval/eval_output_accuracy.py

# L3: reasoning-process embedding cosine
EVAL_SA_RESULTS_DIR=results/single_agent EVAL_CELL_LABEL=run1_SA \
  python3 evaluation/report_eval/eval_log_gpt_embedding.py
```

Each layer writes a per-cell CSV under `evaluation/L1/`, `evaluation/L2/`, and
`evaluation/L3_embed/` respectively.

---

## Citation

If you use GISclaw in academic work, please cite the preprint:

```bibtex
@article{gisclaw2026,
  title   = {GISclaw: A Comprehensive Open-Source LLM Agent System for
             Realistic Multi-Step Geospatial Analysis},
  author  = {Han, Jinzhen and Lee, JinByeong and Shim, Yuri and Kim, Jisung
             and Lee, Jae-Joon},
  journal = {arXiv preprint arXiv:2603.26845},
  year    = {2026},
  url     = {https://arxiv.org/abs/2603.26845}
}
```

---

## License

MIT. See `LICENSE`.

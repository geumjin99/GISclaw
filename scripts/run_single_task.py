#!/usr/bin/env python3
"""GISclaw single-agent runner.

Runs the GISclaw ReAct agent on one or more user-defined tasks. The task
index CSV and per-task input data are located under the directory pointed to
by the `GISCLAW_TASK_ROOT` environment variable (default: `tasks/`). Outputs
are written to `results/<RESULTS_DIR_NAME>/T<NN>/`, and `results.json` is
updated incrementally so the runner can be re-invoked safely.

Expected task layout (relative to GISCLAW_TASK_ROOT):
    tasks.csv                    # task index (id, level, task, instruction, ...)
    <task_id>/dataset/           # input data for that task
    <task_id>/pred_results/      # optional gold reference (copied if present)

Supported model keys (see MODEL_CONFIGS below for the full set):
  gpt-5.4       OpenAI flagship (uses max_completion_tokens, no stop)
  gpt-4.1       GPT-4 series, standard chat completions API
  deepseek      DeepSeek-V3.2 via OpenAI-compatible endpoint
  gemini-pro    Gemini 3.1 Pro via OpenAI-compatible endpoint
  gemini-flash  Gemini 3 Flash via OpenAI-compatible endpoint
  qwen-14b      Ollama-served local Qwen2.5-Coder:14B (no API key needed)
  llama-70b     Llama-3.3-70B via Together AI

Usage:
  python3 scripts/run_single_task.py --model gpt-4.1 --task 1-10
  python3 scripts/run_single_task.py --model qwen-14b --task 5
  python3 scripts/run_single_task.py --model deepseek --task 1 --no-workflow
"""
import os, sys, json, time, csv, shutil, argparse, signal

# Force stdout to line-buffered mode
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# === API keys (loaded from environment) ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")

# === Model configurations ===
MODEL_CONFIGS = {
    "gpt-5.4": {
        "engine": "openai",
        "model_name": "gpt-5.4",
        "label": "gpt54",
        "display": "GPT-5.4",
        "timeout": 300,
        "max_rounds": 50,
        "max_tokens": 4096,
        "cost_per_m": (2.0, 8.0),
    },
    "gpt-4.1": {
        "engine": "openai",
        "model_name": "gpt-4.1",
        "label": "gpt41",
        "display": "GPT-4.1",
        "timeout": 180,
        "max_rounds": 35,
        "max_tokens": 2048,
        "cost_per_m": (2.0, 8.0),
    },
    "qwen-14b": {
        "engine": "ollama",
        "model_name": "qwen2.5-coder:14b",
        "label": "qwen14b",
        "display": "Qwen2.5-Coder:14B",
        "timeout": 180,
        "max_rounds": 35,
        "max_tokens": 2048,
    },
    "deepseek": {
        "engine": "deepseek",
        "model_name": "deepseek-chat",
        "label": "deepseek",
        "display": "DeepSeek-V3.2",
        "timeout": 180,
        "max_rounds": 35,
        "max_tokens": 2048,
        "cost_per_m": (0.28, 0.42),
    },
    "gemini-pro": {
        "engine": "gemini",
        "model_name": "gemini-3.1-pro-preview",
        "label": "gemini_pro",
        "display": "Gemini-3.1-Pro",
        "timeout": 300,
        "max_rounds": 50,
        "max_tokens": 4096,
        "cost_per_m": (1.25, 10.0),  # Gemini 3.1 Pro pricing
    },
    "gemini-flash": {
        "engine": "gemini",
        "model_name": "gemini-3-flash-preview",
        "label": "gemini_flash",
        "display": "Gemini-3-Flash",
        "timeout": 180,
        "max_rounds": 35,
        "max_tokens": 2048,
        "cost_per_m": (0.15, 0.60),  # Gemini 3 Flash pricing
    },
    "gemini-2.5-pro": {
        "engine": "gemini",
        "model_name": "gemini-2.5-pro",
        "label": "gemini_25pro",
        "display": "Gemini-2.5-Pro",
        "timeout": 300,
        "max_rounds": 50,
        "max_tokens": 4096,
        "cost_per_m": (1.25, 10.0),  # Gemini 2.5 Pro pricing
    },
    "llama-70b": {
        "engine": "together",
        "model_name": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "label": "llama70b",
        "display": "Llama-3.3-70B",
        "timeout": 300,
        "max_rounds": 50,
        "max_tokens": 4096,
        "cost_per_m": (0.88, 0.88),  # Together AI serverless pricing
    },
}

RESULTS_DIR_NAME = os.getenv("RESULTS_DIR_NAME", "single_agent")

# Root directory holding task definitions and per-task input data.
# Default layout (relative to TASK_DATA_ROOT):
#     <task_root>/tasks.csv
#     <task_root>/<task_id>/dataset/      (input data)
#     <task_root>/<task_id>/pred_results/ (optional gold reference)
TASK_DATA_ROOT = os.getenv("GISCLAW_TASK_ROOT", os.path.join(PROJECT_ROOT, "tasks"))
TASK_INDEX_CSV = os.getenv("GISCLAW_TASK_INDEX",
                           os.path.join(TASK_DATA_ROOT, "tasks.csv"))

# ============================================================
# System prompt for the ReAct loop
# ============================================================

V4_SYSTEM_PROMPT = """You are an expert GIS analyst agent. You solve complex geospatial analysis tasks by thinking step-by-step, planning your approach, and using tools to interact with data.

## Available Tools

{tool_descriptions}

## Response Format

You MUST respond in this EXACT format every time:

Thought: <your reasoning about what to do next, based on observations so far>
Action: <tool_name>
Args: <arguments as JSON object>

Examples:
Thought: I need to see what data files are available.
Action: list_files
Args: {{}}

Thought: Let me load the shapefile.
Action: load_vector
Args: {{"path": "dataset/stations.geojson", "var_name": "stations"}}

Thought: Now I need to buffer and overlay.
Action: execute
Args: {{"code": "buffered = stations.buffer(1000)\\nprint(f'Buffered: {{len(buffered)}} features')"}}

Thought: Analysis complete, results saved.
Action: finish
Args: {{"summary": "Analysis complete. Results saved to pred_results/"}}

## Core Analysis Rules

### Planning (think BEFORE coding)
1. **Start by listing files and loading data** — always call list_files first, then load and inspect ALL datasets (print columns, dtypes, head(2), CRS, total_bounds).
2. **Plan your approach mentally** — in your first Thought after data inspection, outline all steps you will take (like a Planner would). This helps you stay organized.
3. **Execute step by step** — write 3-15 lines of code per execute() call, print intermediate results to verify.

### GIS Best Practices
4. **CRS first** — any operation involving distance or area (buffer, spatial join, overlay) MUST use a projected CRS (meters). Always reproject with `.to_crs()` before such operations.
5. **Overlay, not clip** — use `gpd.overlay(how='difference')` for set difference. Do NOT use clip when difference is intended.
6. **Schema-driven** — do NOT hardcode column names. After loading data, read actual column names from output and use those. If the task mentions a concept (e.g., "poverty rate"), find the matching column from the data.
7. **Visualization** — `plt.savefig()` MUST be in the SAME execute() call as the plot creation. NEVER use plt.show(). Always: `plt.savefig('pred_results/xxx.png', dpi=150, bbox_inches='tight'); plt.close()`
8. **Overlay visualization** — when overlaying multiple factors on one map, plot them on the SAME axes using the `ax` parameter with transparent colors (alpha), NOT as separate subplots.

### Output Rules
9. **Save all outputs to pred_results/** before calling finish().
10. **If the instruction requests a specific file format** (shapefile, CSV, GeoTIFF), you MUST save in that exact format. Do NOT substitute a PNG for a data file.
11. After saving, verify the file exists: `print(os.listdir('pred_results/'))`

## Available Packages (ONLY use these)
geopandas, rasterio, shapely, fiona, pyproj, numpy, pandas, scipy, matplotlib,
sklearn, libpysal, esda, mgwr, xarray, rasterstats, networkx, osmnx,
seaborn, mapclassify, h3, momepy, pointpats, spaghetti, openpyxl, rtree,
geoplot, cartopy, imblearn

NOT available (do NOT import): pykrige, skimage, arcpy
Alternatives:
- pykrige -> scipy.interpolate.griddata or scipy.interpolate.Rbf
- skimage -> scipy.ndimage or numpy
- arcpy -> geopandas + rasterio + shapely (translate ArcPy tools to open-source equivalents)
"""

# ============================================================
# Utility helpers
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="GISclaw single-agent runner")
    p.add_argument("--model", required=True,
                   choices=list(MODEL_CONFIGS.keys()),
                   help="LLM backend (e.g. gpt-5.4, gpt-4.1, deepseek, qwen-14b)")
    p.add_argument("--task", required=True, help="task id (e.g. 1 or 1-10 or 1,2,3)")
    p.add_argument("--task-timeout", type=int, default=0,
                   help="per-task wall-clock timeout in seconds; 0 disables (default 0)")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip task if results.json already has a non-timeout entry for this model")
    p.add_argument("--no-workflow", action="store_true",
                   help="ablation: blank out the workflow field before passing to agent")
    return p.parse_args()

class TaskTimeoutError(Exception):
    pass

def _timeout_signal_handler(signum, frame):
    raise TaskTimeoutError("task wall-clock budget exceeded")

def _write_timeout_fail(task_id, cfg, timeout_s):
    """Append a fail-by-timeout entry to results.json without disturbing other models."""
    label = cfg["label"]
    output_dir = os.path.join(PROJECT_ROOT, "results", RESULTS_DIR_NAME, f"T{task_id:02d}")
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "results.json")
    existing = {}
    if os.path.exists(results_path):
        try:
            with open(results_path) as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    existing[label] = {
        "task_id": task_id,
        "model": cfg["display"],
        "model_name": cfg["model_name"],
        "architecture": "single_agent_react",
        "max_rounds": cfg["max_rounds"],
        "rounds_used": 0,
        "self_corrections": 0,
        "success": False,
        "output_files": [],
        "elapsed_s": float(timeout_s),
        "cost": {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fail_reason": f"timeout_>{timeout_s}s",
    }
    with open(results_path, "w") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    # Best-effort cleanup of any stray work_dir to avoid leaking GBs across re-runs
    work_dir = os.path.join(output_dir, f"work_{label}")
    shutil.rmtree(work_dir, ignore_errors=True)

def load_task(csv_path, task_id):
    """Load the full record for a single task"""
    with open(csv_path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if int(row['id']) == task_id:
                return {
                    'task': row.get('Task', '')[:80],
                    'instruction': row.get('Instruction', ''),
                    'workflow': row.get('Human Designed Workflow', '') or row.get('Workflow', '') or '',
                    'dataset_description': row.get('Dataset Description', '') or '',
                    'domain_knowledge': row.get('Domain Knowledge', '') or '',
                    'category': row.get('Task Categories1', ''),
                }
    return None

def init_llm(cfg):
    """ engine Initialize LLM"""
    if cfg["engine"] == "openai":
        from src.agent.llm_engine import OpenAIEngine
        llm = OpenAIEngine(
            model=cfg["model_name"],
            api_key=OPENAI_API_KEY,
            temperature=0.1,
            max_tokens=cfg["max_tokens"],
            cost_per_m=cfg.get("cost_per_m", (2.5, 10.0)),
        )
    elif cfg["engine"] == "deepseek":
        from src.agent.llm_engine import OpenAIEngine
        if not DEEPSEEK_API_KEY:
            raise ValueError("DEEPSEEK_API_KEY environment variable is not set")
        llm = OpenAIEngine(
            model=cfg["model_name"],
            api_key=DEEPSEEK_API_KEY,
            temperature=0.1,
            max_tokens=cfg["max_tokens"],
            base_url="https://api.deepseek.com",
            cost_per_m=cfg.get("cost_per_m", (0.28, 0.42)),
        )
    elif cfg["engine"] == "claude":
        from src.agent.llm_engine import ClaudeEngine
        llm = ClaudeEngine(
            model=cfg["model_name"],
            api_key=CLAUDE_API_KEY,
            temperature=0.1,
            max_tokens=cfg["max_tokens"],
            cost_per_m=cfg.get("cost_per_m", (3.0, 15.0)),
        )
    elif cfg["engine"] == "gemini":
        # Gemini OpenAI API 
        from src.agent.llm_engine import OpenAIEngine
        llm = OpenAIEngine(
            model=cfg["model_name"],
            api_key=GEMINI_API_KEY,
            temperature=0.1,
            max_tokens=cfg["max_tokens"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            cost_per_m=cfg.get("cost_per_m", (0.15, 0.60)),
        )
    elif cfg["engine"] == "together":
        # Together AI OpenAI API 
        from src.agent.llm_engine import OpenAIEngine
        llm = OpenAIEngine(
            model=cfg["model_name"],
            api_key=TOGETHER_API_KEY,
            temperature=0.1,
            max_tokens=cfg["max_tokens"],
            base_url="https://api.together.xyz/v1",
            cost_per_m=cfg.get("cost_per_m", (0.88, 0.88)),
        )
    elif cfg["engine"] == "ollama":
        from src.agent.llm_engine import OllamaEngine
        llm = OllamaEngine(
            model=cfg["model_name"],
            temperature=0.1,
            max_tokens=cfg["max_tokens"],
        )
    else:
        raise ValueError(f"Unknown engine: {cfg['engine']}")
    llm.load_model()
    return llm

# ============================================================
# GIS 
# ============================================================

REPORT_SYSTEM_PROMPT = """You are a GIS analyst writing a data-driven technical analysis report.
Write in Markdown format. Be professional, factual, and QUANTITATIVE.

CRITICAL RULES:
- You MUST cite EXACT numbers from the Execution Log. Do NOT paraphrase or round.
- Copy-paste specific values: bounds, CRS codes, feature counts, column names, statistics.
- Use bold (**value**) for all numeric values cited from the log.
- List output files as bullet points with exact filenames.

Keep the report between 300-600 words. Structure it as:

# [Task Title]

## Objective
Briefly describe what was analyzed and why.

## Data
For EACH dataset used, list:
- Dataset name/variable (e.g. `bus_gdf`)
- Valid geometry: **True/False**
- Bounds: **[exact values from log]**
- Feature count: **N**
- CRS: **EPSG:XXXX**
- Key columns and their statistics (mean, median, 75th percentile, max) as reported in log

## Methodology
Describe the GIS operations performed, referencing specific function calls and parameters.
Include exact threshold values, formulas, and quantile cutoffs used.

## Results
Summarize key findings with EXACT numbers from the log.
List ALL output files produced:
- `pred_results/filename.ext`

## Conclusion
One or two sentences on the significance of the results.
"""

def generate_report(llm, task_info, code, output_files, task_id, execution_log=""):
    """After success, call the LLM to generate a GIS analysis report from the execution log"""
    # Keep only the most essential information
    log_text = execution_log[-4000:] if execution_log else "(no execution log available)"
    code_text = code[-2000:] if code else "(no code available)"

    user_msg = (
        f"Write a concise GIS analysis report for the following completed task.\n\n"
        f"## Task Instruction\n{task_info['instruction']}\n\n"
        f"## Execution Log (stdout from actual run)\n```\n{log_text}\n```\n\n"
        f"## Code Executed\n```python\n{code_text}\n```\n\n"
        f"## Output Files Produced\n{', '.join(output_files)}\n\n"
        f"Write the report now. Remember: cite ONLY numbers from the Execution Log above."
    )
    try:
        resp = llm.generate(
            prompt="",
            system_prompt=REPORT_SYSTEM_PROMPT,
            user_message=user_msg,
            max_tokens=4096,
            stop=None,
        )
        text = resp.get("text", "") if isinstance(resp, dict) else str(resp)
        return text.strip()
    except Exception as e:
        print(f" [warn] failure: {e}")
        return ""

# ============================================================
# ============================================================

def run_task(task_id, cfg):
    """Single-agent main loop"""
    os.chdir(PROJECT_ROOT)

    label = cfg["label"]

    # Load tasks
    task = load_task(TASK_INDEX_CSV, task_id)
    if not task:
        print(f"  Task {task_id} not found in {TASK_INDEX_CSV}")
        return None

    # Directory setup
    output_dir = os.path.join(PROJECT_ROOT, "results", RESULTS_DIR_NAME, f"T{task_id:02d}")
    os.makedirs(output_dir, exist_ok=True)
    work_dir = os.path.join(output_dir, f"work_{label}")
    # Clean the previous work_dir
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)
    pred_dir = os.path.join(work_dir, "pred_results")
    os.makedirs(pred_dir, exist_ok=True)

    data_dir = os.path.join(TASK_DATA_ROOT, str(task_id), "dataset")
    ds_link = os.path.join(work_dir, "dataset")
    if os.path.islink(ds_link):
        os.unlink(ds_link)
    if not os.path.exists(ds_link):
        os.symlink(os.path.abspath(data_dir), ds_link)

    dataset_desc = task['dataset_description'] or ""
    try:
        data_files = sorted(os.listdir(data_dir))
        file_list = "\n".join(f"  - {f}" for f in data_files if not f.startswith('.'))
        dataset_desc += f"\n\n## Available files in dataset/:\n{file_list}"
    except:
        pass

    print(f"\n{'='*65}")
    print(f"  T{task_id:02d} [{time.strftime('%Y-%m-%d %H:%M:%S')}]: {task['task']}")
    print(f"  Category : {task['category']}")
    print(f"  Model    : {cfg['display']}")
    print(f"  Output   : {output_dir}")
    print(f"{'='*65}")

    # Initialize LLM
    llm = init_llm(cfg)
    print(f"  [ok] {cfg['display']} ready")

    # Initialize agent
    from src.agent.error_memory import ErrorMemory
    from src.agent.react_agent import GISReActAgent

    agent = GISReActAgent(
        llm_engine=llm,
        timeout=cfg["timeout"],
        max_rounds=cfg["max_rounds"],
        verbose=True,
        rag=None,
        error_memory=ErrorMemory(),
    )

    # System Prompt
    from src.agent.tools import GISToolkit
    from src.agent.sandbox import PythonSandbox

    _temp_sandbox = PythonSandbox(work_dir=work_dir, timeout=cfg["timeout"])
    _temp_toolkit = GISToolkit(_temp_sandbox, data_dir="dataset")
    v4_system_prompt = V4_SYSTEM_PROMPT.format(
        tool_descriptions=_temp_toolkit.tool_descriptions,
    )
    del _temp_sandbox, _temp_toolkit

    import src.agent.prompts as prompts_module
    _original_build = prompts_module.build_system_prompt
    prompts_module.build_system_prompt = lambda **kwargs: v4_system_prompt

    # === Execute the agent loop ===
    print(f"\n  Running agent (max {cfg['max_rounds']} rounds)...")
    t0 = time.time()

    workflow_str = "" if globals().get("NO_WORKFLOW", False) else task['workflow']
    result = agent.run(
        task_id=task_id + 3000,
        instruction=task['instruction'],
        workflow=workflow_str,
        data_dir=data_dir,
        work_dir=work_dir,
        domain_knowledge=task['domain_knowledge'],
        dataset_description=dataset_desc,
        rag_context="",
        skill_text="",
    )

    elapsed = time.time() - t0
    exec_result = result.to_dict()

    # Restore
    prompts_module.build_system_prompt = _original_build

    # === ===
    result_exts = ('.png', '.csv', '.geojson', '.tif', '.shp', '.dbf',
                   '.prj', '.shx', '.cpg', '.json')
    for f in os.listdir(work_dir):
        if f.endswith(result_exts) and not f.startswith('.') and not f.startswith('_react'):
            src = os.path.join(work_dir, f)
            dst = os.path.join(pred_dir, f)
            if os.path.isfile(src) and not os.path.exists(dst):
                shutil.move(src, dst)

    output_files = sorted([
        f for f in os.listdir(pred_dir) if not f.startswith('.')
    ]) if os.path.exists(pred_dir) else []

    model_dir = os.path.join(output_dir, cfg["model_name"])
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    if os.path.exists(pred_dir):
        shutil.copytree(pred_dir, model_dir)

    # Save the merged code
    code_src = os.path.join(work_dir, f"_react_t{task_id + 3000}.py")
    if os.path.exists(code_src):
        shutil.copy2(code_src, os.path.join(model_dir, "code.py"))

    # Optional gold reference (copy if present)
    gold_dir = os.path.join(TASK_DATA_ROOT, str(task_id), "pred_results")
    if os.path.exists(gold_dir):
        dest = os.path.join(output_dir, "gold")
        if not os.path.exists(dest):
            shutil.copytree(gold_dir, dest)

    # === Cost / token statistics ===
    cost_info = {}
    if hasattr(llm, 'get_stats'):
        stats = llm.get_stats()
        cost_info = {
            "total_api_calls": stats.get("total_calls", 0),
            "total_input_tokens": stats.get("total_input_tokens", 0),
            "total_output_tokens": stats.get("total_output_tokens", 0),
            "total_tokens": stats.get("total_tokens", 0),
            "estimated_cost_usd": stats.get("estimated_cost_usd", 0),
            "avg_latency_ms": round(stats.get("avg_latency_ms", 0), 1),
        }

    emergency_names = {'emergency_plot.png', 'emergency_output.geojson', 'emergency_output.csv'}
    real_outputs = [f for f in output_files if f not in emergency_names and f != 'code.py']

    result_entry = {
        "task_id": task_id,
        "model": cfg["display"],
        "model_name": cfg["model_name"],
        "architecture": "single_agent_react",
        "max_rounds": cfg["max_rounds"],
        "rounds_used": exec_result.get("total_rounds", 0),
        "self_corrections": exec_result.get("self_corrections", 0),
        "success": len(real_outputs) > 0,
        "output_files": output_files,
        "elapsed_s": round(elapsed, 1),
        "cost": cost_info,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Save results.json (multiple models share the file)
    results_path = os.path.join(output_dir, "results.json")
    existing = {}
    if os.path.exists(results_path):
        try:
            with open(results_path) as f:
                existing = json.load(f)
        except:
            existing = {}
    existing[label] = result_entry
    with open(results_path, "w") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    status = "[ok]" if real_outputs else "[fail]"
    parts = [
        f"T{task_id:02d}",
        f"{exec_result.get('total_rounds', 0)} rounds",
        f"{len(real_outputs)} files",
        f"{elapsed:.1f}s",
    ]
    if exec_result.get('self_corrections', 0):
        parts.append(f"SC={exec_result['self_corrections']}")
    if cost_info.get('estimated_cost_usd'):
        parts.append(f"${cost_info['estimated_cost_usd']:.4f}")
    if cost_info.get('total_input_tokens'):
        parts.append(f"in={cost_info['total_input_tokens']:,} out={cost_info.get('total_output_tokens', 0):,}")
    print(f"\n  {status} " + " | ".join(parts))
    print(f"  Outputs: {output_files}")

    # === agent history observation===
    exec_log_lines = []
    if hasattr(result, 'history') and result.history:
        for rd in result.history:
            if isinstance(rd, dict) and rd.get('observation'):
                exec_log_lines.append(f"[Round {rd.get('round','')}] {rd['observation'][:500]}")
    # result.to_dict() 
    if not exec_log_lines and exec_result.get('history'):
        for rd in exec_result['history']:
            if isinstance(rd, dict) and rd.get('observation'):
                exec_log_lines.append(f"[Round {rd.get('round','')}] {rd['observation'][:500]}")
    execution_log = "\n".join(exec_log_lines)

    # Save the execution log
    if execution_log:
        log_path = os.path.join(model_dir, "execution_log.txt")
        with open(log_path, "w") as f:
            f.write(execution_log)

    # === Generate the GIS analysis report on success ===
    if len(real_outputs) > 0:
        print(f"  Generating analysis report ...")
        code_text = ""
        code_path = os.path.join(model_dir, "code.py")
        if os.path.exists(code_path):
            with open(code_path) as f:
                code_text = f.read()
        report = generate_report(llm, task, code_text, real_outputs, task_id,
                                 execution_log=execution_log)
        if report:
            report_path = os.path.join(model_dir, "report.md")
            with open(report_path, "w") as f:
                f.write(report)
            print(f"  Report written to: {report_path}")
            result_entry["has_report"] = True
        else:
            result_entry["has_report"] = False
        # Update results.json
        existing[label] = result_entry
        with open(results_path, "w") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

    # Clean work_dir
    shutil.rmtree(work_dir, ignore_errors=True)

    return result_entry

def main():
    args = parse_args()
    cfg = MODEL_CONFIGS[args.model]
    # propagate ablation flag to module scope so run_task() can read it
    globals()["NO_WORKFLOW"] = bool(args.no_workflow)

    # Parse task IDs
    task_str = args.task
    if "-" in task_str and "," not in task_str:
        start, end = task_str.split("-")
        task_ids = list(range(int(start), int(end) + 1))
    else:
        task_ids = [int(t.strip()) for t in task_str.split(",")]

    task_timeout = max(0, int(args.task_timeout or 0))
    if task_timeout > 0:
        signal.signal(signal.SIGALRM, _timeout_signal_handler)

    print(f"\n{'='*65}")
    print(f"  GISclaw Single-Agent runner")
    print(f"  Tasks            : {task_ids}")
    print(f"  Model            : {cfg['display']} ({cfg['model_name']})")
    print(f"  Max rounds       : {cfg['max_rounds']}")
    if task_timeout:
        print(f"  Timeout          : {cfg['timeout']}s per execute, {task_timeout}s per task")
    else:
        print(f"  Timeout          : {cfg['timeout']}s per execute")
    print(f"  Results dir name : {RESULTS_DIR_NAME}")
    print(f"{'='*65}")

    results = []
    for tid in task_ids:
        if args.skip_existing:
            rp = os.path.join(PROJECT_ROOT, "results", RESULTS_DIR_NAME, f"T{tid:02d}", "results.json")
            if os.path.exists(rp):
                try:
                    with open(rp) as f:
                        existing = json.load(f)
                    prev = existing.get(cfg["label"])
                    if prev and prev.get("success") and "timeout" not in (prev.get("fail_reason") or ""):
                        print(f"\n  Skipping T{tid:02d}: prior successful run exists (--skip-existing)")
                        continue
                except Exception:
                    pass
        if task_timeout > 0:
            signal.alarm(task_timeout)
        try:
            r = run_task(tid, cfg)
            if r:
                results.append(r)
        except TaskTimeoutError:
            print(f"\n [timeout] T{tid:02d} TIMEOUT (>{task_timeout}s) — fail ")
            try:
                _write_timeout_fail(tid, cfg, task_timeout)
            except Exception as we:
                print(f" (warn: failure: {we})")
        except Exception as e:
            print(f"\n  [fail] T{tid:02d} crashed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if task_timeout > 0:
                signal.alarm(0)

    # Aggregate
    if results:
        ok = sum(1 for r in results if r["success"])
        total_cost = sum(r.get("cost", {}).get("estimated_cost_usd", 0) for r in results)
        total_input = sum(r.get("cost", {}).get("total_input_tokens", 0) for r in results)
        total_output = sum(r.get("cost", {}).get("total_output_tokens", 0) for r in results)
        total_time = sum(r.get("elapsed_s", 0) for r in results)
        avg_rounds = sum(r.get("rounds_used", 0) for r in results) / len(results)

        print(f"\n{'='*65}")
        print(f"  {cfg['display']} aggregate ({len(results)} tasks)")
        print(f"  Success rate    : {ok}/{len(results)} ({ok/len(results)*100:.0f}%)")
        print(f"  Average rounds  : {avg_rounds:.1f}")
        print(f"  Wall time       : {total_time:.0f}s ({total_time/len(results):.0f}s/task)")
        print(f"  Tokens          : input={total_input:,} output={total_output:,}")
        print(f"  Estimated cost  : ${total_cost:.4f}")
        print(f"{'='*65}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""GISclaw dual-agent runner.

Implements the Plan-Execute-Replan pipeline. A Planner decomposes the task
into 3-7 ordered analytical steps, a Worker executes each step inside the
shared persistent Python sandbox (with up to 10 self-correction rounds per
step), and the Planner is re-invoked as a Replanner if a step fails. Both
roles share the same underlying LLM and sandbox namespace so that variables
and intermediate results persist across the pipeline.

Tasks are read from the task index pointed to by `GISCLAW_TASK_INDEX`
(default: `<GISCLAW_TASK_ROOT>/tasks.csv`); per-task input data is expected
under `<GISCLAW_TASK_ROOT>/<task_id>/dataset/`. Outputs are written to
`results/<RESULTS_DIR_NAME>/T<NN>/` (default: `dual_agent`).

Usage:
  python3 scripts/run_dual_agent_task.py --model deepseek --task 1
  python3 scripts/run_dual_agent_task.py --model gpt-4.1 --task 1-10
"""
import os, sys, json, time, csv, shutil, argparse, hashlib, re, io, signal

# Force stdout to line-buffered mode (block buffering hides progress under nohup) 
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

RESULTS_DIR_NAME = os.getenv("RESULTS_DIR_NAME", "dual_agent")

# === API keys (loaded from environment) ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")

MODEL_CONFIG = {
    "gpt4o": {
        "engine": "openai", "model_name": "gpt-4o-mini",
        "planner_engine": "openai", "planner_model": "gpt-4o-mini",
        "label": "gpt4o", "display": "GPT-4o-mini (dual)",
        "timeout": 180, "step_max_rounds": 8,
    },
    "gpt-5.4": {
        "engine": "openai", "model_name": "gpt-5.4",
        "planner_engine": "openai", "planner_model": "gpt-5.4",
        "label": "gpt-5.4", "display": "GPT-5.4 (dual)",
        "timeout": 300, "step_max_rounds": 8,
    },
    "gpt-4.1": {
        "engine": "openai", "model_name": "gpt-4.1",
        "planner_engine": "openai", "planner_model": "gpt-4.1",
        "label": "gpt-4.1", "display": "GPT-4.1 (dual)",
        "timeout": 300, "step_max_rounds": 8,
    },
    "deepseek": {
        "engine": "deepseek", "model_name": "deepseek-chat",
        "planner_engine": "deepseek", "planner_model": "deepseek-chat",
        "label": "deepseek-chat", "display": "DeepSeek-V3.2",
        "timeout": 300, "step_max_rounds": 8,
    },
    "gemini-flash": {
        "engine": "gemini", "model_name": "gemini-3-flash-preview",
        "planner_engine": "gemini", "planner_model": "gemini-3-flash-preview",
        "label": "gemini-3-flash-preview", "display": "Gemini-3-Flash (dual)",
        "timeout": 300, "step_max_rounds": 8,
    },
    "llama-70b": {
        "engine": "together", "model_name": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "planner_engine": "together", "planner_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "label": "llama-3.3-70b", "display": "Llama-3.3-70B (dual)",
        "timeout": 300, "step_max_rounds": 8,
    },
    "14b": {
        "engine": "ollama", "model_name": "qwen2.5-coder:14b",
        "planner_engine": "ollama", "planner_model": "qwen2.5-coder:14b",
        "label": "14b", "display": "Qwen2.5-Coder:14B (dual instance)",
        "timeout": 120, "step_max_rounds": 6,
    },
}

# ============================================================
# Utility helpers
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, help="task id (e.g. 1 or 1,2,3)")
    p.add_argument("--model", required=True, choices=list(MODEL_CONFIG.keys()))
    p.add_argument("--task-timeout", type=int, default=0,
                   help="per-task wall-clock timeout in seconds; 0 disables (default 0)")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip task if results.json already has a non-timeout entry for this model")
    return p.parse_args()

class TaskTimeoutError(Exception):
    pass

def _timeout_signal_handler(signum, frame):
    raise TaskTimeoutError("task wall-clock budget exceeded")

def _write_timeout_fail(task_id, args, timeout_s):
    """Append a fail-by-timeout entry to results.json without disturbing other models."""
    cfg = MODEL_CONFIG[args.model]
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
        "model": label,
        "model_name": cfg["model_name"],
        "architecture": "dual_agent_plan_execute_replan",
        "plan_steps": 0,
        "total_rounds": 0,
        "self_corrections": 0,
        "elapsed": float(timeout_s),
        "output_files": [],
        "success": False,
        "step_results": [],
        "plan": [],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fail_reason": f"timeout_>{timeout_s}s",
    }
    with open(results_path, "w") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    work_dir = os.path.join(output_dir, f"work_{label}")
    shutil.rmtree(work_dir, ignore_errors=True)

def load_task(csv_path, task_id):
    """Load the full record for a single task"""
    with open(csv_path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if int(row['id']) == task_id:
                return {
                    'instruction': row.get('Instruction', ''),
                    'workflow': row.get('Workflow', ''),
                    'dataset_description': row.get('Dataset Description', '') or '',
                    'domain_knowledge': row.get('Domain Knowledge', '') or '',
                }
    return None

def init_llm(model_key=None, engine=None, model_name=None):
    """Initialize LLM. model_key engine+model_name"""
    if model_key:
        cfg = MODEL_CONFIG[model_key]
        engine = cfg["engine"]
        model_name = cfg["model_name"]
    if engine == "openai":
        from src.agent.llm_engine import OpenAIEngine
        llm = OpenAIEngine(model=model_name, api_key=OPENAI_API_KEY)
    elif engine == "deepseek":
        from src.agent.llm_engine import OpenAIEngine
        llm = OpenAIEngine(model=model_name, api_key=DEEPSEEK_API_KEY,
                          base_url="https://api.deepseek.com",
                          cost_per_m=(0.28, 0.42))
    elif engine == "gemini":
        from src.agent.llm_engine import OpenAIEngine
        llm = OpenAIEngine(model=model_name, api_key=GEMINI_API_KEY,
                          base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                          cost_per_m=(0.15, 0.60))
    elif engine == "together":
        from src.agent.llm_engine import OpenAIEngine
        llm = OpenAIEngine(model=model_name, api_key=TOGETHER_API_KEY,
                          base_url="https://api.together.xyz/v1",
                          cost_per_m=(0.88, 0.88))
    elif engine == "ollama":
        from src.agent.llm_engine import OllamaEngine
        llm = OllamaEngine(model=model_name)
    else:
        raise ValueError(f"Unknown engine: {engine}")
    llm.load_model()
    return llm

# RAG — GIS Agent 

# ============================================================
# Planner verify() invoked between steps
# ============================================================

VERIFY_PROMPT = """You are a GIS analysis verification expert. Evaluate whether a Worker completed a step correctly.

You will receive:
- The original task description
- The step that was supposed to be executed  
- The Worker's output (stdout from code execution)

Evaluate:
1. Did the Worker complete the intended operation?
2. Are output values reasonable (no NaN explosions, no wrong CRS, no empty results)?
3. Did it produce the expected output variables/files?

Respond STRICTLY in JSON:
```json
{"pass": true, "reason": "Successfully loaded and reprojected data", "fix_hint": ""}
```
If it failed:
```json
{"pass": false, "reason": "Buffer distance seems wrong - 1m instead of 1000m", "fix_hint": "Check units, CRS should be projected (meters), use buffer(1000)"}
```
"""

def verify_step(llm, step, worker_stdout, original_instruction):
    """Planner Worker """
    user_msg = (
        f"## Original Task\n{original_instruction[:500]}\n\n"
        f"## Step {step.get('id','?')}: {step.get('action','')}\n"
        f"Instruction: {step.get('instruction','')}\n"
        f"Expected output: {step.get('output','')}\n\n"
        f"## Worker Output\n{worker_stdout[:2000]}\n\n"
        f"Evaluate this step. Respond in JSON."
    )
    response = llm.generate(
        prompt="", system_prompt=VERIFY_PROMPT,
        user_message=user_msg, max_tokens=512,
    )
    raw = response.get("text", "") if isinstance(response, dict) else str(response)
    
    # Strip Qwen3 <think> tags
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
    
    # JSON
    json_match = re.search(r'\{[^{}]*"pass"[^{}]*\}', raw, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except:
            pass
    # failure
    return {"pass": True, "reason": "verification parse failed, defaulting to pass", "fix_hint": ""}

# ============================================================
# Worker.run_step() executes within the shared sandbox
# ============================================================

def run_step_in_sandbox(llm, step, sandbox, toolkit, system_prompt,
                        max_rounds=8, verbose=True, step_history=None, task_start_time=None):
    """
     sandbox Planner 
    
    :
        (success: bool, stdout_log: str, code_accumulated: str, rounds_used: int, self_corrections: int)
    """
    from src.agent.react_agent import _parse_action
    
    # User Message
    step_instruction = (
        f"Execute Step {step.get('id','?')}: {step.get('action','')}\n"
        f"Instruction: {step.get('instruction','')}\n"
        f"Expected output: {step.get('output','')}\n\n"
        f"IMPORTANT: After completing this step, call finish() to indicate step completion.\n"
        f"Do NOT proceed to other steps — only complete the current step."
    )
    if step_history:
        step_instruction += f"\n\n## Context from previous steps\n{step_history}"
    
    history = []
    stdout_log = []
    code_accumulated = []
    self_corrections = 0
    last_code_hash = None
    
    TASK_TIMEOUT_S = 600
    for r in range(1, max_rounds + 1):
        # round 
        if task_start_time and (time.time() - task_start_time) > TASK_TIMEOUT_S:
            if verbose:
                print(f" [{r}] [timeout] ")
            return False, "\n".join(stdout_log), "\n".join(code_accumulated), r, self_corrections
        conv_parts = [f"User: {step_instruction}"]
        for role, content in history:
            if role == "assistant":
                conv_parts.append(f"Assistant: {content}")
            else:
                conv_parts.append(f"User: {content}")
        conversation = "\n\n".join(conv_parts)
        
        response = llm.generate(
            prompt="", system_prompt=system_prompt,
            user_message=conversation, max_tokens=2048,
            stop=["Observation:"],
        )
        raw_text = response.get("text", "") if isinstance(response, dict) else str(response)
        thought, action_name, args = _parse_action(raw_text)
        
        if verbose:
            t_short = thought[:60] + "..." if len(thought) > 60 else thought
            print(f"      [{r}] [think] {t_short}")
        
        if not action_name:
            history.append(("assistant", raw_text))
            history.append(("user", "Observation: [warn] Format error. Use: Thought/Action/Args format."))
            continue
        
        if action_name == "finish":
            if verbose:
                print(f"      [{r}] [done] Step complete")
            return True, "\n".join(stdout_log), "\n".join(code_accumulated), r, self_corrections
        
        # search_docs — RAG
        if action_name == "search_docs":
            observation = "Knowledge base not available. Please proceed with your existing knowledge."
            history.append(("assistant", raw_text))
            history.append(("user", f"Observation: {observation}"))
            continue
        
        observation = toolkit.run(action_name, args)
        
        if action_name == "execute" and isinstance(args, dict) and "code" in args:
            code = args["code"]
            code_accumulated.append(code)
            code_hash = hashlib.md5(code.strip().encode()).hexdigest()
            if code_hash == last_code_hash:
                if verbose:
                    print(f" [{r}] [stop] Code dedup -> ")
                return True, "\n".join(stdout_log), "\n".join(code_accumulated), r, self_corrections
            last_code_hash = code_hash
        
        # observation
        if len(observation) > 3000:
            observation = observation[:1500] + "\n...(truncated)...\n" + observation[-1500:]
        
        # -> 
        is_error = ("Error" in observation or "Traceback" in observation or "[fail]" in observation)
        if is_error:
            self_corrections += 1
            if verbose:
                err_short = observation.split('\n')[-1][:60] if observation else ""
                print(f"       [fail] Error -> self-correction #{self_corrections}: {err_short}")
            # RAG hint 
        else:
            stdout_log.append(observation[:500])
            if verbose:
                print(f"       [ok] Execution successful")
        
        history.append(("assistant", raw_text))
        history.append(("user", f"Observation: {observation}"))
    
    # -> 
    return False, "\n".join(stdout_log), "\n".join(code_accumulated), max_rounds, self_corrections

# ============================================================

# ============================================================

class TeeStream:
    """ stdout StringIO """
    def __init__(self, original, buffer):
        self.original = original
        self.buffer = buffer
    def write(self, text):
        self.original.write(text)
        self.buffer.write(text)
    def flush(self):
        self.original.flush()

def run_dual_agent_task(task_id, args):
    """Dual-agent main loop"""
    log_buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = TeeStream(old_stdout, log_buffer)
    
    # cwd work_dir cwd 
    os.chdir(PROJECT_ROOT)
    cfg = MODEL_CONFIG[args.model]
    label = cfg["label"]
    
    # Load tasks
    csv_path = os.getenv("GISCLAW_TASK_INDEX", os.path.join(os.getenv("GISCLAW_TASK_ROOT", os.path.join(PROJECT_ROOT, "tasks")), "tasks.csv"))
    task = load_task(csv_path, task_id)
    if not task:
        print(f"  [fail] Task {task_id} not found")
        return None
    
    # Directory setup
    output_dir = os.path.join(PROJECT_ROOT, "results", RESULTS_DIR_NAME, f"T{task_id:02d}")
    os.makedirs(output_dir, exist_ok=True)
    work_dir = os.path.join(output_dir, f"work_{label}")
    os.makedirs(work_dir, exist_ok=True)
    pred_dir = os.path.join(work_dir, "pred_results")
    os.makedirs(pred_dir, exist_ok=True)
    
    data_dir = os.path.join(os.getenv("GISCLAW_TASK_ROOT", os.path.join(PROJECT_ROOT, "tasks")), str(task_id), "dataset")
    if not os.path.exists(data_dir):
        data_dir = os.path.join(os.getenv("GISCLAW_TASK_ROOT", os.path.join(PROJECT_ROOT, "tasks")), str(task_id), "dataset")
    ds_link = os.path.join(work_dir, "dataset")
    if os.path.islink(ds_link):
        os.unlink(ds_link)
    if not os.path.exists(ds_link):
        os.symlink(os.path.abspath(data_dir), ds_link)
    
    # Initialize LLM Dual Agent: Planner Worker 
    worker_llm = init_llm(engine=cfg["engine"], model_name=cfg["model_name"])
    planner_llm = init_llm(engine=cfg["planner_engine"], model_name=cfg["planner_model"])

    
    print(f"  [reason] Planner: {cfg['planner_model']}")
    print(f"  [tool] Worker:  {cfg['model_name']}")
    
    dataset_desc = task['dataset_description'] or ""
    try:
        data_files = sorted(os.listdir(data_dir))
        file_list = "\n".join(f"  - {f}" for f in data_files if not f.startswith('.'))
        dataset_desc += f"\n\n## Available files in dataset/:\n{file_list}"
    except:
        pass
    
    print(f"\n{'='*40}")
    print(f"[run] T{task_id:02d} [{time.strftime('%Y-%m-%d %H:%M:%S')}] - {cfg['display']} (dual-agent Plan-Execute-Replan)")
    print(f"{'='*40}")
    
    t0 = time.time()
    
    # ===== Phase 1: Plan ( planner_llm) =====
    from src.agent.planner import TaskPlanner
    planner = TaskPlanner(planner_llm)
    steps = planner.plan(
        instruction=task['instruction'],
        dataset_description=dataset_desc,
        workflow=task['workflow'],
        domain_knowledge=task.get('domain_knowledge', ''),
    )
    
    print(f"\n [list] Plan ({len(steps)} ):")
    for s in steps:
        print(f"    {s.get('id','?')}. [{s.get('action','')}] {s.get('instruction','')[:70]}")
    
    # ===== Phase 2: Execute + Verify =====
    from src.agent.sandbox import PythonSandbox
    from src.agent.tools import GISToolkit
    from src.agent.prompts import build_system_prompt
    
    sandbox = PythonSandbox(work_dir=work_dir, timeout=cfg["timeout"])
    toolkit = GISToolkit(sandbox, data_dir="dataset")
    
    search_docs_desc = (
        "\n\nsearch_docs: Query the knowledge base for API/methodology help."
        "\n  Args: {\"query\": \"what to look up\"}"
        "\n  Returns: relevant knowledge documents"
    )
    system_prompt = build_system_prompt(
        tool_descriptions=toolkit.tool_descriptions + search_docs_desc,
    )
    
    step_results = []
    total_rounds = 0
    total_self_corrections = 0
    all_code = []
    step_context = ""
    # main SIGALRM 
    TASK_TIMEOUT_S = int(getattr(args, "task_timeout", 0) or 0) or 600
    
    for step in steps:
        # 10 
        elapsed = time.time() - t0
        if elapsed > TASK_TIMEOUT_S:
            print(f"\n [timeout] ({elapsed:.0f}s > {TASK_TIMEOUT_S}s)")
            break
        
        step_id = step.get('id', '?')
        step_action = step.get('action', '')
        print(f"\n  > Step {step_id}: {step_action}")
        
        # Execute ( worker_llm)
        success, stdout, code, rounds, sc = run_step_in_sandbox(
            llm=worker_llm, step=step, sandbox=sandbox, toolkit=toolkit,
            system_prompt=system_prompt,
            max_rounds=cfg["step_max_rounds"], verbose=True,
            step_history=step_context, task_start_time=t0,
        )
        total_rounds += rounds
        total_self_corrections += sc
        all_code.append(code)
        
        # Verify ( planner_llm)
        verdict = verify_step(planner_llm, step, stdout, task['instruction'])
        passed = verdict.get("pass", True)
        reason = verdict.get("reason", "")
        fix_hint = verdict.get("fix_hint", "")
        
        status = "[ok]" if passed else "[fail]"
        print(f"    {status} Verify: {reason[:60]}")
        
        # failure -> 
        retry_result = None
        if not passed and fix_hint:
            print(f"    [loop] Retry with fix hint: {fix_hint[:60]}")
            retry_step = dict(step)
            retry_step['instruction'] = f"{step['instruction']}\n\nFIX: {fix_hint}"
            success2, stdout2, code2, rounds2, sc2 = run_step_in_sandbox(
                llm=worker_llm, step=retry_step, sandbox=sandbox, toolkit=toolkit,
                system_prompt=system_prompt,
                max_rounds=5, verbose=True,
                step_history=step_context, task_start_time=t0,
            )
            total_rounds += rounds2
            total_self_corrections += sc2
            all_code.append(code2)
            
            verdict2 = verify_step(planner_llm, retry_step, stdout2, task['instruction'])
            passed = verdict2.get("pass", True)
            retry_result = {
                "rounds": rounds2, "self_corrections": sc2,
                "passed": passed, "reason": verdict2.get("reason", ""),
            }
            status2 = "[ok]" if passed else "[fail]"
            print(f"    {status2} Retry verify: {verdict2.get('reason','')[:60]}")
        
        if stdout:
            step_context += f"\n\nStep {step_id} ({step_action}) output:\n{stdout[:500]}"
        
        step_results.append({
            "step_id": step_id,
            "action": step_action,
            "instruction": step.get('instruction', '')[:200],
            "rounds": rounds,
            "self_corrections": sc,
            "success": success,
            "verify_pass": passed,
            "verify_reason": reason[:200],
            "retry": retry_result,
        })
    
    elapsed = time.time() - t0
    
    # ===== Phase 3: =====
    # ( work_dir pred_results)
    result_exts = ('.png', '.csv', '.geojson', '.tif', '.shp', '.dbf', '.prj', '.shx', '.cpg', '.json')
    for f in os.listdir(work_dir):
        if f.endswith(result_exts) and not f.startswith('.'):
            src = os.path.join(work_dir, f)
            dst = os.path.join(pred_dir, f)
            if os.path.isfile(src) and not os.path.exists(dst):
                shutil.move(src, dst)
    
    output_files = sorted([f for f in os.listdir(pred_dir) if not f.startswith('.')]) if os.path.exists(pred_dir) else []
    
    model_dir_name = cfg["model_name"].replace(":", "-")
    pred_dst = os.path.join(output_dir, model_dir_name)
    if os.path.exists(pred_dst):
        shutil.rmtree(pred_dst)
    if os.path.exists(pred_dir):
        shutil.copytree(pred_dir, pred_dst)
    
    merged_code = "\n\n".join(c for c in all_code if c)
    code_path = os.path.join(pred_dst, "code.py")
    with open(code_path, "w") as f:
        f.write(merged_code)
    
    # Save the execution log
    sys.stdout = old_stdout
    log_content = log_buffer.getvalue()
    log_path = os.path.join(pred_dst, "execution_log.txt")
    with open(log_path, "w") as f:
        f.write(log_content)
    log_buffer.close()
    
    # Gold
    for gold_dir in [
        os.path.join(os.getenv("GISCLAW_TASK_ROOT", os.path.join(PROJECT_ROOT, "tasks")), str(task_id), "pred_results"),
        os.path.join(os.getenv("GISCLAW_TASK_ROOT", os.path.join(PROJECT_ROOT, "tasks")), str(task_id), "pred_results"),
    ]:
        if os.path.exists(gold_dir):
            dest = os.path.join(output_dir, "gold")
            if not os.path.exists(dest):
                shutil.copytree(gold_dir, dest)
            break
    
    # ===== =====
    result = {
        "task_id": task_id,
        "model": label,
        "architecture": "dual_agent_plan_execute_replan",
        "plan_steps": len(steps),
        "total_rounds": total_rounds,
        "self_corrections": total_self_corrections,
        "elapsed": round(elapsed, 1),
        "output_files": output_files,
        "success": len([f for f in output_files if f != 'code.py']) > 0,
        "step_results": step_results,
        "plan": [{"id": s.get("id"), "action": s.get("action"), "instruction": s.get("instruction","")[:100]} for s in steps],
    }
    
    results_path = os.path.join(output_dir, "results.json")
    existing = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            existing = json.load(f)
    existing[label] = result
    with open(results_path, "w") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    
    outs = [f for f in output_files if f != 'code.py']
    sc_str = f"self-corrections {total_self_corrections}" if total_self_corrections else ""
    icon = "[ok]" if outs else "[fail]"
    print(f"\n  {icon} Done: {total_rounds} rounds | {len(outs)} outputs | {sc_str} | {elapsed:.1f}s")
    print(f" : {output_files}")
    step_summary = [f"S{sr['step_id']}:{'[ok]' if sr['verify_pass'] else '[fail]'}" for sr in step_results]
    print(f" : {step_summary}")
    
    # Clean work_dir
    shutil.rmtree(work_dir, ignore_errors=True)
    
    return result

def main():
    args = parse_args()
    cfg = MODEL_CONFIG[args.model]
    # : "3-50" : "1,2,3"
    task_str = args.task
    if "-" in task_str and "," not in task_str:
        start, end = task_str.split("-")
        task_ids = list(range(int(start), int(end) + 1))
    else:
        task_ids = [int(t.strip()) for t in task_str.split(",")]

    task_timeout = max(0, int(args.task_timeout or 0))
    if task_timeout > 0:
        signal.signal(signal.SIGALRM, _timeout_signal_handler)
    print(f"\n  [path] RESULTS_DIR_NAME = {RESULTS_DIR_NAME}")
    if task_timeout:
        print(f"  [timeout] Task timeout = {task_timeout}s")

    for tid in task_ids:
        if args.skip_existing:
            rp = os.path.join(PROJECT_ROOT, "results", RESULTS_DIR_NAME, f"T{tid:02d}", "results.json")
            if os.path.exists(rp):
                try:
                    with open(rp) as f:
                        existing = json.load(f)
                    prev = existing.get(cfg["label"])
                    if prev and prev.get("success") and "timeout" not in (prev.get("fail_reason") or ""):
                        print(f"\n [skip] T{tid:02d} success--skip-existing ")
                        continue
                except Exception:
                    pass
        if task_timeout > 0:
            signal.alarm(task_timeout)
        try:
            run_dual_agent_task(tid, args)
        except TaskTimeoutError:
            print(f"\n [timeout] T{tid:02d} TIMEOUT (>{task_timeout}s) — fail ")
            try:
                _write_timeout_fail(tid, args, task_timeout)
            except Exception as we:
                print(f" (warn: failure: {we})")
        except Exception as e:
            print(f"\n  [fail] T{tid} failed with exception: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if task_timeout > 0:
                signal.alarm(0)

if __name__ == "__main__":
    main()

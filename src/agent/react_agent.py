"""ReAct loop driving the GISclaw single-agent.

Design notes:
A. Observations from the sandbox truncate stdout from the front and stderr
   from the tail so the agent always sees the most recent output and the
   root-cause line of any traceback.
B. The Action JSON is parsed with json.loads, falling back through regex
   extraction, ast.literal_eval, and a bare key=value parser before giving up.
   Parse failures never crash the loop -- the agent receives a hint and is
   given another turn to self-correct.
C. load_vector and load_raster auto-inject the loaded objects into the
   sandbox namespace so subsequent execute() steps can reference them.
D. The system prompt forbids overwriting original input variables, which
   keeps the working namespace stable across rounds.

The self-correction count is exposed on AgentResult and is used by the
companion evaluation pipeline.
"""
import os
import re
import json
import time
import ast
from typing import Dict, Any, Optional, List, Tuple
import glob

from src.agent.sandbox import PythonSandbox
from src.agent.tools import GISToolkit
from src.agent.prompts import build_system_prompt
from src.agent.error_memory import ErrorMemory

class AgentResult:
    """Agent """
    def __init__(self, task_id: int, success: bool = False,
                 code: str = "", history: list = None,
                 output_files: list = None, time_ms: float = 0,
                 total_rounds: int = 0, self_corrections: int = 0):
        self.task_id = task_id
        self.success = success
        self.code = code
        self.history = history or []
        self.output_files = output_files or []
        self.time_ms = time_ms
        self.total_rounds = total_rounds
        self.self_corrections = self_corrections
    
    def to_dict(self):
        return {
            "task_id": self.task_id,
            "success": self.success,
            "code_length": len(self.code),
            "output_files": self.output_files,
            "time_ms": round(self.time_ms, 1),
            "total_rounds": self.total_rounds,
            "self_corrections": self.self_corrections,
            "history_length": len(self.history),
        }

def _parse_action(response_text: str) -> Tuple[str, str, Dict[str, Any]]:
    """Parse the LLM output into (thought, action_name, args)
    
    Robustness notes (failure mode B):
    - JSON //
    - Do not crash on parse failure; return an error hint so the model can self-correct
    """
    thought = ""
    action_name = ""
    args = {}
    
    # Thought
    thought_match = re.search(r'Thought:\s*(.*?)(?=\nAction:|\Z)', response_text, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()
    
    # Action
    action_match = re.search(r'Action:\s*(\w+)', response_text)
    if action_match:
        action_name = action_match.group(1).strip()
    
    # Args — 
    args_match = re.search(r'Args:\s*(.*?)$', response_text, re.DOTALL | re.MULTILINE)
    if args_match:
        args_text = args_match.group(1).strip()
        
        # If more lines follow, stop at the next blank line or before Thought
        lines = args_text.split('\n')
        json_lines = []
        for line in lines:
            if line.strip().startswith('Thought:') or line.strip().startswith('Action:'):
                break
            json_lines.append(line)
        args_text = '\n'.join(json_lines).strip()
        
        # Strip any markdown code fences
        args_text = re.sub(r'^```\w*\n?', '', args_text)
        args_text = re.sub(r'\n?```$', '', args_text)
        args_text = args_text.strip()
        
        if not args_text or args_text == '{}':
            args = {}
        else:
            # Attempt 1: standard json.loads
            try:
                args = json.loads(args_text)
            except (json.JSONDecodeError, ValueError):
                # Attempt 2: regex-extract a JSON object
                json_match = re.search(r'\{.*\}', args_text, re.DOTALL)
                if json_match:
                    try:
                        args = json.loads(json_match.group())
                    except (json.JSONDecodeError, ValueError):
                        # Attempt 3: ast.literal_eval (handles single quotes etc.)
                        try:
                            args = ast.literal_eval(json_match.group())
                        except (ValueError, SyntaxError):
                            pass
                
                # Attempt 4: bare key=value format
                if not args and '=' in args_text:
                    for part in args_text.split(','):
                        if '=' in part:
                            k, v = part.split('=', 1)
                            args[k.strip()] = v.strip().strip('"').strip("'")
    
    # : execute 
    if action_name == "execute" and "code" not in args:
        # Fallback 1: extract from markdown code block
        code_match = re.search(r'```(?:python)?\n(.*?)```', response_text, re.DOTALL)
        if code_match:
            args = {"code": code_match.group(1).strip()}
        else:
            # Fallback 2: treat post-Args: text as raw code
            after_args = re.search(r'Args:\s*\{?\}?\s*\n(.*?)$', response_text, re.DOTALL)
            if after_args:
                raw_code = after_args.group(1).strip()
                if raw_code and not raw_code.startswith('{') and len(raw_code) > 5:
                    args = {"code": raw_code}
    
    return thought, action_name, args

class GISReActAgent:
    """GIS ReAct Agent"""
    
    def __init__(self, llm_engine, timeout: int = 60,
                 max_rounds: int = 25, verbose: bool = True,
                 rag=None, error_memory: ErrorMemory = None):
        self.llm = llm_engine
        self.timeout = timeout
        self.max_rounds = max_rounds
        self.verbose = verbose
        self.rag = rag
        self._loaded_crs = {}
        self.error_memory = error_memory or ErrorMemory()
    
    def run(self, task_id: int, instruction: str,
            workflow: str, data_dir: str, work_dir: str,
            domain_knowledge: str = "",
            dataset_description: str = "",
            rag_context: str = "",
            skill_text: str = "") -> AgentResult:
        """ GIS 
        
        Notes:
        - System Prompt RAG/Skill/Workflow
        - Workflow User Message
        - RAG search_docs 
        """
        t0 = time.time()
        
        # 1. 
        os.makedirs(work_dir, exist_ok=True)
        pred_dir = os.path.join(work_dir, "pred_results")
        os.makedirs(pred_dir, exist_ok=True)
        
        # Create the dataset symlink
        ds_link = os.path.join(work_dir, "dataset")
        if os.path.islink(ds_link):
            os.unlink(ds_link)
        if not os.path.exists(ds_link):
            os.symlink(os.path.abspath(data_dir), ds_link)
        
        sandbox = PythonSandbox(work_dir=work_dir, timeout=self.timeout)
        toolkit = GISToolkit(sandbox, data_dir="dataset")
        
        # 2. Build the system prompt (no RAG/skill/workflow injection at this layer)
        # search_docs tool_descriptions
        search_docs_desc = (
            "\n\nsearch_docs: Query the knowledge base. Use when unsure about a library's API or a GIS methodology."
            "\n  Args: {\"query\": \"what to look up, e.g. 'rasterio read specific band' or 'NBR formula Landsat 8'\"}"
            "\n  Returns: relevant knowledge documents"
        )
        system_prompt = build_system_prompt(
            tool_descriptions=toolkit.tool_descriptions + search_docs_desc,
        )
        
        # 3. Build the user message: task info + workflow + dataset description + domain knowledge
        user_msg = f"Task: {instruction}\n"
        if workflow:
            user_msg += f"\n## Analysis Steps\n{workflow}\n"
        if dataset_description:
            user_msg += f"\n## Dataset Description\n{dataset_description}\n"
        if domain_knowledge:
            user_msg += f"\n## Domain Knowledge\n{domain_knowledge}\n"
        
        if self.verbose:
            print(f"  ReAct loop start (max {self.max_rounds} rounds)")
        
        # 4. ReAct loop
        history = []  # [(role, content), ...]
        round_details = []
        self_corrections = 0
        last_action_failed = False
        finished = False
        round_num = 0
        
        # : 
        consecutive_same_tool = 0
        last_tool_fingerprint = ""
        self._recent_code_hashes = []
        
        for round_num in range(1, self.max_rounds + 1):
            # === Approaching task deadline: persist whatever has been computed ===
            remaining = self.max_rounds - round_num
            if remaining == 2 and history:
                deadline_msg = (
                    "\n\n[alert] DEADLINE WARNING: You have only 2 rounds left! "
                    "You MUST save results NOW. Even if analysis is incomplete, "
                    "save whatever you have IMMEDIATELY:\n"
                    "1. If you have any plot: plt.savefig('pred_results/analysis.png', dpi=150, bbox_inches='tight')\n"
                    "2. If you have any DataFrame: df.to_csv('pred_results/results.csv')\n"
                    "3. If you have any GeoDataFrame: gdf.to_file('pred_results/output.geojson', driver='GeoJSON')\n"
                    "Do NOT continue complex analysis. SAVE IMMEDIATELY, then call finish()."
                )
                # history observation 
                last_role, last_content = history[-1]
                history[-1] = (last_role, last_content + deadline_msg)
                if self.verbose:
                    print(f"\n [!] [alert] Deadline Warning ( {remaining} )")
            
            # Build conversation text
            conversation_text = self._format_conversation(user_msg, history)
            
            # LLM 
            gen_t0 = time.time()
            response = self.llm.generate(
                prompt="",
                system_prompt=system_prompt,
                user_message=conversation_text,
                max_tokens=2048,
                stop=["Observation:"],
            )
            gen_time = (time.time() - gen_t0) * 1000
            
            raw_text = response.get("text", "") if isinstance(response, dict) else str(response)
            
            # API 429/500 ReAct loop
            if raw_text.startswith("Error during API call") or raw_text.startswith("Error during"):
                if not hasattr(self, '_api_error_count'):
                    self._api_error_count = 0
                self._api_error_count += 1
                if self.verbose:
                    err_short = raw_text[:100].replace('\n', ' ')
                    print(f"\n  [{round_num:>2}] [blocked] API error (#{self._api_error_count}): {err_short}")
                if self._api_error_count >= 3:
                    if self.verbose:
                        print(f" [dead] {self._api_error_count} API ")
                    break
                # Wait 5 s and retry (likely a transient rate limit)
                time.sleep(5)
                continue
            
            # Parse Thought + Action
            thought, action_name, args = _parse_action(raw_text)
            
            if self.verbose:
                thought_short = thought[:80] + "..." if len(thought) > 80 else thought
                print(f"\n  [{round_num:>2}] [think] {thought_short}")
            
            # Check whether format parsing failed (failure mode B)
            if not action_name:
                # Format-parse failures: terminate after more than 10 consecutive failures
                if not hasattr(self, '_consecutive_parse_fails'):
                    self._consecutive_parse_fails = 0
                self._consecutive_parse_fails += 1
                if self._consecutive_parse_fails >= 10:
                    if self.verbose:
                        print(f" [dead] {self._consecutive_parse_fails} failure")
                    break
                
                observation = ("[warn] Format error: Could not parse Action. Please respond strictly in this format:\n"
                              "Thought: <your reasoning>\n"
                              "Action: <tool_name>\n"
                              "Args: {<JSON arguments>}")
                history.append(("assistant", raw_text))
                history.append(("user", f"Observation: {observation}"))
                if self.verbose:
                    print(f"       [warn] Format parse failed, asking model to correct")
                continue
            else:
                # Reset the counter on successful parse
                self._consecutive_parse_fails = 0
                self._api_error_count = 0
            
            # Check for finish
            if action_name == "finish":
                # : Agent pred_results/
                pred_dir_check = os.path.join(work_dir, "pred_results")
                result_exts = ('.png', '.csv', '.geojson', '.tif', '.shp', '.dbf', '.prj', '.shx', '.cpg')
                for f in os.listdir(work_dir):
                    if f.endswith(result_exts) and not f.startswith('.') and not f.startswith('_react'):
                        src_f = os.path.join(work_dir, f)
                        dst_f = os.path.join(pred_dir_check, f)
                        if os.path.isfile(src_f) and not os.path.exists(dst_f):
                            import shutil
                            shutil.move(src_f, dst_f)
                            if self.verbose:
                                print(f"       [path] Auto-moved: {f} -> pred_results/")

                # Output Guard: finish pred_results 
                has_outputs = False
                if os.path.exists(pred_dir_check):
                    has_outputs = len([f for f in os.listdir(pred_dir_check) if not f.startswith('.')]) > 0
                
                if not has_outputs and getattr(self, '_output_guard_retries', 0) < 3:
                    self._output_guard_retries = getattr(self, '_output_guard_retries', 0) + 1
                    observation = (
                        "[warn] Output Guard: pred_results/ directory is empty! Your analysis has not saved any result files yet.\n"
                        "Before calling finish(), you must save at least one result file:\n"
                        "- Visualization: plt.savefig('pred_results/analysis.png', dpi=150, bbox_inches='tight')\n"
                        "- Data table: df.to_csv('pred_results/results.csv', index=False)\n"
                        "- Vector data: gdf.to_file('pred_results/result.geojson', driver='GeoJSON')\n"
                        "- Raster data: rasterio.open('pred_results/output.tif', 'w', ...) to write out\n"
                        "Please save your analysis results before calling finish()."
                    )
                    history.append(("assistant", raw_text))
                    history.append(("user", f"Observation: {observation}"))
                    if self.verbose:
                        print(f"       [guard]️ Output Guard: pred_results empty (hint {self._output_guard_retries}/3)")
                    continue
                
                # P0-B: — finish 
                validation_issues = self._validate_outputs(work_dir)
                if validation_issues and getattr(self, '_finish_retries', 0) < 2:
                    self._finish_retries = getattr(self, '_finish_retries', 0) + 1
                    observation = (
                        f"[warn] Output validation failed, please fix and call finish() again:\n{validation_issues}\n\n"
                        "Please check the above issues and regenerate output files, ensuring:\n"
                        "1. Images are not blank (have actual data rendered)\n"
                        "2. Data files have content (rows > 0)\n"
                        "3. If matplotlib image, check ax.collections/ax.patches is not empty"
                    )
                    history.append(("assistant", raw_text))
                    history.append(("user", f"Observation: {observation}"))
                    if self.verbose:
                        print(f"       [warn] Output validation failed (retry {self._finish_retries}/2): {validation_issues[:80]}")
                    continue
                
                observation = toolkit.run("finish", args)
                finished = True
                if self.verbose:
                    print(f"       finish: {args.get('summary', '')[:60]}")
                
                round_details.append({
                    "round": round_num,
                    "thought": thought[:200],
                    "action": action_name,
                    "success": True,
                    "gen_time_ms": gen_time,
                })
                break
            
            # ===== Code-deduplication check =====
            # 1) Sequential repeat detection by tool name + args (legacy logic, retained)
            if action_name == "execute" and isinstance(args, dict) and "code" in args:
                code_lines = args['code'].strip().split('\n')[:5]
                fingerprint = hash('\n'.join(line.strip() for line in code_lines))
            else:
                fingerprint = hash((action_name, str(args)[:200]))
            current_fp = f"{action_name}:{fingerprint}"
            if current_fp == last_tool_fingerprint:
                consecutive_same_tool += 1
            else:
                consecutive_same_tool = 0
            last_tool_fingerprint = current_fp
            
            # 2) Sliding-window deduplication by full-code hash (handles T25-style semantic repeats)
            code_dedup_triggered = False
            if action_name == "execute" and isinstance(args, dict) and "code" in args:
                # Normalize code: drop blank lines, trim whitespace, strip comment-only lines
                raw_code = args['code']
                norm_lines = [l.strip() for l in raw_code.split('\n')
                              if l.strip() and not l.strip().startswith('#')]
                code_hash = hash('\n'.join(norm_lines))
                
                # Check the recent execute window (last 6 calls)
                if not hasattr(self, '_recent_code_hashes'):
                    self._recent_code_hashes = []
                
                dup_count = self._recent_code_hashes.count(code_hash)
                self._recent_code_hashes.append(code_hash)
                # Keep the window size at 6
                if len(self._recent_code_hashes) > 6:
                    self._recent_code_hashes = self._recent_code_hashes[-6:]
                
                if dup_count >= 2:
                    # 3 -> 
                    code_dedup_triggered = True
                    if self.verbose:
                        print(f"       [stop] Code dedup: identical code submitted {dup_count+1} times, force-terminating")
                    observation = toolkit.run("finish", {"summary": f"System auto-terminated: identical code submitted {dup_count+1} times. Intermediate results in pred_results/"})
                    finished = len([f for f in os.listdir(os.path.join(work_dir, "pred_results")) if not f.startswith('.')]) > 0 if os.path.exists(os.path.join(work_dir, "pred_results")) else False
                    round_details.append({
                        "round": round_num,
                        "thought": f"(code dedup: identical code x{dup_count+1})",
                        "action": "finish",
                        "success": finished,
                        "gen_time_ms": gen_time,
                    })
                    break
                elif dup_count == 1:
                    # 2 -> 
                    code_dedup_triggered = True
                    completed_steps = [rd['action'] for rd in round_details if rd.get('success')]
                    step_summary = ', '.join(f"R{rd['round']}:{rd['action']}" for rd in round_details[-5:])
                    
                    observation = (
                        f"[warn] CODE DUPLICATE DETECTED: You submitted nearly identical code as a previous round. "
                        f"This is the 2nd time — do NOT submit it again or the system will terminate.\n\n"
                        f"=== Your recent actions ===\n  {step_summary}\n\n"
                        f"=== What you must do ===\n"
                        f"You have ALREADY completed this step successfully. Move on to the NEXT step in your plan.\n"
                        f"Review the task instruction and your plan, then write code for the next uncompleted step."
                    )
                    history.append(("assistant", raw_text))
                    history.append(("user", f"Observation: {observation}"))
                    if self.verbose:
                        print(f"       [warn] Code dedup: 2nd identical submission, progress reminder injected")
                    continue
            
            # 3) Consecutive identical tool names (legacy)
            if consecutive_same_tool >= 3 and not code_dedup_triggered:
                # First interrupt (4th time): inject full variable info, force model to see actual column names
                if consecutive_same_tool == 3:
                    # Collect detailed column info for all sandbox variables
                    var_details = []
                    for vn in sandbox._list_user_vars():
                        try:
                            desc = sandbox._describe_var(vn, detailed=True)
                            var_details.append(f"  {vn}: {desc[:300]}")
                        except:
                            var_details.append(f"  {vn}: (cannot describe)")
                    var_info = '\n'.join(var_details) if var_details else '(no variables)'
                    
                    observation = (
                        f"[warn] System interrupt: You have called {action_name} {consecutive_same_tool+1} times consecutively. This is ineffective.\n\n"
                        f"=== Detailed info for all current variables ===\n{var_info}\n\n"
                        "=== You must change your strategy immediately ===\n"
                        "1. If you cannot find column names: all variable columns are listed above, use the correct ones\n"
                        "2. If file is unreadable (.lyrx/.ppkx): call finish() and report unsupported data format\n"
                        "3. If CRS mismatch: use gdf.to_crs(epsg=XXXX) to align before operations\n"
                        "4. If code errors: use execute() to try a completely different approach\n"
                        f"5. Do NOT call {action_name}() again, or the system will force-terminate the task"
                    )
                # Second interrupt (5th time): force end, don't wait for model
                elif consecutive_same_tool >= 4:
                    if self.verbose:
                        print(f"       [stop] {consecutive_same_tool+1} consecutive repeats, system force-terminating")
                    # Execute finish and break
                    observation = toolkit.run("finish", {"summary": f"System auto-terminated due to consecutive repeated operations. Intermediate results in pred_results/"})
                    finished = len([f for f in os.listdir(os.path.join(work_dir, "pred_results")) if not f.startswith('.')]) > 0 if os.path.exists(os.path.join(work_dir, "pred_results")) else False
                    round_details.append({
                        "round": round_num,
                        "thought": "(system force-terminated)",
                        "action": "finish",
                        "success": finished,
                        "gen_time_ms": gen_time,
                    })
                    break
                
                history.append(("assistant", raw_text))
                history.append(("user", f"Observation: {observation}"))
                if self.verbose:
                    print(f"       [warn] {consecutive_same_tool+1} consecutive same calls, forced interrupt")
                continue
            
            # JSON — execute 
            if action_name == "execute" and "code" not in args and args:
                # Case: the model emitted a malformed JSON such as {'{"code": "..."': 'value'}
                # Try to reconstruct the code from args keys/values
                all_text = ' '.join(str(k) + ' ' + str(v) for k, v in args.items())
                # "code" 
                code_extract = re.search(r'"code"\s*:\s*"(.*?)"', all_text, re.DOTALL)
                if code_extract:
                    args = {"code": code_extract.group(1).replace('\\n', '\n')}
                else:
                    # Concatenate all values and treat the result as code
                    code_parts = []
                    for k, v in args.items():
                        code_parts.append(str(k).strip('"\'{} '))
                        if v and str(v).strip() != k:
                            code_parts.append(str(v).strip('"\'{} '))
                    combined = '\n'.join(p for p in code_parts if p)
                    if len(combined) > 10:
                        args = {"code": combined}
                if self.verbose and "code" in args:
                    print(f"       [tool] JSON fix: extracted code from malformed args")
            
            if self.verbose:
                args_short = str(args)[:60]
                print(f"       [tool] {action_name}({args_short})")
            
            # v2: search_docs — RAG
            if action_name == "search_docs":
                query = args.get("query", "") if isinstance(args, dict) else str(args)
                if self.rag and query:
                    observation = self.rag.format_context(query, n=3)
                    if observation:
                        observation = f"[docs] Search results:\n{observation}"
                    else:
                        observation = "[docs] No relevant knowledge found. Try other keywords, or continue with your existing knowledge."
                else:
                    observation = "[docs] Knowledge base unavailable. Continue analysis with your existing knowledge."
                if self.verbose:
                    obs_preview = observation[:100].replace('\n', ' ')
                    print(f"       [docs] search_docs: {obs_preview}...")
            else:
                observation = toolkit.run(action_name, args)
            
            # CRS observation CRS 
            if action_name in ('load_vector', 'load_raster'):
                crs_match = re.search(r'CRS:\s*(\S+)', observation)
                var_name = args.get('var_name', '')
                if crs_match and var_name:
                    crs_val = crs_match.group(1)
                    self._loaded_crs[var_name] = crs_val
                    # Check whether the CRS conflicts with an already-loaded dataset
                    unique_crs = set(v for v in self._loaded_crs.values() if v and v != 'None')
                    if len(unique_crs) > 1:
                        crs_list = ', '.join(f"{k}={v}" for k, v in self._loaded_crs.items() if v and v != 'None')
                        observation += (
                            f"\n\n[warn] CRS mismatch warning! Currently loaded dataset CRS:\n  {crs_list}\n"
                            "Before spatial operations (sjoin/clip/mask), you MUST first unify CRS using .to_crs() or rasterio.warp.reproject()!"
                        )
            
            # Track self-correction events (the agent recovers from a prior error)
            action_success = "[fail]" not in observation[:10]  # heuristic: fail tag emitted by toolkit
            if last_action_failed and action_success:
                self_corrections += 1
                if self.verbose:
                    print(f"       [loop] Self-correction success! (#{self_corrections})")
            last_action_failed = not action_success
            
            if self.verbose:
                obs_lines = observation.strip().split("\n")
                for line in obs_lines[:4]:
                    print(f"       > {line[:80]}")
                if len(obs_lines) > 4:
                    print(f"       > ... ({len(obs_lines)} lines)")
            
            # Errors trigger Error Memory and a secondary RAG lookup
            if not action_success:
                # Error Memory
                mem_hint = self.error_memory.lookup(observation)
                if mem_hint:
                    observation += f"\n\n[reason] Cross-task experience hint:\n{mem_hint}"
                    if self.verbose:
                        print(f"       [reason] Error Memory hit")
                # RAG
                if self.rag:
                    rag_hint = self._error_triggered_rag(observation)
                    if rag_hint:
                        observation += f"\n\n[tip] Relevant reference knowledge:\n{rag_hint}"
                        if self.verbose:
                            print(f"       [docs] RAG secondary retrieval hit")
            
            # success Error Memory
            if last_action_failed and action_success and round_num > 1:
                # Extract error keywords from the previous round's observation
                prev_obs = history[-2][1] if len(history) >= 2 else ""
                if "[fail]" in prev_obs:
                    error_key = prev_obs.split('\n')[0][:80]
                    fix_key = observation.split('\n')[0][:80]
                    self.error_memory.record(error_key, f"Fix method: {fix_key}")
            
            # Record history
            history.append(("assistant", raw_text))
            history.append(("user", f"Observation: {observation}"))
            
            round_details.append({
                "round": round_num,
                "thought": thought[:200],
                "action": action_name,
                "args_summary": str(args)[:100],
                "observation": observation[:500] if isinstance(observation, str) else str(observation)[:500],
                "success": action_success,
                "gen_time_ms": gen_time,
            })
        
        # ===== 3: Auto-save partial results when the loop budget is exhausted =====
        if not finished:
            emergency_code = (
                "import os\n"
                "os.makedirs('pred_results', exist_ok=True)\n"
                "_saved = []\n"
                "# 1) Save the current matplotlib figure\n"
                "try:\n"
                "    import matplotlib.pyplot as plt\n"
                "    if plt.get_fignums():\n"
                "        plt.savefig('pred_results/emergency_plot.png', dpi=150, bbox_inches='tight')\n"
                "        _saved.append('emergency_plot.png')\n"
                "except: pass\n"
                "# 2) Save the largest GeoDataFrame in the sandbox\n"
                "try:\n"
                "    import geopandas as _gpd\n"
                "    _gdf_candidates = []\n"
                "    for _n, _v in list(globals().items()):\n"
                "        if isinstance(_v, _gpd.GeoDataFrame) and len(_v) > 0 and not _n.startswith('_'):\n"
                "            _gdf_candidates.append((_n, _v))\n"
                "    if _gdf_candidates:\n"
                "        _name, _gdf = max(_gdf_candidates, key=lambda x: len(x[1]))\n"
                "        _gdf.to_file('pred_results/emergency_output.geojson', driver='GeoJSON')\n"
                "        _saved.append(f'{_name}.geojson')\n"
                "except: pass\n"
                "# 3) Save the largest DataFrame in the sandbox\n"
                "try:\n"
                "    import pandas as _pd\n"
                "    _df_candidates = []\n"
                "    for _n, _v in list(globals().items()):\n"
                "        if isinstance(_v, _pd.DataFrame) and not isinstance(_v, _gpd.GeoDataFrame) and len(_v) > 0 and not _n.startswith('_'):\n"
                "            _df_candidates.append((_n, _v))\n"
                "    if _df_candidates:\n"
                "        _name, _df = max(_df_candidates, key=lambda x: len(x[1]))\n"
                "        _df.to_csv('pred_results/emergency_output.csv', index=False)\n"
                "        _saved.append(f'{_name}.csv')\n"
                "except: pass\n"
                "print(f'Emergency save: {_saved}')\n"
            )
            try:
                emg_result = sandbox.execute(emergency_code)
                if self.verbose:
                    print(f"\n  [save] Emergency Save completed: {getattr(emg_result, 'stdout', '')[:100]}")
            except Exception as e:
                if self.verbose:
                    print(f"\n  [warn] Emergency Save failed: {e}")
        
        # 5. Collect results
        output_files = []
        for d in ["pred_results", "output"]:
            out_dir = os.path.join(work_dir, d)
            if os.path.exists(out_dir):
                for fn in os.listdir(out_dir):
                    if not fn.startswith('.'):
                        output_files.append(os.path.join(d, fn))
        
        # Save the full script
        script_path = os.path.join(work_dir, f"_react_t{task_id}.py")
        full_code = sandbox.get_full_code()
        with open(script_path, "w") as f:
            f.write(full_code)
        
        # Distinguish agent-driven outputs from emergency-save outputs
        emergency_names = {'emergency_plot.png', 'emergency_output.geojson', 'emergency_output.csv'}
        real_output_files = [f for f in output_files
                             if os.path.basename(f) not in emergency_names]
        
        success = finished and len(real_output_files) > 0
        # Emergency save success
        if not finished and len(real_output_files) > 0:
            success = True
        total_time = (time.time() - t0) * 1000
        
        if self.verbose:
            marker = "[ok]" if success else "[fail]"
            print(f"\n  {marker} Done: {round_num} rounds | "
                  f"{len(output_files)} outputs | "
                  f"self-corrections {self_corrections} | "
                  f"{total_time/1000:.1f}s")
        
        return AgentResult(
            task_id=task_id,
            success=success,
            code=full_code,
            history=round_details,
            output_files=output_files,
            time_ms=total_time,
            total_rounds=round_num,
            self_corrections=self_corrections,
        )
    
    def _validate_outputs(self, work_dir: str) -> str:
        """ pred_results/ """
        issues = []
        pred_dir = os.path.join(work_dir, "pred_results")
        if not os.path.exists(pred_dir):
            return "pred_results/ directory does not exist, please save output files first"
        
        files = [f for f in os.listdir(pred_dir) if not f.startswith('.')]
        if not files:
            return "pred_results/ directory is empty, please save output files first"
        
        for fn in files:
            fpath = os.path.join(pred_dir, fn)
            size = os.path.getsize(fpath)
            ext = fn.lower().rsplit('.', 1)[-1] if '.' in fn else ''
            
            # PNG/JPG: < 5KB
            if ext in ('png', 'jpg', 'jpeg'):
                if size < 5000:
                    issues.append(f"- {fn}: File is only {size} bytes, may be a blank image. Please check that plot data is correctly rendered.")
            
            # CSV: 
            elif ext == 'csv':
                try:
                    with open(fpath) as f:
                        lines = sum(1 for _ in f)
                    if lines <= 1:
                        issues.append(f"- {fn}: Only {lines} line(s) (may be header only), please ensure data is present.")
                except:
                    pass
            
            # SHP: .dbf 
            elif ext == 'shp':
                dbf = fpath.replace('.shp', '.dbf')
                if os.path.exists(dbf) and os.path.getsize(dbf) < 100:
                    issues.append(f"- {fn}: .dbf file is extremely small, may have no attribute data.")
        
        return '\n'.join(issues)
    
    def _error_triggered_rag(self, observation: str) -> str:
        """On error, run a second RAG search over stderr and return the most relevant fix"""
        if not self.rag or not hasattr(self.rag, 'collections'):
            return ""
        
        # stderr 
        lines = observation.strip().split('\n')
        error_lines = [l for l in lines[-5:] if l.strip() and not l.startswith('>')]
        if not error_lines:
            error_lines = lines[-3:]
        error_query = ' '.join(error_lines)[:200]
        
        if not error_query.strip():
            return ""
        
        hints = []
        seen = set()
        
        # Look up api_pitfalls (most precise error fixes)
        if hasattr(self.rag, 'collections') and 'api_pitfalls' in self.rag.collections:
            col = self.rag.collections['api_pitfalls']
            if col.count() > 0:
                results = col.query(query_texts=[error_query], n_results=1)
                if results['documents'][0]:
                    doc = results['documents'][0][0][:300]
                    if doc[:60] not in seen:
                        seen.add(doc[:60])
                        hints.append(doc)
        
        # Look up code_cookbook (canonical code patterns)
        if hasattr(self.rag, 'collections') and 'code_cookbook' in self.rag.collections:
            col = self.rag.collections['code_cookbook']
            if col.count() > 0:
                results = col.query(query_texts=[error_query], n_results=1)
                if results['documents'][0]:
                    doc = results['documents'][0][0][:400]
                    if doc[:60] not in seen:
                        seen.add(doc[:60])
                        hints.append(doc)
        
        # Look up api_reference (canonical API signatures)
        if hasattr(self.rag, 'collections') and 'api_reference' in self.rag.collections:
            col = self.rag.collections['api_reference']
            if col.count() > 0:
                results = col.query(query_texts=[error_query], n_results=1)
                if results['documents'][0]:
                    doc = results['documents'][0][0][:300]
                    if doc[:60] not in seen:
                        seen.add(doc[:60])
                        hints.append(doc)
        
        # Look up task_workflows (methodology-level guidance)
        if hasattr(self.rag, 'collections') and 'task_workflows' in self.rag.collections:
            col = self.rag.collections['task_workflows']
            if col.count() > 0:
                results = col.query(query_texts=[error_query], n_results=1)
                if results['documents'][0]:
                    doc = results['documents'][0][0]
                    # pitfalls 
                    if ' pitfalls' in doc:
                        pitfall_section = doc[doc.index(' pitfalls'):][:300]
                        if pitfall_section[:60] not in seen:
                            hints.append(pitfall_section)
                    elif 'pitfalls' in doc.lower():
                        idx = doc.lower().index('pitfalls')
                        pitfall_section = doc[idx:][:300]
                        if pitfall_section[:60] not in seen:
                            hints.append(pitfall_section)
        
        return '\n---\n'.join(hints) if hints else ""
    
    def _format_conversation(self, user_msg: str, history: list) -> str:
        """Flatten conversation history into a single user message (Ollama single-turn API)"""
        parts = [user_msg]
        
        for role, content in history:
            if role == "assistant":
                parts.append(f"\nAssistant:\n{content}")
            else:
                parts.append(f"\n{content}")
        
        # ReAct 
        parts.append("\nNow continue with the next step. Respond with Thought/Action/Args:")
        
        # Truncate over-long history (keep the most recent turns)
        full_text = "\n".join(parts)
        if len(full_text) > 16000:
            # Keep the leading task description plus the most recent turns (14 turns for 25 rounds)
            header = parts[0]
            recent = "\n".join(parts[-14:])
            full_text = header + "\n\n... (earlier steps omitted) ...\n\n" + recent
        
        return full_text

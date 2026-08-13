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

"""
The ReAct loop: think, act, observe, repeat.

A few decisions here are load-bearing and easy to undo by accident:

- Observations truncate stdout from the front and stderr from the tail. The
  real error is at the *end* of a stack trace, and the interesting output is
  usually at the start of a print run.
- Replies are parsed leniently — regex, then a JSON parse, then
  ast.literal_eval, then a few shape-specific rescues. A model that gets the
  format slightly wrong should be told, not silently dropped.
- The loop guards its own exits: it refuses to finish with nothing saved, it
  breaks out of repeated identical actions, and it saves whatever is in the
  sandbox if it runs out of rounds.
"""
import os
import re
import json
import time
import ast
from typing import Dict, Any, Tuple

from src.agent.sandbox import PythonSandbox
from src.agent.tools import GISToolkit
from src.agent.error_memory import ErrorMemory


DEFAULT_SYSTEM_PROMPT = """You are a GIS analyst agent. You solve geospatial analysis tasks by thinking step-by-step and using tools to interact with data.

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

Thought: Now extract band 5 (NIR).
Action: execute
Args: {{"code": "src = rasterio.open('dataset/image.tif')\\nnir = src.read(5)\\nprint(f'Band 5 shape: {{nir.shape}}')"}}

Thought: Analysis complete, results saved.
Action: finish
Args: {{"summary": "Analysis complete. Results saved to pred_results/"}}

## Basic Guidelines
1. Start by calling list_files to see what data files are available.
2. Load and inspect data (print columns, CRS, shape, head) before writing analysis code.
3. Save all outputs to pred_results/ directory before calling finish().
4. NEVER use plt.show(). Always use plt.savefig('pred_results/<filename>.png', dpi=150, bbox_inches='tight') then plt.close(). plt.show() does NOT save the figure.

## Available Packages (ONLY use these)
geopandas, rasterio, shapely, fiona, pyproj, rtree, numpy, pandas, scipy,
matplotlib, seaborn, sklearn, libpysal, esda, momepy, rasterstats, mapclassify,
networkx, osmnx, h3, openpyxl

NOT available (do NOT import): arcpy, pykrige, skimage, geoplot, cartopy, xarray, mgwr
Alternatives: pykrige -> scipy.interpolate.griddata; skimage -> scipy.ndimage
"""


class AgentResult:
    """What one run produced."""
    def __init__(self, success: bool = False,
                 code: str = "", history: list = None,
                 output_files: list = None, time_ms: float = 0,
                 total_rounds: int = 0, self_corrections: int = 0):
        self.success = success
        self.code = code
        self.history = history or []
        self.output_files = output_files or []
        self.time_ms = time_ms
        self.total_rounds = total_rounds
        # Times the agent hit an error and recovered by itself on the next
        # round. Surfaced in the run record, so keep counting it.
        self.self_corrections = self_corrections

    def to_dict(self):
        return {
            "success": self.success,
            "code_length": len(self.code),
            "output_files": self.output_files,
            "time_ms": round(self.time_ms, 1),
            "total_rounds": self.total_rounds,
            "self_corrections": self.self_corrections,
            "history_length": len(self.history),
        }


def _escape_newlines_in_strings(text: str) -> str:
    """Escape literal newlines and tabs that sit inside JSON string values.

    Writing a block of Python into a JSON string without escaping the line
    breaks is the single most common way a reply fails to parse — small
    open-weight models do it constantly. The text is otherwise valid JSON, so
    repairing just the control characters recovers the whole call.
    """
    out = []
    in_string = False
    escaped = False
    for ch in text:
        if not in_string:
            in_string = ch == '"'
            out.append(ch)
            continue
        if escaped:
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == '"':
            in_string = False
        elif ch in "\n\r\t":
            out.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}[ch])
            continue
        out.append(ch)
    return "".join(out)


def _parse_action(response_text: str) -> Tuple[str, str, Dict[str, Any]]:
    """Split a reply into (thought, action_name, args).

    Accepts double quotes, single quotes and bare key=value, and never raises:
    an unparseable reply comes back with an empty action so the caller can ask
    the model to try again.
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
    
    # Args — try progressively looser readings
    # Everything after "Args:" to the end of the reply. `$` must not be
    # multiline here: with MULTILINE it ends at the first line break, which
    # silently truncates every multi-line argument block to its first line.
    args_match = re.search(r'Args:\s*(.*)', response_text, re.DOTALL)
    if args_match:
        args_text = args_match.group(1).strip()
        
        # Stop at the next Thought/Action if the model ran on
        lines = args_text.split('\n')
        json_lines = []
        for line in lines:
            if line.strip().startswith('Thought:') or line.strip().startswith('Action:'):
                break
            json_lines.append(line)
        args_text = '\n'.join(json_lines).strip()
        
        # Strip markdown fencing
        args_text = re.sub(r'^```\w*\n?', '', args_text)
        args_text = re.sub(r'\n?```$', '', args_text)
        args_text = args_text.strip()
        
        if not args_text or args_text == '{}':
            args = {}
        else:
            # 1: plain JSON
            try:
                args = json.loads(args_text)
            except (json.JSONDecodeError, ValueError):
                # 2: pull the object out of surrounding prose
                json_match = re.search(r'\{.*\}', args_text, re.DOTALL)
                if json_match:
                    try:
                        args = json.loads(json_match.group())
                    except (json.JSONDecodeError, ValueError):
                        # 3: the same JSON with its line breaks escaped
                        try:
                            args = json.loads(
                                _escape_newlines_in_strings(json_match.group()))
                        except (json.JSONDecodeError, ValueError):
                            # 4: Python literal (single quotes, etc.)
                            try:
                                args = ast.literal_eval(json_match.group())
                            except (ValueError, SyntaxError):
                                pass
                
                # 5: bare key=value
                if not args and '=' in args_text:
                    for part in args_text.split(','):
                        if '=' in part:
                            k, v = part.split('=', 1)
                            args[k.strip()] = v.strip().strip('"').strip("'")
    
    # execute() with no code: the model often put it outside the Args block
    if action_name == "execute" and "code" not in args:
        # from a fenced code block
        code_match = re.search(r'```(?:python)?\n(.*?)```', response_text, re.DOTALL)
        if code_match:
            args = {"code": code_match.group(1).strip()}
        else:
            # or as plain text after "Args:"
            after_args = re.search(r'Args:\s*\{?\}?\s*\n(.*?)$', response_text, re.DOTALL)
            if after_args:
                raw_code = after_args.group(1).strip()
                # The opening brace of an unparseable Args block was consumed
                # above, so its closing one is still here and would be executed
                # as Python — a bare `}` that fails the whole round.
                while raw_code.endswith("}") and raw_code.count("}") > raw_code.count("{"):
                    raw_code = raw_code[:-1].rstrip()
                if raw_code and not raw_code.startswith('{') and len(raw_code) > 5:
                    args = {"code": raw_code}
    
    return thought, action_name, args


class GISReActAgent:
    """Runs one analysis task to completion."""

    def __init__(self, llm_engine, timeout: int = 60,
                 max_rounds: int = 25, verbose: bool = True,
                 error_memory: ErrorMemory = None,
                 system_prompt_builder=None, extra_tools: dict = None):
        """
        system_prompt_builder: optional callable taking the toolkit's tool
            descriptions and returning the system prompt. Lets a caller wrap
            the agent in its own prompt without editing this module.
        extra_tools: optional {name: (callable, description)} added to the
            toolkit for this agent, for tools the caller owns.
        """
        self.llm = llm_engine
        self.timeout = timeout
        self.max_rounds = max_rounds
        self.verbose = verbose
        self._loaded_crs = {}  # {var_name: crs_string}
        self.error_memory = error_memory or ErrorMemory()
        self.system_prompt_builder = system_prompt_builder
        self.extra_tools = extra_tools or {}
    
    def run(self, instruction: str, data_dir: str, work_dir: str,
            on_step=None) -> AgentResult:
        """Work through `instruction` against the data in `data_dir`.

        Everything happens under `work_dir`: `data_dir` is linked in as
        `dataset/`, results are written to `pred_results/`, and the code the
        agent ran ends up in `code.py`.

        on_step: optional callable invoked once per round, as it happens, for
        live display. It receives the round record plus `code` (the full
        snippet, for execute) and `observation_full` (untruncated). Exceptions
        raised by it are swallowed — display must never break a run.
        """
        t0 = time.time()

        def _emit(action_name=None, args=None, observation=None):
            """Push the round just recorded to the on_step callback."""
            if not on_step or not round_details:
                return
            ev = dict(round_details[-1])
            if action_name == "execute" and isinstance(args, dict) and args.get("code"):
                ev["code"] = args["code"]
            if observation is not None:
                ev["observation_full"] = observation if isinstance(observation, str) else str(observation)
            try:
                on_step(ev)
            except Exception:
                pass
        
        # Sandbox and tools
        os.makedirs(work_dir, exist_ok=True)
        pred_dir = os.path.join(work_dir, "pred_results")
        os.makedirs(pred_dir, exist_ok=True)
        
        # Expose the project's data as dataset/
        ds_link = os.path.join(work_dir, "dataset")
        if os.path.islink(ds_link):
            os.unlink(ds_link)
        if not os.path.exists(ds_link):
            os.symlink(os.path.abspath(data_dir), ds_link)
        
        sandbox = PythonSandbox(work_dir=work_dir, timeout=self.timeout)
        toolkit = GISToolkit(sandbox, data_dir="dataset")
        for name, (fn, desc) in self.extra_tools.items():
            toolkit.tools[name] = fn
            toolkit.tool_descriptions += desc

        if self.system_prompt_builder:
            system_prompt = self.system_prompt_builder(toolkit.tool_descriptions)
        else:
            system_prompt = DEFAULT_SYSTEM_PROMPT.format(
                tool_descriptions=toolkit.tool_descriptions)

        user_msg = f"Task: {instruction}\n"

        if self.verbose:
            print(f"  Agent starting (max {self.max_rounds} rounds)")

        history = []  # [(role, content), ...]
        round_details = []
        self_corrections = 0
        last_action_failed = False
        finished = False
        round_num = 0
        
        # Repetition detection, by tool signature and by code fingerprint.
        consecutive_same_tool = 0
        last_tool_fingerprint = ""
        self._recent_code_hashes = []
        
        for round_num in range(1, self.max_rounds + 1):
            # Two rounds left: tell it to save now.
            remaining = self.max_rounds - round_num
            if remaining == 2 and history:
                deadline_msg = (
                    "\n\n🚨 DEADLINE WARNING: You have only 2 rounds left! "
                    "You MUST save results NOW. Even if analysis is incomplete, "
                    "save whatever you have IMMEDIATELY:\n"
                    "1. If you have any plot: plt.savefig('pred_results/analysis.png', dpi=150, bbox_inches='tight')\n"
                    "2. If you have any DataFrame: df.to_csv('pred_results/results.csv')\n"
                    "3. If you have any GeoDataFrame: gdf.to_file('pred_results/output.geojson', driver='GeoJSON')\n"
                    "Do NOT continue complex analysis. SAVE IMMEDIATELY, then call finish()."
                )
                last_role, last_content = history[-1]
                history[-1] = (last_role, last_content + deadline_msg)
                if self.verbose:
                    print(f"\n  [!] deadline warning injected ({remaining} rounds left)")
            
            conversation_text = self._format_conversation(user_msg, history)
            
            gen_t0 = time.time()
            gen_kwargs = {}
            if getattr(self.llm, "supports_segmented_user", False):
                segments = self._split_conversation(user_msg, history)
                if segments:
                    gen_kwargs["user_segments"] = segments
            response = self.llm.generate(
                prompt="",
                system_prompt=system_prompt,
                user_message=conversation_text,
                max_tokens=2048,
                stop=["Observation:"],
                **gen_kwargs,
            )
            gen_time = (time.time() - gen_t0) * 1000
            
            raw_text = response.get("text", "") if isinstance(response, dict) else str(response)
            
            # A transport error is not a model reply; do not feed it to the loop.
            if raw_text.startswith("Error during API call") or raw_text.startswith("Error during"):
                if not hasattr(self, '_api_error_count'):
                    self._api_error_count = 0
                self._api_error_count += 1
                if self.verbose:
                    err_short = raw_text[:100].replace('\n', ' ')
                    print(f"\n  [{round_num:>2}] API error (#{self._api_error_count}): {err_short}")
                if self._api_error_count >= 3:
                    if self.verbose:
                        print(f"       {self._api_error_count} API errors in a row, giving up")
                    break
                # Could be a passing rate limit — back off once and retry.
                time.sleep(5)
                continue
            
            thought, action_name, args = _parse_action(raw_text)
            
            if self.verbose:
                thought_short = thought[:80] + "..." if len(thought) > 80 else thought
                print(f"\n  [{round_num:>2}] 💭 {thought_short}")
            
            if not action_name:
                # Unparseable reply: ask for the format again, but not forever.
                if not hasattr(self, '_consecutive_parse_fails'):
                    self._consecutive_parse_fails = 0
                self._consecutive_parse_fails += 1
                if self._consecutive_parse_fails >= 10:
                    if self.verbose:
                        print(f"       {self._consecutive_parse_fails} unparseable replies in a row, giving up")
                    break
                
                observation = ("⚠️ Format error: Could not parse Action. Please respond strictly in this format:\n"
                              "Thought: <your reasoning>\n"
                              "Action: <tool_name>\n"
                              "Args: {<JSON arguments>}")
                history.append(("assistant", raw_text))
                history.append(("user", f"Observation: {observation}"))
                if self.verbose:
                    print("       format parse failed, asking the model to correct it")
                continue
            else:
                self._consecutive_parse_fails = 0
                self._api_error_count = 0
            
            if action_name == "finish":
                # Results written to the work dir instead of pred_results/.
                pred_dir_check = os.path.join(work_dir, "pred_results")
                result_exts = ('.png', '.csv', '.geojson', '.tif', '.shp', '.dbf', '.prj', '.shx', '.cpg')
                for f in os.listdir(work_dir):
                    if f.endswith(result_exts) and not f.startswith('.'):
                        src_f = os.path.join(work_dir, f)
                        dst_f = os.path.join(pred_dir_check, f)
                        if os.path.isfile(src_f) and not os.path.exists(dst_f):
                            import shutil
                            shutil.move(src_f, dst_f)
                            if self.verbose:
                                print(f"       📁 Auto-moved: {f} → pred_results/")

                # Refuse to finish with nothing saved.
                has_outputs = False
                if os.path.exists(pred_dir_check):
                    has_outputs = len([f for f in os.listdir(pred_dir_check) if not f.startswith('.')]) > 0
                
                if not has_outputs and getattr(self, '_output_guard_retries', 0) < 3:
                    self._output_guard_retries = getattr(self, '_output_guard_retries', 0) + 1
                    observation = (
                        "⚠️ Output Guard: pred_results/ directory is empty! Your analysis has not saved any result files yet.\n"
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
                        print(f"       🛡️ Output Guard: pred_results empty (hint {self._output_guard_retries}/3)")
                    continue
                
                # Files exist, but a blank image or a header-only CSV is not a result.
                validation_issues = self._validate_outputs(work_dir)
                if validation_issues and getattr(self, '_finish_retries', 0) < 2:
                    self._finish_retries = getattr(self, '_finish_retries', 0) + 1
                    observation = (
                        f"⚠️ Output validation failed, please fix and call finish() again:\n{validation_issues}\n\n"
                        "Please check the above issues and regenerate output files, ensuring:\n"
                        "1. Images are not blank (have actual data rendered)\n"
                        "2. Data files have content (rows > 0)\n"
                        "3. If matplotlib image, check ax.collections/ax.patches is not empty"
                    )
                    history.append(("assistant", raw_text))
                    history.append(("user", f"Observation: {observation}"))
                    if self.verbose:
                        print(f"       ⚠️ Output validation failed (retry {self._finish_retries}/2): {validation_issues[:80]}")
                    continue
                
                observation = toolkit.run("finish", args)
                finished = True
                if self.verbose:
                    print(f"       🏁 finish: {args.get('summary', '')[:60]}")
                
                round_details.append({
                    "round": round_num,
                    "thought": thought[:200],
                    "action": action_name,
                    "success": True,
                    "gen_time_ms": gen_time,
                })
                _emit(action_name, args, observation)
                break
            
            # Repetition: same tool and arguments, round after round.
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
            
            # Repetition: the same code coming back within a short window.
            code_dedup_triggered = False
            if action_name == "execute" and isinstance(args, dict) and "code" in args:
                # Normalise away blank lines, indentation and comments.
                raw_code = args['code']
                norm_lines = [l.strip() for l in raw_code.split('\n')
                              if l.strip() and not l.strip().startswith('#')]
                code_hash = hash('\n'.join(norm_lines))
                
                if not hasattr(self, '_recent_code_hashes'):
                    self._recent_code_hashes = []
                
                dup_count = self._recent_code_hashes.count(code_hash)
                self._recent_code_hashes.append(code_hash)
                if len(self._recent_code_hashes) > 6:
                    self._recent_code_hashes = self._recent_code_hashes[-6:]
                
                if dup_count >= 2:
                    # Third time: stop it.
                    code_dedup_triggered = True
                    if self.verbose:
                        print(f"       🛑 Code dedup: identical code submitted {dup_count+1} times, force-terminating")
                    observation = toolkit.run("finish", {"summary": f"System auto-terminated: identical code submitted {dup_count+1} times. Intermediate results in pred_results/"})
                    finished = len([f for f in os.listdir(os.path.join(work_dir, "pred_results")) if not f.startswith('.')]) > 0 if os.path.exists(os.path.join(work_dir, "pred_results")) else False
                    round_details.append({
                        "round": round_num,
                        "thought": f"(code dedup: identical code x{dup_count+1})",
                        "action": "finish",
                        "success": finished,
                        "gen_time_ms": gen_time,
                    })
                    _emit("finish", None, observation)
                    break
                elif dup_count == 1:
                    # Second time: remind it where it already got to.
                    code_dedup_triggered = True
                    step_summary = ', '.join(f"R{rd['round']}:{rd['action']}" for rd in round_details[-5:])
                    
                    observation = (
                        f"⚠️ CODE DUPLICATE DETECTED: You submitted nearly identical code as a previous round. "
                        f"This is the 2nd time — do NOT submit it again or the system will terminate.\n\n"
                        f"=== Your recent actions ===\n  {step_summary}\n\n"
                        f"=== What you must do ===\n"
                        f"You have ALREADY completed this step successfully. Move on to the NEXT step in your plan.\n"
                        f"Review the task instruction and your plan, then write code for the next uncompleted step."
                    )
                    history.append(("assistant", raw_text))
                    history.append(("user", f"Observation: {observation}"))
                    if self.verbose:
                        print("       duplicate code (2nd time), progress reminder injected")
                    continue
            
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
                        f"⚠️ System interrupt: You have called {action_name} {consecutive_same_tool+1} times consecutively. This is ineffective.\n\n"
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
                        print(f"       🛑 {consecutive_same_tool+1} consecutive repeats, system force-terminating")
                    # Execute finish and break
                    observation = toolkit.run("finish", {"summary": "Stopped automatically after repeating the same action. Intermediate results are in pred_results/"})
                    finished = len([f for f in os.listdir(os.path.join(work_dir, "pred_results")) if not f.startswith('.')]) > 0 if os.path.exists(os.path.join(work_dir, "pred_results")) else False
                    round_details.append({
                        "round": round_num,
                        "thought": "(system force-terminated)",
                        "action": "finish",
                        "success": finished,
                        "gen_time_ms": gen_time,
                    })
                    _emit("finish", None, observation)
                    break
                
                history.append(("assistant", raw_text))
                history.append(("user", f"Observation: {observation}"))
                if self.verbose:
                    print(f"       ⚠️ {consecutive_same_tool+1} consecutive same calls, forced interrupt")
                continue
            
            # Malformed execute args, e.g. {'{"code": "..."': 'value'} —
            # reassemble the code from whatever came through.
            if action_name == "execute" and "code" not in args and args:
                all_text = ' '.join(str(k) + ' ' + str(v) for k, v in args.items())
                code_extract = re.search(r'"code"\s*:\s*"(.*?)"', all_text, re.DOTALL)
                if code_extract:
                    args = {"code": code_extract.group(1).replace('\\n', '\n')}
                else:
                    code_parts = []
                    for k, v in args.items():
                        code_parts.append(str(k).strip('"\'{} '))
                        if v and str(v).strip() != k:
                            code_parts.append(str(v).strip('"\'{} '))
                    combined = '\n'.join(p for p in code_parts if p)
                    if len(combined) > 10:
                        args = {"code": combined}
                if self.verbose and "code" in args:
                    print("       recovered code from malformed args")
            
            if self.verbose:
                args_short = str(args)[:60]
                print(f"       🔧 {action_name}({args_short})")
            
            observation = toolkit.run(action_name, args)

            # Track the CRS of everything loaded, and say so when they diverge.
            if action_name in ('load_vector', 'load_raster'):
                crs_match = re.search(r'CRS:\s*(\S+)', observation)
                var_name = args.get('var_name', '')
                if crs_match and var_name:
                    crs_val = crs_match.group(1)
                    self._loaded_crs[var_name] = crs_val
                    unique_crs = set(v for v in self._loaded_crs.values() if v and v != 'None')
                    if len(unique_crs) > 1:
                        crs_list = ', '.join(f"{k}={v}" for k, v in self._loaded_crs.items() if v and v != 'None')
                        observation += (
                            f"\n\n⚠️ CRS mismatch warning! Currently loaded dataset CRS:\n  {crs_list}\n"
                            "Before spatial operations (sjoin/clip/mask), you MUST first unify CRS using .to_crs() or rasterio.warp.reproject()!"
                        )
            
            # An error followed by a success is a self-correction.
            action_success = "❌" not in observation[:10]
            if last_action_failed and action_success:
                self_corrections += 1
                if self.verbose:
                    print(f"       self-correction (#{self_corrections})")
            last_action_failed = not action_success
            
            if self.verbose:
                obs_lines = observation.strip().split("\n")
                for line in obs_lines[:4]:
                    print(f"       > {line[:80]}")
                if len(obs_lines) > 4:
                    print(f"       > ... ({len(obs_lines)} lines)")
            
            # On failure, offer whatever this run already learned about it.
            if not action_success:
                mem_hint = self.error_memory.lookup(observation)
                if mem_hint:
                    observation += f"\n\nHint from an earlier error in this run:\n{mem_hint}"
                    if self.verbose:
                        print("       error-memory hit")

            # And remember the fix that worked.
            if last_action_failed and action_success and round_num > 1:
                prev_obs = history[-2][1] if len(history) >= 2 else ""
                if "❌" in prev_obs:
                    error_key = prev_obs.split('\n')[0][:80]
                    fix_key = observation.split('\n')[0][:80]
                    self.error_memory.record(error_key, f"Fix method: {fix_key}")
            
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
            _emit(action_name, args, observation)

        # Out of rounds: save whatever is in the sandbox rather than nothing.
        if not finished:
            emergency_code = (
                "import os\n"
                "os.makedirs('pred_results', exist_ok=True)\n"
                "_saved = []\n"
                "# current matplotlib figure\n"
                "try:\n"
                "    import matplotlib.pyplot as plt\n"
                "    if plt.get_fignums():\n"
                "        plt.savefig('pred_results/emergency_plot.png', dpi=150, bbox_inches='tight')\n"
                "        _saved.append('emergency_plot.png')\n"
                "except: pass\n"
                "# largest GeoDataFrame\n"
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
                "# largest DataFrame\n"
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
                    print(f"\n  emergency save: {getattr(emg_result, 'stdout', '')[:100]}")
            except Exception as e:
                if self.verbose:
                    print(f"\n  emergency save failed: {e}")
        
        output_files = []
        for d in ["pred_results", "output"]:
            out_dir = os.path.join(work_dir, d)
            if os.path.exists(out_dir):
                for fn in os.listdir(out_dir):
                    if not fn.startswith('.'):
                        output_files.append(os.path.join(d, fn))
        
        # Everything the agent ran, as one script.
        full_code = sandbox.get_full_code()
        with open(os.path.join(work_dir, "code.py"), "w") as f:
            f.write(full_code)

        # An emergency save is not the agent finishing the job.
        emergency_names = {'emergency_plot.png', 'emergency_output.geojson', 'emergency_output.csv'}
        real_output_files = [f for f in output_files
                             if os.path.basename(f) not in emergency_names]
        
        success = finished and len(real_output_files) > 0
        if not finished and len(real_output_files) > 0:
            success = True   # ran out of rounds, but produced real results
        total_time = (time.time() - t0) * 1000
        
        if self.verbose:
            marker = "ok" if success else "failed"
            print(f"\n  {marker}: {round_num} rounds | "
                  f"{len(output_files)} outputs | "
                  f"self-corrections {self_corrections} | "
                  f"{total_time/1000:.1f}s")
        
        return AgentResult(
            success=success,
            code=full_code,
            history=round_details,
            output_files=output_files,
            time_ms=total_time,
            total_rounds=round_num,
            self_corrections=self_corrections,
        )
    
    def _validate_outputs(self, work_dir: str) -> str:
        """Report anything in pred_results/ that looks empty. '' if all fine."""
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
            
            # A blank figure is almost always under 5 KB.
            if ext in ('png', 'jpg', 'jpeg'):
                if size < 5000:
                    issues.append(f"- {fn}: File is only {size} bytes, may be a blank image. Please check that plot data is correctly rendered.")
            
            elif ext == 'csv':
                try:
                    with open(fpath) as f:
                        lines = sum(1 for _ in f)
                    if lines <= 1:
                        issues.append(f"- {fn}: Only {lines} line(s) (may be header only), please ensure data is present.")
                except:
                    pass
            
            # An empty .dbf means no attributes came through.
            elif ext == 'shp':
                dbf = fpath.replace('.shp', '.dbf')
                if os.path.exists(dbf) and os.path.getsize(dbf) < 100:
                    issues.append(f"- {fn}: .dbf file is extremely small, may have no attribute data.")
        
        return '\n'.join(issues)
    
    def _split_conversation(self, user_msg: str, history: list):
        """Same text as _format_conversation, split for prompt caching.

        Everything except the newest exchange is byte-for-byte what the previous
        round already sent, so it can be cached and read back instead of paid
        for again. The newest exchange stays outside the cached span because the
        deadline warning rewrites history[-1] in place.

        Returns None when there is nothing settled yet, or when the transcript
        is long enough that _format_conversation would truncate it — the
        truncated form is not a growing prefix, so it cannot be cached.
        """
        parts = [user_msg]
        for role, content in history:
            if role == "assistant":
                parts.append(f"\nAssistant:\n{content}")
            else:
                parts.append(f"\n{content}")
        parts.append("\nNow continue with the next step. Respond with Thought/Action/Args:")

        if len("\n".join(parts)) > 16000 or len(parts) < 4:
            return None
        # stable + volatile must re-join to exactly what _format_conversation
        # produces, or the model sees a different prompt than it used to.
        stable = "\n".join(parts[:-2])
        volatile = "\n" + "\n".join(parts[-2:])
        return [stable, volatile]

    def _format_conversation(self, user_msg: str, history: list) -> str:
        """Flatten the exchange into a single user message."""
        parts = [user_msg]
        
        for role, content in history:
            if role == "assistant":
                parts.append(f"\nAssistant:\n{content}")
            else:
                parts.append(f"\n{content}")
        
        parts.append("\nNow continue with the next step. Respond with Thought/Action/Args:")
        
        # Keep the task and the most recent rounds when it gets long.
        full_text = "\n".join(parts)
        if len(full_text) > 16000:
            header = parts[0]
            recent = "\n".join(parts[-14:])
            full_text = header + "\n\n... (earlier steps omitted) ...\n\n" + recent
        
        return full_text

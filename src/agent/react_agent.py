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
ReAct 循环引擎 — GIS Agent 核心

设计要点（来自用户反馈）:
A. Observation 截断: stdout 前截，stderr 尾截
B. JSON 解析健壮性: 正则 + ast.literal_eval 回退 + 格式错误反馈给模型
C. 工具 load_vector/load_raster 自动注入 namespace
D. Prompt 强调不覆盖原始变量

新增统计: Self-Correction Rate（论文指标）
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
    """Agent 执行结果"""
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
        self.self_corrections = self_corrections  # 论文关键指标
    
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
    """解析 LLM 输出为 (thought, action_name, args)
    
    健壮性设计（风险点 B）:
    - 支持多种 JSON 格式（双引号/单引号/无引号）
    - 解析失败时不崩溃，返回错误提示让模型纠正
    """
    thought = ""
    action_name = ""
    args = {}
    
    # 提取 Thought
    thought_match = re.search(r'Thought:\s*(.*?)(?=\nAction:|\Z)', response_text, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()
    
    # 提取 Action
    action_match = re.search(r'Action:\s*(\w+)', response_text)
    if action_match:
        action_name = action_match.group(1).strip()
    
    # 提取 Args — 多级回退策略
    args_match = re.search(r'Args:\s*(.*?)$', response_text, re.DOTALL | re.MULTILINE)
    if args_match:
        args_text = args_match.group(1).strip()
        
        # 如果还有后续行，只取到下一个空行或 Thought 之前
        lines = args_text.split('\n')
        json_lines = []
        for line in lines:
            if line.strip().startswith('Thought:') or line.strip().startswith('Action:'):
                break
            json_lines.append(line)
        args_text = '\n'.join(json_lines).strip()
        
        # 去掉可能的 markdown 代码标记
        args_text = re.sub(r'^```\w*\n?', '', args_text)
        args_text = re.sub(r'\n?```$', '', args_text)
        args_text = args_text.strip()
        
        if not args_text or args_text == '{}':
            args = {}
        else:
            # 尝试 1: 标准 json.loads
            try:
                args = json.loads(args_text)
            except (json.JSONDecodeError, ValueError):
                # 尝试 2: 用正则提取 JSON 对象
                json_match = re.search(r'\{.*\}', args_text, re.DOTALL)
                if json_match:
                    try:
                        args = json.loads(json_match.group())
                    except (json.JSONDecodeError, ValueError):
                        # 尝试 3: ast.literal_eval（支持单引号等）
                        try:
                            args = ast.literal_eval(json_match.group())
                        except (ValueError, SyntaxError):
                            pass
                
                # 尝试 4: 简单 key=value 格式
                if not args and '=' in args_text:
                    for part in args_text.split(','):
                        if '=' in part:
                            k, v = part.split('=', 1)
                            args[k.strip()] = v.strip().strip('"').strip("'")
    
    # 修复: execute 工具参数为空时，尝试从响应中提取代码
    if action_name == "execute" and "code" not in args:
        # 回退 1: 从 markdown 代码块提取
        code_match = re.search(r'```(?:python)?\n(.*?)```', response_text, re.DOTALL)
        if code_match:
            args = {"code": code_match.group(1).strip()}
        else:
            # 回退 2: Args: 后的非 JSON 文本当作代码
            after_args = re.search(r'Args:\s*\{?\}?\s*\n(.*?)$', response_text, re.DOTALL)
            if after_args:
                raw_code = after_args.group(1).strip()
                if raw_code and not raw_code.startswith('{') and len(raw_code) > 5:
                    args = {"code": raw_code}
    
    return thought, action_name, args


class GISReActAgent:
    """GIS 领域 ReAct Agent"""
    
    def __init__(self, llm_engine, timeout: int = 60,
                 max_rounds: int = 25, verbose: bool = True,
                 rag=None, error_memory: ErrorMemory = None):
        self.llm = llm_engine
        self.timeout = timeout
        self.max_rounds = max_rounds
        self.verbose = verbose
        self.rag = rag
        self._loaded_crs = {}  # CRS 跟踪: {var_name: crs_string}
        self.error_memory = error_memory or ErrorMemory()
    
    def run(self, task_id: int, instruction: str,
            workflow: str, data_dir: str, work_dir: str,
            domain_knowledge: str = "",
            dataset_description: str = "",
            rag_context: str = "",
            skill_text: str = "",
            on_step=None) -> AgentResult:
        """执行 GIS 分析任务

        v2 架构变更:
        - System Prompt 不再注入 RAG/Skill/Workflow
        - Workflow 移入 User Message
        - RAG 通过 search_docs 工具按需检索

        on_step: 可选回调 Callable[[dict], None]，每轮 round_details 落定后实时触发，
        用于前端逐轮流式显示 Thought/Action/Observation。默认 None，完全向后兼容。
        事件 dict 在 round_details 基础上增补 `code`(execute 的完整代码) 与
        `observation_full`(未截断的观察结果)。回调异常被吞掉，绝不影响 Agent 主流程。
        """
        t0 = time.time()

        def _emit(action_name=None, args=None, observation=None):
            """把刚刚 append 的 round_details[-1] 增补后推给 on_step 回调。"""
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
        
        # 1. 初始化沙箱和工具
        os.makedirs(work_dir, exist_ok=True)
        pred_dir = os.path.join(work_dir, "pred_results")
        os.makedirs(pred_dir, exist_ok=True)
        
        # 创建 dataset 符号链接
        ds_link = os.path.join(work_dir, "dataset")
        if os.path.islink(ds_link):
            os.unlink(ds_link)
        if not os.path.exists(ds_link):
            os.symlink(os.path.abspath(data_dir), ds_link)
        
        sandbox = PythonSandbox(work_dir=work_dir, timeout=self.timeout)
        toolkit = GISToolkit(sandbox, data_dir="dataset")
        
        # 2. 构建 System Prompt — v2: 纯净版，不注入 RAG/Skill/Workflow
        # search_docs 工具描述追加到 tool_descriptions
        search_docs_desc = (
            "\n\nsearch_docs: Query the knowledge base. Use when unsure about a library's API or a GIS methodology."
            "\n  Args: {\"query\": \"what to look up, e.g. 'rasterio read specific band' or 'NBR formula Landsat 8'\"}"
            "\n  Returns: relevant knowledge documents"
        )
        system_prompt = build_system_prompt(
            tool_descriptions=toolkit.tool_descriptions + search_docs_desc,
        )
        
        # 3. 构建 User Message — v2: 包含任务信息 + Workflow + DD + DK
        user_msg = f"Task: {instruction}\n"
        if workflow:
            user_msg += f"\n## Analysis Steps\n{workflow}\n"
        if dataset_description:
            user_msg += f"\n## Dataset Description\n{dataset_description}\n"
        if domain_knowledge:
            user_msg += f"\n## Domain Knowledge\n{domain_knowledge}\n"
        
        if self.verbose:
            print(f"  🤖 ReAct Agent 启动 (max {self.max_rounds} rounds)")
        
        # 4. ReAct 循环
        history = []  # [(role, content), ...]
        round_details = []
        self_corrections = 0  # 自修复次数
        last_action_failed = False
        finished = False
        round_num = 0
        
        # 修复: 重复工具调用检测（改进：基于代码指纹）
        consecutive_same_tool = 0
        last_tool_fingerprint = ""
        self._recent_code_hashes = []  # 代码去重：滑动窗口
        
        for round_num in range(1, self.max_rounds + 1):
            # ===== 改动1: Deadline Panic-Save =====
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
                # 注入到最后一条 history observation 中
                last_role, last_content = history[-1]
                history[-1] = (last_role, last_content + deadline_msg)
                if self.verbose:
                    print(f"\n  [!] 🚨 Deadline Warning 注入 (剩余 {remaining} 轮)")
            
            # 构建对话文本
            conversation_text = self._format_conversation(user_msg, history)
            
            # LLM 生成
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
            
            # API 错误检测：429/500 等错误不应进入 ReAct 循环
            if raw_text.startswith("Error during API call") or raw_text.startswith("Error during"):
                if not hasattr(self, '_api_error_count'):
                    self._api_error_count = 0
                self._api_error_count += 1
                if self.verbose:
                    err_short = raw_text[:100].replace('\n', ' ')
                    print(f"\n  [{round_num:>2}] 🚫 API 错误 (#{self._api_error_count}): {err_short}")
                if self._api_error_count >= 3:
                    if self.verbose:
                        print(f"       💀 连续 {self._api_error_count} 次 API 错误，终止任务")
                    break
                # 等待 5 秒后重试（可能是临时 rate limit）
                time.sleep(5)
                continue
            
            # 解析 Thought + Action
            thought, action_name, args = _parse_action(raw_text)
            
            if self.verbose:
                thought_short = thought[:80] + "..." if len(thought) > 80 else thought
                print(f"\n  [{round_num:>2}] 💭 {thought_short}")
            
            # 检查是否格式解析失败（风险点 B）
            if not action_name:
                # 连续 format parse fail 检测：超过 10 次连续失败则终止
                if not hasattr(self, '_consecutive_parse_fails'):
                    self._consecutive_parse_fails = 0
                self._consecutive_parse_fails += 1
                if self._consecutive_parse_fails >= 10:
                    if self.verbose:
                        print(f"       💀 连续 {self._consecutive_parse_fails} 次解析失败，终止任务")
                    break
                
                observation = ("⚠️ Format error: Could not parse Action. Please respond strictly in this format:\n"
                              "Thought: <your reasoning>\n"
                              "Action: <tool_name>\n"
                              "Args: {<JSON arguments>}")
                history.append(("assistant", raw_text))
                history.append(("user", f"Observation: {observation}"))
                if self.verbose:
                    print(f"       ⚠️ Format parse failed, asking model to correct")
                continue
            else:
                # 成功解析则重置计数器
                self._consecutive_parse_fails = 0
                self._api_error_count = 0
            
            # 检查是否 finish
            if action_name == "finish":
                # 自动搬文件: Agent 可能把结果存到了根目录而非 pred_results/
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
                                print(f"       📁 Auto-moved: {f} → pred_results/")

                # Output Guard: finish 前检查 pred_results 是否为空
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
                
                # P0-B: 输出验证机制 — finish 前检查输出质量（非空但可能空白）
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
            
            # ===== 代码去重检测 =====
            # 1) 基于工具名+参数的连续重复检测（旧逻辑保留）
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
            
            # 2) 基于全代码内容 hash 的滑动窗口去重（解决 T25 类语义重复）
            code_dedup_triggered = False
            if action_name == "execute" and isinstance(args, dict) and "code" in args:
                # 规范化代码：去空行、去首尾空格、去注释行
                raw_code = args['code']
                norm_lines = [l.strip() for l in raw_code.split('\n')
                              if l.strip() and not l.strip().startswith('#')]
                code_hash = hash('\n'.join(norm_lines))
                
                # 检查近期 execute 窗口（最近 6 次）
                if not hasattr(self, '_recent_code_hashes'):
                    self._recent_code_hashes = []
                
                dup_count = self._recent_code_hashes.count(code_hash)
                self._recent_code_hashes.append(code_hash)
                # 保持窗口大小为 6
                if len(self._recent_code_hashes) > 6:
                    self._recent_code_hashes = self._recent_code_hashes[-6:]
                
                if dup_count >= 2:
                    # 第 3 次提交相同代码 → 强制终止
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
                    # 第 2 次提交相同代码 → 进度提醒
                    code_dedup_triggered = True
                    # 收集已完成步骤信息
                    completed_steps = [rd['action'] for rd in round_details if rd.get('success')]
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
                        print(f"       ⚠️ Code dedup: 2nd identical submission, progress reminder injected")
                    continue
            
            # 3) 连续工具名重复（旧逻辑）
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
                    observation = toolkit.run("finish", {"summary": f"System auto-terminated due to consecutive repeated operations. Intermediate results in pred_results/"})
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
            
            # JSON 参数修复器 — 修复 execute 工具的畸形参数
            if action_name == "execute" and "code" not in args and args:
                # 场景: 模型输出了 {'{"code": "..."': 'value'} 这种畸形 JSON
                # 尝试从 args 的 keys/values 中拼凑出代码
                all_text = ' '.join(str(k) + ' ' + str(v) for k, v in args.items())
                # 提取 "code" 后面的内容
                code_extract = re.search(r'"code"\s*:\s*"(.*?)"', all_text, re.DOTALL)
                if code_extract:
                    args = {"code": code_extract.group(1).replace('\\n', '\n')}
                else:
                    # 把所有值拼在一起当代码
                    code_parts = []
                    for k, v in args.items():
                        code_parts.append(str(k).strip('"\'{} '))
                        if v and str(v).strip() != k:
                            code_parts.append(str(v).strip('"\'{} '))
                    combined = '\n'.join(p for p in code_parts if p)
                    if len(combined) > 10:
                        args = {"code": combined}
                if self.verbose and "code" in args:
                    print(f"       🔧 JSON fix: extracted code from malformed args")
            
            # 执行工具
            if self.verbose:
                args_short = str(args)[:60]
                print(f"       🔧 {action_name}({args_short})")
            
            # v2: search_docs 工具 — 按需检索 RAG
            if action_name == "search_docs":
                query = args.get("query", "") if isinstance(args, dict) else str(args)
                if self.rag and query:
                    observation = self.rag.format_context(query, n=3)
                    if observation:
                        observation = f"📚 Search results:\n{observation}"
                    else:
                        observation = "📚 No relevant knowledge found. Try other keywords, or continue with your existing knowledge."
                else:
                    observation = "📚 Knowledge base unavailable. Continue analysis with your existing knowledge."
                if self.verbose:
                    obs_preview = observation[:100].replace('\n', ' ')
                    print(f"       📚 search_docs: {obs_preview}...")
            else:
                observation = toolkit.run(action_name, args)
            
            # CRS 跟踪：从 observation 中提取 CRS 信息
            if action_name in ('load_vector', 'load_raster'):
                crs_match = re.search(r'CRS:\s*(\S+)', observation)
                var_name = args.get('var_name', '')
                if crs_match and var_name:
                    crs_val = crs_match.group(1)
                    self._loaded_crs[var_name] = crs_val
                    # 检查是否与已有数据集 CRS 不一致
                    unique_crs = set(v for v in self._loaded_crs.values() if v and v != 'None')
                    if len(unique_crs) > 1:
                        crs_list = ', '.join(f"{k}={v}" for k, v in self._loaded_crs.items() if v and v != 'None')
                        observation += (
                            f"\n\n⚠️ CRS mismatch warning! Currently loaded dataset CRS:\n  {crs_list}\n"
                            "Before spatial operations (sjoin/clip/mask), you MUST first unify CRS using .to_crs() or rasterio.warp.reproject()!"
                        )
            
            # 检测自修复（论文关键指标）
            action_success = "❌" not in observation[:10]
            if last_action_failed and action_success:
                self_corrections += 1
                if self.verbose:
                    print(f"       🔄 Self-correction success! (#{self_corrections})")
            last_action_failed = not action_success
            
            if self.verbose:
                obs_lines = observation.strip().split("\n")
                for line in obs_lines[:4]:
                    print(f"       > {line[:80]}")
                if len(obs_lines) > 4:
                    print(f"       > ... ({len(obs_lines)} lines)")
            
            # 错误触发 Error Memory + RAG 二次检索
            if not action_success:
                # 先查 Error Memory（跨题经验）
                mem_hint = self.error_memory.lookup(observation)
                if mem_hint:
                    observation += f"\n\n🧠 Cross-task experience hint:\n{mem_hint}"
                    if self.verbose:
                        print(f"       🧠 Error Memory hit")
                # 再查 RAG
                if self.rag:
                    rag_hint = self._error_triggered_rag(observation)
                    if rag_hint:
                        observation += f"\n\n💡 Relevant reference knowledge:\n{rag_hint}"
                        if self.verbose:
                            print(f"       📚 RAG secondary retrieval hit")
            
            # 自修复成功时记录到 Error Memory
            if last_action_failed and action_success and round_num > 1:
                # 从上一轮的 observation 提取错误关键词
                prev_obs = history[-2][1] if len(history) >= 2 else ""
                if "❌" in prev_obs:
                    error_key = prev_obs.split('\n')[0][:80]
                    fix_key = observation.split('\n')[0][:80]
                    self.error_memory.record(error_key, f"Fix method: {fix_key}")
            
            # 记录历史
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

        # ===== 改动3: 循环耗尽时自动保存中间结果 =====
        if not finished:
            emergency_code = (
                "import os\n"
                "os.makedirs('pred_results', exist_ok=True)\n"
                "_saved = []\n"
                "# 1) 保存当前 matplotlib figure\n"
                "try:\n"
                "    import matplotlib.pyplot as plt\n"
                "    if plt.get_fignums():\n"
                "        plt.savefig('pred_results/emergency_plot.png', dpi=150, bbox_inches='tight')\n"
                "        _saved.append('emergency_plot.png')\n"
                "except: pass\n"
                "# 2) 保存 sandbox 中最大的 GeoDataFrame\n"
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
                "# 3) 保存 sandbox 中最大的 DataFrame\n"
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
                    print(f"\n  💾 Emergency Save completed: {getattr(emg_result, 'stdout', '')[:100]}")
            except Exception as e:
                if self.verbose:
                    print(f"\n  ⚠️ Emergency Save failed: {e}")
        
        # 5. 收集结果
        output_files = []
        for d in ["pred_results", "output"]:
            out_dir = os.path.join(work_dir, d)
            if os.path.exists(out_dir):
                for fn in os.listdir(out_dir):
                    if not fn.startswith('.'):
                        output_files.append(os.path.join(d, fn))
        
        # 保存完整脚本
        script_path = os.path.join(work_dir, f"_react_t{task_id}.py")
        full_code = sandbox.get_full_code()
        with open(script_path, "w") as f:
            f.write(full_code)
        
        # 区分 Agent 主动产出 vs Emergency Save 产出
        emergency_names = {'emergency_plot.png', 'emergency_output.geojson', 'emergency_output.csv'}
        real_output_files = [f for f in output_files
                             if os.path.basename(f) not in emergency_names]
        
        success = finished and len(real_output_files) > 0
        # Emergency save 不算成功（避免假阳性）
        if not finished and len(real_output_files) > 0:
            success = True  # Agent 未 finish 但有非 emergency 产出
        total_time = (time.time() - t0) * 1000
        
        if self.verbose:
            marker = "✅" if success else "❌"
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
        """验证 pred_results/ 中输出文件的质量，返回问题描述或空串"""
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
            
            # PNG/JPG: 空白图通常 < 5KB
            if ext in ('png', 'jpg', 'jpeg'):
                if size < 5000:
                    issues.append(f"- {fn}: File is only {size} bytes, may be a blank image. Please check that plot data is correctly rendered.")
            
            # CSV: 检查行数
            elif ext == 'csv':
                try:
                    with open(fpath) as f:
                        lines = sum(1 for _ in f)
                    if lines <= 1:
                        issues.append(f"- {fn}: Only {lines} line(s) (may be header only), please ensure data is present.")
                except:
                    pass
            
            # SHP: 检查 .dbf 伴随文件大小
            elif ext == 'shp':
                dbf = fpath.replace('.shp', '.dbf')
                if os.path.exists(dbf) and os.path.getsize(dbf) < 100:
                    issues.append(f"- {fn}: .dbf file is extremely small, may have no attribute data.")
        
        return '\n'.join(issues)
    
    def _error_triggered_rag(self, observation: str) -> str:
        """出错时用 stderr 做 RAG 二次检索，返回最相关的修复建议"""
        if not self.rag or not hasattr(self.rag, 'collections'):
            return ""
        
        # 提取错误关键信息（取 stderr 最后几行）
        lines = observation.strip().split('\n')
        error_lines = [l for l in lines[-5:] if l.strip() and not l.startswith('>')]
        if not error_lines:
            error_lines = lines[-3:]
        error_query = ' '.join(error_lines)[:200]
        
        if not error_query.strip():
            return ""
        
        hints = []
        seen = set()
        
        # 查 api_pitfalls（最精准的错误修复）
        if hasattr(self.rag, 'collections') and 'api_pitfalls' in self.rag.collections:
            col = self.rag.collections['api_pitfalls']
            if col.count() > 0:
                results = col.query(query_texts=[error_query], n_results=1)
                if results['documents'][0]:
                    doc = results['documents'][0][0][:300]
                    if doc[:60] not in seen:
                        seen.add(doc[:60])
                        hints.append(doc)
        
        # 查 code_cookbook（提供正确的代码写法）
        if hasattr(self.rag, 'collections') and 'code_cookbook' in self.rag.collections:
            col = self.rag.collections['code_cookbook']
            if col.count() > 0:
                results = col.query(query_texts=[error_query], n_results=1)
                if results['documents'][0]:
                    doc = results['documents'][0][0][:400]
                    if doc[:60] not in seen:
                        seen.add(doc[:60])
                        hints.append(doc)
        
        # 查 api_reference（提供正确的 API 签名）
        if hasattr(self.rag, 'collections') and 'api_reference' in self.rag.collections:
            col = self.rag.collections['api_reference']
            if col.count() > 0:
                results = col.query(query_texts=[error_query], n_results=1)
                if results['documents'][0]:
                    doc = results['documents'][0][0][:300]
                    if doc[:60] not in seen:
                        seen.add(doc[:60])
                        hints.append(doc)
        
        # 查 task_workflows（方法论层面的指导）
        if hasattr(self.rag, 'collections') and 'task_workflows' in self.rag.collections:
            col = self.rag.collections['task_workflows']
            if col.count() > 0:
                results = col.query(query_texts=[error_query], n_results=1)
                if results['documents'][0]:
                    doc = results['documents'][0][0]
                    # 只取 pitfalls 部分（最实用）
                    if '常见 pitfalls' in doc:
                        pitfall_section = doc[doc.index('常见 pitfalls'):][:300]
                        if pitfall_section[:60] not in seen:
                            hints.append(pitfall_section)
                    elif 'pitfalls' in doc.lower():
                        idx = doc.lower().index('pitfalls')
                        pitfall_section = doc[idx:][:300]
                        if pitfall_section[:60] not in seen:
                            hints.append(pitfall_section)
        
        return '\n---\n'.join(hints) if hints else ""
    
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
        """将对话历史格式化为单个 user message（适配 Ollama 单轮接口）"""
        parts = [user_msg]
        
        for role, content in history:
            if role == "assistant":
                parts.append(f"\nAssistant:\n{content}")
            else:
                parts.append(f"\n{content}")
        
        # 在末尾加提示，引导模型继续 ReAct 格式
        parts.append("\nNow continue with the next step. Respond with Thought/Action/Args:")
        
        # 截断过长的历史（保留最近的对话）
        full_text = "\n".join(parts)
        if len(full_text) > 16000:
            # 保留开头的任务描述 + 最近的对话（加大到 14 轮适配 25 rounds）
            header = parts[0]
            recent = "\n".join(parts[-14:])
            full_text = header + "\n\n... (earlier steps omitted) ...\n\n" + recent
        
        return full_text

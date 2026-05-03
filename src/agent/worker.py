"""
Worker Agent — 

:
1. Receive a sub-task from the Planner
2. Invoke a real GIS tool
3. Handle tool execution errors
4. Return a structured result
"""
import json
import time
import traceback
from typing import Dict, Any, Optional

from ..tools.registry import tool_registry

class ToolResult:
    """Tool execution result"""
    def __init__(self, step_num: int, tool: str, params: Dict,
                 result: Any = None, error: str = "", latency_ms: float = 0):
        self.step_num = step_num
        self.tool = tool
        self.params = params
        self.result = result
        self.error = error
        self.latency_ms = latency_ms
        self.success = error == ""

    def to_dict(self):
        return {
            "step": self.step_num,
            "tool": self.tool,
            "params": self.params,
            "result": str(self.result)[:500] if self.result else None,
            "error": self.error,
            "success": self.success,
            "latency_ms": round(self.latency_ms, 1),
        }

class WorkerAgent:
    """Tool-execution agent"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def execute(self, step: Dict[str, Any]) -> ToolResult:
        """"""
        step_num = step.get("step", 0)
        tool_name = step.get("tool", "")
        params = step.get("params", {})
        desc = step.get("description", "")

        if self.verbose:
            print(f"  [tool] Step {step_num}: {tool_name}({json.dumps(params, default=str)[:80]})")

        t0 = time.time()
        try:
            # Check whether the tool exists
            tool = tool_registry.get_tool(tool_name)
            if tool is None:
                available = tool_registry.list_tools()
                return ToolResult(
                    step_num, tool_name, params,
                    error=f"Tool '{tool_name}' not found. Available: {available}",
                )

            result = tool_registry.execute_tool(tool_name, params)
            latency = (time.time() - t0) * 1000

            # Check whether the result contains an error
            if isinstance(result, dict) and "error" in result:
                return ToolResult(step_num, tool_name, params,
                                  error=result["error"], latency_ms=latency)
            if isinstance(result, str):
                try:
                    parsed = json.loads(result)
                    if isinstance(parsed, dict) and "error" in parsed:
                        return ToolResult(step_num, tool_name, params,
                                          error=parsed["error"], latency_ms=latency)
                except:
                    pass

            if self.verbose:
                result_preview = str(result)[:120]
                print(f"    [ok] -> {result_preview}")

            return ToolResult(step_num, tool_name, params,
                              result=result, latency_ms=latency)

        except Exception as e:
            latency = (time.time() - t0) * 1000
            error_msg = f"{type(e).__name__}: {str(e)}"
            if self.verbose:
                print(f"    [fail] {error_msg}")
            return ToolResult(step_num, tool_name, params,
                              error=error_msg, latency_ms=latency)

    def execute_batch(self, steps: list) -> list:
        """"""
        results = []
        for step in steps:
            result = self.execute(step)
            results.append(result)
            if not result.success:
                if self.verbose:
                    print(f"  [warn] Step {result.step_num} failed, stopping.")
                break
        return results

"""
Orchestrator — Planner Worker

:
1. Planner query -> 
2. Worker executes step by step
3. The Planner validates the result
4. Re-plan on failure
5. Return the full final result
"""
import json
import time
from typing import Dict, Any, List, Optional

from .planner import PlannerAgent
from .worker import WorkerAgent, ToolResult

class AgentResult:
    """Agent """
    def __init__(self, query: str, plan: Dict, steps: List[ToolResult],
                 final_answer: str = "", success: bool = True,
                 total_time_ms: float = 0, replans: int = 0):
        self.query = query
        self.plan = plan
        self.steps = steps
        self.final_answer = final_answer
        self.success = success
        self.total_time_ms = total_time_ms
        self.replans = replans

    def to_dict(self):
        return {
            "query": self.query,
            "plan": self.plan,
            "steps": [s.to_dict() for s in self.steps],
            "final_answer": self.final_answer,
            "success": self.success,
            "total_time_ms": round(self.total_time_ms, 1),
            "replans": self.replans,
            "num_steps": len(self.steps),
        }

class Orchestrator:
    """Planner-Worker """

    def __init__(self, planner: PlannerAgent, worker: WorkerAgent,
                 max_replans: int = 2, verbose: bool = True):
        self.planner = planner
        self.worker = worker
        self.max_replans = max_replans
        self.verbose = verbose

    def run(self, query: str) -> AgentResult:
        """ Agent """
        t0 = time.time()
        replans = 0
        all_steps = []

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[list] Query: {query[:100]}")
            print(f"{'='*60}")

        # 1. Planner builds the plan
        if self.verbose:
            print("\n[reason] Planner: ...")
        plan = self.planner.plan(query)

        if "error" in plan:
            return AgentResult(
                query=query, plan=plan, steps=[],
                final_answer=f"Planning failed: {plan.get('error', 'unknown')}",
                success=False, total_time_ms=(time.time()-t0)*1000
            )

        if self.verbose:
            steps = plan.get("plan", [])
            print(f" [write] : {len(steps)} ")
            for s in steps:
                print(f"    Step {s.get('step')}: {s.get('tool')} — {s.get('description','')[:60]}")

        # 2. Worker executes step by step
        while replans <= self.max_replans:
            steps = plan.get("plan", [])
            if not steps:
                break

            if self.verbose:
                print(f"\n[build] Worker: ... (attempt {replans + 1})")

            step_results = self.worker.execute_batch(steps)
            all_steps.extend(step_results)

            # 3. Check whether all steps succeeded
            failed = [r for r in step_results if not r.success]
            if not failed:
                if self.verbose:
                    print(f"\n[ok] {len(step_results)} success!")
                break

            # 4. failure -> re-plan
            fail = failed[0]
            if self.verbose:
                print(f"\n[warn] Step {fail.step_num} failure: {fail.error}")

            if replans >= self.max_replans:
                if self.verbose:
                    print(f" [fail] ({self.max_replans})")
                total_ms = (time.time() - t0) * 1000
                return AgentResult(
                    query=query, plan=plan, steps=all_steps,
                    final_answer=f"Failed after {replans+1} attempts: {fail.error}",
                    success=False, total_time_ms=total_ms, replans=replans
                )

            if self.verbose:
                print(f"  [loop] Planner: re-planning...")
            plan = self.planner.replan(query, fail.step_num, fail.error, plan)
            replans += 1

        # 5. Build the final answer
        total_ms = (time.time() - t0) * 1000
        final_answer = self._build_answer(all_steps)

        if self.verbose:
            print(f"\n[stats] : {final_answer[:200]}")
            print(f"[time] : {total_ms:.0f}ms | : {len(all_steps)} | : {replans}")

        return AgentResult(
            query=query, plan=plan, steps=all_steps,
            final_answer=final_answer, success=True,
            total_time_ms=total_ms, replans=replans
        )

    def _build_answer(self, steps: List[ToolResult]) -> str:
        """Extract the final answer from the execution trace"""
        answers = []
        for s in steps:
            if s.success and s.result:
                try:
                    r = json.loads(s.result) if isinstance(s.result, str) else s.result
                    if isinstance(r, dict):
                        for key in ['count', 'features_count', 'mean', 'total',
                                    'min', 'max', 'area_km2', 'distance_km',
                                    'contour_lines', 'expression', 'output_file']:
                            if key in r:
                                answers.append(f"{key}: {r[key]}")
                except:
                    answers.append(str(s.result)[:100])
        return "; ".join(answers) if answers else "No results"

def create_orchestrator(
    llm_engine=None,
    rag=None,
    use_rag: bool = True,
    max_replans: int = 2,
    verbose: bool = True,
) -> Orchestrator:
    """Helper to construct a complete agent stack"""
    from ..tools.registry import tool_registry
    # Trigger tool registration
    from ..tools import vector_tools, raster_tools, analysis_tools  # noqa
    from ..tools import advanced_tools, viz_tools, terrain_tools, conversion_tools  # noqa

    tools_desc = tool_registry.get_tools_description()
    planner = PlannerAgent(llm_engine, rag if use_rag else None, tools_desc)
    worker = WorkerAgent(verbose=verbose)

    return Orchestrator(planner, worker, max_replans=max_replans, verbose=verbose)

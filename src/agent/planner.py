"""
GIS Task Planner — Complex task decomposer

Uses the LLM to split complex tasks into sub-tasks,
each executed by the Worker in a ReAct loop.

Design:
- Planner only decomposes, does not execute code
- Extracts operation sequences from Skills (no code templates, avoiding answer leaks)
- Queries gis_methodology + task_workflows RAG collections
- Outputs structured sub-task list
"""
import json
import re
from typing import List, Dict, Optional

PLANNER_SYSTEM_PROMPT = """You are a GIS analysis task planning expert. Your job is to decompose complex GIS analysis instructions into 3-7 simple sub-steps.

Each sub-step must satisfy:
1. Independently executable — a single GIS operation (load/reproject/buffer/overlay/statistics/visualize)
2. Sequential dependency — later steps can reference variables from earlier steps
3. Clear output — specify what variable/file the step produces

You must output strictly in JSON format:
```json
{
  "steps": [
    {"id": 1, "action": "Load data", "instruction": "Use gpd.read_file to load dataset/xxx.shp into variable gdf", "output": "gdf"},
    {"id": 2, "action": "Reproject", "instruction": "Reproject gdf to EPSG:32618 for distance calculations", "output": "gdf_proj"},
    {"id": 3, "action": "Buffer analysis", "instruction": "Create 1000m buffer around gdf_proj", "output": "buffered"},
    {"id": 4, "action": "Overlay analysis", "instruction": "Use gpd.overlay(how='difference') for set difference", "output": "result"},
    {"id": 5, "action": "Save results", "instruction": "Save result to pred_results/output.csv and pred_results/map.png", "output": "files"}
  ]
}
```

Key rules:
- Operations involving distance/area **must** have a reprojection step first (to_crs to a projected CRS)
- Overlay operations must explicitly specify the how parameter (intersection/difference/union)
- Do not use clip instead of overlay(how='difference')
- The final step must save results to pred_results/ directory
- If visualization is needed, ensure a plt.savefig step is included
- NEVER use plt.show(). Always use plt.savefig() then plt.close()
- OUTPUT FORMAT RULE: If the instruction explicitly requests saving a specific file format (e.g. "save as shapefile", "save as CSV", "save as GeoTIFF"), you MUST include a dedicated save step for that exact format. Do NOT substitute visualization (PNG) for data file output.
- VISUALIZATION RULE: plt.savefig() MUST be in the SAME step as the plot creation code. Do NOT split "Visualize" and "Save" into separate steps — the matplotlib figure object does not persist between steps. Combine them into one step like: "Create choropleth map and save to pred_results/xxx.png"
- COLUMN NAME RULE: Do NOT hardcode column names in step instructions. Instead, instruct the Worker to inspect available columns first (e.g., "use the appropriate population/density column"). The Worker has access to the data schema.
- VARIABLE NAME RULE: Use short variable names (e.g., gdf, result, overlay) to avoid truncation issues in the execution environment.
- SCHEMA ANALYSIS RULE: Your FIRST step must ALWAYS be "Load and inspect data". In this step, instruct the Worker to: (1) load all datasets, (2) print column names and dtypes, (3) print sample values (head(2)), (4) print CRS and spatial extent (total_bounds). This gives the Worker concrete column names for subsequent steps instead of guessing.
- SEMANTIC MAPPING RULE: When the task mentions concepts like "poverty", "population density", or "vehicle access", your step instructions must say "identify the column that represents [concept] from the loaded data" rather than assuming a specific column name.
- OVERLAY VISUALIZATION RULE: When the task says "overlay" multiple factors on one map, you MUST instruct the Worker to plot ALL factors on the SAME axes (ax) using transparent colors (alpha), NOT as separate subplots. Use ax parameter to layer plots.
- PACKAGE CONSTRAINT RULE: The execution environment does NOT have arcpy, ArcGIS, pykrige, or skimage. NEVER reference these packages in your step instructions. Instead use: geopandas (vector), rasterio (raster), scipy.interpolate (interpolation), shapely (geometry), numpy/scipy.ndimage (image processing). If the task's workflow mentions arcpy tools (e.g., ExtractByMask, Project_management, Con), translate them to equivalent open-source operations.
"""

class TaskPlanner:
    """GIS task decomposer"""
    
    def __init__(self, llm_engine, rag=None):
        """
        Args:
            llm_engine: LLM for the Planner
            rag: ()
        """
        self.llm = llm_engine
    
    def plan(self, instruction: str, dataset_description: str = "",
             domain_knowledge: str = "", workflow: str = "",
             skill_text: str = "") -> List[Dict]:
        """
        Decompose complex instruction into sub-task list
        
        Returns:
            [{"id": 1, "action": "...", "instruction": "...", "output": "..."}, ...]
        """
        # Build prompt
        prompt_parts = [f"## Task Instruction\n{instruction}"]
        
        if dataset_description:
            prompt_parts.append(f"## Data Description\n{dataset_description[:1500]}")
        
        if workflow:
            prompt_parts.append(f"## Reference Workflow\n{workflow}")
        
        if domain_knowledge:
            prompt_parts.append(f"## Domain Knowledge\n{domain_knowledge}")
        
        if skill_text:
            # Extract operation sequence from Skill (no full code templates)
            ops_section = self._extract_ops_from_skill(skill_text)
            if ops_section:
                prompt_parts.append(f"## Key Operation Reference\n{ops_section}")
        
        # RAG — GIS Agent 
        
        user_prompt = "\n\n".join(prompt_parts)
        user_prompt += "\n\nPlease decompose the above task into 3-7 ordered sub-steps, output strictly in JSON format."
        
        # Call LLM
        response = self.llm.generate(
            prompt="",
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_message=user_prompt,
            max_tokens=2048,
        )
        
        # OllamaEngine returns dict {"text": "...", ...}
        raw_text = response.get("text", "") if isinstance(response, dict) else str(response)
        
        # Parse JSON
        steps = self._parse_steps(raw_text)
        return steps
    
    def format_step_for_worker(self, step: Dict, previous_results: str = "") -> str:
        """Format a single Planner step as a Worker instruction"""
        parts = [f"Execute the following sub-task (Step {step.get('id', '?')}):"]
        parts.append(f"Action: {step.get('action', '')}")
        parts.append(f"Instruction: {step.get('instruction', '')}")
        parts.append(f"Expected output: {step.get('output', '')}")
        
        if previous_results:
            parts.append(f"\nVariables from previous steps are available in the environment:\n{previous_results}")
        
        return "\n".join(parts)
    
    def _extract_ops_from_skill(self, skill_text: str) -> str:
        """Extract 'Key Operation Sequence' and 'Workflow' sections from Skill content (no code)"""
        lines = skill_text.split('\n')
        result_parts = []
        in_section = False
        in_code = False
        
        for line in lines:
            # Skip code blocks
            if line.strip().startswith('```'):
                in_code = not in_code
                continue
            if in_code:
                continue
            
            # Match target sections
            if any(kw in line for kw in ['', 'Workflow', '', 'Key Operation', 'Required Libraries']):
                in_section = True
                continue
            
            # End section at next ## heading or code block
            if line.startswith('## ') and in_section:
                in_section = False
                continue
            
            if in_section and line.strip():
                result_parts.append(line)
        
        return '\n'.join(result_parts).strip()[:800]
    
    def _query_rag(self, query: str) -> str:
        """Query methodology-level RAG"""
        if not self.rag or not hasattr(self.rag, 'collections'):
            return ""
        ctx_parts = []
        for col_name in ["gis_methodology", "task_workflows"]:
            if col_name in self.rag.collections:
                col = self.rag.collections[col_name]
                if col.count() > 0:
                    results = col.query(query_texts=[query], n_results=min(2, col.count()))
                    for doc in results["documents"][0]:
                        ctx_parts.append(doc[:400])
        return "\n---\n".join(ctx_parts) if ctx_parts else ""
    
    def _parse_steps(self, response: str) -> List[Dict]:
        """Parse sub-steps JSON from LLM response"""
        # Strip Qwen3 <think> tags
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        # Try to extract JSON block
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to extract { ... } directly
            json_match = re.search(r'\{[\s\S]*"steps"[\s\S]*\}', response)
            if json_match:
                json_str = json_match.group(0)
            else:
                # fallback: return single step
                return [{"id": 1, "action": "Execute full task",
                         "instruction": "Complete all analysis per the instruction and save to pred_results/",
                         "output": "result files"}]
        
        try:
            data = json.loads(json_str)
            steps = data.get("steps", [])
            if isinstance(steps, list) and len(steps) > 0:
                return steps
        except json.JSONDecodeError:
            pass
        
        return [{"id": 1, "action": "Execute full task",
                 "instruction": "Complete all analysis per the instruction and save to pred_results/",
                 "output": "result files"}]

    def replan(self, instruction: str, failed_steps: str,
               skill_text: str = "") -> List[Dict]:
        """When Worker fails consecutively, generate a simplified plan"""
        user_prompt = (
            f"## Original Task\n{instruction}\n\n"
            f"## Failed Steps\n{failed_steps}\n\n"
            "The above steps failed. Please re-plan using a simpler approach. "
            "Avoid the failed operations and use more basic alternatives. "
            "Output strictly in JSON format."
        )
        response = self.llm.generate(
            prompt="", system_prompt=PLANNER_SYSTEM_PROMPT,
            user_message=user_prompt, max_tokens=1024,
        )
        raw_text = response.get("text", "") if isinstance(response, dict) else str(response)
        return self._parse_steps(raw_text)

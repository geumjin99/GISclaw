"""
(Tool Registry)

GIS JSON LLM 
@register_tool 
"""
import json
import inspect
from typing import Callable, Dict, Any, Optional, List
from functools import wraps

class ToolRegistry:
    """GIS """

    def __init__(self):
        # name -> {func, description, parameters}
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        returns: str = "",
    ) -> Callable:
        """
         GIS 

        Args:
            name: LLM 
            description: LLM 
            parameters: JSON Schema
            returns: 
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            self._tools[name] = {
                "func": wrapper,
                "name": name,
                "description": description,
                "parameters": parameters,
                "returns": returns,
            }
            return wrapper
        return decorator

    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """"""
        return self._tools.get(name)

    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """
        

        Args:
            name: 
            arguments: 

        Returns:
            Tool execution result
        """
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"Unknown tool: {name}. Available tools: {list(self._tools.keys())}"}

        try:
            result = tool["func"](**arguments)
            return result
        except Exception as e:
            return {"error": f"Tool '{name}' execution failed: {str(e)}"}

    def get_tools_description(self) -> str:
        """
         System Prompt 

        Returns:
            
        """
        descriptions = []
        for name, tool in self._tools.items():
            params_str = json.dumps(tool["parameters"], indent=2)
            desc = (
                f"### {name}\n"
                f"Description: {tool['description']}\n"
                f"Parameters:\n```json\n{params_str}\n```\n"
                f"Returns: {tool['returns']}"
            )
            descriptions.append(desc)
        return "\n\n".join(descriptions)

    def get_tools_json_schema(self) -> List[Dict[str, Any]]:
        """
         JSON Schema function calling 

        Returns:
             schema 
        """
        schemas = []
        for name, tool in self._tools.items():
            schema = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool["description"],
                    "parameters": {
                        "type": "object",
                        "properties": tool["parameters"],
                        "required": [
                            k for k, v in tool["parameters"].items()
                            if not v.get("optional", False)
                        ],
                    },
                },
            }
            schemas.append(schema)
        return schemas

    def list_tools(self) -> List[str]:
        """"""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"ToolRegistry({len(self._tools)} tools: {self.list_tools()})"

tool_registry = ToolRegistry()

from __future__ import annotations
import uuid
from typing import Dict, List, Type, Any
from tunix.rl.agentic.tools.base_tool import BaseTool, ToolCall, ToolOutput
from concurrent.futures import ThreadPoolExecutor, as_completed

class ToolManager:
    """
    ToolManager is used to route and execute multiple tools 
    (only supports explicit `tool_map` registration).
    """

    def __init__(self, tool_map: Dict[str, Type[BaseTool]], *, desc_fallback: str = ""):
        """
        Args:
            tool_map: A mapping of tool names to tool classes,
                      e.g., {"search": SearchTool, "calc": CalculatorTool}
            desc_fallback: Used as default description if the tool class has no __doc__.
        """
        self._tool_dict: Dict[str, BaseTool] = {
            name: cls(name=name, description=getattr(cls, "__doc__", desc_fallback))
            for name, cls in tool_map.items()
        }

    # ---------- Basic Properties ----------
    @property
    def names(self) -> List[str]:
        return list(self._tool_dict.keys())

    @property
    def json(self) -> List[dict]:
        """Returns the JSON Schemas of all tools, for prompt template injection."""
        return [tool.json for tool in self._tool_dict.values()]
    
    @property
    def mcp_json(self) -> List[dict]:
        """Returns MCP-compatible tool metadata (Gemini/Claude standard)."""
        return [tool.to_mcp_json() for tool in self._tool_dict.values()]
    
    # ---------- Tool Registration ----------
    def register_mcp_tool(self, tool: BaseTool):
        """
        Register a MCP-compatible tool instance directly.
        """
        self._tool_dict[tool.name] = tool

    # ---------- Single Tool Execution ----------
    def run(self, tool_name: str, **kwargs) -> ToolOutput:
        """
        Invoke a tool by its name.

        Args:
            tool_name: The name of the tool to invoke.
            kwargs: Parameters for the tool.

        Returns:
            ToolOutput: The result of the tool execution.
        """
        tool = self._tool_dict.get(tool_name)
        if tool is None:
            return ToolOutput(name=tool_name, error=f"Tool '{tool_name}' not registered.")
        try:
            return tool(**kwargs)
        except Exception as e:
            return ToolOutput(name=tool_name, error=f"{type(e).__name__}: {e}")

    # ---------- Batch Execution ----------
    def execute_calls(self, calls: List[ToolCall], parallel: bool = True) -> Dict[str, str]:
        """
        Execute a batch of tool calls.
        Args:
            calls: List[ToolCall], each containing a tool name and arguments.
            parallel: Whether to execute in parallel using threads.
        Returns:
            Dict[str, str]: Mapping from call_id to ToolOutput.to_string() results.
        """
        outputs = {}

        if not parallel:
            for call in calls:
                cid = getattr(call, "id", None) or str(uuid.uuid4())
                res = self.run(tool_name=call.name, **call.arguments)
                outputs[cid] = res.to_string()
            return outputs

        with ThreadPoolExecutor() as executor:
            future_to_id = {}
            for call in calls:
                cid = getattr(call, "id", None) or str(uuid.uuid4())
                future = executor.submit(self.run, tool_name=call.name, **call.arguments)
                future_to_id[future] = cid

            for future in as_completed(future_to_id):
                cid = future_to_id[future]
                try:
                    res = future.result()
                    outputs[cid] = res.to_string()
                except Exception as e:
                    outputs[cid] = f"Error: {type(e).__name__}: {e}"

        return outputs

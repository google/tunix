from __future__ import annotations
import queue
import threading
import uuid
from typing import Dict, List, Type, Any
from tunix.rl.multi_turn.tools.base_tool import BaseTool, ToolCall, ToolOutput


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
        outputs, q = {}, queue.Queue()

        def _worker(call: ToolCall, cid: str):
            res = self.run(tool_name=call.name, **call.arguments)
            q.put((cid, res.to_string()))

        threads = []
        for call in calls:
            cid = getattr(call, "id", None) or str(uuid.uuid4())
            if parallel:
                t = threading.Thread(target=_worker, args=(call, cid))
                threads.append(t)
                t.start()
            else:
                _worker(call, cid)

        # Wait for all threads to complete if running in parallel
        for t in threads:
            t.join()

        while not q.empty():
            k, v = q.get()
            outputs[k] = v

        return outputs

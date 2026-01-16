# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tools for agentic LLM training.

This module provides tool implementations that can be used during LLM training
to enable tool-augmented reasoning. Tools follow a consistent interface defined
by BaseTool and can be executed by ToolManager.

Available tools:
- CalculatorTool: Basic arithmetic with explicit operands and operator
- ExpressionCalculatorTool: Evaluates mathematical expressions (recommended)

Example usage:
  ```python
  from tunix.rl.agentic.tools import ExpressionCalculatorTool, ToolManager

  # Define available tools
  tool_map = {"calculator": ExpressionCalculatorTool}

  # Create a tool manager
  manager = ToolManager(tool_map)

  # Execute a tool
  result = manager.run("calculator", expression="2 + 3 * 4")
  print(result.output)  # "14"
  ```
"""

from tunix.rl.agentic.tools import base_tool
from tunix.rl.agentic.tools import calculator_tool
from tunix.rl.agentic.tools import expression_calculator_tool
from tunix.rl.agentic.tools import tool_manager
from tunix.rl.agentic.tools.base_tool import BaseTool
from tunix.rl.agentic.tools.base_tool import ToolCall
from tunix.rl.agentic.tools.base_tool import ToolOutput
from tunix.rl.agentic.tools.calculator_tool import CalculatorTool
from tunix.rl.agentic.tools.expression_calculator_tool import (
    ExpressionCalculatorTool,
)
from tunix.rl.agentic.tools.tool_manager import ToolManager

__all__ = [
    "BaseTool",
    "CalculatorTool",
    "ExpressionCalculatorTool",
    "ToolCall",
    "ToolManager",
    "ToolOutput",
]


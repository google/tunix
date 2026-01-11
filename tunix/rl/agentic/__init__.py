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

"""Agentic RL module for tool-augmented training of LLMs.

This module provides components for training LLMs to use external tools
during reasoning through reinforcement learning. Key components include:

- **AgenticGRPOLearner**: GRPO trainer with multi-turn tool interaction support
- **Tools**: Reusable tool implementations (calculator, etc.)
- **Environments**: RL environments for tool execution
- **Agents**: Agent implementations for tool-aware generation

Example usage:
  ```python
  from tunix.rl.agentic import AgenticGRPOConfig, AgenticGRPOLearner
  from tunix.rl.agentic.tools import ExpressionCalculatorTool

  # Define available tools
  tool_map = {"calculator": ExpressionCalculatorTool}

  # Configure the learner
  config = AgenticGRPOConfig(
      num_generations=4,
      system_prompt="You can use tools...",
      max_tool_steps=5,
  )

  # Create and train
  learner = AgenticGRPOLearner(
      rl_cluster=rl_cluster,
      reward_fns=[...],
      algo_config=config,
      tool_map=tool_map,
  )
  learner.train(train_dataset)
  ```
"""

from tunix.rl.agentic.agentic_grpo_learner import AgenticGRPOConfig
from tunix.rl.agentic.agentic_grpo_learner import AgenticGRPOLearner

# Aliases for backwards compatibility
GRPOConfig = AgenticGRPOConfig
GRPOLearner = AgenticGRPOLearner

__all__ = [
    "AgenticGRPOConfig",
    "AgenticGRPOLearner",
    "GRPOConfig",
    "GRPOLearner",
]


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

"""DEPRECATED: This module has graduated to tunix.rl.agentic.

This module is maintained for backwards compatibility only. Please update your
imports to use the new location:

  # Old (deprecated):
  from tunix.rl.experimental.agentic_grpo_learner import GRPOConfig, GRPOLearner

  # New (recommended):
  from tunix.rl.agentic import AgenticGRPOConfig, AgenticGRPOLearner

The new module also adds tool calling support through the `tool_map` parameter.
"""

import warnings

# Re-export from the new location for backwards compatibility
from tunix.rl.agentic.agentic_grpo_learner import AgenticGRPOConfig
from tunix.rl.agentic.agentic_grpo_learner import AgenticGRPOLearner
from tunix.rl.agentic.agentic_grpo_learner import agentic_grpo_loss_fn
from tunix.rl.agentic.agentic_grpo_learner import compute_agentic_advantages
from tunix.rl.experimental import agentic_rl_learner

# Type aliases from base module
TrainingInputT = agentic_rl_learner.TrainingInputT
RewardFn = agentic_rl_learner.RewardFn
MetricFn = agentic_rl_learner.MetricFn
TrainExample = agentic_rl_learner.TrainExample



# Backwards compatible aliases
class GRPOConfig(AgenticGRPOConfig):
  """DEPRECATED: Use AgenticGRPOConfig from tunix.rl.agentic instead."""

  def __init__(self, *args, **kwargs):
    warnings.warn(
        "GRPOConfig is deprecated. Use AgenticGRPOConfig from "
        "tunix.rl.agentic instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    super().__init__(*args, **kwargs)


class GRPOLearner(AgenticGRPOLearner):
  """DEPRECATED: Use AgenticGRPOLearner from tunix.rl.agentic instead."""

  def __init__(self, *args, **kwargs):
    warnings.warn(
        "GRPOLearner is deprecated. Use AgenticGRPOLearner from "
        "tunix.rl.agentic instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    super().__init__(*args, **kwargs)


# Legacy aliases
GrpoConfig = GRPOConfig
GrpoLearner = GRPOLearner
grpo_loss_fn = agentic_grpo_loss_fn
compute_advantages = compute_agentic_advantages

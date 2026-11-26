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
"""DAPO learner and re-exported DAPO configuration helpers."""

from collections.abc import Sequence
from typing import List
from absl import logging
from tunix.rl import algorithm_config as algo_config_lib
from tunix.rl import rl_cluster as rl_engine_lib
from tunix.rl import rl_learner
from tunix.rl.grpo import grpo_learner as grpo_learner_lib
from tunix.utils import compat

TrainingInputT = rl_learner.TrainingInputT
RewardFn = rl_learner.RewardFn
MetricFn = rl_learner.MetricFn

# Re-export canonical DAPOConfig and reward_shaping from algorithm_config for
# backward compatibility while keeping algorithm_config JAX-free for dist_rl.
DAPOConfig = algo_config_lib.DAPOConfig
reward_shaping = algo_config_lib.reward_shaping


class DAPOLearner(grpo_learner_lib.GrpoLearner[DAPOConfig]):
  """DAPO learner."""

  _allow_unimplemented_drop_zero_variance: bool = True

  @compat.alias_init_param("rl_cluster", "rl_engine")
  def __init__(
      self,
      rl_engine: rl_engine_lib.RLEngine,
      algo_config: DAPOConfig,
      reward_fns: RewardFn | List[RewardFn],
      metric_fns: Sequence[MetricFn] | None = None,
      data_shuffle_seed: int | None = None,
  ):
    """Initializes the `DAPOLearner`."""
    if algo_config.drop_zero_variance_groups:
      logging.warning(
          "DAPOConfig.drop_zero_variance_groups=True is not yet implemented in"
          " non-distributed DAPOLearner and will have no effect; use"
          " distributed StandardRLProgram to drop zero-variance groups."
      )
    super().__init__(
        rl_engine=rl_engine,
        algo_config=algo_config,
        reward_fns=reward_fns,
        metric_fns=metric_fns,
        data_shuffle_seed=data_shuffle_seed,
    )

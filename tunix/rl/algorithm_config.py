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

import dataclasses


@dataclasses.dataclass(slots=True, kw_only=True)
class AlgorithmConfig:
  """Configuration for RL algorithms.

  Parameters:
    algo_variant: The core algorithm variant to use.
    advantage_estimator: The advantage estimator to use.
    policy_loss_fn: The policy loss function to use.
  """

  algo_variant: str = "grpo"
  advantage_estimator: str = "grpo"
  policy_loss_fn: str = "grpo"

  def __post_init__(self):
    valid_algo_variants = [
        "grpo",
        "gspo",
        "ppo",
    ]
    valid_advantage_estimators = ["grpo", "gae"]
    valid_policy_loss_fns = ["grpo", "ppo"]
    if self.algo_variant not in valid_algo_variants:
      raise ValueError(
          f"algo_variant must be one of {valid_algo_variants}. "
          f"Received: {self.algo_variant!r}"
      )
    if self.advantage_estimator not in valid_advantage_estimators:
      raise ValueError(
          f"advantage_estimator must be one of {valid_advantage_estimators}."
          f" Received: {self.advantage_estimator}"
      )
    if self.policy_loss_fn not in valid_policy_loss_fns:
      raise ValueError(
          f"policy_loss_fn must be one of {valid_policy_loss_fns}."
          f" Received: {self.policy_loss_fn}"
      )

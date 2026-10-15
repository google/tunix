# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common enums and datatypes for Tunix."""

import dataclasses
import enum
from typing import Literal


@dataclasses.dataclass(frozen=True)
class CacheConfig:
  """Configuration for the KV cache."""

  cache_size: int
  num_layers: int
  num_kv_heads: int
  head_dim: int


ROUTED_EXPERTS_ENCODE_MODES: tuple[str, ...] = (
    "auto",
    "uint8",
    "uint10",
    "int16",
)
RoutedExpertsEncodeMode = Literal["auto", "uint8", "uint10", "int16"]


def validate_routed_experts_encode_config(
    encode_mode: str,
    num_experts: int | None,
) -> None:
  """Validates `routed_experts_encode_mode` and `routed_experts_num_experts`."""
  if encode_mode not in ROUTED_EXPERTS_ENCODE_MODES:
    raise ValueError(
        "routed_experts_encode_mode must be one of"
        f" {ROUTED_EXPERTS_ENCODE_MODES}. Received: {encode_mode!r}"
    )
  if num_experts is not None and (
      isinstance(num_experts, bool)
      or not isinstance(num_experts, int)
      or int(num_experts) <= 0
  ):
    raise ValueError(
        "routed_experts_num_experts must be a positive int or None."
        f" Received: {num_experts!r}"
    )


class Mode(enum.Enum):
  """Mode of RolloutConfig."""

  TRAIN = "train"
  EVAL = "eval"

  def __str__(self):
    return self.value


class Role(str, enum.Enum):
  """Role of the model."""

  ACTOR = "actor"  # policy model
  CRITIC = "critic"  # value model (only for PPO-style algos, not for GRPO)
  REFERENCE = "reference"  # kept fixed during training
  REWARD = "reward"
  ROLLOUT = "rollout"

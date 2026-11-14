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


@dataclasses.dataclass(frozen=True, kw_only=True)
class CacheGeometry:
  """The shape and reach of one KV cache.

  Attributes:
    num_kv_heads: The number of KV heads in the model.
    head_dim: The dimension of each KV head.
    window_size: The window size of the KV cache. None if the cache is global.
  """

  num_kv_heads: int
  head_dim: int
  window_size: int | None = None

  def __post_init__(self):
    positive_checks = {
        "num_kv_heads": self.num_kv_heads,
        "head_dim": self.head_dim,
    }
    if self.window_size is not None:
      positive_checks["window_size"] = self.window_size

    for field_name, value in positive_checks.items():
      if value <= 0:
        raise ValueError(f"{field_name} must be positive, got {value}.")


ROUTED_EXPERTS_ENCODE_MODES: tuple[str, ...] = (
    "auto",
    "uint8",
    "uint10",
    "int16",
)
RoutedExpertsEncodeMode = Literal["auto", "uint8", "uint10", "int16"]
RoutedExpertsWireEncoding = Literal["int16", "uint8", "uint10"]


@dataclasses.dataclass(frozen=True, kw_only=True)
class RoutedExpertsRef:
  """Opaque handle referencing a staged routed_experts array on a RolloutWorker."""

  worker_id: str
  instance_id: str
  ref_id: int

  def __post_init__(self) -> None:
    if not isinstance(self.worker_id, str) or not self.worker_id:
      raise ValueError("RoutedExpertsRef.worker_id must be a non-empty str.")
    if not isinstance(self.instance_id, str) or not self.instance_id:
      raise ValueError("RoutedExpertsRef.instance_id must be a non-empty str.")
    if (
        isinstance(self.ref_id, bool)
        or not isinstance(self.ref_id, int)
        or self.ref_id < 0
        or self.ref_id >= (1 << 63)
    ):
      raise ValueError(
          "RoutedExpertsRef.ref_id must be an int in [0, 2^63), got"
          f" {self.ref_id!r}."
      )


def resolve_routed_experts_wire_encoding(
    encode_mode: str,
    num_experts: int | None,
) -> RoutedExpertsWireEncoding:
  """Statically resolves the wire encoding tier from `encode_mode` and `num_experts`."""
  if encode_mode not in ROUTED_EXPERTS_ENCODE_MODES:
    raise ValueError(
        "routed_experts_encode_mode must be one of"
        f" {ROUTED_EXPERTS_ENCODE_MODES}. Received: {encode_mode!r}"
    )
  if num_experts is not None and (
      isinstance(num_experts, bool)
      or not isinstance(num_experts, int)
      or num_experts <= 0
  ):
    raise ValueError(
        "routed_experts_num_experts must be a positive int or None."
        f" Received: {num_experts!r}"
    )
  if encode_mode == "uint8":
    if num_experts is not None and num_experts > 255:
      raise ValueError(
          "routed_experts_encode_mode='uint8' supports at most 255 experts,"
          f" got routed_experts_num_experts={num_experts}."
      )
    return "uint8"
  if encode_mode == "uint10":
    if num_experts is not None and num_experts > 1023:
      raise ValueError(
          "routed_experts_encode_mode='uint10' supports at most 1023 experts,"
          f" got routed_experts_num_experts={num_experts}."
      )
    return "uint10"
  if encode_mode == "int16":
    return "int16"
  if num_experts is None:
    raise ValueError(
        "routed_experts_encode_mode='auto' requires a positive"
        " routed_experts_num_experts to statically determine wire encoding."
    )
  if num_experts <= 255:
    return "uint8"
  if num_experts <= 1023:
    return "uint10"
  return "int16"


def uint10_trailing_shape(top_k: int) -> tuple[int, int]:
  """Returns the 2D trailing shape `(G, 5)` or `(1, ceil(10 * K / 8))` for `uint10`."""
  if top_k % 4 == 0:
    return (top_k // 4, 5)
  return (1, (10 * top_k + 7) // 8)


def routed_experts_bytes_per_token(
    num_layers: int,
    top_k: int,
    encoding: RoutedExpertsWireEncoding,
) -> int:
  """Returns exact encoded byte count per token for `(num_layers, top_k, encoding)`."""
  if (
      isinstance(num_layers, bool)
      or not isinstance(num_layers, int)
      or isinstance(top_k, bool)
      or not isinstance(top_k, int)
      or num_layers <= 0
      or top_k <= 0
  ):
    raise ValueError(
        "num_layers and top_k must be positive ints, got"
        f" ({num_layers!r}, {top_k!r})."
    )
  if encoding == "uint8":
    return num_layers * top_k
  if encoding == "uint10":
    d0, d1 = uint10_trailing_shape(top_k)
    return num_layers * d0 * d1
  if encoding == "int16":
    return num_layers * top_k * 2
  raise ValueError(f"Unsupported routed_experts wire encoding: {encoding!r}")


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


class WeightSyncMode(str, enum.Enum):
  """Modes for weight synchronization between trainer and rollout workers.

  Attributes:
    NONE: Disables weight synchronization; rollout workers keep their initial
      weights across training steps (`sync_weights=False`).
    NOOP: Runs full weight-sync barrier handshakes (`NullHandler`) and advances
      `policy_version` every step without transferring model weights.
    RAIDEN: Multi-process TPU transport via Raiden host-staged RMA/ICI channels
      (wired in `WeightSyncCoordinator` via `RaidenHandler`).
    IN_MEMORY: Direct in-process tensor transfer via `WeightSyncRequest.weights`
      when trainer and rollout samplers share a Python process (used by
      in-process `VanillaSampler` / `InprocessVllmSampler` sync; not used by the
      multi-process `WeightSyncCoordinator`).
    FS: Shared-filesystem / checkpoint-staged weight transfer (schema value
      accepted by config resolution; coordinator handler not yet wired).
    NCCL: GPU collective / NCCL broadcast weight transfer (schema value accepted
      by config resolution; coordinator handler not yet wired).
    GCS: Experimental Orbax checkpoint-staged transfer through a shared
      `gs://` or filesystem staging directory (wired via
      `tunix.experimental.weight_sync`).
  """

  NONE = "none"
  NOOP = "noop"
  RAIDEN = "raiden"
  IN_MEMORY = "in_memory"
  FS = "fs"
  NCCL = "nccl"
  GCS = "gcs"


DEFAULT_WEIGHT_SYNC_MODE = WeightSyncMode.NONE

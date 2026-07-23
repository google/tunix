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

"""InferenceWorker: frozen-model scoring (reference log-probs and reward scores).

Adapts a reference/reward inference core (e.g.
``tunix.rl.inference.inference_worker.InferenceWorker``) to the numpy wire DTOs:
requests arrive as host arrays, are converted to device arrays and scored,
and results are returned as host numpy. This version hosts frozen weights only;
there is no weight sync.
"""

from typing import Any, Protocol

import jax
from tunix.experimental.common import datatypes
from tunix.experimental.worker import abstract_worker


class ReferenceScoringCore(Protocol):
  """Structural type of the frozen inference core an InferenceWorker wraps."""

  def get_ref_per_token_logps(
      self,
      prompt_tokens: jax.typing.ArrayLike,
      completion_tokens: jax.typing.ArrayLike,
      pad_id: int,
      eos_id: int,
      temperature: float = 1.0,
  ) -> jax.Array:
    ...

  def get_rewards(
      self,
      prompt_tokens: jax.typing.ArrayLike,
      completion_tokens: jax.typing.ArrayLike,
      pad_id: int,
      eos_id: int,
  ) -> jax.Array:
    ...


class InferenceWorker(abstract_worker.Worker):
  """Worker exposing frozen-model scoring to the orchestrator.

  Hosts frozen weights (a reference model and, optionally, a reward model) and
  answers scoring requests. It never trains and only runs inference.
  """

  def __init__(
      self,
      core: ReferenceScoringCore,
      *,
      pad_id: int,
      eos_id: int,
      model_version: int = 0,
  ):
    """Initializes the worker.

    Args:
      core: The frozen inference core to score against.
      pad_id: Padding token id used in the request arrays.
      eos_id: End-of-sequence token id.
      model_version: Version tag for the hosted weights; constant while frozen.
    """
    self._core = core
    self._pad_id = pad_id
    self._eos_id = eos_id
    self._model_version = model_version

  def initialize(self) -> None:
    pass

  def compile(self, dummy_data: Any) -> None:
    del dummy_data

  def start(self) -> None:
    pass

  def stop(self) -> None:
    pass

  def compute_logps(
      self, req: datatypes.LogprobsRequest
  ) -> datatypes.LogprobsResult:
    """Scores per-token log-probs for a batch under the frozen reference model."""
    raise NotImplementedError()

  def score(self, req: datatypes.ScoreRequest) -> datatypes.ScoreResult:
    """Scores one scalar per row under a hosted (frozen) reward model."""
    raise NotImplementedError()

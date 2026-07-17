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
requests arrive as host arrays, are converted to device arrays and scored in
worker-internal micro-batches, and results are returned as host numpy. This
version hosts frozen weights only; there is no weight sync.
"""

from typing import Any, Callable, Protocol

import jax.numpy as jnp
import numpy as np

from tunix.experimental.common import datatypes
from tunix.experimental.worker import abstract_worker


class ReferenceScoringCore(Protocol):
  """Structural type of the frozen inference core an InferenceWorker wraps."""

  def get_ref_per_token_logps(
      self,
      prompt_tokens: Any,
      completion_tokens: Any,
      pad_id: int,
      eos_id: int,
      temperature: float = 1.0,
  ) -> Any:
    ...

  def get_rewards(
      self,
      prompt_tokens: Any,
      completion_tokens: Any,
      pad_id: int,
      eos_id: int,
  ) -> Any:
    ...


class InferenceWorker(abstract_worker.Worker):
  """Worker exposing frozen-model scoring to the orchestrator.

  Hosts frozen weights (a reference model and, optionally, a reward model) and
  answers scoring requests. It never trains and takes no weight sync in this
  version; versioned policy recompute is a later addition.
  """

  def __init__(
      self,
      core: ReferenceScoringCore,
      *,
      pad_id: int,
      eos_id: int,
      model_version: int = 0,
      worker_id: str = "inference",
  ):
    """Initializes the worker.

    Args:
      core: The frozen inference core to score against.
      pad_id: Padding token id used in the request arrays.
      eos_id: End-of-sequence token id.
      model_version: Version tag for the hosted weights; constant while frozen.
      worker_id: Unique id reported via `info()`.
    """
    self._core = core
    self._pad_id = pad_id
    self._eos_id = eos_id
    self._model_version = model_version
    self._worker_id = worker_id
    self._is_running = False

  def initialize(self) -> None:
    pass

  def compile(self, shape_config: datatypes.ShapeConfig) -> None:
    del shape_config

  def start(self) -> None:
    self._is_running = True

  def stop(self) -> None:
    self._is_running = False

  def health(self) -> datatypes.HealthReport:
    """Returns a liveness/status snapshot."""
    return datatypes.HealthReport(
        state="READY" if self._is_running else "STOPPED",
        policy_version=self._model_version,
    )

  def info(self) -> datatypes.WorkerInfo:
    """Returns the worker's static description."""
    return datatypes.WorkerInfo(
        worker_id=self._worker_id, roles=frozenset({"inference"})
    )

  def compute_logprobs(
      self, req: datatypes.LogprobsRequest
  ) -> datatypes.LogprobsResult:
    """Scores per-token log-probs for a batch under the frozen reference model."""
    if req.model_role != "reference":
      raise NotImplementedError(
          "compute_logprobs hosts the frozen reference model only; got "
          f"model_role={req.model_role!r} (versioned policy recompute is a "
          "later addition)."
      )

    def _score(prompt_chunk: np.ndarray, completion_chunk: np.ndarray) -> Any:
      return self._core.get_ref_per_token_logps(
          prompt_tokens=jnp.asarray(prompt_chunk),
          completion_tokens=jnp.asarray(completion_chunk),
          pad_id=self._pad_id,
          eos_id=self._eos_id,
          temperature=req.temperature,
      )

    logps = self._run_microbatched(
        req.prompt_tokens, req.completion_tokens, req.micro_batch_size, _score
    )
    return datatypes.LogprobsResult(
        request_id=req.request_id,
        per_token_logps=np.asarray(logps, dtype=np.float32),
        model_version=self._model_version,
    )

  def score(self, req: datatypes.ScoreRequest) -> datatypes.ScoreResult:
    """Scores one scalar per row under a hosted (frozen) reward model."""

    def _score(prompt_chunk: np.ndarray, completion_chunk: np.ndarray) -> Any:
      return self._core.get_rewards(
          prompt_tokens=jnp.asarray(prompt_chunk),
          completion_tokens=jnp.asarray(completion_chunk),
          pad_id=self._pad_id,
          eos_id=self._eos_id,
      )

    scores = self._run_microbatched(
        req.prompt_tokens, req.completion_tokens, req.micro_batch_size, _score
    )
    return datatypes.ScoreResult(
        request_id=req.request_id,
        scores=np.asarray(scores, dtype=np.float32),
        model_version=self._model_version,
    )

  def _run_microbatched(
      self,
      prompt_tokens: np.ndarray,
      completion_tokens: np.ndarray,
      micro_batch_size: int | None,
      fn: Callable[[np.ndarray, np.ndarray], Any],
  ) -> Any:
    """Applies `fn` over the batch, splitting into micro-batches when requested.

    Splitting is over the batch (axis 0) only, so each micro-batch is scored
    independently and the concatenation is identical to a single pass.
    """
    batch_size = prompt_tokens.shape[0]
    if not micro_batch_size or micro_batch_size >= batch_size:
      return fn(prompt_tokens, completion_tokens)
    chunks = []
    for start in range(0, batch_size, micro_batch_size):
      end = start + micro_batch_size
      chunks.append(fn(prompt_tokens[start:end], completion_tokens[start:end]))
    return jnp.concatenate(chunks, axis=0)

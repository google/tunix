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
import jax.numpy as jnp
import numpy as np
from tunix.dist_rl.common import batch_utils
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.worker import abstract_worker

WorkerState = datatypes.WorkerState


class ReferenceScoringCore(Protocol):
  """Structural type of the frozen inference core an InferenceWorker wraps."""

  def get_ref_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      pad_id: int,
      eos_id: int,
      temperature: float = 1.0,
      segment_ids: jax.Array | None = None,
      segment_positions: jax.Array | None = None,
  ) -> jax.Array:
    ...

  def get_rewards(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      pad_id: int,
      eos_id: int,
  ) -> jax.Array:
    ...


class InferenceWorker(abstract_worker.Worker):
  """Worker exposing frozen-model scoring to the orchestrator.

  This class wraps an already-initialized inference core (provided via the
  `core` argument) which is responsible for loading and materializing the
  frozen weights (e.g. a reference model and, optionally, a reward model).
  It answers scoring requests by routing them to the core. It never trains and
  only runs inference.
  """

  def __init__(
      self,
      core: ReferenceScoringCore,
      *,
      worker_id: str,
      pad_id: int,
      eos_id: int,
      model_version: int = 0,
      chunk_size: int | None = None,
      execution_context: Any = None,
  ):
    """Initializes the worker.

    Args:
      core: An already-initialized inference core holding the frozen weights.
        The core is responsible for model materialization.
      worker_id: The unique identifier for this worker.
      pad_id: Padding token id used in the request arrays.
      eos_id: End-of-sequence token id.
      model_version: Version tag for the hosted weights; constant while frozen.
      chunk_size: Optional maximum batch size for scoring to reduce peak memory.
      execution_context: Optional context manager or zero-arg callable returning
        a context manager (e.g., a JAX Mesh) to enter during worker method
        execution.
    """
    self._execution_context = execution_context
    self._worker_id = worker_id
    self._core = core
    self._pad_id = pad_id
    self._eos_id = eos_id
    self._model_version = model_version
    self._chunk_size = chunk_size

  def info(self) -> datatypes.WorkerInfo:
    return datatypes.WorkerInfo(
        worker_id=self._worker_id, roles=frozenset({"inference"})
    )

  @abstract_worker.offload_to_thread
  def initialize(self) -> datatypes.Response:
    self.state = WorkerState.INITIALIZING
    try:
      return datatypes.Response()
    finally:
      self.state = WorkerState.READY

  @abstract_worker.offload_to_thread
  def compile(self, dummy_data: Any) -> datatypes.Response:
    del dummy_data
    self.state = WorkerState.COMPILING
    try:
      return datatypes.Response()
    finally:
      self.state = WorkerState.READY

  @abstract_worker.offload_to_thread
  def start(self) -> datatypes.Response:
    return datatypes.Response()

  @abstract_worker.offload_to_thread
  def stop(self) -> datatypes.Response:
    self.state = WorkerState.STOPPED
    return datatypes.Response()

  def heartbeat(self) -> datatypes.HealthReport:
    return datatypes.HealthReport(state=self.state)

  @abstract_worker.offload_to_thread
  def compute_logps(
      self, req: datatypes.LogprobsRequest
  ) -> datatypes.LogprobsResponse:
    """Scores per-token log-probs for a batch under the frozen reference model."""
    try:
      if req.model_role != "reference":
        raise NotImplementedError(
            "compute_logprobs hosts the frozen reference model only; got "
            f"model_role={req.model_role!r} (versioned policy recompute is a "
            "later addition)."
        )

      temperature = float(req.temperature)
      if temperature <= 1e-5:
        raise ValueError("Temperature must be strictly positive.")

      prompt_tokens = np.asarray(req.prompt_tokens, dtype=np.int32)
      completion_tokens = np.asarray(req.completion_tokens, dtype=np.int32)
      segment_ids = (
          None
          if req.segment_ids is None
          else np.asarray(req.segment_ids, dtype=np.int32)
      )
      segment_positions = (
          None
          if req.segment_positions is None
          else np.asarray(req.segment_positions, dtype=np.int32)
      )
      if (segment_ids is None) != (segment_positions is None):
        raise ValueError(
            "LogprobsRequest must specify both segment_ids and"
            " segment_positions or neither."
        )

      def _score(
          prompt_chunk: np.ndarray,
          completion_chunk: np.ndarray,
          segment_ids_chunk: np.ndarray | None = None,
          segment_positions_chunk: np.ndarray | None = None,
      ) -> jax.Array:
        segment_kwargs: dict[str, jax.Array] = {}
        if (
            segment_ids_chunk is not None
            and segment_positions_chunk is not None
        ):
          segment_kwargs = {
              "segment_ids": jnp.asarray(segment_ids_chunk, dtype=jnp.int32),
              "segment_positions": jnp.asarray(
                  segment_positions_chunk, dtype=jnp.int32
              ),
          }
        return self._core.get_ref_per_token_logps(
            prompt_tokens=jnp.asarray(prompt_chunk, dtype=jnp.int32),
            completion_tokens=jnp.asarray(completion_chunk, dtype=jnp.int32),
            pad_id=self._pad_id,
            eos_id=self._eos_id,
            temperature=temperature,
            **segment_kwargs,
        )

      chunk_inputs = [prompt_tokens, completion_tokens]
      if segment_ids is not None and segment_positions is not None:
        chunk_inputs.extend([segment_ids, segment_positions])
      logps = batch_utils.apply_chunked(_score, self._chunk_size, *chunk_inputs)
      return datatypes.LogprobsResponse(
          request_id=req.request_id,
          per_token_logps=np.asarray(logps, dtype=np.float32),
          model_version=self._model_version,
      )
    except Exception as e:
      return datatypes.LogprobsResponse(
          request_id=req.request_id,
          per_token_logps=np.zeros((0, 0), dtype=np.float32),
          model_version=self._model_version,
          error=datatypes.ErrorInfo(
              error_type=type(e).__name__, message=str(e), traceback=repr(e)
          ),
      )

  @abstract_worker.offload_to_thread
  def per_token_logps(
      self,
      items: datatypes.LogprobsRequest,
  ) -> np.ndarray:
    """Scores reference log-probs for an already padded LogprobsRequest."""
    if not isinstance(items, datatypes.LogprobsRequest):
      raise TypeError(
          "InferenceWorker.per_token_logps expects a padded LogprobsRequest. "
          "Assemble microbatches in BatchAssembler before scoring."
      )
    resp = self.compute_logps(items)
    if resp.error is not None:
      raise RuntimeError(resp.error.message)
    return np.asarray(resp.per_token_logps, dtype=np.float32)

  @abstract_worker.offload_to_thread
  def score(self, req: datatypes.ScoreRequest) -> datatypes.ScoreResponse:
    """Scores one scalar per row under a hosted (frozen) reward model."""
    try:
      if req.model_role != "reward":
        raise NotImplementedError(
            "score hosts the frozen reward model only; got "
            f"model_role={req.model_role!r}."
        )

      def _score(
          prompt_chunk: np.ndarray, completion_chunk: np.ndarray
      ) -> jax.Array:
        return self._core.get_rewards(
            prompt_tokens=jnp.asarray(prompt_chunk, dtype=jnp.int32),
            completion_tokens=jnp.asarray(completion_chunk, dtype=jnp.int32),
            pad_id=self._pad_id,
            eos_id=self._eos_id,
        )

      scores = batch_utils.apply_chunked(
          _score, self._chunk_size, req.prompt_tokens, req.completion_tokens
      )
      scores_array = np.asarray(scores, dtype=np.float32)
      if scores_array.ndim == 2:
        scores_array = np.squeeze(scores_array, axis=-1)
      return datatypes.ScoreResponse(
          request_id=req.request_id,
          scores=scores_array,
          model_version=self._model_version,
      )
    except Exception as e:
      return datatypes.ScoreResponse(
          request_id=req.request_id,
          scores=np.zeros((0,), dtype=np.float32),
          model_version=self._model_version,
          error=datatypes.ErrorInfo(
              error_type=type(e).__name__, message=str(e), traceback=repr(e)
          ),
      )

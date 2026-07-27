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
from tunix.experimental.common import batch_utils
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
      pad_id: int,
      eos_id: int,
      model_version: int = 0,
      chunk_size: int | None = None,
  ):
    """Initializes the worker.

    Args:
      core: An already-initialized inference core holding the frozen weights.
        The core is responsible for model materialization.
      pad_id: Padding token id used in the request arrays.
      eos_id: End-of-sequence token id.
      model_version: Version tag for the hosted weights; constant while frozen.
      chunk_size: Optional maximum batch size for scoring to reduce peak memory.
    """
    self._core = core
    self._pad_id = pad_id
    self._eos_id = eos_id
    self._model_version = model_version
    self._chunk_size = chunk_size

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
    try:
      if req.model_role != "reference":
        raise NotImplementedError(
            "compute_logprobs hosts the frozen reference model only; got "
            f"model_role={req.model_role!r} (versioned policy recompute is a "
            "later addition)."
        )

      if req.temperature <= 1e-5:
        raise ValueError("Temperature must be strictly positive.")

      def _score(prompt_chunk: np.ndarray, completion_chunk: np.ndarray) -> Any:
        return self._core.get_ref_per_token_logps(
            prompt_tokens=jnp.asarray(prompt_chunk, dtype=jnp.int32),
            completion_tokens=jnp.asarray(completion_chunk, dtype=jnp.int32),
            pad_id=self._pad_id,
            eos_id=self._eos_id,
            temperature=req.temperature,
        )

      logps = batch_utils.apply_chunked(
          _score, self._chunk_size, req.prompt_tokens, req.completion_tokens
      )
      return datatypes.LogprobsResult(
          request_id=req.request_id,
          per_token_logps=np.asarray(logps, dtype=np.float32),
          model_version=self._model_version,
      )
    except Exception as e:
      return datatypes.LogprobsResult(
          request_id=req.request_id,
          per_token_logps=np.zeros((0, 0), dtype=np.float32),
          model_version=self._model_version,
          error=datatypes.ErrorInfo(
              error_type=type(e).__name__, message=str(e), traceback=repr(e)
          ),
      )

  def score(self, req: datatypes.ScoreRequest) -> datatypes.ScoreResult:
    """Scores one scalar per row under a hosted (frozen) reward model."""
    try:
      if req.model_role != "reward":
        raise NotImplementedError(
            "score hosts the frozen reward model only; got "
            f"model_role={req.model_role!r}."
        )

      def _score(prompt_chunk: np.ndarray, completion_chunk: np.ndarray) -> Any:
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
      return datatypes.ScoreResult(
          request_id=req.request_id,
          scores=scores_array,
          model_version=self._model_version,
      )
    except Exception as e:
      return datatypes.ScoreResult(
          request_id=req.request_id,
          scores=np.zeros((0,), dtype=np.float32),
          model_version=self._model_version,
          error=datatypes.ErrorInfo(
              error_type=type(e).__name__, message=str(e), traceback=repr(e)
          ),
      )

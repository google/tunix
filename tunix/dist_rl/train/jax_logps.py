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

"""JAX/NNX per-token log-probability helper for JAX trainer backends."""

import functools
from typing import Any, cast
import weakref

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.train import jax_pytrees  # pylint: disable=unused-import
from tunix.rl import common as rl_common

# Tracks `(chunk_size, bound_fwd_only_fn)` per trainer instance so calling
# `trainer.with_fwd_only_fn(custom_fn)` externally invalidates the cached
# log-probability scorer even when `chunk_size` is unchanged.
_LOGPS_BINDINGS: weakref.WeakKeyDictionary[Any, tuple[int, Any]] = (
    weakref.WeakKeyDictionary()
)


def compute_per_token_logps_from_payload(
    model: nnx.Module,
    payload: datatypes.LogprobsScoringPayload,
    *,
    chunk_size: int = 0,
) -> jax.Array:
  """Computes per-token log-probabilities on a read-only NNX model copy.

  Args:
    model: The NNX Module instance to score with.
    payload: The `LogprobsScoringPayload` micro-batch containing prompt,
      completion, static scoring parameters (`pad_id`, `eos_id`, `temperature`),
      and optional packing/routing arrays.
    chunk_size: Vocabulary chunk size for computing log-probabilities (`0`
      computes all at once).

  Returns:
    A JAX Array containing the computed per-token log-probabilities.
  """
  graphdef, state = nnx.split(model)
  logps = rl_common.compute_per_token_logps(
      graphdef,
      state,
      prompt_tokens=jnp.asarray(payload.prompt_ids),
      completion_tokens=jnp.asarray(payload.completion_ids),
      pad_id=payload.pad_id,
      eos_id=payload.eos_id,
      stop_gradient=True,
      return_entropy=False,
      segment_ids=(
          None
          if payload.segment_ids is None
          else jnp.asarray(payload.segment_ids)
      ),
      segment_positions=(
          None
          if payload.segment_positions is None
          else jnp.asarray(payload.segment_positions)
      ),
      temperature=payload.temperature,
      chunk_size=chunk_size,
      routed_experts=(
          None
          if payload.routed_experts is None
          else jnp.asarray(payload.routed_experts)
      ),
  )
  return cast(jax.Array, logps)


# TODO(igorts): Drop legacy callback once
# oss/requirements/maxtext_requirements.txt advances past cl/1018794127.
def compute_nnx_per_token_logps(
    model: nnx.Module, *args: Any, **kwargs: Any
) -> jax.Array:
  """Computes per-token log-probabilities under live NNX model weights.

  Args:
    model: The NNX Module instance to score with.
    *args: Positional arguments forwarded to
      `rl_common.compute_per_token_logps`.
    **kwargs: Keyword arguments forwarded to
      `rl_common.compute_per_token_logps`.

  Returns:
    A JAX Array containing the computed per-token log-probabilities.
  """
  jax_args = tuple(
      jnp.asarray(arg) if isinstance(arg, np.ndarray) else arg for arg in args
  )
  jax_kwargs = {
      key: jnp.asarray(val) if isinstance(val, np.ndarray) else val
      for key, val in kwargs.items()
  }
  graphdef, state = nnx.split(model)
  return cast(
      jax.Array,
      rl_common.compute_per_token_logps(
          graphdef, state, *jax_args, stop_gradient=True, **jax_kwargs
      ),
  )


def compute_trainer_per_token_logps(
    trainer: Any,
    prompt_tokens: np.ndarray,
    completion_tokens: np.ndarray,
    *,
    pad_id: int,
    eos_id: int,
    temperature: float = 1.0,
    chunk_size: int = 0,
    segment_ids: np.ndarray | None = None,
    segment_positions: np.ndarray | None = None,
    routed_experts: np.ndarray | None = None,
) -> jax.Array | np.ndarray:
  """Scores per-token log-probabilities via a JAX trainer's `fwd_only` API.

  Returns the raw `fwd_only` array without an eager `np.asarray` copy so callers
  such as `TrainerWorker.per_token_logps` can enqueue multiple micro-batches
  asynchronously before materializing results to host memory.

  Args:
    trainer: Trainer or training engine exposing `with_fwd_only_fn` and
      `fwd_only`.
    prompt_tokens: Int32 prompt token array `[B, P]`.
    completion_tokens: Int32 completion token array `[B, C]`.
    pad_id: Padding token ID.
    eos_id: End-of-sequence token ID.
    temperature: Sampling softmax temperature.
    chunk_size: Optional vocabulary chunk size for memory-bounded log-probs.
    segment_ids: Optional packed sequence segment IDs `[B, P + C]`.
    segment_positions: Optional packed sequence positions `[B, P + C]`.
    routed_experts: Optional MoE router replay tensor.

  Returns:
    Array of completion per-token log-probabilities `[B, C]`.
  """
  if hasattr(trainer, "with_fwd_only_fn"):
    cur_fn = getattr(trainer, "_fwd_only_fn", None)
    if cur_fn is None or _LOGPS_BINDINGS.get(trainer) != (chunk_size, cur_fn):
      trainer.with_fwd_only_fn(
          functools.partial(
              compute_per_token_logps_from_payload,
              chunk_size=chunk_size,
          )
      )
      _LOGPS_BINDINGS[trainer] = (
          chunk_size,
          getattr(trainer, "_fwd_only_fn", None),
      )
    payload = datatypes.LogprobsScoringPayload(
        prompt_ids=prompt_tokens,
        completion_ids=completion_tokens,
        pad_id=int(pad_id),
        eos_id=int(eos_id),
        temperature=temperature,
        segment_ids=segment_ids,
        segment_positions=segment_positions,
        routed_experts=routed_experts,
    )
    return trainer.fwd_only(payload)
  # TODO(igorts): Drop legacy fallback once
  # oss/requirements/maxtext_requirements.txt advances past cl/1018794127.
  return trainer.fwd_only(
      compute_nnx_per_token_logps,
      prompt_tokens,
      completion_tokens,
      pad_id=pad_id,
      eos_id=eos_id,
      temperature=temperature,
      chunk_size=chunk_size,
      segment_ids=segment_ids,
      segment_positions=segment_positions,
      routed_experts=routed_experts,
  )

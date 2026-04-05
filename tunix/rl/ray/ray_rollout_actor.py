# Copyright 2026 Google LLC
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

"""Ray remote actor that owns the rollout engine.

``RolloutActor`` is the counterpart of ``TrainerActor``: it lives in its own
Ray process, owns the rollout engine (vLLM, SGLang-JAX, or vanilla), handles
token generation requests, and accepts weight updates from the trainer.

Design notes:

  * All inputs/outputs that cross the Ray process boundary are numpy arrays
    or plain Python objects (lists of strings, ints, etc.) to avoid
    serialisation surprises with JAX arrays.
  * ``update_weights`` is the single entry-point for all weight-sync
    strategies; strategies that use the file system or object store call
    ``load_weights_from_file`` / ``update_weights_from_ref`` instead.
  * The rollout cluster is constructed via a factory callable so that JAX
    device initialisation happens inside the actor process.
"""

from __future__ import annotations

import io
from typing import Any, Callable

from absl import logging
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from tunix.rl.rollout import base_rollout


class RolloutActor:
  """Owns the rollout engine; serves generate requests and accepts weight syncs.

  Args:
    rollout_factory: Zero-argument callable that returns a fully initialised
      ``RLCluster`` (or any object whose ``.rollout`` attribute is a
      ``BaseRollout`` instance and whose ``.tokenizer`` is set).  Called
      **inside** the actor process.
  """

  def __init__(self, rollout_factory: Callable[[], Any]):
    logging.info("RolloutActor: initialising rollout inside actor process.")
    self._cluster = rollout_factory()
    logging.info(
        "RolloutActor: rollout initialised. JAX devices: %s", jax.devices()
    )

  # ---------------------------------------------------------------------------
  # Generation
  # ---------------------------------------------------------------------------

  def generate(
      self,
      prompts: list[str] | list[list[dict[str, str]]],
      apply_chat_template: bool = False,
      mode_str: str = "train",
      micro_batch_size: int | None = None,
      trace_tags: dict[str, Any] | None = None,
  ) -> dict[str, Any]:
    """Generate responses for ``prompts``.

    Returns a plain-Python dict that mirrors ``base_rollout.RolloutOutput``
    but with numpy leaves so it can cross the Ray boundary safely.

    Args:
      prompts: Prompts to generate from.
      apply_chat_template: Whether to apply the tokenizer's chat template.
      mode_str: ``"train"`` or ``"eval"`` (string form of ``Mode``).
      micro_batch_size: Optional micro-batch size.
      trace_tags: Optional perf tracing tags.

    Returns:
      A dict with keys ``text``, ``tokens``, ``logits``, ``logprobs``, and
      ``left_padded_prompt_tokens``.  All array-like values are numpy.
    """
    from tunix.rl import rl_cluster as rl_cluster_lib  # pylint: disable=g-import-not-at-top

    mode = (
        rl_cluster_lib.Mode.TRAIN
        if mode_str == "train"
        else rl_cluster_lib.Mode.EVAL
    )
    output: base_rollout.RolloutOutput = self._cluster.generate(
        prompts=prompts,
        apply_chat_template=apply_chat_template,
        mode=mode,
        micro_batch_size=micro_batch_size,
        trace_tags=trace_tags,
    )

    # Convert JAX arrays → numpy before returning across the Ray boundary.
    return {
        "text": output.text,
        "tokens": [np.asarray(t) for t in output.tokens],
        "logits": (
            [np.asarray(l) for l in output.logits]
            if output.logits is not None
            else None
        ),
        "logprobs": (
            [np.asarray(lp) for lp in output.logprobs]
            if output.logprobs is not None
            else None
        ),
        "left_padded_prompt_tokens": np.asarray(output.left_padded_prompt_tokens),
    }

  # ---------------------------------------------------------------------------
  # Weight sync receivers
  # ---------------------------------------------------------------------------

  def update_weights(self, weights_numpy: Any) -> None:
    """Update rollout model weights from a numpy pytree.

    This is the receiver side of ``NumpyDirectSync`` and
    ``FileWeightSync``.  It converts numpy leaves back to JAX arrays sharded
    on the rollout mesh and calls ``rollout.update_params``.

    Args:
      weights_numpy: Pytree of ``np.ndarray`` leaves with the same structure
        as ``nnx.state(actor_model, filter_types)``.
    """
    rollout = self._cluster.rollout
    rollout_model = rollout.model()
    rollout_state = nnx.state(rollout_model)

    # Re-shard: each leaf gets the sharding of the corresponding rollout param.
    def _put_with_rollout_sharding(np_leaf, rollout_leaf):
      return jax.device_put(
          jnp.asarray(np_leaf), rollout_leaf.sharding
      )

    # The incoming state may carry only LoRA params or all Params; match
    # by structure rather than forcing an exact match.
    try:
      new_params = jax.tree_util.tree_map(
          _put_with_rollout_sharding, weights_numpy, rollout_state
      )
    except ValueError:
      # Fallback: update_params handles mismatched filter_types internally.
      new_params = jax.tree_util.tree_map(jnp.asarray, weights_numpy)

    rollout.update_params(new_params)
    logging.info("RolloutActor: weights updated.")

  def update_weights_from_ref(self, obj_ref: Any) -> None:
    """Receive weights via Ray object store reference (``RayObjectStoreSync``).

    Args:
      obj_ref: A Ray ``ObjectRef`` that resolves to a numpy pytree.
    """
    import ray  # pylint: disable=g-import-not-at-top
    weights_numpy = ray.get(obj_ref)
    self.update_weights(weights_numpy)

  def load_weights_from_file(
      self,
      path: str,
      fmt: str = "npz",
      use_fsspec: bool = False,
  ) -> None:
    """Load weights from a shared filesystem path (``FileWeightSync``).

    Args:
      path: Path written by the trainer actor's ``FileWeightSync.transfer``.
      fmt: Serialisation format; must match what the trainer used.
      use_fsspec: Whether to use ``fsspec`` for filesystem access.
    """
    import pickle  # pylint: disable=g-import-not-at-top
    import jax  # pylint: disable=g-import-not-at-top

    if fmt == "npz":
      meta_path = path + ".treedef"
      if use_fsspec:
        import fsspec  # pylint: disable=g-import-not-at-top
        with fsspec.open(path, "rb") as f:
          data = np.load(io.BytesIO(f.read()), allow_pickle=False)
        with fsspec.open(meta_path, "rb") as f:
          treedef = pickle.loads(f.read())
      else:
        data = np.load(path, allow_pickle=False)
        with open(meta_path, "rb") as f:
          treedef = pickle.loads(f.read())
      n = len(data.files)
      leaves = [data[str(i)] for i in range(n)]
      weights_numpy = jax.tree_util.tree_unflatten(treedef, leaves)

    elif fmt == "msgpack":
      import msgpack  # pylint: disable=g-import-not-at-top
      import msgpack_numpy  # pylint: disable=g-import-not-at-top
      msgpack_numpy.patch()
      if use_fsspec:
        import fsspec  # pylint: disable=g-import-not-at-top
        with fsspec.open(path, "rb") as f:
          packed = f.read()
      else:
        with open(path, "rb") as f:
          packed = f.read()
      payload = msgpack.unpackb(packed, object_hook=msgpack_numpy.decode)
      treedef = pickle.loads(payload["treedef"])
      flat = payload["flat"]
      n = len(flat)
      leaves = [flat[str(i)] for i in range(n)]
      weights_numpy = jax.tree_util.tree_unflatten(treedef, leaves)
    else:
      raise ValueError(f"Unsupported fmt: {fmt!r}")

    self.update_weights(weights_numpy)
    logging.info("RolloutActor: weights loaded from %s", path)

  # ---------------------------------------------------------------------------
  # Misc helpers
  # ---------------------------------------------------------------------------

  def pad_id(self) -> int:
    return self._cluster.rollout.pad_id()

  def eos_id(self) -> int:
    return self._cluster.rollout.eos_id()

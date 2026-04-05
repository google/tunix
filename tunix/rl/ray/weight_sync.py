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

"""Weight synchronization strategies for distributed RL training.

This module provides a pluggable abstraction for transferring model weights
from the trainer process to the rollout engine process. Multiple strategies
are supported:

  - ``NumpyDirectSync``: Pass numpy arrays directly as Ray remote-call
    arguments (simplest; good for moderate model sizes where object-store
    serialisation overhead is acceptable).
  - ``RayObjectStoreSync``: Stage weights in Ray's distributed object store
    so that the rollout actor fetches them by reference rather than having
    them pushed inline.
  - ``FileWeightSync``: Serialise weights to a shared filesystem path
    (local, NFS, or GCS via ``gcsfs``/``fsspec``).  Useful when processes
    run on different nodes without shared GPU memory.
  - ``JaxDevicePutSync``: Use ``jax.device_put`` to move arrays between
    meshes within the **same** process.  This mirrors the existing Pathways/
    ``reshard_pytree`` behaviour and is provided mainly for testing and
    single-process fallback scenarios.

Usage::

    from tunix.rl.ray.weight_sync import NumpyDirectSync
    strategy = NumpyDirectSync()
    strategy.sync(trainer_actor_handle, rollout_actor_handle)
"""

from __future__ import annotations

import abc
import io
import os
import tempfile
from typing import Any

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np

# NpPyTree: a pytree whose leaves are numpy arrays (possibly nested dicts).
NpPyTree = Any


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class WeightSyncStrategy(abc.ABC):
  """Abstract strategy for syncing model weights from trainer to rollout.

  Subclasses implement the three-step protocol:

  1. ``extract(trainer_actor)`` – pull raw numpy weights from the trainer
     actor (runs in the orchestrator process).
  2. ``transfer(weights, rollout_actor)`` – deliver those weights to the
     rollout actor using the strategy-specific transport.

  The public ``sync`` method chains both steps; callers can also invoke
  them individually when fine-grained control is needed.
  """

  @abc.abstractmethod
  def extract(self, trainer_actor: Any) -> NpPyTree:
    """Extract current weights from the trainer actor as numpy arrays.

    Args:
      trainer_actor: A ``TrainerActor`` Ray remote handle (or any object with
        a ``get_weights_numpy()`` method that returns a numpy pytree).

    Returns:
      A pytree of ``np.ndarray`` leaves representing the current policy
      parameters.
    """

  @abc.abstractmethod
  def transfer(self, weights: NpPyTree, rollout_actor: Any) -> None:
    """Deliver ``weights`` to ``rollout_actor`` using this strategy.

    Args:
      weights: Numpy pytree returned by ``extract``.
      rollout_actor: A ``RolloutActor`` Ray remote handle (or any object
        with an ``update_weights(weights)`` method).
    """

  def sync(self, trainer_actor: Any, rollout_actor: Any) -> None:
    """Convenience method: extract then transfer.

    Args:
      trainer_actor: Trainer actor handle.
      rollout_actor: Rollout actor handle.
    """
    weights = self.extract(trainer_actor)
    self.transfer(weights, rollout_actor)


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

class NumpyDirectSync(WeightSyncStrategy):
  """Send numpy arrays inline as Ray remote-call arguments.

  This is the default strategy. Ray serialises the numpy arrays via the
  plasma object store automatically when the argument size exceeds the
  inline threshold (≈ 100 KB by default).  For very large models consider
  ``RayObjectStoreSync`` to get explicit control over placement.
  """

  def extract(self, trainer_actor: Any) -> NpPyTree:
    import ray  # pylint: disable=g-import-not-at-top
    return ray.get(trainer_actor.get_weights_numpy.remote())

  def transfer(self, weights: NpPyTree, rollout_actor: Any) -> None:
    import ray  # pylint: disable=g-import-not-at-top
    ray.get(rollout_actor.update_weights.remote(weights))


class RayObjectStoreSync(WeightSyncStrategy):
  """Stage weights in Ray's distributed object store, transfer by reference.

  The trainer actor serialises weights into the object store and returns an
  ``ObjectRef``.  The rollout actor fetches those weights by resolving the
  ref inside its own process, avoiding a redundant copy through the
  orchestrator.
  """

  def extract(self, trainer_actor: Any) -> Any:
    """Returns a Ray ObjectRef pointing to the numpy weight pytree."""
    import ray  # pylint: disable=g-import-not-at-top
    # get_weights_numpy_ref returns an ObjectRef directly, skipping an
    # unnecessary round-trip through the orchestrator process.
    return trainer_actor.get_weights_numpy_ref.remote()

  def transfer(self, obj_ref: Any, rollout_actor: Any) -> None:
    import ray  # pylint: disable=g-import-not-at-top
    ray.get(rollout_actor.update_weights_from_ref.remote(obj_ref))


class FileWeightSync(WeightSyncStrategy):
  """Serialise weights to a shared path; rollout actor reads from there.

  Supports any filesystem reachable by both trainer and rollout actors.
  By default uses ``numpy.savez``/``numpy.load`` for the NPZ format.
  Pass ``fmt="msgpack"`` to use ``msgpack_numpy`` for faster serialisation
  (requires ``msgpack`` and ``msgpack_numpy`` packages).

  Args:
    path: Directory (or exact file prefix) to write weight files into.  Must
      be accessible from both the trainer actor and the rollout actor.  Can be
      a local path (for single-node setups) or a GCS/S3 path when using
      ``fsspec``-backed open.
    fmt: Serialisation format.  One of ``"npz"`` (default) or ``"msgpack"``.
    use_fsspec: If True, open files via ``fsspec`` so that cloud paths like
      ``gs://`` work transparently.
  """

  def __init__(
      self,
      path: str,
      fmt: str = "npz",
      use_fsspec: bool = False,
  ):
    if fmt not in ("npz", "msgpack"):
      raise ValueError(f"Unsupported fmt: {fmt!r}. Choose 'npz' or 'msgpack'.")
    self._path = path
    self._fmt = fmt
    self._use_fsspec = use_fsspec
    self._step = 0

  def _weight_path(self) -> str:
    return os.path.join(self._path, f"weights_step_{self._step}.{self._fmt}")

  # -- helpers ---------------------------------------------------------------

  @staticmethod
  def _flatten(weights: NpPyTree) -> dict[str, np.ndarray]:
    """Flatten a pytree of np.ndarrays into a dict with string keys."""
    leaves, treedef = jax.tree_util.tree_flatten(weights)
    keys = [str(i) for i in range(len(leaves))]
    return dict(zip(keys, leaves)), treedef

  @staticmethod
  def _unflatten(flat: dict[str, np.ndarray], treedef) -> NpPyTree:
    """Reconstruct a pytree from flat dict + treedef."""
    n = len(flat)
    leaves = [flat[str(i)] for i in range(n)]
    return jax.tree_util.tree_unflatten(treedef, leaves)

  def _open(self, path: str, mode: str):
    if self._use_fsspec:
      import fsspec  # pylint: disable=g-import-not-at-top
      return fsspec.open(path, mode)
    return open(path, mode)  # pylint: disable=unspecified-encoding

  # -- protocol --------------------------------------------------------------

  def extract(self, trainer_actor: Any) -> NpPyTree:
    import ray  # pylint: disable=g-import-not-at-top
    return ray.get(trainer_actor.get_weights_numpy.remote())

  def transfer(self, weights: NpPyTree, rollout_actor: Any) -> None:
    import ray  # pylint: disable=g-import-not-at-top
    path = self._weight_path()
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    if self._fmt == "npz":
      flat, treedef = self._flatten(weights)
      # Also save treedef structure (as a string) so rollout can reconstruct.
      buf = io.BytesIO()
      np.savez(buf, **flat)
      buf_bytes = buf.getvalue()
      import pickle  # pylint: disable=g-import-not-at-top
      treedef_bytes = pickle.dumps(treedef)
      meta_path = path + ".treedef"
      if self._use_fsspec:
        import fsspec  # pylint: disable=g-import-not-at-top
        with fsspec.open(path, "wb") as f:
          f.write(buf_bytes)
        with fsspec.open(meta_path, "wb") as f:
          f.write(treedef_bytes)
      else:
        with open(path, "wb") as f:
          f.write(buf_bytes)
        with open(meta_path, "wb") as f:
          f.write(treedef_bytes)
    elif self._fmt == "msgpack":
      import msgpack  # pylint: disable=g-import-not-at-top
      import msgpack_numpy  # pylint: disable=g-import-not-at-top
      msgpack_numpy.patch()
      flat, treedef = self._flatten(weights)
      import pickle  # pylint: disable=g-import-not-at-top
      payload = {"flat": flat, "treedef": pickle.dumps(treedef)}
      packed = msgpack.packb(payload, default=msgpack_numpy.encode)
      if self._use_fsspec:
        import fsspec  # pylint: disable=g-import-not-at-top
        with fsspec.open(path, "wb") as f:
          f.write(packed)
      else:
        with open(path, "wb") as f:
          f.write(packed)

    logging.info("FileWeightSync: wrote weights to %s", path)
    ray.get(rollout_actor.load_weights_from_file.remote(path, self._fmt, self._use_fsspec))
    self._step += 1


class JaxDevicePutSync(WeightSyncStrategy):
  """Transfer weights via ``jax.device_put`` between meshes in one process.

  This replicates the existing ``reshard_pytree`` / Pathways behaviour and
  is useful for:

  * Single-process testing where trainer and rollout share the same Python
    runtime.
  * Environments where Pathways is unavailable but both meshes are visible.

  .. warning::
      This strategy requires the trainer actor and rollout actor to be plain
      Python objects (not Ray remote actors) living in the **same** process.
      It will not work across process boundaries.

  Args:
    dst_mesh: Target JAX mesh for the rollout engine.  If ``None``, the
      sharding of each leaf is inferred from the existing rollout model.
  """

  def __init__(self, dst_mesh: jax.sharding.Mesh | None = None):
    self._dst_mesh = dst_mesh

  def extract(self, trainer_actor: Any) -> NpPyTree:
    """Get JAX arrays (not numpy) directly from the trainer object."""
    from flax import nnx  # pylint: disable=g-import-not-at-top
    return nnx.state(trainer_actor.model)

  def transfer(self, state, rollout_actor: Any) -> None:
    """Apply via jax.device_put to the rollout actor's model."""
    rollout_model = rollout_actor.rollout.model()
    from flax import nnx  # pylint: disable=g-import-not-at-top
    rollout_state = nnx.state(rollout_model)
    if self._dst_mesh is not None:
      from tunix.rl import reshard  # pylint: disable=g-import-not-at-top
      dst_shardings = jax.tree_util.tree_map(
          lambda x: jax.sharding.NamedSharding(self._dst_mesh, x.sharding.spec),
          rollout_state,
      )
      new_state = reshard.reshard_pytree(state, dst_shardings)
    else:
      new_state = jax.tree_util.tree_map(
          lambda src, dst: jax.device_put(src, dst.sharding),
          state,
          rollout_state,
      )
    rollout_actor.rollout.update_params(new_state)

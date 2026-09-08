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

"""Keeps sub-batch ledger payloads in the Trajectory Store.

Sub-batch checkpointing requires a Trajectory Store: `TrajectoryCollectEngine`
writes every finished trajectory to it and tags its Token-mode dict with the
store id. `TrajectoryStorePayloads` lets `SubBatchCheckpointManager` record
only that id (plus a digest and the item metadata) per active trajectory;
restore rebuilds the dict from the store with the same
`token_dict.build_token_dict` that produced it.

Every id is verified before it is first referenced: the store is flushed, the
trajectory is read back, rebuilt, and its digest compared to the live dict. A
snapshot therefore never references a trajectory that was not durably written
(the file store writes asynchronously and only logs failures) or that does not
rebuild to exactly what training consumed.
"""

from collections.abc import Sequence
import enum
import hashlib
import json
from typing import Any, Final

import jax
import numpy as np
from tunix.experimental.trajectory import store as store_lib
from tunix.rl.agentic.trajectory import token_dict

# Keys of a payload reference; all values are str so Orbax stores them as-is.
_TRAJECTORY_ID: Final[str] = "trajectory_id"
_DIGEST: Final[str] = "digest"
_METADATA_JSON: Final[str] = "metadata_json"

Payload = tuple[dict[str, Any], dict[str, Any]]


def _feed(hasher: Any, value: Any) -> None:
  """Feeds a type-tagged canonical encoding of `value` into `hasher`.

  Lists and tuples encode identically, NumPy scalars encode as the equivalent
  Python scalar, and dict entries are ordered by their encoding. Everything
  else that affects training (array dtype and shape, int vs float, str vs
  bytes) is part of the encoding.

  Args:
    hasher: A hashlib hash object.
    value: The value to encode.

  Raises:
    TypeError: For values with no canonical encoding (object arrays, arbitrary
      objects); digesting them would silently treat unequal values as equal.
  """
  if isinstance(value, jax.Array):
    value = np.asarray(value)
  if isinstance(value, np.generic):
    value = value.item()
  if value is None:
    hasher.update(b"N;")
  elif isinstance(value, enum.Enum):
    name = value.name.encode()
    hasher.update(b"E%d:" % len(name) + name)
  elif isinstance(value, bool):
    hasher.update(b"B1;" if value else b"B0;")
  elif isinstance(value, int):
    hasher.update(b"I%d;" % value)
  elif isinstance(value, float):
    hasher.update(b"F" + value.hex().encode() + b";")
  elif isinstance(value, str):
    data = value.encode()
    hasher.update(b"S%d:" % len(data) + data)
  elif isinstance(value, bytes):
    hasher.update(b"Y%d:" % len(value) + value)
  elif isinstance(value, np.ndarray):
    if value.dtype.hasobject or value.dtype.kind == "V":
      raise TypeError(f"Cannot digest a NumPy array of dtype {value.dtype}.")
    data = np.ascontiguousarray(value).tobytes()
    hasher.update(
        f"A{value.dtype.str}{value.shape}{len(data)}:".encode() + data
    )
  elif isinstance(value, dict):
    entries = sorted(
        (canonical_digest(k), canonical_digest(v)) for k, v in value.items()
    )
    hasher.update(b"D%d:" % len(entries))
    for key_digest, value_digest in entries:
      hasher.update(key_digest.encode() + value_digest.encode())
  elif isinstance(value, (list, tuple)):
    hasher.update(b"L%d:" % len(value))
    for item in value:
      _feed(hasher, item)
  else:
    raise TypeError(f"Cannot digest a value of type {type(value).__name__}.")


def canonical_digest(value: Any) -> str:
  """Returns a hex digest that is equal iff `value`s train identically."""
  hasher = hashlib.sha256()
  _feed(hasher, value)
  return hasher.hexdigest()


def _top_level_diff(actual: dict[str, Any], expected: dict[str, Any]) -> str:
  """Names the top-level keys whose values digest differently."""
  keys = sorted(
      key
      for key in actual.keys() | expected.keys()
      if key not in actual
      or key not in expected
      or canonical_digest(actual[key]) != canonical_digest(expected[key])
  )
  return ", ".join(keys) or "<key order or nesting only>"


class TrajectoryStorePayloads:
  """`sub_batch_checkpoint.TrajectoryPayloadStore` backed by a Trajectory Store.

  The store must be the one the collect engines write to, and must be
  readable by every process that restores (not the process-local in-memory
  backend).
  """

  def __init__(self, store: store_lib.TrajectoryStore) -> None:
    self._store = store
    # trajectory_id -> digest of the verified Token-mode dict. Pruned to the
    # ids of the latest commit, which bounds it by the active rollout window.
    self._verified: dict[str, str] = {}

  def _rebuild(self, trajectory_id: str) -> dict[str, Any]:
    (record,) = self._store.get_trajectories([trajectory_id])
    trajectory, context = token_dict.from_store_record(record)
    return token_dict.build_token_dict(
        trajectory, context, trajectory_id=trajectory_id
    )

  def commit(self, payloads: Sequence[Payload]) -> list[dict[str, str]]:
    """Returns references to `payloads`, verifying unseen ids first.

    Args:
      payloads: (traj, metadata) of each active trajectory item, where traj is
        a Token-mode dict tagged with `token_dict.TRAJECTORY_ID_KEY`.

    Returns:
      One str-valued reference per payload, in order.

    Raises:
      ValueError: If a traj carries no store id, its metadata does not
        round-trip through JSON, or the store's copy does not rebuild to the
        same dict.
    """
    entries: list[tuple[str, dict[str, Any], str, str]] = []
    for traj, metadata in payloads:
      if not isinstance(traj, dict) or token_dict.TRAJECTORY_ID_KEY not in traj:
        raise ValueError(
            "Trajectory Store payloads require Token-mode trajectories tagged"
            f" with {token_dict.TRAJECTORY_ID_KEY!r}; configure the collect"
            " engines with the same trajectory store."
        )
      metadata_json = json.dumps(metadata, sort_keys=True)
      if canonical_digest(json.loads(metadata_json)) != canonical_digest(
          metadata
      ):
        raise ValueError(
            f"Trajectory item metadata {metadata!r} does not round-trip"
            " through JSON."
        )
      entries.append((
          traj[token_dict.TRAJECTORY_ID_KEY],
          traj,
          canonical_digest(traj),
          metadata_json,
      ))

    unverified = [
        (trajectory_id, traj, digest)
        for trajectory_id, traj, digest, _ in entries
        if self._verified.get(trajectory_id) != digest
    ]
    if unverified:
      self._store.flush()
    for trajectory_id, traj, digest in unverified:
      rebuilt = self._rebuild(trajectory_id)
      if canonical_digest(rebuilt) != digest:
        raise ValueError(
            f"Trajectory {trajectory_id!r} in the Trajectory Store does not"
            " rebuild to the collected Token-mode dict; differing keys:"
            f" {_top_level_diff(rebuilt, traj)}."
        )
      self._verified[trajectory_id] = digest

    self._verified = {
        trajectory_id: digest for trajectory_id, _, digest, _ in entries
    }
    return [
        {
            _TRAJECTORY_ID: trajectory_id,
            _DIGEST: digest,
            _METADATA_JSON: metadata_json,
        }
        for trajectory_id, _, digest, metadata_json in entries
    ]

  def load(self, refs: Sequence[dict[str, str]]) -> list[Payload]:
    """Rebuilds the payloads `commit` returned `refs` for.

    Args:
      refs: References returned by `commit`.

    Returns:
      (traj, metadata) per reference, in order.

    Raises:
      ValueError: If a rebuilt traj does not match its recorded digest.
      store_lib.TrajectoryNotFoundError: If a referenced trajectory is gone.
    """
    payloads = []
    for ref in refs:
      trajectory_id = str(ref[_TRAJECTORY_ID])
      traj = self._rebuild(trajectory_id)
      if canonical_digest(traj) != str(ref[_DIGEST]):
        raise ValueError(
            f"Trajectory {trajectory_id!r} rebuilt from the Trajectory Store"
            " does not match the digest recorded at save."
        )
      payloads.append((traj, json.loads(str(ref[_METADATA_JSON]))))
    return payloads

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

"""Composite Trajectory Store for multi-backend write fan-out."""

from collections.abc import Mapping, Sequence
import contextlib
from typing import Any, ClassVar, TypeVar

from tunix.experimental.trajectory import base_store as store
from tunix.experimental.trajectory import trajectory as trajectory_lib

MetadataT = TypeVar("MetadataT", bound=trajectory_lib.TrajectoryMetadata)


def _close_stores(stores: Sequence[store.TrajectoryStore[Any]]) -> None:
  """Closes all stores in `stores`, ensuring every store's `close` is called."""
  # Register callbacks in reverse so ExitStack's LIFO unwinding closes stores in
  # forward order (`stores[0]`, `stores[1]`, ...) even if an earlier `close()`
  # raises an exception.
  with contextlib.ExitStack() as stack:
    for child in reversed(stores):
      stack.callback(child.close)


class CompositeTrajectoryStore(store.TrajectoryStore[MetadataT]):
  """TrajectoryStore that fans out writes across multiple child backends.

  `CompositeTrajectoryStore` implements the GoF Composite pattern over
  `TrajectoryStore`, allowing a run to persist trajectories to multiple
  backends simultaneously (for example, `FileTrajectoryStore` +
  `SqlTrajectoryStore`) while exposing a single `TrajectoryStore` interface to
  rollout workers.

  Architectural Decisions & Invariants:
    1. Strict Decoupling:
       Interacts with child stores exclusively through the `TrajectoryStore`
       contract (`add_step`, `update_metadata`, `get_trajectories_metadata`,
       `get_trajectories`, `flush`, `close`, `to_config`, and
       `to_redacted_config`), so any combination of built-in or custom store
       backends can be composed without backend-specific coupling.
    2. Independent Backend Queues & Worker Threads:
       Asynchronous backends (`FileTrajectoryStore`, `SqlTrajectoryStore`) own
       their own bounded FIFO queues and background daemon threads, while
       `trajectory.deep_copy` shares immutable scalar payloads (`str`, `int`,
       `float`, `bool`) by reference across snapshots. Slow I/O or transient
       backpressure on one backend never blocks or drops writes on another.
    3. Primary-Store Read Delegation:
       Synchronous read queries (`get_trajectories_metadata()` and
       `get_trajectories()`) delegate to the primary child store (`stores[0]`).
  """

  BACKEND: ClassVar[str] = "composite"

  def __init__(
      self,
      stores: Sequence[store.TrajectoryStore[MetadataT]],
  ) -> None:
    """Initializes CompositeTrajectoryStore with one or more child stores.

    Args:
      stores: Non-empty sequence of child `TrajectoryStore` instances. Reads
        delegate to `stores[0]`; writes fan out to all stores in order.

    Raises:
      ValueError: If `stores` is empty.
      TypeError: If any element of `stores` is not a `TrajectoryStore` instance.
    """
    super().__init__()
    if isinstance(stores, (str, bytes)) or not stores:
      raise ValueError(
          "CompositeTrajectoryStore requires at least one child store."
      )
    child_stores = tuple(stores)
    for child in child_stores:
      if not isinstance(child, store.TrajectoryStore):
        raise TypeError(
            "CompositeTrajectoryStore requires TrajectoryStore instances, got"
            f" {type(child).__name__}."
        )
    self._stores: tuple[store.TrajectoryStore[MetadataT], ...] = child_stores

  @property
  def stores(self) -> tuple[store.TrajectoryStore[MetadataT], ...]:
    """Returns the child stores wrapped by this composite store."""
    return self._stores

  @classmethod
  def _from_config(
      cls,
      config: Mapping[str, Any],
  ) -> "CompositeTrajectoryStore[MetadataT]":
    """Builds a CompositeTrajectoryStore recursively from `config["stores"]`.

    Child configs with `"enabled": False` are skipped via
    `TrajectoryStore.from_config`. At least one child store must remain enabled.

    Args:
      config: Configuration mapping requiring a `"stores"` sequence of child
        store configuration dicts.

    Returns:
      A new `CompositeTrajectoryStore` wrapping the enabled child stores.

    Raises:
      ValueError: If `"stores"` is missing, not a sequence of mappings, or
        produces no enabled child stores.
    """
    raw_stores = config.get("stores")
    if (
        not isinstance(raw_stores, Sequence)
        or isinstance(raw_stores, (str, bytes))
        or not raw_stores
    ):
      raise ValueError(
          "CompositeTrajectoryStore requires a non-empty 'stores' sequence."
      )

    with contextlib.ExitStack() as stack:
      children: list[store.TrajectoryStore[Any]] = []
      for child_cfg in raw_stores:
        if not isinstance(child_cfg, Mapping):
          raise ValueError(
              "Each entry in 'stores' must be a configuration mapping."
          )
        child = store.TrajectoryStore.from_config(child_cfg)
        if child is not None:
          stack.callback(child.close)
          children.append(child)
      if not children:
        raise ValueError(
            "CompositeTrajectoryStore requires at least one enabled child"
            " store in 'stores'."
        )
      instance = cls(stores=children)
      # Transfer ownership of the open child stores to `instance` so exiting the
      # `with` block does not close them; if an exception was raised above,
      # `ExitStack` automatically closes all already-created children.
      stack.pop_all()
      return instance

  def to_config(self) -> dict[str, Any]:
    """Returns the config dict that rebuilds an equivalent composite store."""
    return {
        "enabled": True,
        "backend": self.BACKEND,
        "stores": [child.to_config() for child in self._stores],
    }

  def to_redacted_config(self) -> dict[str, Any]:
    """Returns `to_config()` with secrets masked in every child store."""
    return {
        "enabled": True,
        "backend": self.BACKEND,
        "stores": [child.to_redacted_config() for child in self._stores],
    }

  def add_step(
      self,
      step: trajectory_lib.Step,
      metadata: MetadataT,
  ) -> None:
    """Logs a turn step and its trajectory metadata across all child stores.

    Args:
      step: Step object to log.
      metadata: TrajectoryMetadata containing trajectory_id and run metadata.

    Raises:
      TypeError: If metadata is not a TrajectoryMetadata instance or its type
        conflicts with the store's bound metadata type.
      ValueError: If any child store rejects the metadata or step identifiers.
      RuntimeError: If any child store has already been closed.
    """
    self._validate_and_bind_metadata(metadata)
    for child in self._stores:
      child.add_step(step, metadata)

  def update_metadata(
      self,
      metadata: MetadataT,
  ) -> None:
    """Updates or creates trajectory metadata across all child stores.

    Args:
      metadata: TrajectoryMetadata containing trajectory_id and run metadata.

    Raises:
      TypeError: If metadata is not a TrajectoryMetadata instance or its type
        conflicts with the store's bound metadata type.
      ValueError: If any child store rejects the metadata identifiers.
      RuntimeError: If any child store has already been closed.
    """
    self._validate_and_bind_metadata(metadata)
    for child in self._stores:
      child.update_metadata(metadata)

  def get_trajectories_metadata(
      self, trajectory_ids: list[str] | None = None
  ) -> list[MetadataT]:
    """Retrieves trajectory metadata from the primary child store (`stores[0]`).

    Args:
      trajectory_ids: Optional list of unique trajectory identifiers. If
        specified, only metadata for these IDs is returned. If None, metadata
        for all trajectories in the run is returned.

    Returns:
      A list of TrajectoryMetadata objects from `stores[0]`.
    """
    metas = self._stores[0].get_trajectories_metadata(trajectory_ids)
    for meta in metas:
      self._validate_and_bind_metadata(meta)
    return metas

  def get_trajectories(
      self, trajectory_ids: list[str]
  ) -> list[trajectory_lib.Trajectory[Any]]:
    """Retrieves full trajectories from the primary child store (`stores[0]`).

    Args:
      trajectory_ids: List of unique trajectory identifiers to load.

    Returns:
      A list of full Trajectory objects from `stores[0]`.
    """
    trajs = self._stores[0].get_trajectories(trajectory_ids)
    for traj in trajs:
      self._rehydrate_metadata(traj.to_atif_metadata())
    return trajs

  def flush(self) -> None:
    """Flushes pending writes across all child stores."""
    for child in self._stores:
      child.flush()

  def close(self) -> None:
    """Flushes and closes all child stores."""
    _close_stores(self._stores)

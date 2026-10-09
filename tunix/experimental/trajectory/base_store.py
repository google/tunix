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

"""Base protocols and abstract class defining Trajectory Store interfaces."""

import abc
from typing import Any, ClassVar, Mapping, Protocol, TypeVar, runtime_checkable

from tunix.experimental.trajectory import trajectory as trajectory_lib

MetadataT = TypeVar("MetadataT", bound=trajectory_lib.TrajectoryMetadata)

# Config key naming the registered `METADATA_TYPE` of the TrajectoryMetadata
# subclass a store reads back. Required by `TrajectoryStore.from_config`.
METADATA_TYPE_KEY = "metadata_type"

# ==============================================================================
# Custom Exceptions
# ==============================================================================


class TrajectoryNotFoundError(KeyError):
  """Raised when a requested trajectory ID is not found in the store."""

  def __init__(self, trajectory_id: str) -> None:
    super().__init__(f"Trajectory with ID '{trajectory_id}' not found.")
    self.trajectory_id = trajectory_id


class TrajectoryMetadataNotFoundError(KeyError):
  """Raised when requested trajectory metadata is not found in the store."""

  def __init__(self, trajectory_id: str) -> None:
    super().__init__(f"Trajectory metadata for ID '{trajectory_id}' not found.")
    self.trajectory_id = trajectory_id


# ==============================================================================
# Protocols (Structural Interfaces)
# ==============================================================================


@runtime_checkable
class TrajectoryReader(Protocol[MetadataT]):
  """Structural protocol defining read-only Trajectory Store operations."""

  def get_trajectories_metadata(
      self, trajectory_ids: list[str] | None = None
  ) -> list[MetadataT]:
    """Retrieves metadata for trajectories in the run.

    Args:
      trajectory_ids: Optional list of unique trajectory identifiers. If
        specified, only metadata for these IDs is returned. If None, metadata
        for all trajectories in the run is returned.

    Returns:
      A list of TrajectoryMetadata objects for the requested trajectories.

    Raises:
      TrajectoryMetadataNotFoundError: If any requested trajectory ID does not
        exist.
    """
    ...

  def get_trajectories(
      self, trajectory_ids: list[str]
  ) -> list[trajectory_lib.Trajectory[Any]]:
    """Retrieves full trajectories for a list of trajectory IDs.

    Args:
      trajectory_ids: List of unique trajectory identifiers to load.

    Returns:
      A list of full Trajectory objects corresponding to the requested IDs.

    Raises:
      TrajectoryNotFoundError: If any requested trajectory ID does not exist.
    """
    ...


@runtime_checkable
class TrajectoryWriter(Protocol[MetadataT]):
  """Structural protocol defining write Trajectory Store operations."""

  def add_step(
      self,
      step: trajectory_lib.Step,
      metadata: MetadataT,
  ) -> None:
    """Logs a turn step and its trajectory metadata.

    Depending on the backend implementation, writes may be queued
    asynchronously. Readers never observe partially written or inconsistent
    state.

    Implementations snapshot `step` and `metadata` at call time, so callers may
    keep mutating those objects afterwards without affecting what was logged.

    Args:
      step: Step object to log.
      metadata: TrajectoryMetadata containing trajectory_id and run metadata.
    """
    ...

  def update_metadata(
      self,
      metadata: MetadataT,
  ) -> None:
    """Updates (or creates) trajectory metadata.

    Implementations snapshot `metadata` at call time, so callers may keep
    mutating it afterwards without affecting what was logged.

    Args:
      metadata: TrajectoryMetadata containing trajectory_id and run metadata.
    """
    ...

  def flush(self) -> None:
    """Flushes any pending or asynchronous writes to persistent storage.

    Users do not need to call flush() in normal usage; it is primarily for
    testing.
    """
    ...

  def close(self) -> None:
    """Flushes pending writes and releases the writer's resources.

    Implementations must be idempotent, and must not be used for writing after
    being closed. Backends that write asynchronously also close themselves at
    interpreter exit, so calling `close()` is only required to release
    resources earlier, e.g. for a writer created inside a loop.
    """
    ...


# ==============================================================================
# Base class (config <-> instance)
# ==============================================================================


class TrajectoryStore(
    TrajectoryReader[MetadataT],
    TrajectoryWriter[MetadataT],
    abc.ABC,
):
  """Base class pairing a store implementation with its own configuration.

  Every backend owns both directions of its configuration: `_from_config`
  builds an instance from a plain dict and `to_config` rebuilds that dict, so a
  new backend adds its own keys and its own validation in one place instead of
  growing a shared config object that knows about every backend's fields.

  `from_config` is the single construction entry point for the processes that
  make up a run, and doubles as the on/off gate: a config of None, or one whose
  "enabled" is false, yields None, leaving every store-guarded call site a
  no-op.

  Every store is constructed with a required `metadata_cls`, the
  TrajectoryMetadata subclass it reads metadata back as. The argument is typed
  `type[MetadataT]`, so the type checker infers `MetadataT` from it and the
  metadata type is stated exactly once:

      store = FileTrajectoryStore(
          root_dir=root, run_id=run_id, metadata_cls=TunixTrajectoryMetadata)
      # Inferred: FileTrajectoryStore[TunixTrajectoryMetadata].

  Because it is required, a store annotated with one metadata type but built
  with another, or with none, is a static type error rather than a store that
  silently reads a different type. A store that reads base metadata says so
  with `metadata_cls=TrajectoryMetadata`.
  """

  # The value of the config's "backend" key that selects this class.
  BACKEND: ClassVar[str]
  _REGISTRY: ClassVar[dict[str, type["TrajectoryStore[Any]"]]] = {}

  def __init_subclass__(cls, **kwargs: Any) -> None:
    super().__init_subclass__(**kwargs)
    backend = cls.__dict__.get("BACKEND")
    if backend:
      cls._REGISTRY[backend] = cls

  def __init__(self, *, metadata_cls: type[MetadataT]) -> None:
    """Records the metadata class this store reads back.

    Args:
      metadata_cls: The TrajectoryMetadata subclass to read stored metadata back
        as. The type checker infers `MetadataT` from it. Must declare a
        `METADATA_TYPE` registered in `TrajectoryMetadata._REGISTRY`.

    Raises:
      TypeError: If `metadata_cls` is not a TrajectoryMetadata subclass.
      ValueError: If `metadata_cls` is not registered in
        `TrajectoryMetadata._REGISTRY`.
    """
    if (
        not isinstance(metadata_cls, type)
        or not issubclass(metadata_cls, trajectory_lib.TrajectoryMetadata)
        or issubclass(metadata_cls, trajectory_lib.Trajectory)
    ):
      raise TypeError(
          f"{type(self).__name__} requires metadata_cls to be a"
          f" TrajectoryMetadata subclass; got {metadata_cls}."
      )
    self._metadata_type: str = self._get_metadata_type_name(metadata_cls)
    self._metadata_cls: type[MetadataT] = metadata_cls

  @classmethod
  def _resolve_metadata_type_name(
      cls, name: str
  ) -> type[trajectory_lib.TrajectoryMetadata]:
    """Resolves a registered metadata_type name to its TrajectoryMetadata class."""
    registry = (
        trajectory_lib.TrajectoryMetadata._REGISTRY  # pylint: disable=protected-access
    )
    if name in registry:
      return registry[name]

    valid = sorted(registry)
    raise ValueError(
        f"Unknown Trajectory Store metadata_type {name}; expected one of"
        f" {valid}."
    )

  @classmethod
  def _get_metadata_type_name(
      cls, metadata_cls: type[trajectory_lib.TrajectoryMetadata]
  ) -> str:
    """Returns the registered METADATA_TYPE name for `metadata_cls`."""
    meta_type = metadata_cls.__dict__.get("METADATA_TYPE")
    registry = (
        trajectory_lib.TrajectoryMetadata._REGISTRY  # pylint: disable=protected-access
    )
    if (
        not isinstance(meta_type, str)
        or not meta_type
        or registry.get(meta_type) is not metadata_cls
    ):
      raise ValueError(
          f"{metadata_cls.__name__} must declare a registered"
          " METADATA_TYPE in TrajectoryMetadata._REGISTRY."
      )
    return meta_type

  @classmethod
  @abc.abstractmethod
  def _from_config(
      cls,
      config: Mapping[str, Any],
      *,
      metadata_cls: type[trajectory_lib.TrajectoryMetadata],
  ) -> "TrajectoryStore[Any]":
    """Builds an instance of this backend from `config`.

    Implementations read the keys they care about and raise ValueError for a
    config this backend cannot honour. Called only by `from_config`, which has
    already established that `config` selects this backend and resolved its
    "metadata_type" into `metadata_cls`.
    """

  @abc.abstractmethod
  def to_config(self) -> dict[str, Any]:
    """Returns the config dict that rebuilds an equivalent store.

    `type(store).from_config(store.to_config())` must produce a store reading
    and writing the same data as `store`. Use it to report what a process
    actually built — two processes in one run that log different dicts are
    reading and writing different data.
    """

  def to_redacted_config(self) -> dict[str, Any]:
    """Returns `to_config()` with secrets masked, for logging and reporting.

    `to_config()` must round-trip, so it may carry credentials (e.g. a password
    in a database URL). Log this instead. Backends whose config holds secrets
    override it; the default returns `to_config()` unchanged.

    Returns:
      A dict with the same keys as `to_config()`. It is not guaranteed to be
      accepted by `from_config`.
    """
    return self.to_config()

  @classmethod
  def from_config(
      cls, config: Mapping[str, Any] | None
  ) -> "TrajectoryStore[Any] | None":
    """Builds the store described by `config`, or None when it is disabled.

    Call once per process and hold onto the result: the process that built a
    store owns closing it. Calling this twice in one process builds two
    independent stores (and, for the file backend, two background writer
    threads); nothing prevents that, the guard is simply that callers
    construct once.

    Args:
      config: Configuration mapping, or None. The "backend" key selects the
        implementation; "enabled" turns the store off without removing the rest
        of the config.

    Returns:
      A store instance, or None if `config` is None or not enabled.

    Raises:
      ValueError: If "backend" names no known implementation, "metadata_type"
        is missing or names an unknown metadata type, or the selected backend
        rejects the rest of the config.
    """
    if config is None or not config.get("enabled", False):
      return None

    backend = config.get("backend")
    if backend not in cls._REGISTRY:
      raise ValueError(
          f"Unknown Trajectory Store backend {backend!r}; expected one of"
          f" {sorted(cls._REGISTRY)}."
      )
    if not config.get(METADATA_TYPE_KEY):
      raise ValueError(
          f"Trajectory Store config requires a non-empty '{METADATA_TYPE_KEY}';"
          " use 'base' for plain TrajectoryMetadata."
      )
    metadata_cls = cls._resolve_metadata_type_name(config[METADATA_TYPE_KEY])
    return cls._REGISTRY[backend]._from_config(  # pylint: disable=protected-access
        config, metadata_cls=metadata_cls
    )

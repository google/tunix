"""Protocols defining Trajectory Store interfaces."""

import abc
from typing import Any, ClassVar, Mapping, Protocol, runtime_checkable

from tunix.experimental.trajectory import trajectory as trajectory_lib

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
class TrajectoryReader(Protocol):
  """Structural protocol defining read-only Trajectory Store operations."""

  def get_trajectories_metadata(
      self, trajectory_ids: list[str] | None = None
  ) -> list[trajectory_lib.TrajectoryMetadata]:
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
  ) -> list[trajectory_lib.Trajectory]:
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
class TrajectoryWriter(Protocol):
  """Structural protocol defining write Trajectory Store operations."""

  def add_step(
      self,
      step: trajectory_lib.Step,
      metadata: trajectory_lib.TrajectoryMetadata,
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
      metadata: trajectory_lib.TrajectoryMetadata,
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


class TrajectoryStore(TrajectoryReader, TrajectoryWriter, abc.ABC):
  """Base class pairing a store implementation with its own configuration.

  Every backend owns both directions of its configuration: `_from_config`
  builds an instance from a plain dict, and `to_config` returns the dict that
  would rebuild an equivalent one. Keeping the two next to the backend's
  `__init__` means a new backend adds its own keys and its own validation in
  one place, instead of growing a shared config object that has to know about
  every backend's fields.

  `TrajectoryStore.from_config` is the single construction entry point for
  building a store from a dictionary configuration.
  """

  # The value of the config's "backend" key that selects this class.
  BACKEND: ClassVar[str]
  _REGISTRY: ClassVar[dict[str, type["TrajectoryStore"]]] = {}

  def __init_subclass__(cls, **kwargs: Any) -> None:
    super().__init_subclass__(**kwargs)
    backend = getattr(cls, "BACKEND", None)
    if backend:
      cls._REGISTRY[backend] = cls

  @classmethod
  @abc.abstractmethod
  def _from_config(cls, config: Mapping[str, Any]) -> "TrajectoryStore":
    """Builds an instance of this backend from `config`.

    Implementations read the keys they care about and raise ValueError for a
    config this backend cannot honour. Called only by `from_config`, which has
    already established that `config` selects this backend.

    Args:
      config: Configuration mapping for this backend.

    Returns:
      A new store instance.
    """

  @abc.abstractmethod
  def to_config(self) -> dict[str, Any]:
    """Returns the config dict that rebuilds an equivalent store.

    For any store constructed via `from_config`,
    `type(store).from_config(store.to_config())` must produce a store reading
    and writing the same data as `store`. Use it to report what a process
    actually built — two processes in one run that log different dicts are
    reading and writing different data.

    Returns:
      A dict accepted by `from_config`, including "backend".
    """

  @classmethod
  def from_config(cls, config: Mapping[str, Any]) -> "TrajectoryStore":
    """Builds the store described by `config`.

    Call once per process and hold onto the result: the process that built a
    store owns closing it. Calling this twice in one process builds two
    independent stores (and, for the file backend, two background writer
    threads); nothing prevents that, the guard is simply that callers
    construct once.

    Args:
      config: Configuration mapping. The "backend" key selects the
        implementation.

    Returns:
      A constructed store instance.

    Raises:
      ValueError: If `config` is empty, if "backend" names no known
        implementation, or if the selected backend rejects the rest of the
        config.
    """
    if not config:
      raise ValueError("TrajectoryStore config must be a non-empty mapping.")

    # Ensure built-in backends are imported so their __init_subclass__ hooks
    # have registered them in _REGISTRY before lookup. Imported here rather
    # than at module level because both implementations import this module.
    from tunix.experimental.trajectory import file_store  # pylint: disable=g-import-not-at-top,unused-import
    from tunix.experimental.trajectory import in_memory_store  # pylint: disable=g-import-not-at-top,unused-import

    backend = config.get("backend")
    if backend not in cls._REGISTRY:
      raise ValueError(
          f"Unknown Trajectory Store backend {backend!r}; expected one of"
          f" {sorted(cls._REGISTRY)}."
      )
    return cls._REGISTRY[backend]._from_config(config)  # pylint: disable=protected-access

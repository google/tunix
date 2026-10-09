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
from collections.abc import Mapping
from typing import Any, ClassVar, Protocol, Self, TypeVar, cast, get_args, get_origin, runtime_checkable

from tunix.experimental.trajectory import trajectory as trajectory_lib

MetadataT = TypeVar("MetadataT", bound=trajectory_lib.TrajectoryMetadata)

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
      A list of TrajectoryMetadata objects for the requested trajectories, in
      the order of `trajectory_ids` if specified.

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
      A list of full Trajectory objects corresponding to the requested IDs, in
      the order of `trajectory_ids`.

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
  """

  # The value of the config's "backend" key that selects this class.
  BACKEND: ClassVar[str]
  _REGISTRY: ClassVar[dict[str, type["TrajectoryStore[Any]"]]] = {}

  # Bound metadata class for this store. Starts as `None` on unparameterized
  # stores, is set on the class when a subclass specifies a concrete metadata
  # type (e.g. `class MyStore(InMemoryStore[TunixMetadata])`), and is set on the
  # instance (`self._metadata_cls`) either from `self.__orig_class__` (e.g.
  # `InMemoryStore[TunixMetadata]()`) or on the first read/write operation.
  _metadata_cls: type[MetadataT] | None = None

  def __init_subclass__(cls, **kwargs: Any) -> None:
    super().__init_subclass__(**kwargs)
    # Check `cls.__dict__` (not `getattr`) so a subclass that inherits `BACKEND`
    # without declaring its own does not overwrite its parent in `_REGISTRY`.
    backend = cls.__dict__.get("BACKEND")
    if backend:
      cls._REGISTRY[backend] = cls

    # When a class inherits from a parameterized generic such as
    # `TrajectoryStore[TunixTrajectoryMetadata]`, Python erases the `[...]` type
    # argument from `cls.__bases__` and preserves the parameterized alias object
    # in `cls.__orig_bases__` (PEP 560).
    bases = cls.__dict__.get("__orig_bases__") or cls.__bases__
    metadata_classes = {
        meta_cls
        for base in bases
        if (meta_cls := cls._extract_metadata_cls(base)) is not None
    }
    if len(metadata_classes) > 1:
      names = ", ".join(sorted(m.__name__ for m in metadata_classes))
      raise TypeError(
          f"Conflicting TrajectoryMetadata types in {cls.__name__}: {names}."
      )
    cls._metadata_cls = next(iter(metadata_classes), None)

  @classmethod
  def _extract_metadata_cls(cls, base_or_alias: Any) -> type[MetadataT] | None:
    """Returns the concrete TrajectoryMetadata class bound to `base_or_alias`.

    Args:
      base_or_alias: Either a `TrajectoryStore` class (e.g.
        `TunixTrajectoryStore`) or a parameterized generic alias (e.g.
        `InMemoryTrajectoryStore[TunixTrajectoryMetadata]`).
    """
    # For a parameterized alias like `Store[Meta]`, `get_origin` returns `Store`
    # and `get_args` returns `(Meta,)`. For a plain class, `get_origin` returns
    # `None` and `get_args` returns `()`.
    origin = get_origin(base_or_alias) or base_or_alias
    if not (isinstance(origin, type) and issubclass(origin, TrajectoryStore)):
      return None
    for type_arg in get_args(base_or_alias):
      if (
          isinstance(type_arg, type)
          and issubclass(type_arg, trajectory_lib.TrajectoryMetadata)
          and not issubclass(type_arg, trajectory_lib.Trajectory)
      ):
        return cast(type[MetadataT], type_arg)
    # If `base_or_alias` was not directly parameterized (or was parameterized
    # with an unbound TypeVar), inherit any metadata class already bound on
    # `origin` (e.g. when subclassing `TunixTrajectoryStore`).
    return getattr(origin, "_metadata_cls", None)

  def _get_bound_metadata_cls(self) -> type[MetadataT] | None:
    """Returns the metadata class bound to this store instance, if any."""
    if self._metadata_cls is None:
      # CPython's `typing._GenericAlias.__call__` sets `self.__orig_class__` on
      # the instance *after* `__init__` returns, so subscripted instantiations
      # like `InMemoryTrajectoryStore[TunixTrajectoryMetadata]()` are resolved
      # lazily here on first read or write.
      orig_class = getattr(self, "__orig_class__", None)
      if orig_class is not None:
        self._metadata_cls = self._extract_metadata_cls(orig_class)
    return self._metadata_cls

  def _validate_and_bind_metadata(self, metadata: MetadataT) -> None:
    """Validates and binds the store instance's single TrajectoryMetadata type.

    Args:
      metadata: The metadata instance being written to the store.

    Raises:
      TypeError: If `metadata` is not a `TrajectoryMetadata` instance (or is a
        `Trajectory`), or if its type does not match the metadata type already
        bound to this store instance.
    """
    meta_cls = type(metadata)
    if self._metadata_cls is not None and meta_cls is self._metadata_cls:
      return
    if not isinstance(metadata, trajectory_lib.TrajectoryMetadata) or (
        isinstance(metadata, trajectory_lib.Trajectory)
    ):
      raise TypeError(
          "Expected a TrajectoryMetadata instance (not Trajectory), got"
          f" {meta_cls.__name__}."
      )
    bound_cls = self._get_bound_metadata_cls()
    if bound_cls is None:
      self._metadata_cls = meta_cls
    elif meta_cls is not bound_cls:
      raise TypeError(
          f"{type(self).__name__} is bound to metadata type"
          f" {bound_cls.__name__}, got {meta_cls.__name__}."
      )

  def _rehydrate_metadata(
      self, atif_metadata: trajectory_lib.TrajectoryMetadata
  ) -> MetadataT:
    """Rehydrates base ATIF metadata and enforces the store's metadata type.

    Args:
      atif_metadata: The deserialized base ATIF `TrajectoryMetadata` instance.

    Returns:
      The rehydrated metadata instance matching the store's bound metadata type.

    Raises:
      TypeError: If the resolved metadata class does not match the metadata type
        already bound to this store instance.
    """
    bound_cls = self._get_bound_metadata_cls()
    resolved_cls = cast(
        type[MetadataT],
        (bound_cls or trajectory_lib.TrajectoryMetadata).resolve_subclass(
            atif_metadata
        ),
    )
    if bound_cls is None:
      self._metadata_cls = resolved_cls
    elif resolved_cls is not bound_cls:
      raise TypeError(
          f"{type(self).__name__} is bound to metadata type"
          f" {bound_cls.__name__}, got {resolved_cls.__name__}."
      )
    return resolved_cls.from_atif_metadata(atif_metadata)

  @classmethod
  @abc.abstractmethod
  def _from_config(
      cls, config: Mapping[str, Any]
  ) -> "TrajectoryStore[MetadataT]":
    """Builds an instance of this backend from `config`.

    Implementations read the keys they care about and raise ValueError for a
    config this backend cannot honour. Called only by `from_config`, which has
    already established that `config` selects this backend.
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
  def from_config(cls, config: Mapping[str, Any] | None) -> Self | None:
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
      ValueError: If "backend" names no known implementation, or the selected
        backend rejects the rest of the config.
    """
    if config is None or not config.get("enabled", False):
      return None

    backend = config.get("backend")
    if backend not in cls._REGISTRY:
      raise ValueError(
          f"Unknown Trajectory Store backend {backend!r}; expected one of"
          f" {sorted(cls._REGISTRY)}."
      )
    store = cls._REGISTRY[backend]._from_config(config)  # pylint: disable=protected-access
    metadata_cls = getattr(cls, "_metadata_cls", None)
    if metadata_cls is not None:
      store._metadata_cls = metadata_cls  # pylint: disable=protected-access
    return cast(Self, store)


# TODO(tunix-dev): Consider overriding __instancecheck__ (via a metaclass) so
# `isinstance(store, TunixTrajectoryStore)` returns True for stores bound to
# `TunixTrajectoryMetadata`.
class TunixTrajectoryStore(
    TrajectoryStore[trajectory_lib.TunixTrajectoryMetadata],
    abc.ABC,
):
  """TrajectoryStore bound to TunixTrajectoryMetadata.

  `TunixTrajectoryStore.from_config(config)` instantiates the configured backend
  (`FileTrajectoryStore`, `SqlTrajectoryStore`, or `InMemoryTrajectoryStore`)
  and binds it to `TunixTrajectoryMetadata`. Note that the returned instance is
  a subclass of `TrajectoryStore` (and of the selected backend), not a runtime
  subclass of `TunixTrajectoryStore`.
  """

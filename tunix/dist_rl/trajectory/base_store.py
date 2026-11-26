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
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, ClassVar, Protocol, Self, TypeVar, cast, get_args, get_origin, runtime_checkable

from absl import logging
import pydantic
from tunix.dist_rl.trajectory import trajectory as trajectory_lib

MetadataT = TypeVar("MetadataT", bound=trajectory_lib.TrajectoryMetadata)
_RawStepT = TypeVar("_RawStepT")


def parse_raw_steps(
    raw_steps: Iterable[_RawStepT],
    validate_step: Callable[[_RawStepT], trajectory_lib.Step],
    *,
    strict: bool,
    trajectory_id: str | None = None,
    source_desc: str = "step payload",
) -> tuple[list[trajectory_lib.Step], bool]:
  """Validates serialized step items, skipping corrupt entries when tolerant.

  Args:
    raw_steps: Iterable of raw serialized step payloads (JSON text or dicts).
    validate_step: Callable converting one raw step item into a `Step`.
    strict: Whether to re-raise validation errors immediately.
    trajectory_id: Optional trajectory identifier included in warning logs.
    source_desc: Human-readable label for the step source in warning logs.

  Returns:
    A tuple `(parsed_steps, had_corrupt)` containing validated `Step` instances
    and whether at least one corrupt item was skipped under `strict=False`.

  Raises:
    pydantic.ValidationError: If `strict` is True and step validation fails.
    ValueError: If `strict` is True and raw payload parsing fails.
    TypeError: If `strict` is True and a raw payload has an invalid type.
  """
  steps: list[trajectory_lib.Step] = []
  had_corrupt = False
  for raw_step in raw_steps:
    try:
      steps.append(validate_step(raw_step))
    except (pydantic.ValidationError, ValueError, TypeError):
      if strict:
        raise
      had_corrupt = True
      logging.warning(
          "Skipping corrupt %s in trajectory %s.",
          source_desc,
          trajectory_id,
          exc_info=True,
      )
  return steps, had_corrupt


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


class TrajectoryMetricsNotFoundError(KeyError):
  """Raised when requested trajectory metrics are not found in the store."""

  def __init__(self, trajectory_id: str) -> None:
    super().__init__(f"Trajectory metrics for ID '{trajectory_id}' not found.")
    self.trajectory_id = trajectory_id


# ==============================================================================
# Protocols (Structural Interfaces)
# ==============================================================================


@runtime_checkable
class TrajectoryReader(Protocol[MetadataT]):
  """Structural protocol defining read-only Trajectory Store operations."""

  def get_trajectories_metadata(
      self,
      trajectory_ids: Sequence[str] | None = None,
      *,
      ignore_missing: bool = False,
  ) -> dict[str, MetadataT]:
    """Retrieves metadata for trajectories in the run.

    Args:
      trajectory_ids: Optional sequence of unique trajectory identifiers. If
        specified, only metadata for these IDs is returned. If None, metadata
        for all trajectories in the run is returned.
      ignore_missing: If True, trajectory IDs with no stored metadata are
        omitted from the returned dictionary instead of raising
        TrajectoryMetadataNotFoundError.

    Returns:
      A dictionary mapping trajectory IDs to TrajectoryMetadata objects for the
      present trajectories, in the order of `trajectory_ids` if specified.

    Raises:
      TrajectoryMetadataNotFoundError: If `ignore_missing` is False and any
        requested trajectory ID does not exist.
    """
    ...

  def get_trajectories(
      self,
      trajectory_ids: Sequence[str],
      *,
      ignore_missing: bool = False,
      strict: bool = True,
  ) -> dict[str, trajectory_lib.Trajectory[Any]]:
    """Retrieves full trajectories for a sequence of trajectory IDs.

    Args:
      trajectory_ids: Sequence of unique trajectory identifiers to load.
      ignore_missing: If True, trajectory IDs with no stored records are omitted
        from the returned dictionary instead of raising TrajectoryNotFoundError.
      strict: If False, retains the longest contiguous valid prefix [1..k] for
        any trajectory with a corrupt step, step-ID gap, or step-count mismatch
        instead of raising ValidationError, and resets completion fields
        (`status = None`, `total_reward = None`) on the returned Trajectory
        while preserving lineage metadata.

    Returns:
      A dictionary mapping trajectory IDs to full Trajectory objects for the
      present trajectories, in the order of `trajectory_ids`.

    Raises:
      TrajectoryNotFoundError: If `ignore_missing` is False and any requested
        trajectory ID does not exist.
      pydantic.ValidationError: If `strict` is True and steps contain gaps or
        fail model validation.
    """
    ...

  def get_trajectories_metrics(
      self, trajectory_ids: Sequence[str] | None = None
  ) -> dict[str, trajectory_lib.TrajectoryMetrics]:
    """Retrieves metrics for trajectories in the run.

    Args:
      trajectory_ids: Optional list of unique trajectory identifiers. If
        specified, only metrics for these IDs are returned. If None, metrics for
        all trajectories in the run with recorded metrics are returned.

    Returns:
      A dictionary mapping trajectory IDs to TrajectoryMetrics objects for the
      requested trajectories, in the order of `trajectory_ids` if specified.

    Raises:
      TrajectoryMetricsNotFoundError: If any requested trajectory ID does not
        have recorded metrics.
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

    Implementations always snapshot `metadata` at call time. Unless defensive
    step copying is enabled on the backend, ownership of `step` transfers to
    the writer on `add_step()`, so callers must not mutate `step` or its
    nested containers afterwards.

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

  def update_metrics(
      self,
      trajectory_id: str,
      metrics: trajectory_lib.TrajectoryMetrics,
  ) -> None:
    """Updates (or creates) trajectory metrics.

    If metrics already exist for `trajectory_id`, implementations perform a
    shallow top-level dictionary merge (`existing.extra.update(metrics.extra)`)
    into the existing `extra` dictionary, allowing rollout workers and RL
    trainers to record metrics independently without overwriting each other's
    keys. Top-level keys in `metrics.extra` (including nested dicts) overwrite
    matching top-level keys in the existing `extra` dictionary. If
    `metrics.extra` is `None` when metrics already exist for `trajectory_id`,
    the existing `extra` dictionary remains unchanged.

    Implementations snapshot `metrics` at call time, so callers may keep
    mutating it afterwards without affecting what was logged.

    Args:
      trajectory_id: Unique identifier for the trajectory.
      metrics: TrajectoryMetrics containing the metrics data to record.
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
  make up a run, and doubles as the on/off gate: a config of None yields None,
  leaving every store-guarded call site a no-op.
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

  # Whether `exclude_step_arrays` has been validated against the bound metadata
  # class's step schema. Set on the instance by `_validate_and_bind_metadata`.
  _step_array_exclusions_validated: bool = False

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

  @property
  def bound_metadata_cls(self) -> type[MetadataT] | None:
    """The TrajectoryMetadata class bound to this store, or None if unbound."""
    return self._get_bound_metadata_cls()

  @property
  def exclude_step_arrays(self) -> frozenset[str]:
    """Step array fields this store drops before writing steps.

    Defaults to none; backends that support `exclude_step_arrays` override it.
    """
    return frozenset()

  def _validate_step_array_exclusions(
      self, metadata_cls: type[MetadataT]
  ) -> None:
    """Checks `exclude_step_arrays` against `metadata_cls`'s step schema.

    Args:
      metadata_cls: The TrajectoryMetadata class bound to this store.

    Raises:
      ValueError: If `exclude_step_arrays` names fields that no step class of
        `metadata_cls` declares in `EXCLUDABLE_ARRAY_FIELDS`.
    """
    if self.exclude_step_arrays:
      try:
        trajectory_lib.validate_step_array_exclusions(
            self.exclude_step_arrays,
            allowed_fields=metadata_cls.excludable_step_array_fields(),
        )
      except ValueError as e:
        raise ValueError(
            f"{type(self).__name__} bound to {metadata_cls.__name__}: {e}"
        ) from e
    self._step_array_exclusions_validated = True

  def _validate_and_bind_metadata(self, metadata: MetadataT) -> None:
    """Validates and binds the store instance's single TrajectoryMetadata type.

    On the first write, also validates `exclude_step_arrays` against the bound
    metadata class's step schema (`TrajectoryMetadata.STEP_CLASSES`).

    Args:
      metadata: The metadata instance being written to the store.

    Raises:
      TypeError: If `metadata` is not a `TrajectoryMetadata` instance (or is a
        `Trajectory`), or if its type does not match the metadata type already
        bound to this store instance.
      ValueError: If `exclude_step_arrays` names fields that the bound metadata
        class's step classes cannot exclude.
    """
    meta_cls = type(metadata)
    if (
        self._metadata_cls is not None
        and meta_cls is self._metadata_cls
        and self._step_array_exclusions_validated
    ):
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
    if not self._step_array_exclusions_validated:
      self._validate_step_array_exclusions(meta_cls)

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

  def update_metrics(
      self,
      trajectory_id: str,
      metrics: trajectory_lib.TrajectoryMetrics,
  ) -> None:
    """Updates (or creates) trajectory-level metrics.

    Args:
      trajectory_id: Unique identifier of the trajectory.
      metrics: TrajectoryMetrics containing trajectory-level metrics.
    """
    raise NotImplementedError(
        f"{type(self).__name__} does not implement update_metrics."
    )

  def get_trajectories_metrics(
      self, trajectory_ids: Sequence[str] | None = None
  ) -> dict[str, trajectory_lib.TrajectoryMetrics]:
    """Retrieves metrics for trajectories in the run.

    Args:
      trajectory_ids: Optional list of unique trajectory identifiers. If
        specified, only metrics for these IDs is returned. If None, metrics for
        all trajectories with recorded metrics is returned.

    Returns:
      A mapping from trajectory_id to TrajectoryMetrics.

    Raises:
      TrajectoryMetricsNotFoundError: If any requested trajectory ID does not
        exist.
    """
    raise NotImplementedError(
        f"{type(self).__name__} does not implement get_trajectories_metrics."
    )

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
      config: Configuration mapping, or None to run without a store. The
        "backend" key selects the implementation.

    Returns:
      A store instance, or None if `config` is None.

    Raises:
      ValueError: If `config` contains the removed "enabled" key, "backend"
        names no known implementation, "exclude_step_arrays" names fields the
        bound metadata class's step classes cannot exclude, or the selected
        backend rejects the rest of the config.
      TypeError: If "exclude_step_arrays" is a string or contains non-string
        entries.
    """
    if config is None:
      return None
    # Backends ignore keys they do not read, so a leftover "enabled": False
    # would otherwise build an active store. Reject the key so such configs
    # fail loudly instead.
    if "enabled" in config:
      raise ValueError(
          "The 'enabled' key is no longer supported in TrajectoryStore"
          " configs; pass None to disable the store."
      )

    backend = config.get("backend")
    if backend not in cls._REGISTRY:
      raise ValueError(
          f"Unknown Trajectory Store backend {backend!r}; expected one of"
          f" {sorted(cls._REGISTRY)}."
      )
    metadata_cls = getattr(cls, "_metadata_cls", None)
    # Fail fast at startup for schema-bound stores (e.g. TunixTrajectoryStore);
    # unbound stores validate on the first write instead.
    exclude = config.get("exclude_step_arrays")
    if metadata_cls is not None and exclude is not None:
      trajectory_lib.validate_step_array_exclusions(
          exclude, allowed_fields=metadata_cls.excludable_step_array_fields()
      )
    store = cls._REGISTRY[backend]._from_config(config)  # pylint: disable=protected-access
    if metadata_cls is not None:
      store._metadata_cls = metadata_cls  # pylint: disable=protected-access
    return cast(Self, store)


class TunixTrajectoryStore(
    TrajectoryStore[trajectory_lib.TunixTrajectoryMetadata],
):
  """TrajectoryStore bound to TunixTrajectoryMetadata.

  `TunixTrajectoryStore.from_config(config)` instantiates the configured backend
  (`FileTrajectoryStore`, `SqlTrajectoryStore`, or `InMemoryTrajectoryStore`)
  and binds it to `TunixTrajectoryMetadata`.
  """

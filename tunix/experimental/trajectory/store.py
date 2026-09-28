"""Protocols defining Trajectory Store interfaces."""

import abc
import importlib
import typing
from typing import Any, ClassVar, Mapping, Protocol, TypeVar, cast
from typing import runtime_checkable

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
      Each is really the subclass paired with `MetadataT`, which no annotation
      can name; isinstance-check for it if you need the narrower type.

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
# Generic argument resolution
# ==============================================================================


def _metadata_arg_of(alias: Any) -> type[Any] | None:
  """Returns the class `alias` binds to `MetadataT`, or None if none does.

  None covers everything naming no concrete class: an unsubscripted `alias`, a
  subscript omitting `MetadataT`, and one binding it to another type variable.

  Args:
    alias: A type alias or class reference to inspect.

  Returns:
    The concrete class bound to `MetadataT`, or None if none does.
  """
  origin = typing.get_origin(alias)
  if origin is None:
    return None
  # Match by type variable name or identity rather than by position: a subclass
  # may declare type parameters of its own, and `MetadataT` need not be first
  # among them.
  params = getattr(origin, "__parameters__", ())
  target_param = None
  for p in params:
    if p is MetadataT or getattr(p, "__name__", None) == "MetadataT":
      target_param = p
      break
  if target_param is None:
    return None
  args = typing.get_args(alias)
  index = params.index(target_param)
  if index >= len(args):
    return None
  arg = args[index]
  if isinstance(arg, TypeVar):
    # A still-generic subclass binds the parameter to another type variable,
    # which names no class; the caller keeps searching.
    return None
  # Unwrap a subscripted argument, e.g. `Trajectory[Step]` -> `Trajectory`.
  return typing.get_origin(arg) or arg


def _resolve_metadata_cls(instance: Any) -> type[Any] | None:
  """Recovers the class bound to `MetadataT` for `instance`, or None.

  Type variables are erased, so the only runtime trace of `MetadataT` is the
  subscript the caller wrote, which the interpreter records in exactly two
  places, checked here in order of specificity: `__orig_class__`, set for a
  subscripted construction `FileTrajectoryStore[Meta](...)` but only *after*
  `__init__` returns, and `__orig_bases__`, set for a subscripted base class
  `class ConcreteStore(FileTrajectoryStore[Meta])`. A bare
  `FileTrajectoryStore(...)` has neither, so callers must supply a default.

  Args:
    instance: The TrajectoryStore instance to inspect.

  Returns:
    The concrete class bound to `MetadataT`, or None if unresolved.
  """
  if (
      resolved := _metadata_arg_of(getattr(instance, "__orig_class__", None))
  ) is not None:
    return resolved

  for klass in type(instance).__mro__:
    # Read `__orig_bases__` out of each class's own __dict__: a plain subclass
    # has none of its own and would otherwise inherit its parent's by ordinary
    # attribute lookup, which is right only by accident.
    for base in vars(klass).get("__orig_bases__", ()):
      if (resolved := _metadata_arg_of(base)) is not None:
        return resolved
  return None


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
  __orig_class__: Any
  _configured_metadata_type: str | None = None

  def __init_subclass__(cls, **kwargs: Any) -> None:
    super().__init_subclass__(**kwargs)
    # Only a class that declares its own BACKEND claims that key: `getattr`
    # would follow the MRO and see one inherited from a base, letting a
    # subclass that never asked to be a backend displace its base in the
    # registry for the rest of the process. `cls._REGISTRY` is the opposite —
    # one dict, on TrajectoryStore, that every subclass registers into.
    backend = cls.__dict__.get("BACKEND")
    if backend:
      cls._REGISTRY[backend] = cls

  @classmethod
  def _resolve_metadata_type_name(
      cls, name: str
  ) -> type[trajectory_lib.TrajectoryMetadata]:
    """Resolves a metadata type name or qualified class path to a class."""
    registry = (
        trajectory_lib.TrajectoryMetadata._REGISTRY  # pylint: disable=protected-access
    )
    if name in registry:
      return registry[name]

    if "." in name:
      try:
        module_path, class_name = name.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        resolved = getattr(mod, class_name, None)
        if isinstance(resolved, type) and issubclass(
            resolved, trajectory_lib.TrajectoryMetadata
        ):
          return resolved
      except (ImportError, AttributeError):
        pass

    valid = sorted(registry)
    raise ValueError(
        f"Unknown Trajectory Store metadata_type {name!r}; expected one of"
        f" {valid} or a valid fully-qualified class path."
    )

  @classmethod
  def _get_metadata_type_name(
      cls, metadata_cls: type[trajectory_lib.TrajectoryMetadata]
  ) -> str:
    """Returns the registered name or qualified path for a metadata class."""
    if metadata_cls is trajectory_lib.TrajectoryMetadata:
      return "base"
    for (
        name,
        registered_cls,
    ) in (
        trajectory_lib.TrajectoryMetadata._REGISTRY.items()  # pylint: disable=protected-access
    ):
      if registered_cls is metadata_cls:
        return name
    if (meta_type := metadata_cls.__dict__.get("METADATA_TYPE")) is not None:
      return meta_type
    if (
        metadata_cls.__module__ != "__main__"
        and "<locals>" not in metadata_cls.__qualname__
    ):
      return f"{metadata_cls.__module__}.{metadata_cls.__qualname__}"
    return metadata_cls.__name__

  @property
  def _metadata_type(self) -> str | None:
    """Returns the serialized metadata_type string for this store's config."""
    meta_cls = self._metadata_cls
    if meta_cls is trajectory_lib.TrajectoryMetadata:
      return getattr(self, "_configured_metadata_type", None)
    return self._get_metadata_type_name(meta_cls)

  @property
  def _metadata_cls(self) -> type[MetadataT]:
    """Returns the concrete class to deserialize stored metadata into.

    Type variables are erased, so `MetadataT` alone supplies no class to parse
    with. This recovers one from whatever subscript the caller wrote, stating
    the metadata type once, at the construction site:

        store = FileTrajectoryStore[TunixTrajectoryMetadata](
            root_dir=root, run_id=run_id)

    A subscripted base class works too; an unparameterized store falls back to
    `TrajectoryMetadata`. Annotating instead of subscripting (`s:
    FileTrajectoryStore[Meta] = FileTrajectoryStore(...)`) reaches that fallback
    because the interpreter records no subscript for annotations, which
    surfaces as a `pydantic.ValidationError` on the first read rather than as
    bad data, since `TrajectoryMetadata` forbids extra fields. For distributed
    construction, `from_config` recovers `MetadataT` when the configuration
    provides a `metadata_type` key.

    Raises:
      TypeError: If the subscript binds `MetadataT` to something that is not a
        `TrajectoryMetadata` subclass.
    """
    # Not a `cached_property`: `__orig_class__` does not exist until after
    # `__init__` returns, so caching an access made during construction would
    # pin the fallback forever. Hoist it into a local to read it in a loop.
    resolved = _resolve_metadata_cls(self)
    if resolved is None:
      return cast(type[MetadataT], trajectory_lib.TrajectoryMetadata)
    if not (
        isinstance(resolved, type)
        and issubclass(resolved, trajectory_lib.TrajectoryMetadata)
    ):
      raise TypeError(
          f"{type(self).__name__} is parameterized with"
          f" {resolved!r} as its metadata type, which is not a"
          " TrajectoryMetadata subclass."
      )
    return cast(type[MetadataT], resolved)

  @classmethod
  @abc.abstractmethod
  def _from_config(cls, config: Mapping[str, Any]) -> "TrajectoryStore[Any]":
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

  @classmethod
  def from_config(
      cls, config: Mapping[str, Any] | None
  ) -> "TrajectoryStore[Any] | None":
    """Builds the store described by `config`, or None when it is disabled.

    "backend" selects the implementation; "enabled" turns the store off without
    removing the rest of the config; "metadata_type" optionally selects the
    TrajectoryMetadata subclass. Call once per process and hold onto the
    result: the process that built a store owns closing it. Calling this twice
    builds two independent stores (and, for the file backend, two writer
    threads); nothing prevents that, the guard is that callers construct once.

    Raises:
      ValueError: If "backend" names no known implementation, "metadata_type"
        names an unknown metadata type, or the selected backend rejects the
        rest of the config.
    """
    if config is None or not config.get("enabled", False):
      return None

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
    backend_cls = cls._REGISTRY[backend]
    store = backend_cls._from_config(config)  # pylint: disable=protected-access

    meta_type = config.get("metadata_type")
    if meta_type:
      meta_cls = cls._resolve_metadata_type_name(meta_type)
      if getattr(backend_cls, "__parameters__", ()):
        store.__orig_class__ = cast(Any, backend_cls)[meta_cls]
      elif store._metadata_cls is not meta_cls:  # pylint: disable=protected-access
        raise ValueError(
            f"Backend {backend!r} ({backend_cls.__name__}) has a fixed"
            f" metadata type {store._metadata_cls.__name__!r}, which conflicts"  # pylint: disable=protected-access
            f" with configured metadata_type {meta_type!r}"
            f" ({meta_cls.__name__})."
        )
      store._configured_metadata_type = meta_type  # pylint: disable=protected-access

    return store

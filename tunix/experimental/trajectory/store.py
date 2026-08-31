"""Protocols defining Trajectory Store interfaces."""

from typing import Protocol, runtime_checkable

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

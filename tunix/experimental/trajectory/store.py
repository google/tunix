"""Protocols defining Trajectory Store interfaces."""

import typing
from typing import Protocol

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


@typing.runtime_checkable
class TrajectoryReader(Protocol):
  """Structural protocol defining read-only Trajectory Store operations."""

  def get_trajectories_metadata(
      self,
  ) -> list[trajectory_lib.TrajectoryMetadata]:
    """Retrieves metadata for each trajectory in the run.

    Returns:
      A list of TrajectoryMetadata objects for all trajectories in this run.
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


@typing.runtime_checkable
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

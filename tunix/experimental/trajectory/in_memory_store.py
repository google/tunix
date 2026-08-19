"""In-memory implementation for Trajectory Store."""

import collections

from tunix.experimental.trajectory import store
from tunix.experimental.trajectory import trajectory as trajectory_lib


def _validate_trajectory_id(trajectory_id: str | None) -> str:
  """Validates that trajectory_id is non-empty.

  Args:
    trajectory_id: The trajectory identifier to validate.

  Returns:
    The validated trajectory_id string.

  Raises:
    ValueError: If trajectory_id is None or empty.
  """
  if not trajectory_id:
    raise ValueError("TrajectoryMetadata must have a non-empty trajectory_id.")
  return trajectory_id


class InMemoryTrajectoryStore(store.TrajectoryReader, store.TrajectoryWriter):
  """In-memory implementation satisfying TrajectoryReader and TrajectoryWriter."""

  def __init__(self) -> None:
    """Initializes the InMemoryTrajectoryStore."""
    self._metadata_by_trajectory_id: dict[
        str, trajectory_lib.TrajectoryMetadata
    ] = {}
    self._steps_by_trajectory_id: dict[str, list[trajectory_lib.Step]] = (
        collections.defaultdict(list)
    )

  def get_trajectories_metadata(
      self,
  ) -> list[trajectory_lib.TrajectoryMetadata]:
    """Retrieves metadata for each trajectory in the run."""
    return list(self._metadata_by_trajectory_id.values())

  def get_trajectories(
      self, trajectory_ids: list[str]
  ) -> list[trajectory_lib.Trajectory]:
    """Retrieves full trajectories for a list of trajectory IDs.

    Args:
      trajectory_ids: List of unique trajectory identifiers to load.

    Returns:
      A list of full Trajectory objects corresponding to the requested IDs.

    Raises:
      store.TrajectoryNotFoundError: If any requested trajectory ID does not
      exist.
    """
    result: list[trajectory_lib.Trajectory] = []
    for traj_id in trajectory_ids:
      if traj_id not in self._metadata_by_trajectory_id:
        raise store.TrajectoryNotFoundError(traj_id)
      meta = self._metadata_by_trajectory_id[traj_id]
      steps = list(self._steps_by_trajectory_id.get(traj_id, []))
      traj_data = meta.model_dump()
      traj_data["steps"] = steps
      result.append(trajectory_lib.Trajectory(**traj_data))
    return result

  def add_step(
      self,
      step: trajectory_lib.Step,
      metadata: trajectory_lib.TrajectoryMetadata,
  ) -> None:
    """Atomically logs a turn step and its trajectory metadata.

    Args:
      step: Step object to log.
      metadata: TrajectoryMetadata containing trajectory_id and run metadata.

    Raises:
      ValueError: If metadata.trajectory_id is empty or None.
    """
    traj_id = _validate_trajectory_id(metadata.trajectory_id)
    self.update_metadata(metadata)
    self._steps_by_trajectory_id[traj_id].append(step)

  def update_metadata(
      self,
      metadata: trajectory_lib.TrajectoryMetadata,
  ) -> None:
    """Updates (or creates) trajectory metadata.

    Args:
      metadata: TrajectoryMetadata containing trajectory_id and run metadata.

    Raises:
      ValueError: If metadata.trajectory_id is empty or None.
    """
    traj_id = _validate_trajectory_id(metadata.trajectory_id)
    self._metadata_by_trajectory_id[traj_id] = metadata

  def flush(self) -> None:
    """Flushes any pending or asynchronous writes to persistent storage."""
    pass

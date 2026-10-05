"""In-memory implementation for Trajectory Store."""

import collections
from typing import Any, ClassVar, Mapping

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


class InMemoryTrajectoryStore(
    store.TrajectoryStore, store.TrajectoryReader, store.TrajectoryWriter
):
  """In-memory implementation satisfying TrajectoryReader and TrajectoryWriter.

  Process-local: the steps written here are visible only to the process that
  wrote them. Configuring this backend for a run whose reader and writer live
  in different processes gives the reader an empty store.
  """

  BACKEND: ClassVar[str] = "memory"

  def __init__(self) -> None:
    """Initializes the InMemoryTrajectoryStore."""
    self._metadata_by_trajectory_id: dict[
        str, trajectory_lib.TrajectoryMetadata
    ] = {}
    self._steps_by_trajectory_id: dict[str, list[trajectory_lib.Step]] = (
        collections.defaultdict(list)
    )

  @classmethod
  def _from_config(cls, config: Mapping[str, Any]) -> "InMemoryTrajectoryStore":
    """Builds an in-memory store; this backend takes no configuration.

    Args:
      config: Unused beyond the keys `store.TrajectoryStore.from_config` has
        already read.

    Returns:
      A new, empty InMemoryTrajectoryStore.
    """
    del config
    return cls()

  def to_config(self) -> dict[str, Any]:
    """Returns the config dict that rebuilds an equivalent store."""
    return {"enabled": True, "backend": self.BACKEND}

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
      store.TrajectoryMetadataNotFoundError: If any requested trajectory ID does
        not exist.
    """
    if trajectory_ids is None:
      trajectory_ids = list(self._metadata_by_trajectory_id.keys())
    metas: list[trajectory_lib.TrajectoryMetadata] = []
    for traj_id in trajectory_ids:
      if traj_id not in self._metadata_by_trajectory_id:
        raise store.TrajectoryMetadataNotFoundError(traj_id)
      meta = self._metadata_by_trajectory_id[traj_id].model_copy(deep=True)
      metas.append(meta)
    return metas

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
      steps = [
          s.model_copy(deep=True)
          for s in self._steps_by_trajectory_id.get(traj_id, [])
      ]
      traj_data = meta.model_dump()
      traj_data["steps"] = steps
      if isinstance(meta, trajectory_lib.TunixTrajectoryMetadata):
        result.append(trajectory_lib.TunixTrajectory(**traj_data))  # pyrefly: ignore[bad-argument-type]
      else:
        result.append(trajectory_lib.Trajectory(**traj_data))
    return result

  def add_step(
      self,
      step: trajectory_lib.Step,
      metadata: trajectory_lib.TrajectoryMetadata,
  ) -> None:
    """Atomically logs a turn step and its trajectory metadata.

    The step is deep copied, so the stored trajectory reflects the step as it
    was at call time and is unaffected by later mutations of the caller's
    object. This matches the file-backed store, whose asynchronous writer
    snapshots for the same reason.

    Args:
      step: Step object to log.
      metadata: TrajectoryMetadata containing trajectory_id and run metadata.

    Raises:
      ValueError: If metadata.trajectory_id is empty or None.
    """
    traj_id = _validate_trajectory_id(metadata.trajectory_id)
    self.update_metadata(metadata)
    step_copy = step.model_copy(deep=True)
    steps = self._steps_by_trajectory_id[traj_id]
    for idx, s in enumerate(steps):
      if s.step_id == step_copy.step_id:
        steps[idx] = step_copy
        break
    else:
      steps.append(step_copy)

  def update_metadata(
      self,
      metadata: trajectory_lib.TrajectoryMetadata,
  ) -> None:
    """Updates (or creates) trajectory metadata.

    The metadata is deep copied, so later mutations of the caller's object do
    not retroactively change what this store reports.

    Args:
      metadata: TrajectoryMetadata containing trajectory_id and run metadata.

    Raises:
      ValueError: If metadata.trajectory_id is empty or None.
    """
    traj_id = _validate_trajectory_id(metadata.trajectory_id)
    self._metadata_by_trajectory_id[traj_id] = metadata.model_copy(deep=True)

  def flush(self) -> None:
    """Flushes any pending or asynchronous writes to persistent storage."""
    pass

  def close(self) -> None:
    """No-op: this store holds no background thread or file handle.

    Provided so that callers can treat every TrajectoryWriter uniformly. Data
    stays readable after closing.
    """
    pass

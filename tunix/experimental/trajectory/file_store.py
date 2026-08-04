"""File-based implementation for Trajectory Store."""

import functools
import re
from typing import Final

from etils import epath
import pydantic
from tunix.experimental.trajectory import store
from tunix.experimental.trajectory import trajectory as trajectory_lib

_METADATA_FILENAME: Final[str] = "metadata.json"
_TRAJECTORY_DIR_PREFIX: Final[str] = "traj_"
# Characters allowed in a trajectory_id: ASCII letters, digits, underscores, and
# hyphens. Shared between `_TRAJECTORY_ID_REGEX` and `_TRAJECTORY_DIR_REGEX` so
# that every ID written to disk is discoverable when listing trajectory
# directories.
_TRAJECTORY_ID_PATTERN: Final[str] = r"[a-zA-Z0-9_\-]+"
_TRAJECTORY_ID_REGEX: Final[re.Pattern[str]] = re.compile(
    rf"^{_TRAJECTORY_ID_PATTERN}$"
)
_TRAJECTORY_DIR_REGEX: Final[re.Pattern[str]] = re.compile(
    rf"^{_TRAJECTORY_DIR_PREFIX}(?P<trajectory_id>{_TRAJECTORY_ID_PATTERN})$"
)
_STEP_FILENAME_TEMPLATE: Final[str] = "step_{step_id:06d}.json"
_STEP_FILENAME_REGEX: Final[re.Pattern[str]] = re.compile(r"^step_\d+\.json$")


def _dump_json(model: pydantic.BaseModel) -> str:
  """Serializes a Pydantic model to indented, human-readable JSON excluding None values."""
  return model.model_dump_json(indent=2, exclude_none=True)


class FileTrajectoryStore(store.TrajectoryReader, store.TrajectoryWriter):
  """File-based implementation satisfying TrajectoryReader and TrajectoryWriter.

  Directory Structure:
    <root_dir>/[<run_id>/]/
        └── traj_<trajectory_id>/
            ├── metadata.json
            ├── step_000001.json
            ├── step_000002.json
            └── ...
  """

  def __init__(
      self, root_dir: epath.PathLike, run_id: str | None = None
  ) -> None:
    """Initializes FileTrajectoryStore.

    Args:
      root_dir: Base directory path for storage (supports local paths and GCS
        uris e.g., 'gs://bucket/path').
      run_id: Optional unique identifier for the RL run. If provided, paths are
        scoped under root_dir / run_id. This ID MUST stay the same when
        recovering from failures or process restarts as long as the same RL
        process is being continued.
    """
    self._raw_root_dir = epath.Path(root_dir)
    self._run_id = run_id

  @functools.cached_property
  def root_dir(self) -> epath.Path:
    """Returns the effective root directory path, creating it if needed."""
    root_dir = (
        self._raw_root_dir / self._run_id
        if self._run_id
        else self._raw_root_dir
    )
    root_dir.mkdir(parents=True, exist_ok=True)
    return root_dir

  def get_trajectory_dir(self, trajectory_id: str) -> epath.Path:
    """Returns the directory path for a given trajectory ID."""
    return self.root_dir / f"{_TRAJECTORY_DIR_PREFIX}{trajectory_id}"

  def get_trajectory_metadata_path(self, trajectory_id: str) -> epath.Path:
    """Returns the file path for a given trajectory ID's metadata."""
    return self.get_trajectory_dir(trajectory_id) / _METADATA_FILENAME

  def get_step_path(self, trajectory_id: str, step_id: int) -> epath.Path:
    """Returns the file path for a given trajectory ID and step ID."""
    step_filename = _STEP_FILENAME_TEMPLATE.format(step_id=step_id)
    return self.get_trajectory_dir(trajectory_id) / step_filename

  def get_trajectories_metadata(
      self,
  ) -> list[trajectory_lib.TrajectoryMetadata]:
    """Retrieves metadata for each trajectory in the run."""
    metas: list[trajectory_lib.TrajectoryMetadata] = []

    for entry in self.root_dir.iterdir():
      if not entry.is_dir():
        continue
      if not (match := _TRAJECTORY_DIR_REGEX.match(entry.name)):
        continue

      traj_id = match.group("trajectory_id")
      meta_path = self.get_trajectory_metadata_path(traj_id)
      if not meta_path.exists():
        raise store.TrajectoryMetadataNotFoundError(entry.name)

      meta = trajectory_lib.TrajectoryMetadata.model_validate_json(
          meta_path.read_text()
      )
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
    trajs: list[trajectory_lib.Trajectory] = []

    for traj_id in trajectory_ids:
      traj_dir = self.get_trajectory_dir(traj_id)
      meta_path = self.get_trajectory_metadata_path(traj_id)
      if not meta_path.exists():
        raise store.TrajectoryNotFoundError(traj_id)

      meta = trajectory_lib.TrajectoryMetadata.model_validate_json(
          meta_path.read_text()
      )
      steps: list[trajectory_lib.Step] = []

      for file_entry in traj_dir.iterdir():
        if not _STEP_FILENAME_REGEX.match(file_entry.name):
          continue
        step = trajectory_lib.Step.model_validate_json(file_entry.read_text())
        steps.append(step)

      steps.sort(key=lambda s: s.step_id)
      traj_data = meta.model_dump()
      traj_data["steps"] = steps
      trajs.append(trajectory_lib.Trajectory(**traj_data))

    return trajs

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
      ValueError: If metadata.trajectory_id is empty, None, or contains
        characters that cannot be encoded in a trajectory directory name.
    """
    traj_id = metadata.trajectory_id
    if not traj_id:
      raise ValueError(
          "TrajectoryMetadata must have a non-empty trajectory_id."
      )
    if not _TRAJECTORY_ID_REGEX.match(traj_id):
      raise ValueError(
          f"trajectory_id {traj_id!r} contains unsupported characters; only "
          "letters, digits, underscores, and hyphens are allowed."
      )

    traj_dir = self.get_trajectory_dir(traj_id)
    traj_dir.mkdir(parents=True, exist_ok=True)

    meta_path = self.get_trajectory_metadata_path(traj_id)
    meta_path.write_text(_dump_json(metadata))

    step_path = self.get_step_path(traj_id, step.step_id)
    step_path.write_text(_dump_json(step))

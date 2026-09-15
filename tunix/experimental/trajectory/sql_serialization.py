"""Serialization and deserialization utilities for SQL Trajectory Store."""

from collections.abc import Sequence
import datetime
from typing import Any

from tunix.experimental.trajectory import trajectory as trajectory_lib

_STEP_CLASSES: dict[str, type[trajectory_lib.Step]] = {
    "TunixAgentStep": trajectory_lib.TunixAgentStep,
    "TunixEnvStep": trajectory_lib.TunixEnvStep,
    "Step": trajectory_lib.Step,
}

_METADATA_CLASSES: dict[str, type[trajectory_lib.TrajectoryMetadata]] = {
    "TunixTrajectoryMetadata": trajectory_lib.TunixTrajectoryMetadata,
    "TrajectoryMetadata": trajectory_lib.TrajectoryMetadata,
}


def _to_utc_timestamp(dt: datetime.datetime | None) -> datetime.datetime:
  """Normalizes a datetime to a timezone-aware UTC timestamp.

  Args:
    dt: Datetime to normalize, or None. If naive, assumes UTC time.

  Returns:
    A timezone-aware UTC datetime. Defaults to the current UTC time if dt is
    None.
  """
  if dt is None:
    return datetime.datetime.now(datetime.timezone.utc)
  if dt.tzinfo is not None:
    return dt.astimezone(datetime.timezone.utc)
  return dt.replace(tzinfo=datetime.timezone.utc)


def serialize_metadata(
    metadata: trajectory_lib.TrajectoryMetadata,
) -> dict[str, Any]:
  """Serializes TrajectoryMetadata into a JSON-compatible dictionary.

  Args:
    metadata: TrajectoryMetadata instance to serialize.

  Returns:
    A JSON-serializable dictionary containing the metadata fields and a
    `_metadata_type` discriminator.
  """
  payload = metadata.model_dump(mode="json", exclude_none=True)
  payload["_metadata_type"] = type(metadata).__name__
  return payload


def deserialize_metadata(
    payload: dict[str, Any],
) -> trajectory_lib.TrajectoryMetadata:
  """Deserializes a dictionary into the appropriate TrajectoryMetadata instance.

  Args:
    payload: Raw dictionary containing serialized metadata fields and an
      optional `_metadata_type` discriminator.

  Returns:
    A validated TrajectoryMetadata or TunixTrajectoryMetadata instance.
  """
  data = dict(payload)
  meta_type = data.pop("_metadata_type", None)
  metadata_cls = _METADATA_CLASSES.get(
      meta_type, trajectory_lib.TrajectoryMetadata
  )
  return metadata_cls.model_validate(data)


def serialize_step(
    step: trajectory_lib.Step,
) -> tuple[dict[str, Any], datetime.datetime]:
  """Serializes a Step into a JSON-compatible dictionary and UTC timestamp.

  Args:
    step: Step instance to serialize.

  Returns:
    A tuple of:
      - payload: JSON-serializable dictionary containing step fields and a
        `_step_type` discriminator.
      - timestamp: Non-null, timezone-aware UTC datetime for the step.
  """
  payload_dict = step.model_dump(mode="json", exclude_none=True)
  payload_dict["_step_type"] = type(step).__name__
  return payload_dict, _to_utc_timestamp(step.timestamp)


def deserialize_step(
    payload: dict[str, Any],
) -> trajectory_lib.Step:
  """Deserializes a dictionary into the appropriate Step instance.

  Args:
    payload: Raw dictionary containing serialized step fields and an optional
      `_step_type` discriminator.

  Returns:
    A validated Step, TunixAgentStep, or TunixEnvStep instance.
  """
  data = dict(payload)
  step_type = data.pop("_step_type", None)
  step_cls = _STEP_CLASSES.get(step_type, trajectory_lib.Step)
  return step_cls.model_validate(data)


def build_trajectory(
    metadata: trajectory_lib.TrajectoryMetadata,
    steps: Sequence[trajectory_lib.Step],
) -> trajectory_lib.Trajectory | trajectory_lib.TunixTrajectory:
  """Combines metadata and steps into a complete Trajectory instance.

  Args:
    metadata: Trajectory metadata determining the concrete trajectory class.
    steps: Turn steps associated with the trajectory.

  Returns:
    A validated Trajectory or TunixTrajectory instance with steps sorted by
    `step_id` in ascending order.
  """
  traj_data = metadata.model_dump(exclude_none=True)
  traj_data["steps"] = sorted(steps, key=lambda s: s.step_id)
  trajectory_cls = (
      trajectory_lib.TunixTrajectory
      if isinstance(metadata, trajectory_lib.TunixTrajectoryMetadata)
      else trajectory_lib.Trajectory
  )
  return trajectory_cls.model_validate(traj_data)

"""Lazy data generator for ATIF Trajectory Store benchmarks."""

from collections.abc import Iterator
import dataclasses

from tunix.experimental.trajectory import trajectory as trajectory_lib


@dataclasses.dataclass(frozen=True, kw_only=True)
class WorkloadConfig:
  """Workload configuration parameters for data generation and benchmarking."""

  cumulative_trajectory_checkpoints: list[int] = dataclasses.field(
      default_factory=lambda: [100, 1000, 10000],
      metadata={
          "help": (
              "Sequence of cumulative target trajectory counts for recovery"
              " checks (e.g. 100, 1000, 10000 means write 100, read/recover,"
              " write 900 more to reach 1000 total, read/recover, etc.)."
          )
      },
  )
  steps_per_trajectory: int = dataclasses.field(
      default=20,
      metadata={"help": "Number of steps per trajectory."},
  )
  step_payload_chars: int = dataclasses.field(
      default=20000,
      metadata={"help": "Character length of step message payload (~20KB)."},
  )

  def __post_init__(self) -> None:
    if not self.cumulative_trajectory_checkpoints:
      raise ValueError("cumulative_trajectory_checkpoints cannot be empty.")
    for i, target_count in enumerate(self.cumulative_trajectory_checkpoints):
      if target_count <= 0:
        raise ValueError(
            f"All checkpoints must be positive integers, got {target_count}"
        )
      if (
          i > 0
          and target_count <= self.cumulative_trajectory_checkpoints[i - 1]
      ):
        raise ValueError(
            "cumulative_trajectory_checkpoints must be strictly increasing,"
            f" got {self.cumulative_trajectory_checkpoints}"
        )
    if self.steps_per_trajectory <= 0:
      raise ValueError("steps_per_trajectory must be > 0.")
    if self.step_payload_chars <= 0:
      raise ValueError("step_payload_chars must be > 0.")


def generate_trajectories(
    workload: WorkloadConfig,
) -> Iterator[
    tuple[trajectory_lib.TrajectoryMetadata, list[trajectory_lib.Step]]
]:
  """Lazily yields pre-cached trajectory metadata and steps.

  Generates trajectories indexed 1 through the maximum cumulative checkpoint.

  Args:
    workload: WorkloadConfig containing data generation parameters.

  Yields:
    Tuples of (TrajectoryMetadata, list[Step]).
  """
  total_count = workload.cumulative_trajectory_checkpoints[-1]
  default_agent = trajectory_lib.Agent(
      name="benchmark_agent",
      version="1.0",
  )
  payload = "x" * workload.step_payload_chars
  cached_steps = [
      trajectory_lib.Step(
          step_id=step_id,
          source=trajectory_lib.Source.USER,
          message=payload,
      )
      for step_id in range(1, workload.steps_per_trajectory + 1)
  ]

  for traj_idx in range(1, total_count + 1):
    metadata = trajectory_lib.TrajectoryMetadata(
        trajectory_id=f"traj_{traj_idx:06d}",
        agent=default_agent,
    )
    yield metadata, cached_steps

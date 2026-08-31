from absl.testing import absltest
from tunix.experimental.trajectory.benchmarks import data_generator


class WorkloadConfigTest(absltest.TestCase):

  def test_valid_workload_config(self) -> None:
    _ = data_generator.WorkloadConfig(
        cumulative_trajectory_checkpoints=[10, 50, 100],
        steps_per_trajectory=5,
        step_payload_chars=1000,
    )

  def test_empty_checkpoints_raises(self) -> None:
    with self.assertRaises(ValueError):
      data_generator.WorkloadConfig(cumulative_trajectory_checkpoints=[])

  def test_non_increasing_checkpoints_raises(self) -> None:
    with self.assertRaises(ValueError):
      data_generator.WorkloadConfig(
          cumulative_trajectory_checkpoints=[100, 50, 100]
      )

  def test_non_positive_checkpoints_raises(self) -> None:
    with self.assertRaises(ValueError):
      data_generator.WorkloadConfig(cumulative_trajectory_checkpoints=[0, 100])


class DataGeneratorTest(absltest.TestCase):

  def test_generate_trajectories_count(self) -> None:
    workload = data_generator.WorkloadConfig(
        cumulative_trajectory_checkpoints=[10, 50, 100],
    )
    trajectories = list(data_generator.generate_trajectories(workload))
    self.assertLen(trajectories, 100)

  def test_generate_trajectories_reproducibility(self) -> None:
    workload = data_generator.WorkloadConfig(
        cumulative_trajectory_checkpoints=[5],
        steps_per_trajectory=3,
        step_payload_chars=50,
    )
    run1 = list(data_generator.generate_trajectories(workload))
    run2 = list(data_generator.generate_trajectories(workload))

    self.assertEqual(run1, run2)

  def test_generate_trajectories_steps(self) -> None:
    workload = data_generator.WorkloadConfig(
        cumulative_trajectory_checkpoints=[5],
        steps_per_trajectory=3,
        step_payload_chars=50,
    )
    for _, steps in data_generator.generate_trajectories(workload):
      self.assertLen(steps, 3)
      for step in steps:
        self.assertLen(step.message, 50)


if __name__ == "__main__":
  absltest.main()

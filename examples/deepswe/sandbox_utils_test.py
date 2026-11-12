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

"""Unit tests for tunix.oss.examples.deepswe.sandbox_utils."""

from unittest import mock
from absl.testing import absltest
import numpy as np
from examples.deepswe import sandbox_utils


class _FakePlan:
  """Minimal FleetPlan implementation tracking planned task images."""

  def __init__(self, images: set[str]):
    self._images = set(images)

  def for_image(self, image: str) -> str | None:
    return image if image in self._images else None


class FakeFleet:
  """Fake SandboxFleet that records calls and tracks active pool replica state."""

  def __init__(self):
    self.tasks: list[object] = []
    self.plan_: _FakePlan | None = None
    self._image_rewrite_fn = None
    self.warm_calls: list[tuple[str, int | None, bool]] = []
    self.set_replicas_calls: list[tuple[str, int]] = []
    self.unwarm_calls: list[str] = []
    self.active_pools: dict[str, int] = {}

  def load_tasks(self, tasks: list[object], **kwargs: object) -> None:
    del kwargs
    self.tasks = list(tasks)

  def plan(self) -> _FakePlan:
    self.plan_ = _FakePlan({getattr(t, "image") for t in self.tasks})
    return self.plan_

  def warm_image(
      self, image: str, replicas_override: int | None = None, wait: bool = False
  ) -> None:
    if self.plan_ is None or self.plan_.for_image(image) is None:
      raise AssertionError(
          f"warm_image({image!r}) called before registering image in"
          " fleet.plan_"
      )
    self.warm_calls.append((image, replicas_override, wait))
    self.active_pools[image] = replicas_override or 1

  def set_pool_replicas(self, image: str, replicas: int) -> None:
    self.set_replicas_calls.append((image, replicas))
    self.active_pools[image] = replicas

  def unwarm_image(self, image: str) -> None:
    self.unwarm_calls.append(image)
    self.active_pools.pop(image, None)


class SandboxUtilsTest(absltest.TestCase):

  def test_two_queue_batch_prewarming(self):
    fleet = FakeFleet()
    dataset = [
        {"prompt": "p0", "docker_image": "img_A"},
        {"prompt": "p1", "docker_image": "img_A"},
        {"prompt": "p2", "docker_image": "img_B"},
        {"prompt": "p3", "docker_image": "img_B"},
        {"prompt": "p4", "docker_image": "img_C"},
    ]
    iterator = sandbox_utils.PrewarmDatasetIterator(
        dataset,
        fleet=fleet,
        num_generations=4,
        batch_size=2,
    )

    # 1. Verification at __init__:
    # current_batch has p0, p1 (img_A: 2).
    # Dict maintains samples of both queues:
    # img_A: 2 (8 reps), img_B: 2 (8 reps).
    # Fleet warms both with wait=False.
    self.assertEqual(
        fleet.warm_calls, [("img_A", 8, False), ("img_B", 8, False)]
    )
    self.assertEqual(fleet.active_pools, {"img_A": 8, "img_B": 8})
    self.assertLen(iterator.current_batch, 2)
    self.assertLen(iterator.next_batch, 2)

    # 2. Step 0: pop p0 (img_A). current_batch has p1 left.
    item0 = next(iterator)
    self.assertEqual(item0["prompt"], "p0")
    self.assertEqual(fleet.active_pools, {"img_A": 8, "img_B": 8})
    self.assertLen(iterator.current_batch, 1)

    # 3. Step 1: pop p1 (img_A). current_batch is now empty. Fleet untouched.
    item1 = next(iterator)
    self.assertEqual(item1["prompt"], "p1")
    self.assertEqual(fleet.active_pools, {"img_A": 8, "img_B": 8})
    self.assertEmpty(iterator.current_batch)

    # 4. Step 2: current_batch was empty -> batch transition!
    item2 = next(iterator)
    self.assertEqual(item2["prompt"], "p2")
    self.assertNotIn("img_A", fleet.unwarm_calls)
    self.assertEqual(fleet.active_pools, {"img_A": 8, "img_B": 8, "img_C": 4})

    # 5. Step 3: pop p3 (img_B).
    item3 = next(iterator)
    self.assertEqual(item3["prompt"], "p3")
    self.assertEqual(fleet.active_pools, {"img_A": 8, "img_B": 8, "img_C": 4})

    # 6. Step 4: batch transition -> img_A retired and unwarmed!
    item4 = next(iterator)
    self.assertEqual(item4["prompt"], "p4")
    self.assertIn("img_A", fleet.unwarm_calls)
    self.assertEqual(fleet.active_pools, {"img_B": 8, "img_C": 4})

    # 7. Exhaustion
    with self.assertRaises(StopIteration):
      next(iterator)
    self.assertEqual(fleet.active_pools, {"img_B": 8, "img_C": 4})

    iterator.close()
    self.assertEqual(fleet.active_pools, {})

  def test_batched_items_format_preserved(self):
    fleet = FakeFleet()
    dataset = [
        [
            {"prompt": "b0_0", "docker_image": "img_A"},
            {"prompt": "b0_1", "docker_image": "img_A"},
        ],
        [
            {"prompt": "b1_0", "docker_image": "img_B"},
            {"prompt": "b1_1", "docker_image": "img_B"},
        ],
    ]
    iterator = sandbox_utils.PrewarmDatasetIterator(
        dataset,
        fleet=fleet,
        num_generations=4,
        batch_size=2,
    )

    batch0 = next(iterator)
    self.assertIsInstance(batch0, list)
    self.assertLen(batch0, 2)
    self.assertEqual(batch0[0]["prompt"], "b0_0")

    batch1 = next(iterator)
    self.assertIsInstance(batch1, list)
    self.assertLen(batch1, 2)
    self.assertEqual(batch1[0]["prompt"], "b1_0")

    with self.assertRaises(StopIteration):
      next(iterator)

    iterator.close()
    self.assertEqual(fleet.active_pools, {})

  def test_nested_metadata_image_extraction(self):
    fleet = FakeFleet()
    dataset = [{
        "prompt": "p0",
        "metadata": {
            "env_config": {"entry": {"docker_image": "nested_docker_image"}}
        },
    }]
    iterator = sandbox_utils.PrewarmDatasetIterator(
        dataset, fleet=fleet, num_generations=2, batch_size=1
    )
    self.assertIn("nested_docker_image", fleet.active_pools)
    item = next(iterator)
    self.assertEqual(item["prompt"], "p0")

  def test_defensive_nested_metadata_with_none(self):
    fleet = FakeFleet()
    # Test items where metadata or env_config or entry has explicit None
    dataset = [
        {"prompt": "p0", "metadata": None},
        {"prompt": "p1", "metadata": {"env_config": None}},
        {"prompt": "p2", "metadata": {"env_config": {"entry": None}}},
        {
            "prompt": "p3",
            "metadata": {
                "env_config": {"entry": {"docker_image": "valid_img"}}
            },
        },
    ]
    iterator = sandbox_utils.PrewarmDatasetIterator(
        dataset, fleet=fleet, num_generations=2, batch_size=2
    )
    self.assertIn("valid_img", fleet.active_pools)
    p0 = next(iterator)
    self.assertEqual(p0["prompt"], "p0")

  def test_fallback_to_image_field(self):
    fleet = FakeFleet()
    dataset = [
        {"prompt": "p0", "image": "fallback_image_A"},
        {
            "prompt": "p1",
            "metadata": {
                "env_config": {"entry": {"image": "fallback_image_B"}}
            },
        },
    ]
    _ = sandbox_utils.PrewarmDatasetIterator(
        dataset, fleet=fleet, num_generations=3, batch_size=2
    )
    self.assertIn("fallback_image_A", fleet.active_pools)
    self.assertIn("fallback_image_B", fleet.active_pools)

  def test_none_docker_image_falls_back_to_metadata(self):
    fleet = FakeFleet()
    # item has docker_image=None; should check metadata instead of skipping
    dataset = [{
        "prompt": "p0",
        "docker_image": None,
        "metadata": {"docker_image": "meta_image"},
    }]
    _ = sandbox_utils.PrewarmDatasetIterator(
        dataset, fleet=fleet, num_generations=2, batch_size=1
    )
    self.assertIn("meta_image", fleet.active_pools)

  def test_batched_items_without_docker_image_sequence_length(self):
    fleet = FakeFleet()
    # Batched list containing 3 items without docker_image
    dataset = [
        [{"prompt": "t0"}, {"prompt": "t1"}, {"prompt": "t2"}],
    ]
    iterator = sandbox_utils.PrewarmDatasetIterator(
        dataset, fleet=fleet, num_generations=2, batch_size=3
    )
    # The batch should have 3 samples, satisfying batch_size=3 in a single step
    self.assertEqual(fleet.active_pools, {})
    item = next(iterator)
    self.assertLen(item, 3)

  def test_numpy_array_docker_image_extraction(self):
    fleet = FakeFleet()
    dataset = [{
        "prompt": np.array(["p0", "p1"]),
        "docker_image": np.array(["array_img_A", "array_img_A"]),
    }]
    iterator = sandbox_utils.PrewarmDatasetIterator(
        dataset, fleet=fleet, num_generations=3, batch_size=2
    )
    self.assertEqual(fleet.active_pools.get("array_img_A"), 6)
    item = next(iterator)
    self.assertIsInstance(item["docker_image"], np.ndarray)

  def test_max_warmpool_replicas_capping(self):
    fleet = FakeFleet()
    dataset = [
        {"prompt": f"p{i}", "docker_image": "capped_img"} for i in range(4)
    ]
    iterator = sandbox_utils.PrewarmDatasetIterator(
        dataset,
        fleet=fleet,
        num_generations=8,
        batch_size=4,
        max_warmpool_replicas=10,
    )
    self.assertEqual(fleet.active_pools.get("capped_img"), 10)
    for _ in range(4):
      next(iterator)
    with self.assertRaises(StopIteration):
      next(iterator)

  def test_empty_dataset(self):
    fleet = FakeFleet()
    iterator = sandbox_utils.PrewarmDatasetIterator(
        [], fleet=fleet, num_generations=4, batch_size=2
    )
    with self.assertRaises(StopIteration):
      next(iterator)
    self.assertEqual(fleet.active_pools, {})

  def test_init_global_fleet_registers_subsequent_task_images_in_plan(self):
    class _FakeTask:

      def __init__(self, id, image, metadata=None):  # pylint: disable=redefined-builtin
        self.id = id
        self.image = image
        self.metadata = metadata or {}

    mock_fleet = mock.MagicMock()
    mock_fleet.tasks = []
    mock_fleet._image_rewrite_fn = None
    planned_images = set()

    def _load_tasks(tasks, **kwargs):
      del kwargs
      mock_fleet.tasks = list(tasks)

    def _plan():
      planned_images.clear()
      for t in mock_fleet.tasks:
        planned_images.add(t.image)

    mock_fleet.load_tasks.side_effect = _load_tasks
    mock_fleet.plan.side_effect = _plan
    mock_fleet.plan_.for_image.side_effect = (
        lambda img: img if img in planned_images else None
    )

    mock_as_rl = mock.MagicMock()
    mock_as_rl.Task = _FakeTask
    mock_as_rl.SandboxFleet.return_value = mock_fleet

    with mock.patch.dict("sys.modules", {"agent_sandbox_rl": mock_as_rl}):
      with mock.patch.object(sandbox_utils, "_GLOBAL_FLEET", None):
        fleet1 = sandbox_utils.init_global_fleet(
            tasks=[{"docker_image": "img_A"}],
            num_generations=4,
        )
        self.assertEqual(planned_images, {"img_A"})
        self.assertEqual(mock_fleet.plan.call_count, 1)

        # Second call with a new image updates the singleton's plan
        fleet2 = sandbox_utils.init_global_fleet(
            tasks=[{"docker_image": "img_B"}],
            num_generations=4,
        )
        self.assertIs(fleet1, fleet2)
        self.assertEqual(planned_images, {"img_A", "img_B"})
        self.assertEqual(mock_fleet.plan.call_count, 2)

        # Third call with an already-planned image is a no-op
        _ = sandbox_utils.init_global_fleet(
            tasks=[{"docker_image": "img_B"}],
            num_generations=4,
        )
        self.assertEqual(mock_fleet.plan.call_count, 2)

  def test_get_global_fleet_uninitialized_raises(self):
    sandbox_utils._GLOBAL_FLEET = None
    with self.assertRaises(RuntimeError):
      sandbox_utils.get_global_fleet()

  def test_image_rewrite_fn(self):
    rewrite = sandbox_utils.get_image_rewrite_fn(
        lambda img: f"gcr.io/rewritten/{img}"
    )
    self.assertEqual(
        rewrite("my-image:latest"), "gcr.io/rewritten/my-image:latest"
    )

  def test_init_global_fleet_plans_without_warming_or_preflight(self):
    mock_fleet = mock.MagicMock()
    mock_entry = mock.MagicMock()
    mock_entry.image = "test-image:v1"
    mock_fleet.plan_.entries = [mock_entry]

    mock_as_rl = mock.MagicMock()
    mock_as_rl.SandboxFleet.return_value = mock_fleet
    with mock.patch.dict("sys.modules", {"agent_sandbox_rl": mock_as_rl}):
      with mock.patch.object(sandbox_utils, "_GLOBAL_FLEET", None):
        with mock.patch.object(
            sandbox_utils.atexit, "register"
        ) as mock_register:
          fleet = sandbox_utils.init_global_fleet(
              tasks=[{"docker_image": "test-image:v1"}],
              num_generations=4,
          )
          self.assertEqual(fleet, mock_fleet)
          mock_fleet.plan.assert_called_once()
          mock_fleet.preflight.assert_not_called()
          mock_fleet.warm_images.assert_not_called()
          mock_register.assert_any_call(sandbox_utils.teardown_global_fleet)

  def test_warm_global_fleet_warms_planned_images_and_waits(self):
    mock_fleet = mock.MagicMock()
    mock_entry = mock.MagicMock()
    mock_entry.image = "test-image:v1"
    mock_fleet.plan_.entries = [mock_entry]
    mock_fleet.config.max_warmpool_size = 4

    with mock.patch.object(sandbox_utils, "_GLOBAL_FLEET", mock_fleet):
      sandbox_utils.warm_global_fleet()
      mock_fleet.warm_images.assert_called_once_with(
          ["test-image:v1"],
          replicas_override=4,
          wait=True,
      )

  def test_max_staleness_and_job_scoped_fleet_config(self):
    fleet = FakeFleet()
    dataset = [
        {"prompt": f"p{i}", "docker_image": f"img_{i}"} for i in range(4)
    ]
    iterator = sandbox_utils.PrewarmDatasetIterator(
        dataset, fleet=fleet, num_generations=2, batch_size=1, max_staleness=1
    )
    self.assertEqual(next(iterator)["prompt"], "p0")
    self.assertEqual(next(iterator)["prompt"], "p1")
    self.assertEqual(next(iterator)["prompt"], "p2")
    self.assertNotIn("img_0", fleet.unwarm_calls)
    self.assertEqual(next(iterator)["prompt"], "p3")
    self.assertIn("img_0", fleet.unwarm_calls)

    captured: dict[str, object] = {}
    mock_as_rl = mock.MagicMock()
    mock_as_rl.FleetConfig.side_effect = lambda **kw: (
        captured.update(kw)
        or mock.MagicMock(install_teardown_hooks=kw["install_teardown_hooks"])
    )
    with (
        mock.patch.dict("sys.modules", {"agent_sandbox_rl": mock_as_rl}),
        mock.patch.object(sandbox_utils, "_GLOBAL_FLEET", None),
        mock.patch.object(sandbox_utils.atexit, "register") as mock_register,
    ):
      sandbox_utils.init_global_fleet(
          tasks=None,
          num_generations=8,
          job_id="haoyugao-orch",
          warm_on_init=False,
          teardown_on_exit=False,
      )
      self.assertEqual(captured["template_name_prefix"], "r2e-haoyugao-")
      self.assertLen(str(captured["run_id"]), 8)
      self.assertFalse(captured["install_teardown_hooks"])
      mock_register.assert_not_called()


if __name__ == "__main__":
  absltest.main()

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

from collections.abc import Callable
import dataclasses
import os
import sys
from unittest import mock

from absl.testing import absltest
import numpy as np
from examples.deepswe import sandbox_utils


@dataclasses.dataclass
class FakeTask:
  id: str
  image: str
  metadata: dict[str, object]


@dataclasses.dataclass
class FakePlanEntry:
  image: str
  replicas: int


@dataclasses.dataclass
class FakeFleetPlan:
  entries: list[FakePlanEntry]


@dataclasses.dataclass
class FakeClusterConfig:
  name: str = "default"
  namespace: str = "default"
  node_selector: dict[str, str] | None = None
  in_cluster: bool = False


@dataclasses.dataclass
class FakeFleetConfig:
  clusters: list[FakeClusterConfig] = dataclasses.field(default_factory=list)
  max_concurrent: int = 1
  window_size: int | None = None
  max_warmpool_size: int = 8
  warm_per_task: bool = False
  run_isolation: str = "none"
  run_id: str = ""
  install_teardown_hooks: bool = True


class FakeSandboxFleet:
  """Typed fake matching agent_sandbox_rl.SandboxFleet."""

  plan_error: Exception | None = None
  teardown_error: Exception | None = None

  def __init__(self, config: FakeFleetConfig):
    self.config = config
    self.run_id = config.run_id or "random01"
    self.tasks: list[object] = []
    self.plan_: FakeFleetPlan | None = None
    self.preflight_called = False
    self.teardown_hooks_installed = False
    self.teardown_calls = 0
    self.warm_images_calls: list[tuple[list[str], int | None, bool]] = []
    self._image_rewrite_fn: Callable[[str], str] | None = None
    self._torndown = False

  def load_tasks(
      self,
      source: list[object],
      *,
      image_rewrite: Callable[[str], str] | None = None,
  ) -> list[object]:
    del image_rewrite
    self.tasks = list(source)
    return self.tasks

  def _install_teardown_hooks(self) -> None:
    self.teardown_hooks_installed = True

  def preflight(self) -> dict[str, bool]:
    self.preflight_called = True
    return {"default": True}

  def plan(self) -> FakeFleetPlan:
    if self.plan_error is not None:
      raise self.plan_error
    seen: list[str] = []
    for task in self.tasks:
      img = task.image if isinstance(task, FakeTask) else str(task)
      if img not in seen:
        seen.append(img)
    self.plan_ = FakeFleetPlan(
        entries=[
            FakePlanEntry(image=img, replicas=self.config.max_warmpool_size)
            for img in seen
        ]
    )
    return self.plan_

  def warm_images(
      self,
      images: list[str],
      replicas_override: int | None = None,
      wait: bool = True,
  ) -> None:
    self.warm_images_calls.append((list(images), replicas_override, wait))

  def warm_image(
      self,
      image: str,
      replicas_override: int | None = None,
      wait: bool = True,
  ) -> None:
    self.warm_images_calls.append(([image], replicas_override, wait))

  def set_pool_replicas(self, image: str, replicas: int) -> None:
    del image, replicas

  def unwarm_image(self, image: str) -> None:
    del image

  def acquire(self, task: object) -> object:
    return task

  def release(self, handle: object) -> None:
    del handle

  def teardown(self) -> None:
    self.teardown_calls += 1
    if self.teardown_error is not None:
      raise self.teardown_error


class FakeAgentSandboxRLModule:
  Task = FakeTask  # pylint: disable=invalid-name
  ClusterConfig = FakeClusterConfig  # pylint: disable=invalid-name
  FleetConfig = FakeFleetConfig  # pylint: disable=invalid-name
  SandboxFleet = FakeSandboxFleet  # pylint: disable=invalid-name


class FakeFleet:
  """Fake SandboxFleet that records calls and tracks active pool replica state."""

  def __init__(self):
    self.warm_calls: list[tuple[str, int | None, bool]] = []
    self.set_replicas_calls: list[tuple[str, int]] = []
    self.unwarm_calls: list[str] = []
    self.active_pools: dict[str, int] = {}

  def warm_image(
      self, image: str, replicas_override: int | None = None, wait: bool = False
  ) -> None:
    self.warm_calls.append((image, replicas_override, wait))
    self.active_pools[image] = replicas_override or 1

  def set_pool_replicas(self, image: str, replicas: int) -> None:
    self.set_replicas_calls.append((image, replicas))
    self.active_pools[image] = replicas

  def unwarm_image(self, image: str) -> None:
    self.unwarm_calls.append(image)
    self.active_pools.pop(image, None)


class SandboxUtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    clean_env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in (
            "IMAGE_REWRITE_PREFIX",
            "KUBERNETES_SERVICE_HOST",
            "NAMESPACE",
            "NODE_SELECTOR_KEY",
            "NODE_SELECTOR_VAL",
        )
    }
    self.enter_context(mock.patch.dict(os.environ, clean_env, clear=True))
    self.enter_context(mock.patch.object(sandbox_utils, "_GLOBAL_FLEET", None))
    self.enter_context(
        mock.patch.object(sandbox_utils, "_SHUTDOWN_HOOKS_REGISTERED", False)
    )
    self.enter_context(
        mock.patch.object(sandbox_utils, "_R2EGYM_PATCHED", False)
    )
    self.enter_context(
        mock.patch.object(FakeSandboxFleet, "plan_error", None)
    )
    self.enter_context(
        mock.patch.object(FakeSandboxFleet, "teardown_error", None)
    )

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

  def test_get_global_fleet_uninitialized_raises(self):
    with self.assertRaises(RuntimeError):
      sandbox_utils.get_global_fleet()

  def test_image_rewrite_fn(self):
    rewrite = sandbox_utils.get_image_rewrite_fn(
        lambda img: f"gcr.io/rewritten/{img}"
    )
    assert rewrite is not None
    self.assertEqual(
        rewrite("my-image:latest"), "gcr.io/rewritten/my-image:latest"
    )

  def test_init_global_fleet_plans_without_warming_or_preflight(self):
    self.enter_context(
        mock.patch.dict(
            "sys.modules", {"agent_sandbox_rl": FakeAgentSandboxRLModule}
        )
    )
    self.enter_context(
        mock.patch.object(
            sandbox_utils.threading, "_register_atexit", autospec=True
        )
    )
    self.enter_context(
        mock.patch.object(sandbox_utils.atexit, "register", autospec=True)
    )

    fleet = sandbox_utils.init_global_fleet(
        tasks=[{"instance_id": "id_1", "docker_image": "test-image:v1"}],
        num_generations=4,
        namespace="trellis",
    )
    self.assertIsInstance(fleet, FakeSandboxFleet)
    self.assertEqual(fleet.config.run_isolation, "none")
    self.assertEqual(fleet.config.clusters[0].namespace, "trellis")
    self.assertFalse(fleet.preflight_called)
    self.assertIsNotNone(fleet.plan_)
    assert fleet.plan_ is not None
    self.assertEqual([e.image for e in fleet.plan_.entries], ["test-image:v1"])
    self.assertEqual(fleet.warm_images_calls, [])

    sandbox_utils.warm_global_fleet()
    self.assertEqual(
        fleet.warm_images_calls,
        [(["test-image:v1"], 4, True)],
    )

  def test_init_global_fleet_shared_run_id_and_worker_attach(self):
    self.enter_context(
        mock.patch.dict(
            "sys.modules", {"agent_sandbox_rl": FakeAgentSandboxRLModule}
        )
    )
    mock_threading_atexit = self.enter_context(
        mock.patch.object(
            sandbox_utils.threading, "_register_atexit", autospec=True
        )
    )
    mock_atexit = self.enter_context(
        mock.patch.object(sandbox_utils.atexit, "register", autospec=True)
    )

    with self.assertRaises(ValueError):
      sandbox_utils.init_global_fleet(run_id="   ")

    # Simulated worker process attaching to orchestrator-owned pools: shares
    # deterministic DNS-1123 run_id and skips exit teardown hooks.
    worker_fleet = sandbox_utils.init_global_fleet(
        tasks=[{"instance_id": "id_1", "docker_image": "test-image:v1"}],
        num_generations=2,
        run_id="run_20261110_123456",
        teardown_on_exit=False,
    )
    self.assertIsInstance(worker_fleet, FakeSandboxFleet)
    self.assertEqual(worker_fleet.config.run_isolation, "names")
    self.assertFalse(worker_fleet.config.install_teardown_hooks)
    self.assertFalse(worker_fleet.teardown_hooks_installed)
    self.assertRegex(worker_fleet.run_id, r"^[0-9a-f]{8}$")
    mock_threading_atexit.assert_not_called()
    mock_atexit.assert_not_called()

    # Simulated orchestrator process with the same logical run_id resolves the
    # exact same DNS-1123 fleet run_id and arms exit hooks.
    sandbox_utils._GLOBAL_FLEET = None
    orch_fleet = sandbox_utils.init_global_fleet(
        tasks=[{"instance_id": "id_1", "docker_image": "test-image:v1"}],
        num_generations=2,
        run_id="run_20261110_123456",
        teardown_on_exit=True,
    )
    self.assertEqual(orch_fleet.run_id, worker_fleet.run_id)
    self.assertTrue(orch_fleet.config.install_teardown_hooks)
    self.assertTrue(orch_fleet.teardown_hooks_installed)
    mock_threading_atexit.assert_called_once_with(
        sandbox_utils.teardown_global_fleet
    )
    mock_atexit.assert_called_once_with(sandbox_utils.teardown_global_fleet)

  def test_init_global_fleet_tears_down_on_plan_failure(self):
    created: list[FakeSandboxFleet] = []

    class TrackingFakeFleet(FakeSandboxFleet):

      def __init__(self, config: FakeFleetConfig):
        super().__init__(config)
        created.append(self)

    class TrackingModule(FakeAgentSandboxRLModule):
      SandboxFleet = TrackingFakeFleet  # pylint: disable=invalid-name

    self.enter_context(
        mock.patch.dict("sys.modules", {"agent_sandbox_rl": TrackingModule})
    )
    FakeSandboxFleet.plan_error = RuntimeError("plan boom")

    with self.assertRaisesRegex(RuntimeError, "plan boom"):
      sandbox_utils.init_global_fleet(
          tasks=[{"instance_id": "id_1", "docker_image": "test-image:v1"}],
          teardown_on_exit=True,
      )
    self.assertLen(created, 1)
    self.assertEqual(created[0].teardown_calls, 1)
    self.assertIsNone(sandbox_utils._GLOBAL_FLEET)

    # Attaching worker (teardown_on_exit=False) must NOT tear down shared pools
    # if local plan() fails.
    with self.assertRaisesRegex(RuntimeError, "plan boom"):
      sandbox_utils.init_global_fleet(
          tasks=[{"instance_id": "id_1", "docker_image": "test-image:v1"}],
          teardown_on_exit=False,
      )
    self.assertLen(created, 2)
    self.assertEqual(created[1].teardown_calls, 0)
    self.assertIsNone(sandbox_utils._GLOBAL_FLEET)

  def test_init_global_fleet_registers_exit_hooks_once_and_tears_down_idempotently(
      self,
  ):
    self.enter_context(
        mock.patch.dict(
            "sys.modules", {"agent_sandbox_rl": FakeAgentSandboxRLModule}
        )
    )
    mock_threading_atexit = self.enter_context(
        mock.patch.object(
            sandbox_utils.threading, "_register_atexit", autospec=True
        )
    )
    mock_atexit = self.enter_context(
        mock.patch.object(sandbox_utils.atexit, "register", autospec=True)
    )

    fleet1 = sandbox_utils.init_global_fleet(
        tasks=[{"instance_id": "id_1", "docker_image": "test-image:v1"}],
        num_generations=2,
    )
    self.assertIn("concurrent.futures.thread", sys.modules)
    self.assertIn("_futures_thread", vars(sandbox_utils))
    self.assertTrue(callable(sandbox_utils._futures_thread._python_exit))
    mock_threading_atexit.assert_called_once_with(
        sandbox_utils.teardown_global_fleet
    )
    mock_atexit.assert_called_once_with(sandbox_utils.teardown_global_fleet)

    # First teardown (e.g., threading._shutdown() before ThreadPoolExecutor
    # shutdown) tears down the fleet and clears _GLOBAL_FLEET; second call
    # (e.g., atexit) is an idempotent no-op.
    sandbox_utils.teardown_global_fleet()
    self.assertEqual(fleet1.teardown_calls, 1)
    self.assertIsNone(sandbox_utils._GLOBAL_FLEET)

    sandbox_utils.teardown_global_fleet()
    self.assertEqual(fleet1.teardown_calls, 1)

    # Re-initializing in the same process does not duplicate exit handlers, and
    # exceptions raised by fleet.teardown() are swallowed so threading._shutdown
    # continues cleanly.
    fleet2 = sandbox_utils.init_global_fleet(
        tasks=[{"instance_id": "id_2", "docker_image": "test-image:v2"}],
        num_generations=2,
    )
    self.assertIsNot(fleet2, fleet1)
    self.assertEqual(mock_threading_atexit.call_count, 1)
    self.assertEqual(mock_atexit.call_count, 1)

    FakeSandboxFleet.teardown_error = RuntimeError("k8s api unreachable")
    sandbox_utils.teardown_global_fleet()
    self.assertEqual(fleet2.teardown_calls, 1)
    self.assertIsNone(sandbox_utils._GLOBAL_FLEET)


if __name__ == "__main__":
  absltest.main()

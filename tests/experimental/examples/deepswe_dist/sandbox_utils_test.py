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

"""Unit tests for tunix.experimental.examples.deepswe_dist.sandbox_utils."""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import hashlib
from unittest import mock

from absl.testing import absltest
import numpy as np
from tunix.experimental.examples.deepswe_dist import sandbox_utils


@dataclasses.dataclass
class FakeTask:
  id: str
  image: str
  metadata: dict[str, object]


@dataclasses.dataclass
class FakePlanEntry:
  image: str
  replicas: int


class FakeFleetPlan:

  def __init__(self, entries: list[FakePlanEntry]):
    self.entries = entries
    self._by_image = {e.image: e for e in entries}

  def for_image(self, image: str) -> FakePlanEntry | None:
    return self._by_image.get(image)


@dataclasses.dataclass
class FakeFleetConfig:
  clusters: list[object] = dataclasses.field(default_factory=list)
  max_concurrent: int = 1
  window_size: int | None = None
  max_warmpool_size: int = 8
  warm_per_task: bool = False
  install_teardown_hooks: bool = True
  run_id: str | None = None
  template_name_prefix: str = "r2e-img-"
  labels: dict[str, str] = dataclasses.field(
      default_factory=lambda: {"app": "agent-sandbox-rl"}
  )
  pod_template: dict[str, object] | None = None


@dataclasses.dataclass
class FakeClusterConfig:
  name: str = "default"
  namespace: str = "default"
  node_selector: dict[str, str] | None = None
  tolerations: list[dict[str, object]] | None = None
  in_cluster: bool = False


class FakeSandboxFleet:
  """Typed fake matching `agent_sandbox_rl.fleet.SandboxFleet` at 9afee7039342."""

  def __init__(self, config: FakeFleetConfig | None = None):
    self.config = config or FakeFleetConfig()
    self.run_id = self.config.run_id or "default-run"
    self.plan_: FakeFleetPlan | None = None
    self.tasks: list[object] = []
    self.warm_calls: list[tuple[str, int | None, bool]] = []
    self.warm_images_calls: list[tuple[list[str], int | None, bool]] = []
    self.set_replicas_calls: list[tuple[str, int]] = []
    self.unwarm_calls: list[str] = []
    self.active_pools: dict[str, int] = {}
    self.preflight_called = False
    self.teardown_hooks_installed = False

  def load_tasks(
      self,
      source: list[object],
      *,
      image_rewrite: Callable[[str], str] | None = None,
  ) -> list[object]:
    del image_rewrite
    self.tasks = list(source)
    return self.tasks

  def preflight(self) -> dict[str, object]:
    self.preflight_called = True
    return {"default": True}

  def plan(self) -> FakeFleetPlan:
    seen: list[str] = []
    for t in self.tasks:
      img = t.image if isinstance(t, FakeTask) else str(t)
      if img not in seen:
        seen.append(img)
    entries = [
        FakePlanEntry(image=img, replicas=self.config.max_warmpool_size)
        for img in seen
    ]
    self.plan_ = FakeFleetPlan(entries)
    return self.plan_

  def warm_images(
      self,
      images: list[str],
      *,
      replicas_override: int | None = None,
      wait: bool = True,
  ) -> None:
    self.warm_images_calls.append((list(images), replicas_override, wait))
    reps = (
        replicas_override
        if replicas_override is not None
        else self.config.max_warmpool_size
    )
    for img in images:
      self.active_pools[img] = reps

  def warm_image(
      self,
      image: str,
      *,
      replicas_override: int | None = None,
      wait: bool = True,
  ) -> None:
    if self.plan_ is None or self.plan_.for_image(image) is None:
      raise KeyError(f"image not in plan: {image}")
    self.warm_calls.append((image, replicas_override, wait))
    self.active_pools[image] = (
        replicas_override
        if replicas_override is not None
        else self.config.max_warmpool_size
    )

  def set_pool_replicas(self, image: str, replicas: int) -> None:
    self.set_replicas_calls.append((image, replicas))
    self.active_pools[image] = replicas

  def unwarm_image(self, image: str) -> None:
    self.unwarm_calls.append(image)
    self.active_pools.pop(image, None)

  def teardown(self) -> dict[str, object] | None:
    self.active_pools.clear()
    return None

  def _install_teardown_hooks(self) -> None:
    self.teardown_hooks_installed = True


class FakeAgentSandboxRLModule:
  Task = FakeTask  # pylint: disable=invalid-name
  FleetConfig = FakeFleetConfig  # pylint: disable=invalid-name
  ClusterConfig = FakeClusterConfig  # pylint: disable=invalid-name
  SandboxFleet = FakeSandboxFleet  # pylint: disable=invalid-name


class SandboxUtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._modules_patch = mock.patch.dict(
        "sys.modules", {"agent_sandbox_rl": FakeAgentSandboxRLModule}
    )
    self._modules_patch.start()
    self.addCleanup(self._modules_patch.stop)
    sandbox_utils._GLOBAL_FLEET = None
    sandbox_utils._GLOBAL_IMAGE_REWRITE_FN = None

  def test_prewarm_iterator_registers_plan_and_warms_pools(self):
    fleet = FakeSandboxFleet(FakeFleetConfig(max_warmpool_size=4))
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

    self.assertIsNotNone(fleet.plan_)
    self.assertIsNotNone(fleet.plan_.for_image("img_A"))
    self.assertIsNotNone(fleet.plan_.for_image("img_B"))
    self.assertEqual(
        fleet.warm_calls,
        [("img_A", 8, False), ("img_B", 8, False)],
    )
    self.assertEqual(fleet.active_pools, {"img_A": 8, "img_B": 8})

    # Consume batch 0 (p0, p1)
    self.assertEqual(next(iterator)["prompt"], "p0")
    self.assertEqual(next(iterator)["prompt"], "p1")

    # Step 2 transitions batch 0 -> previous, batch 1 -> current,
    # and batch 2 (img_C: 4) -> next.
    self.assertEqual(next(iterator)["prompt"], "p2")
    self.assertIsNotNone(fleet.plan_.for_image("img_C"))
    self.assertEqual(
        fleet.warm_calls,
        [("img_A", 8, False), ("img_B", 8, False), ("img_C", 4, False)],
    )
    self.assertEqual(
        fleet.active_pools, {"img_A": 8, "img_B": 8, "img_C": 4}
    )
    self.assertEqual(next(iterator)["prompt"], "p3")

    # Step 4 retires img_A (since max_staleness=0 retains 1 previous batch)
    self.assertEqual(next(iterator)["prompt"], "p4")
    self.assertIn("img_A", fleet.unwarm_calls)
    self.assertEqual(fleet.active_pools, {"img_B": 8, "img_C": 4})

    with self.assertRaises(StopIteration):
      next(iterator)

    iterator.close()
    self.assertEqual(fleet.active_pools, {})

  def test_max_staleness_retains_multiple_previous_batches(self):
    fleet = FakeSandboxFleet(FakeFleetConfig(max_warmpool_size=4))
    dataset = [
        {"prompt": "p0", "docker_image": "img_A"},
        {"prompt": "p1", "docker_image": "img_B"},
        {"prompt": "p2", "docker_image": "img_C"},
        {"prompt": "p3", "docker_image": "img_D"},
    ]
    iterator = sandbox_utils.PrewarmDatasetIterator(
        dataset,
        fleet=fleet,
        num_generations=4,
        batch_size=1,
        max_staleness=1,
    )
    self.assertEqual(next(iterator)["prompt"], "p0")
    self.assertEqual(next(iterator)["prompt"], "p1")
    self.assertNotIn("img_A", fleet.unwarm_calls)
    self.assertEqual(next(iterator)["prompt"], "p2")
    self.assertNotIn("img_A", fleet.unwarm_calls)
    self.assertEqual(next(iterator)["prompt"], "p3")
    self.assertIn("img_A", fleet.unwarm_calls)
    self.assertNotIn("img_B", fleet.unwarm_calls)
    iterator.close()

  def test_init_global_fleet_rollout_worker_registers_plan_on_singleton_reuse(
      self,
  ):
    fleet1 = sandbox_utils.init_global_fleet(
        tasks=[{"instance_id": "id_1", "docker_image": "img_1"}],
        num_generations=16,
        job_id="tsbao-mlperf-35b-orch",
        warm_on_init=False,
        teardown_on_exit=False,
    )
    expected_run_id = hashlib.sha256(
        b"tsbao-mlperf-35b-orch"
    ).hexdigest()[:8]
    self.assertEqual(fleet1.run_id, expected_run_id)
    self.assertEqual(
        fleet1.config.template_name_prefix, "r2e-tsbao-mlperf-35b-"
    )
    self.assertEqual(
        fleet1.config.labels["agent-sandbox-rl-run"],
        expected_run_id,
    )
    self.assertEqual(
        fleet1.config.labels["app.kubernetes.io/created-by"],
        "tsbao-mlperf-35b-orch",
    )
    self.assertIsNotNone(fleet1.plan_)
    self.assertIsNotNone(fleet1.plan_.for_image("img_1"))
    self.assertEqual(fleet1.warm_images_calls, [])
    self.assertFalse(fleet1.preflight_called)
    self.assertFalse(getattr(fleet1, "_torndown", True))
    self.assertFalse(fleet1.teardown_hooks_installed)

    # Episode 2+ reuses the singleton and registers the new task image in plan_
    # via load_tasks + plan().
    fleet2 = sandbox_utils.init_global_fleet(
        tasks=[{"instance_id": "id_2", "docker_image": "img_2"}],
        num_generations=16,
        job_id="tsbao-mlperf-35b-orch",
        warm_on_init=False,
        teardown_on_exit=False,
    )
    self.assertIs(fleet2, fleet1)
    self.assertIsNotNone(fleet2.plan_.for_image("img_1"))
    self.assertIsNotNone(fleet2.plan_.for_image("img_2"))

  def test_init_global_fleet_raises_when_fleet_ignores_shared_run_id(self):
    class BrokenSandboxFleet(FakeSandboxFleet):

      def __init__(self, config: FakeFleetConfig | None = None):
        super().__init__(config)
        self.run_id = "random-uuid-ignored-config"

    with mock.patch.object(
        FakeAgentSandboxRLModule, "SandboxFleet", BrokenSandboxFleet
    ):
      with self.assertRaisesRegex(
          RuntimeError, "did not honor shared run_id/template_name_prefix"
      ):
        sandbox_utils.init_global_fleet(
            tasks=[{"instance_id": "id_1", "docker_image": "img_1"}],
            job_id="tsbao-mlperf-35b-orch",
        )

  def test_numpy_array_and_nested_metadata_extraction(self):
    fleet = FakeSandboxFleet(FakeFleetConfig(max_warmpool_size=3))
    dataset = [
        {
            "prompt": np.array(["p0", "p1"]),
            "docker_image": np.array(["array_img_A", "array_img_A"]),
        },
        {
            "prompt": "p2",
            "metadata": {
                "env_config": {"entry": {"docker_image": "nested_img_B"}}
            },
        },
    ]
    iterator = sandbox_utils.PrewarmDatasetIterator(
        dataset, fleet=fleet, num_generations=3, batch_size=2
    )
    self.assertEqual(fleet.active_pools.get("array_img_A"), 6)
    self.assertEqual(fleet.active_pools.get("nested_img_B"), 3)
    iterator.close()


if __name__ == "__main__":
  absltest.main()

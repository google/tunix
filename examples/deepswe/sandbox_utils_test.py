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

import os
import threading
import time
import types
from unittest import mock
from absl.testing import absltest
import numpy as np
from examples.deepswe import sandbox_utils


class FakeFleet:
  """Fake SandboxFleet that records calls and tracks active pool replica state."""

  def __init__(self):
    self._lock = threading.Lock()
    self.warm_calls: list[tuple[str, int | None, bool]] = []
    self.set_replicas_calls: list[tuple[str, int]] = []
    self.unwarm_calls: list[str] = []
    self.active_pools: dict[str, int] = {}

  def warm_image(
      self, image: str, replicas_override: int | None = None, wait: bool = False
  ) -> None:
    with self._lock:
      self.warm_calls.append((image, replicas_override, wait))
      self.active_pools[image] = replicas_override or 1

  def set_pool_replicas(self, image: str, replicas: int) -> None:
    with self._lock:
      self.set_replicas_calls.append((image, replicas))
      self.active_pools[image] = replicas

  def unwarm_image(self, image: str) -> None:
    with self._lock:
      self.unwarm_calls.append(image)
      self.active_pools.pop(image, None)

  def teardown(self) -> None:
    with self._lock:
      self.active_pools.clear()


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
    # Fleet warms both with wait=True (synchronous initial priming barrier).
    self.assertEqual(
        fleet.warm_calls, [("img_A", 8, True), ("img_B", 8, True)]
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

  def test_max_staleness_retains_multiple_previous_batches(self):
    fleet = FakeFleet()
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
        unwarm_on_exhaustion=False,
        max_staleness=1,
    )
    self.assertTrue(iterator.has_next())
    self.assertEqual(fleet.active_pools, {"img_A": 4, "img_B": 4})

    # Batch 0 (p0 / img_A)
    self.assertEqual(next(iterator)["prompt"], "p0")
    self.assertEqual(fleet.active_pools, {"img_A": 4, "img_B": 4})

    # Batch 1 (p1 / img_B): img_A retained in _previous_batches (1/2)
    self.assertEqual(next(iterator)["prompt"], "p1")
    self.assertNotIn("img_A", fleet.unwarm_calls)
    self.assertEqual(
        fleet.active_pools, {"img_A": 4, "img_B": 4, "img_C": 4}
    )

    # Batch 2 (p2 / img_C): both img_A and img_B retained in _previous_batches (2/2)
    self.assertEqual(next(iterator)["prompt"], "p2")
    self.assertNotIn("img_A", fleet.unwarm_calls)
    self.assertNotIn("img_B", fleet.unwarm_calls)
    self.assertEqual(
        fleet.active_pools, {"img_A": 4, "img_B": 4, "img_C": 4, "img_D": 4}
    )

    # Batch 3 (p3 / img_D): img_A evicted as _previous_batches slides to [img_B, img_C]
    self.assertEqual(next(iterator)["prompt"], "p3")
    self.assertIn("img_A", fleet.unwarm_calls)
    self.assertNotIn("img_B", fleet.unwarm_calls)
    self.assertEqual(
        fleet.active_pools, {"img_B": 4, "img_C": 4, "img_D": 4}
    )
    self.assertFalse(iterator.has_next())

    with self.assertRaises(StopIteration):
      next(iterator)
    iterator.close()
    self.assertEqual(fleet.active_pools, {})

  def test_wait_initial_configurable(self):
    fleet = FakeFleet()
    dataset = [{"prompt": "p0", "docker_image": "img_A"}]
    _ = sandbox_utils.PrewarmDatasetIterator(
        dataset,
        fleet=fleet,
        num_generations=2,
        batch_size=1,
        wait_initial=False,
    )
    self.assertEqual(fleet.warm_calls, [("img_A", 2, False)])

  def test_wait_initial_env_default(self):
    dataset = [{"prompt": "p0", "docker_image": "img_A"}]
    for env, explicit, want in (
        ({}, None, True),
        ({"DEEPSWE_PREWARM_WAIT_INITIAL": "0"}, None, False),
        ({"DEEPSWE_PREWARM_WAIT_INITIAL": "1"}, None, True),
        ({"DEEPSWE_PREWARM_WAIT_INITIAL": "0"}, True, True),
    ):
      fleet = FakeFleet()
      with mock.patch.dict(os.environ, env, clear=True):
        sandbox_utils.PrewarmDatasetIterator(
            list(dataset),
            fleet=fleet,
            num_generations=2,
            batch_size=1,
            wait_initial=explicit,
        )
      self.assertEqual(fleet.warm_calls, [("img_A", 2, want)], (env, explicit))

  def test_max_warmpool_replicas_env_default(self):
    dataset = [{"prompt": "p0", "docker_image": "img_A"}]
    key = "DEEPSWE_PREWARM_MAX_REPLICAS"
    for env, explicit, scale_on_hold, want in (
        ({}, None, False, 4),
        ({key: ""}, None, False, 4),
        ({key: "1"}, None, False, 1),
        ({key: "0"}, None, False, 1),  # Raised to 1: claims need the pool.
        ({key: "-3"}, None, False, 1),
        ({key: "abc"}, None, False, 4),
        ({key: "1"}, 3, False, 3),  # An explicit cap wins.
        ({key: "1"}, None, True, 1),
        ({}, None, True, 4),
    ):
      fleet = FakeFleet()
      with mock.patch.dict(os.environ, env, clear=True):
        iterator = sandbox_utils.PrewarmDatasetIterator(
            list(dataset),
            fleet=fleet,
            num_generations=4,
            batch_size=1,
            max_warmpool_replicas=explicit,
            scale_on_hold=scale_on_hold,
        )
      case = (env, explicit, scale_on_hold)
      self.assertEqual(fleet.warm_calls, [("img_A", want, True)], case)
      self.assertEqual(fleet.active_pools.get("img_A"), want, case)
      iterator.close()

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

    with mock.patch.dict(os.environ, {"IMAGE_REWRITE_PREFIX": "europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/tunix/"}):
      rewrite = sandbox_utils.get_image_rewrite_fn()
      self.assertIsNotNone(rewrite)
      self.assertEqual(
          rewrite("namanjain12/aiohttp_final:v1"),
          "europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/tunix/aiohttp_final:v1",
      )
      self.assertEqual(
          rewrite("gcr.io/other/project/my_image:tag"),
          "europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/tunix/my_image:tag",
      )

    # Verify both without trailing slash and with wrapping quotes are normalized
    for test_prefix in (
        "europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/tunix",
        '"europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/tunix/"',
    ):
      with mock.patch.dict(os.environ, {"IMAGE_REWRITE_PREFIX": test_prefix}):
        rewrite = sandbox_utils.get_image_rewrite_fn()
        self.assertIsNotNone(rewrite)
        self.assertEqual(
            rewrite("my_image:v1"),
            "europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/tunix/my_image:v1",
        )

    with mock.patch.dict(os.environ, {}, clear=True):
      rewrite = sandbox_utils.get_image_rewrite_fn()
      self.assertIsNone(rewrite)

  def test_init_global_fleet_plans_without_eager_warming(self):
    mock_fleet = mock.MagicMock()
    mock_entry = mock.MagicMock()
    mock_entry.image = "test-image:v1"
    mock_fleet.plan_.entries = [mock_entry]

    mock_as_rl = mock.MagicMock()
    mock_as_rl.SandboxFleet.return_value = mock_fleet
    with mock.patch.dict("sys.modules", {"agent_sandbox_rl": mock_as_rl}):
      with mock.patch.object(sandbox_utils, "_GLOBAL_FLEET", None):
        fleet = sandbox_utils.init_global_fleet(
            tasks=[{"docker_image": "test-image:v1"}],
            num_generations=4,
        )
        self.assertEqual(fleet, mock_fleet)
        mock_fleet.plan.assert_called_once()
        mock_fleet.warm_images.assert_not_called()

  def test_init_global_fleet_configures_labels_and_teardown_hooks(self):
    mock_fleet = mock.MagicMock()
    mock_as_rl = mock.MagicMock()
    mock_as_rl.SandboxFleet.return_value = mock_fleet
    with mock.patch.dict("sys.modules", {"agent_sandbox_rl": mock_as_rl}):
      with mock.patch.dict(os.environ, {"ORCHESTRATOR_ID": "test-user-orch"}):
        with mock.patch.object(sandbox_utils, "_GLOBAL_FLEET", None):
          _ = sandbox_utils.init_global_fleet(
              tasks=None,
              num_generations=4,
          )
          fleet_cfg_call = mock_as_rl.FleetConfig.call_args[1]
          self.assertTrue(fleet_cfg_call.get("install_teardown_hooks"))
          self.assertEqual(
              fleet_cfg_call.get("labels"),
              {
                  "app": "agent-sandbox-rl",
                  "app.kubernetes.io/created-by": "test-user-orch",
              },
          )

  def test_init_global_fleet_scopes_teardown_to_run_selector(self):
    mock_fleet = mock.MagicMock()
    mock_cluster = mock.MagicMock()
    mock_fleet.registry = [mock_cluster]
    mock_fleet.run_selector.return_value = "agents.x-k8s.io/asrl-run-id=test-run"

    mock_as_rl = mock.MagicMock()
    mock_as_rl.SandboxFleet.return_value = mock_fleet
    with mock.patch.dict("sys.modules", {"agent_sandbox_rl": mock_as_rl}):
      with mock.patch.object(sandbox_utils, "_GLOBAL_FLEET", None):
        _ = sandbox_utils.init_global_fleet(tasks=None, num_generations=4)
        self.assertEqual(
            mock_cluster.resources.managed_selector, mock_fleet.run_selector
        )

  def test_init_global_fleet_derives_template_name_prefix_from_job_prefix(self):
    mock_fleet = mock.MagicMock()
    mock_as_rl = mock.MagicMock()
    mock_as_rl.SandboxFleet.return_value = mock_fleet
    with mock.patch.dict("sys.modules", {"agent_sandbox_rl": mock_as_rl}):
      with mock.patch.dict(os.environ, {"JOB_PREFIX": "atwigg-openhands-rr"}):
        with mock.patch.object(sandbox_utils, "_GLOBAL_FLEET", None):
          _ = sandbox_utils.init_global_fleet(tasks=None, num_generations=4, scaffold="openhands")
          fleet_cfg_call = mock_as_rl.FleetConfig.call_args[1]
          self.assertEqual(
              fleet_cfg_call.get("template_name_prefix"),
              "oh-atwigg-openhands-rr-",
          )

  def test_init_global_fleet_derives_template_name_prefix_from_orchestrator_id(self):
    mock_fleet = mock.MagicMock()
    mock_as_rl = mock.MagicMock()
    mock_as_rl.SandboxFleet.return_value = mock_fleet
    with mock.patch.dict("sys.modules", {"agent_sandbox_rl": mock_as_rl}):
      with mock.patch.dict(os.environ, {"ORCHESTRATOR_ID": "atwigg-mamba-orch"}, clear=True):
        with mock.patch.object(sandbox_utils, "_GLOBAL_FLEET", None):
          _ = sandbox_utils.init_global_fleet(tasks=None, num_generations=4, scaffold="r2egym")
          fleet_cfg_call = mock_as_rl.FleetConfig.call_args[1]
          self.assertEqual(
              fleet_cfg_call.get("template_name_prefix"),
              "r2e-atwigg-mamba-",
          )

  def test_init_global_fleet_respects_custom_prefixes_and_formats(self):
    mock_fleet = mock.MagicMock()
    mock_as_rl = mock.MagicMock()
    mock_as_rl.SandboxFleet.return_value = mock_fleet
    with mock.patch.dict("sys.modules", {"agent_sandbox_rl": mock_as_rl}):
      with mock.patch.dict(
          os.environ,
          {
              "JOB_PREFIX": "my-job",
              "TEMPLATE_NAME_PREFIX": "custom-tmpl-",
              "POOL_NAME_FORMAT": "custom-pool-{image_hash}",
          },
      ):
        with mock.patch.object(sandbox_utils, "_GLOBAL_FLEET", None):
          _ = sandbox_utils.init_global_fleet(tasks=None, num_generations=4)
          fleet_cfg_call = mock_as_rl.FleetConfig.call_args[1]
          self.assertEqual(
              fleet_cfg_call.get("template_name_prefix"),
              "custom-tmpl-",
          )
          self.assertEqual(
              fleet_cfg_call.get("pool_name_format"),
              "custom-pool-{image_hash}",
          )

  def test_init_global_fleet_sets_sandbox_priority_class(self):
    mock_as_rl = mock.MagicMock()
    base_template = mock.MagicMock()
    base_template.extra_pod_spec = {"tolerations": []}
    mock_as_rl.TemplateSpec.return_value = base_template
    with mock.patch.dict("sys.modules", {"agent_sandbox_rl": mock_as_rl}):
      with mock.patch.dict(
          os.environ, {"SANDBOX_PRIORITY_CLASS": "low"}, clear=True
      ):
        with mock.patch.object(sandbox_utils, "_GLOBAL_FLEET", None):
          _ = sandbox_utils.init_global_fleet(
              tasks=None, num_generations=4, scaffold="r2egym"
          )
    base_template.model_copy.assert_called_once_with(
        update={
            "extra_pod_spec": {"tolerations": [], "priorityClassName": "low"}
        }
    )
    self.assertIs(
        mock_as_rl.FleetConfig.call_args[1]["template"],
        base_template.model_copy.return_value,
    )
    self.assertEqual(base_template.extra_pod_spec, {"tolerations": []})

  def test_init_global_fleet_without_priority_class_keeps_default_template(
      self,
  ):
    mock_as_rl = mock.MagicMock()
    with mock.patch.dict("sys.modules", {"agent_sandbox_rl": mock_as_rl}):
      with mock.patch.dict(os.environ, {}, clear=True):
        with mock.patch.object(sandbox_utils, "_GLOBAL_FLEET", None):
          _ = sandbox_utils.init_global_fleet(
              tasks=None, num_generations=4, scaffold="r2egym"
          )
    self.assertNotIn("template", mock_as_rl.FleetConfig.call_args[1])
    mock_as_rl.TemplateSpec.assert_not_called()

  def test_teardown_global_fleet_invokes_teardown_and_reaper(self):
    mock_fleet = mock.MagicMock()
    mock_fleet.run_id = "test-run-1234"
    mock_cluster = mock.MagicMock()
    mock_cluster.namespace = "test-ns"
    mock_cluster.in_cluster = True
    mock_fleet.registry = [mock_cluster]

    mock_as_rl = mock.MagicMock()
    with mock.patch.dict("sys.modules", {"agent_sandbox_rl": mock_as_rl}):
      with mock.patch.object(sandbox_utils, "_GLOBAL_FLEET", mock_fleet):
        sandbox_utils.teardown_global_fleet()
        self.assertIsNone(sandbox_utils._GLOBAL_FLEET)
        mock_fleet.teardown.assert_called_once()
        mock_as_rl.reap.assert_called_once_with(
            run_id="test-run-1234",
            in_cluster=True,
            namespace="test-ns",
            delete_pods=False,
        )

  def test_init_global_fleet_registers_subsequent_task_images_in_plan(self):
    class _FakeTask:

      def __init__(self, id, image, metadata=None):
        self.id = id
        self.image = image
        self.metadata = metadata or {}

    mock_fleet = mock.MagicMock()
    mock_fleet.tasks = []
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

  def test_init_global_fleet_shares_deterministic_run_id_across_job_processes(self):
    mock_fleet_orch = mock.MagicMock()
    mock_fleet_orch.config.labels = {}
    mock_cluster_orch = mock.MagicMock()
    mock_cluster_orch.resources.labels = {}
    mock_fleet_orch.registry = [mock_cluster_orch]

    mock_fleet_roll = mock.MagicMock()
    mock_fleet_roll.config.labels = {}
    mock_cluster_roll = mock.MagicMock()
    mock_cluster_roll.resources.labels = {}
    mock_fleet_roll.registry = [mock_cluster_roll]

    mock_as_rl = mock.MagicMock()
    mock_as_rl.SandboxFleet.side_effect = [mock_fleet_orch, mock_fleet_roll]

    with mock.patch.dict("sys.modules", {"agent_sandbox_rl": mock_as_rl}):
      with mock.patch.dict(
          os.environ,
          {"JOB_PREFIX": "trellis-1024-0929", "ORCHESTRATOR_ID": "trellis-1024-0929-orch"},
          clear=True,
      ):
        with mock.patch.object(sandbox_utils, "_GLOBAL_FLEET", None):
          _ = sandbox_utils.init_global_fleet(tasks=None, num_generations=4)
        with mock.patch.object(sandbox_utils, "_GLOBAL_FLEET", None):
          _ = sandbox_utils.init_global_fleet(tasks=None, num_generations=4)

    self.assertEqual(mock_fleet_orch.run_id, mock_fleet_roll.run_id)
    self.assertLen(mock_fleet_orch.run_id, 12)

  def test_parallel_prewarming_across_images(self):
    fleet = FakeFleet()
    dataset = [
        {"prompt": f"p{i}", "docker_image": f"img_{i}"} for i in range(8)
    ]
    iterator = sandbox_utils.PrewarmDatasetIterator(
        dataset,
        fleet=fleet,
        num_generations=4,
        batch_size=4,
        wait_initial=True,
        max_workers=8,
    )
    # Both current_batch (img_0..3) and next_batch (img_4..7) should be warmed concurrently
    self.assertEqual(len(fleet.active_pools), 8)
    for i in range(8):
      self.assertEqual(fleet.active_pools[f"img_{i}"], 4)
    iterator.close()
    self.assertEqual(fleet.active_pools, {})


_FAIL_FAST_ENV = {
    "FT_SANDBOX_FAIL_FAST": "true",
    "FT_SANDBOX_READY_TIMEOUT_S": "300",
    "FT_SANDBOX_ACQUIRE_RETRIES": "2",
}


class _FakeFleetError(Exception):
  """Stands in for agent_sandbox_rl.FleetError."""


class _FailingWarmFleet(FakeFleet):

  def warm_image(
      self, image: str, replicas_override: int | None = None, wait: bool = False
  ) -> None:
    raise _FakeFleetError(f"pool for {image} never became ready")


class SandboxFailFastTest(absltest.TestCase):

  def _fake_sdk(self) -> mock.MagicMock:
    sdk = mock.MagicMock()
    sdk.FleetError = _FakeFleetError
    return sdk

  def test_config_defaults_to_legacy_when_unset(self):
    with mock.patch.dict(os.environ, {}, clear=True):
      self.assertEqual(
          sandbox_utils.SandboxFailFastConfig.from_env(),
          sandbox_utils.SandboxFailFastConfig(
              enabled=False, ready_timeout_s=None, acquire_retries=5
          ),
      )

  def test_config_fail_fast_reads_knobs(self):
    with mock.patch.dict(os.environ, _FAIL_FAST_ENV, clear=True):
      self.assertEqual(
          sandbox_utils.SandboxFailFastConfig.from_env(),
          sandbox_utils.SandboxFailFastConfig(
              enabled=True, ready_timeout_s=300, acquire_retries=2
          ),
      )

  def test_config_rejects_bad_values(self):
    for env, error in (
        ({"FT_SANDBOX_FAIL_FAST": "1"}, ValueError),
        ({"FT_SANDBOX_FAIL_FAST": "true"}, KeyError),
        ({**_FAIL_FAST_ENV, "FT_SANDBOX_ACQUIRE_RETRIES": "0"}, ValueError),
    ):
      with self.subTest(env=env):
        with mock.patch.dict(os.environ, env, clear=True):
          with self.assertRaises(error):
            sandbox_utils.SandboxFailFastConfig.from_env()

  def test_init_global_fleet_fail_fast_sets_timeout_and_skips_preflight(self):
    sdk = self._fake_sdk()
    with mock.patch.dict("sys.modules", {"agent_sandbox_rl": sdk}):
      with mock.patch.dict(os.environ, _FAIL_FAST_ENV):
        with mock.patch.object(sandbox_utils, "_GLOBAL_FLEET", None):
          sandbox_utils.init_global_fleet(tasks=None, num_generations=4)
    self.assertEqual(sdk.FleetConfig.call_args[1]["ready_timeout"], 300)
    sdk.SandboxFleet.return_value.preflight.assert_not_called()

  def test_init_global_fleet_off_keeps_sdk_timeout_and_skips_preflight(self):
    sdk = self._fake_sdk()
    with mock.patch.dict("sys.modules", {"agent_sandbox_rl": sdk}):
      with mock.patch.dict(os.environ, {}, clear=True):
        with mock.patch.object(sandbox_utils, "_GLOBAL_FLEET", None):
          sandbox_utils.init_global_fleet(tasks=None, num_generations=4)
    self.assertNotIn("ready_timeout", sdk.FleetConfig.call_args[1])
    sdk.SandboxFleet.return_value.preflight.assert_not_called()

  def test_prewarm_fail_fast_raises_fleet_error(self):
    dataset = [{"prompt": "p0", "docker_image": "img_A"}]
    with mock.patch.dict("sys.modules", {"agent_sandbox_rl": self._fake_sdk()}):
      with mock.patch.dict(os.environ, _FAIL_FAST_ENV):
        with self.assertRaisesRegex(_FakeFleetError, "never became ready"):
          sandbox_utils.PrewarmDatasetIterator(
              dataset, fleet=_FailingWarmFleet(), num_generations=2, batch_size=1
          )

  def test_prewarm_off_logs_fleet_error(self):
    dataset = [{"prompt": "p0", "docker_image": "img_A"}]
    with mock.patch.dict(os.environ, {}, clear=True):
      with self.assertLogs(level="WARNING") as logs:
        iterator = sandbox_utils.PrewarmDatasetIterator(
            dataset, fleet=_FailingWarmFleet(), num_generations=2, batch_size=1
        )
    self.assertEqual(next(iterator), dataset[0])
    self.assertTrue(any("Warm note for img_A" in line for line in logs.output))


class ScaleOnHoldTest(absltest.TestCase):

  def _iterator(self, dataset, fleet, **kwargs):
    iterator = sandbox_utils.PrewarmDatasetIterator(
        dataset, fleet=fleet, scale_on_hold=True, **kwargs
    )
    self.addCleanup(iterator.close)
    return iterator

  def _claim(self, image, n=1):
    for _ in range(n):
      sandbox_utils.note_sandbox_acquired(image)

  def test_pool_shrinks_as_claims_land_and_is_not_recreated(self):
    fleet = FakeFleet()
    dataset = [
        {"prompt": "p0", "docker_image": "img_A"},
        {"prompt": "p1", "docker_image": "img_A"},
        {"prompt": "p2", "docker_image": "img_B"},
        {"prompt": "p3", "docker_image": "img_B"},
        {"prompt": "p4", "docker_image": "img_C"},
        {"prompt": "p5", "docker_image": "img_D"},
    ]
    iterator = self._iterator(dataset, fleet, num_generations=4, batch_size=2)
    self.assertIs(sandbox_utils._ACTIVE_PREWARM_ITERATOR, iterator)
    self.assertEqual(fleet.active_pools, {"img_A": 8, "img_B": 8})

    # Batch 0 (img_A x2): every claim shrinks img_A's pool, none is replaced.
    self.assertEqual(next(iterator)["prompt"], "p0")
    self._claim("img_A", 8)
    self.assertEqual(fleet.active_pools, {"img_A": 0, "img_B": 8})
    self.assertEqual(
        [r for img, r in fleet.set_replicas_calls if img == "img_A"],
        [7, 6, 5, 4, 3, 2, 1, 0],
    )
    # A re-acquire after a failed env init: the pool stays at 0.
    self._claim("img_A")
    self.assertLen(fleet.set_replicas_calls, 8)
    self.assertEqual(next(iterator)["prompt"], "p1")

    # Batch 1 (img_B x2): img_A is retained (previous batch) at 0 replicas.
    self.assertEqual(next(iterator)["prompt"], "p2")
    self.assertEqual(
        fleet.active_pools, {"img_A": 0, "img_B": 8, "img_C": 4, "img_D": 4}
    )
    self._claim("img_B", 3)
    self._claim("img_unknown")  # Not prewarmed: ignored.
    self.assertEqual(fleet.active_pools["img_B"], 5)
    self.assertNotIn("img_unknown", fleet.active_pools)
    self.assertEqual(next(iterator)["prompt"], "p3")

    # Batch 2 (img_C, img_D): img_A leaves the window and is unwarmed.
    self.assertEqual(next(iterator)["prompt"], "p4")
    self.assertIn("img_A", fleet.unwarm_calls)
    self.assertEqual(fleet.active_pools, {"img_B": 5, "img_C": 4, "img_D": 4})
    # A late claim of the retired image must not recreate its pool.
    calls = len(fleet.set_replicas_calls)
    self._claim("img_A")
    self.assertLen(fleet.set_replicas_calls, calls)
    self.assertNotIn("img_A", fleet.active_pools)

    iterator.close()
    self.assertEqual(fleet.active_pools, {})
    self.assertIsNone(sandbox_utils._ACTIVE_PREWARM_ITERATOR)
    self._claim("img_C")  # No active iterator: a no-op.
    self.assertNotIn("img_C", fleet.active_pools)

  def test_same_image_in_consecutive_batches_keeps_next_batch_warm(self):
    fleet = FakeFleet()
    dataset = [
        {"prompt": "p0", "docker_image": "img_A"},
        {"prompt": "p1", "docker_image": "img_A"},
        {"prompt": "p2", "docker_image": "img_B"},
    ]
    iterator = self._iterator(dataset, fleet, num_generations=4, batch_size=1)
    self.assertEqual(fleet.active_pools, {"img_A": 8})
    self.assertEqual(next(iterator)["prompt"], "p0")
    self._claim("img_A", 4)
    self.assertEqual(fleet.active_pools, {"img_A": 4})  # p1's sandboxes.
    self.assertEqual(next(iterator)["prompt"], "p1")
    self.assertEqual(fleet.active_pools, {"img_A": 4, "img_B": 4})
    self._claim("img_A", 4)
    self.assertEqual(fleet.active_pools, {"img_A": 0, "img_B": 4})

  def test_concurrent_claims_end_at_zero(self):
    fleet = FakeFleet()
    dataset = [{"prompt": f"p{i}", "docker_image": f"img_{i}"} for i in range(8)]
    iterator = self._iterator(dataset, fleet, num_generations=16, batch_size=4)
    next(iterator)
    threads = [
        threading.Thread(target=self._claim, args=(f"img_{i % 4}",))
        for i in range(64)
    ]
    for t in threads:
      t.start()
    for t in threads:
      t.join()
    for i in range(4):
      self.assertEqual(fleet.active_pools[f"img_{i}"], 0)
    for i in range(4, 8):
      self.assertEqual(fleet.active_pools[f"img_{i}"], 16)

  def test_env_enables_and_default_is_off(self):
    fleet = FakeFleet()
    dataset = [{"prompt": "p0", "docker_image": "img_A"}]
    with mock.patch.dict(os.environ, {}, clear=True):
      off = sandbox_utils.PrewarmDatasetIterator(
          dataset, fleet=fleet, num_generations=4, batch_size=1
      )
    self.addCleanup(off.close)
    self.assertFalse(off.scale_on_hold)
    self.assertIsNone(sandbox_utils._ACTIVE_PREWARM_ITERATOR)
    self._claim("img_A", 2)
    self.assertEqual(fleet.active_pools, {"img_A": 4})
    off.close()
    with mock.patch.dict(os.environ, {"DEEPSWE_PREWARM_SCALE_ON_HOLD": "1"}):
      on = sandbox_utils.PrewarmDatasetIterator(
          list(dataset), fleet=FakeFleet(), num_generations=4, batch_size=1
      )
    self.addCleanup(on.close)
    self.assertTrue(on.scale_on_hold)
    self.assertIs(sandbox_utils._ACTIVE_PREWARM_ITERATOR, on)


class _BlockingAcquireFleet(FakeFleet):
  """FakeFleet whose acquire() blocks until `release` is set, then may raise."""

  def __init__(self, error: Exception | None = None):
    super().__init__()
    self.release = threading.Event()
    self.error = error

  def acquire(self, task):
    self.release.wait(10)
    if self.error is not None:
      raise self.error
    return ("handle", task.image)


class AcquireSandboxTest(absltest.TestCase):

  _TASK = types.SimpleNamespace(image="img_A")

  def setUp(self):
    super().setUp()
    self.addCleanup(mock.patch.stopall)

  def _prewarm(self, fleet):
    iterator = sandbox_utils.PrewarmDatasetIterator(
        [{"prompt": "p0", "docker_image": "img_A"}],
        fleet=fleet,
        num_generations=4,
        batch_size=1,
        scale_on_hold=True,
    )
    self.addCleanup(iterator.close)
    self.assertEqual(fleet.active_pools, {"img_A": 4})

  def _set_delay(self, delay):
    mock.patch.dict(
        os.environ, {"DEEPSWE_PREWARM_EARLY_NOTE_S": delay}
    ).start()

  def _start_acquire(self, fleet):
    out = {}

    def run():
      try:
        out["handle"] = sandbox_utils.acquire_sandbox(fleet, self._TASK)
      except Exception as e:  # pylint: disable=broad-exception-caught
        out["error"] = e

    thread = threading.Thread(target=run)
    thread.start()
    return thread, out

  def _wait_for_replicas(self, fleet, replicas):
    deadline = time.time() + 5
    while fleet.active_pools["img_A"] != replicas and time.time() < deadline:
      time.sleep(0.01)
    self.assertEqual(fleet.active_pools["img_A"], replicas)

  def test_reports_claim_still_waiting_after_delay_once(self):
    fleet = _BlockingAcquireFleet()
    self._prewarm(fleet)
    self._set_delay("0.05")
    thread, out = self._start_acquire(fleet)
    self._wait_for_replicas(fleet, 3)  # Reported while still waiting.
    fleet.release.set()
    thread.join()
    self.assertEqual(out, {"handle": ("handle", "img_A")})
    self.assertEqual(fleet.active_pools["img_A"], 3)  # Not reported twice.

  def test_fast_acquire_reports_once(self):
    fleet = _BlockingAcquireFleet()
    fleet.release.set()
    self._prewarm(fleet)
    self.assertEqual(
        sandbox_utils.acquire_sandbox(fleet, self._TASK), ("handle", "img_A")
    )
    time.sleep(0.1)
    self.assertEqual(fleet.active_pools["img_A"], 3)

  def test_failure_before_delay_reports_nothing(self):
    fleet = _BlockingAcquireFleet(error=RuntimeError("boom"))
    fleet.release.set()
    self._prewarm(fleet)
    self._set_delay("0.2")
    with self.assertRaisesRegex(RuntimeError, "boom"):
      sandbox_utils.acquire_sandbox(fleet, self._TASK)
    time.sleep(0.4)
    self.assertEqual(fleet.active_pools["img_A"], 4)

  def test_failure_after_delay_keeps_the_report(self):
    fleet = _BlockingAcquireFleet(error=RuntimeError("not ready"))
    self._prewarm(fleet)
    self._set_delay("0.05")
    thread, out = self._start_acquire(fleet)
    self._wait_for_replicas(fleet, 3)
    fleet.release.set()
    thread.join()
    self.assertIsInstance(out.get("error"), RuntimeError)
    self.assertEqual(fleet.active_pools["img_A"], 3)

  def test_zero_delay_reports_only_on_success(self):
    fleet = _BlockingAcquireFleet()
    self._prewarm(fleet)
    self._set_delay("0")
    thread, out = self._start_acquire(fleet)
    time.sleep(0.2)
    self.assertEqual(fleet.active_pools["img_A"], 4)
    fleet.release.set()
    thread.join()
    self.assertEqual(out, {"handle": ("handle", "img_A")})
    self.assertEqual(fleet.active_pools["img_A"], 3)

  def test_without_prewarmer_returns_handle(self):
    fleet = _BlockingAcquireFleet()
    fleet.release.set()
    self.assertIsNone(sandbox_utils._ACTIVE_PREWARM_ITERATOR)
    self.assertEqual(
        sandbox_utils.acquire_sandbox(fleet, self._TASK), ("handle", "img_A")
    )


class SandboxRetryOverrideTest(absltest.TestCase):

  def test_legacy_defaults(self):
    with mock.patch.dict(os.environ, {}, clear=True):
      self.assertEqual(
          sandbox_utils.SandboxFailFastConfig.from_env(),
          sandbox_utils.SandboxFailFastConfig(
              enabled=False, ready_timeout_s=None, acquire_retries=5
          ),
      )

  def test_overrides_without_fail_fast(self):
    env = {
        "DEEPSWE_ACQUIRE_RETRIES": "3",
        "DEEPSWE_SANDBOX_READY_TIMEOUT_S": "1200",
    }
    with mock.patch.dict(os.environ, env, clear=True):
      self.assertEqual(
          sandbox_utils.SandboxFailFastConfig.from_env(),
          sandbox_utils.SandboxFailFastConfig(
              enabled=False, ready_timeout_s=1200, acquire_retries=3
          ),
      )

  def test_invalid_overrides(self):
    for env in (
        {"DEEPSWE_ACQUIRE_RETRIES": "0"},
        {"DEEPSWE_SANDBOX_READY_TIMEOUT_S": "-1"},
    ):
      with mock.patch.dict(os.environ, env, clear=True):
        with self.assertRaises(ValueError):
          sandbox_utils.SandboxFailFastConfig.from_env()


if __name__ == "__main__":
  absltest.main()

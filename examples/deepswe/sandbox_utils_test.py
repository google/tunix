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
    # Fleet creates both pools (wait=False); only the current batch's pool
    # (img_A) gets the initial priming barrier (wait=True). The next batch's
    # pool warms in the background.
    self.assertCountEqual(
        fleet.warm_calls,
        [
            ("img_A", 8, False),
            ("img_A", 8, True),
            ("img_B", 8, False),
        ],
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
    # Start-up window = max_staleness + 1 = 2 undispatched batches past current.
    self.assertEqual(
        fleet.active_pools, {"img_A": 4, "img_B": 4, "img_C": 4}
    )

    # Batch 0 (p0 / img_A)
    self.assertEqual(next(iterator)["prompt"], "p0")
    self.assertEqual(
        fleet.active_pools, {"img_A": 4, "img_B": 4, "img_C": 4}
    )

    # Batch 1 (p1 / img_B): img_A retained in _previous_batches (1/2). The
    # window still holds img_C (>= lookahead_steps), so nothing new is warmed.
    self.assertEqual(next(iterator)["prompt"], "p1")
    self.assertNotIn("img_A", fleet.unwarm_calls)
    self.assertEqual(
        fleet.active_pools, {"img_A": 4, "img_B": 4, "img_C": 4}
    )

    # Batch 2 (p2 / img_C): both img_A and img_B retained in _previous_batches
    # (2/2); the window slides by one and img_D is prewarmed.
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

  def test_sliding_window_startup_then_one_batch_ahead(self):
    # max_staleness=2 mimics the orchestrator: batches 0..2 are pulled
    # back-to-back at start-up, then one batch per committed step.
    fleet = FakeFleet()
    dataset = [
        {"prompt": f"p{i}", "docker_image": f"img_{i}"} for i in range(8)
    ]
    iterator = sandbox_utils.PrewarmDatasetIterator(
        dataset,
        fleet=fleet,
        num_generations=4,
        batch_size=1,
        max_staleness=2,
    )
    self.assertEqual(iterator.initial_lookahead_batches, 3)
    self.assertEqual(iterator.lookahead_batches, 1)
    # Before the first pull: batches 0..2 (dispatched at start-up) and the
    # prefetch batch 3 all have pools; only batch 0 is waited on.
    self.assertEqual(set(fleet.active_pools), {f"img_{i}" for i in range(4)})
    self.assertCountEqual(
        [call for call in fleet.warm_calls if call[2]], [("img_0", 4, True)]
    )

    # Start-up dispatch of batches 0..2: no new pools, the window drains to
    # exactly the prefetch batch.
    for i in range(3):
      self.assertEqual(next(iterator)["prompt"], f"p{i}")
    self.assertEqual(set(fleet.active_pools), {f"img_{i}" for i in range(4)})
    self.assertLen(iterator._lookahead, 1)  # pylint: disable=protected-access

    # Each later dispatch (one committed step) slides the window by one: the
    # batch after the one just dispatched is prewarmed, never more.
    self.assertEqual(next(iterator)["prompt"], "p3")
    self.assertIn("img_4", fleet.active_pools)
    self.assertNotIn("img_5", fleet.active_pools)
    self.assertEqual(next(iterator)["prompt"], "p4")
    self.assertIn("img_5", fleet.active_pools)
    self.assertNotIn("img_6", fleet.active_pools)
    # Previous window (S + 1 = 3 batches: img_1..img_3) slid past img_0.
    self.assertNotIn("img_0", fleet.active_pools)
    self.assertIn("img_1", fleet.active_pools)
    # Pools created after start-up never block.
    self.assertCountEqual(
        [call for call in fleet.warm_calls if call[2]], [("img_0", 4, True)]
    )
    iterator.close()

  def test_lookahead_steps_overrides_when_larger(self):
    fleet = FakeFleet()
    dataset = [
        {"prompt": f"p{i}", "docker_image": f"img_{i}"} for i in range(5)
    ]
    iterator = sandbox_utils.PrewarmDatasetIterator(
        dataset,
        fleet=fleet,
        num_generations=2,
        batch_size=1,
        lookahead_steps=3,
        max_staleness=0,
    )
    self.assertEqual(iterator.lookahead_batches, 3)
    self.assertEqual(iterator.initial_lookahead_batches, 3)
    self.assertLen(fleet.active_pools, 4)
    iterator.close()

  def test_all_pools_created_before_readiness_barrier(self):
    # With one worker thread, the old single-phase flow created and waited on
    # each pool in turn, so lookahead pools waited behind the barrier. Now
    # every pool is created first and only then is the barrier run.
    fleet = FakeFleet()
    dataset = [
        {"prompt": f"p{i}", "docker_image": f"img_{i}"} for i in range(4)
    ]
    iterator = sandbox_utils.PrewarmDatasetIterator(
        dataset,
        fleet=fleet,
        num_generations=2,
        batch_size=1,
        max_staleness=2,
        max_workers=1,
    )
    self.assertEqual(
        fleet.warm_calls,
        [
            ("img_0", 2, False),
            ("img_1", 2, False),
            ("img_2", 2, False),
            ("img_3", 2, False),
            ("img_0", 2, True),
        ],
    )
    iterator.close()

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

  def test_async_initial_runs_in_background_until_wait_for_initial(self):
    warm_started = threading.Event()
    allow_warm_finish = threading.Event()
    fleet = FakeFleet()
    orig_warm = fleet.warm_image

    def _blocking_warm(
        image: str, replicas_override: int | None = None, wait: bool = False
    ) -> None:
      orig_warm(image, replicas_override=replicas_override, wait=wait)
      if wait:
        warm_started.set()
        allow_warm_finish.wait(timeout=5.0)

    fleet.warm_image = _blocking_warm
    dataset = [{"prompt": "p0", "docker_image": "img_A"}]
    iterator = sandbox_utils.PrewarmDatasetIterator(
        dataset,
        fleet=fleet,
        num_generations=2,
        batch_size=1,
        wait_initial=True,
        async_initial=True,
    )
    self.assertTrue(warm_started.wait(timeout=5.0))
    self.assertIsNotNone(iterator._initial_warm_thread)
    self.assertTrue(iterator._initial_warm_thread.is_alive())

    allow_warm_finish.set()
    iterator.wait_for_initial()
    self.assertIsNone(iterator._initial_warm_thread)
    self.assertEqual(
        fleet.warm_calls, [("img_A", 2, False), ("img_A", 2, True)]
    )
    self.assertEqual(next(iterator)["prompt"], "p0")

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

  def test_patch_agent_sandbox_rl_templates_sets_unmanaged_network_policy(self):
    class _FakeResources:

      def _template_manifest(self, image, template_name, template):
        del template
        return {
            "apiVersion": "extensions.agents.x-k8s.io/v1beta1",
            "kind": "SandboxTemplate",
            "metadata": {"name": template_name},
            "spec": {
                "podTemplate": {
                    "metadata": {"labels": {"sandbox": template_name}},
                    "spec": {"containers": [{"image": image}]},
                }
            },
        }

    fake_resources_mod = mock.MagicMock()
    fake_resources_mod.Resources = _FakeResources
    fake_asrl = mock.MagicMock()
    fake_asrl.resources = fake_resources_mod
    with mock.patch.dict(
        "sys.modules",
        {
            "agent_sandbox_rl": fake_asrl,
            "agent_sandbox_rl.resources": fake_resources_mod,
        },
    ):
      sandbox_utils.patch_agent_sandbox_rl_templates()
      res = _FakeResources()
      manifest = res._template_manifest("img:v1", "oh-test-123", None)
      self.assertEqual(
          manifest["spec"]["networkPolicyManagement"], "Unmanaged"
      )


_FAIL_FAST_ENV = {
    "FT_SANDBOX_FAIL_FAST": "true",
    "FT_SANDBOX_READY_TIMEOUT_S": "300",
    "FT_SANDBOX_ACQUIRE_RETRIES": "2",
}


class _FakeFleetError(Exception):
  """Stands in for agent_sandbox_rl.FleetError."""


class _FailingWarmFleet(FakeFleet):
  """Pool creation itself fails (e.g. name collision with another run)."""

  def warm_image(
      self, image: str, replicas_override: int | None = None, wait: bool = False
  ) -> None:
    raise _FakeFleetError(f"pool for {image} belongs to another run")


class _SlowReadyWarmFleet(FakeFleet):
  """Pool creation succeeds but the readiness barrier times out."""

  def warm_image(
      self, image: str, replicas_override: int | None = None, wait: bool = False
  ) -> None:
    super().warm_image(image, replicas_override=replicas_override, wait=wait)
    if wait:
      raise _FakeFleetError(
          f"warm pool for {image} did not become ready within 600s"
      )


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

  def test_prewarm_fail_fast_raises_on_pool_creation_error(self):
    dataset = [{"prompt": "p0", "docker_image": "img_A"}]
    with mock.patch.dict("sys.modules", {"agent_sandbox_rl": self._fake_sdk()}):
      with mock.patch.dict(os.environ, _FAIL_FAST_ENV):
        with self.assertRaisesRegex(_FakeFleetError, "belongs to another run"):
          sandbox_utils.PrewarmDatasetIterator(
              dataset, fleet=_FailingWarmFleet(), num_generations=2, batch_size=1
          )

  def test_prewarm_fail_fast_tolerates_readiness_timeout(self):
    # A pool that exists but is slow to fill must not end the run: the
    # orchestrator keeps the pool as active (so it is still scaled/unwarmed
    # later) and proceeds; rollout workers degrade per trajectory.
    dataset = [{"prompt": "p0", "docker_image": "img_A"}]
    fleet = _SlowReadyWarmFleet()
    with mock.patch.dict("sys.modules", {"agent_sandbox_rl": self._fake_sdk()}):
      with mock.patch.dict(os.environ, _FAIL_FAST_ENV):
        with self.assertLogs(level="WARNING") as logs:
          iterator = sandbox_utils.PrewarmDatasetIterator(
              dataset, fleet=fleet, num_generations=2, batch_size=1
          )
    self.assertEqual(
        fleet.warm_calls, [("img_A", 2, False), ("img_A", 2, True)]
    )
    self.assertEqual(fleet.active_pools, {"img_A": 2})
    self.assertEqual(iterator._active_replicas, {"img_A": 2})  # pylint: disable=protected-access
    self.assertTrue(
        any("not fully ready yet" in line for line in logs.output), logs.output
    )
    self.assertEqual(next(iterator), dataset[0])
    iterator.close()
    self.assertEqual(fleet.unwarm_calls, ["img_A"])

  def test_prewarm_off_logs_fleet_error(self):
    dataset = [{"prompt": "p0", "docker_image": "img_A"}]
    with mock.patch.dict(os.environ, {}, clear=True):
      with self.assertLogs(level="WARNING") as logs:
        iterator = sandbox_utils.PrewarmDatasetIterator(
            dataset, fleet=_FailingWarmFleet(), num_generations=2, batch_size=1
        )
  def test_lazy_initial_defers_iteration_and_priming(self):
    dataset_iterated = []

    def _gen():
      for i in range(5):
        dataset_iterated.append(i)
        yield {"prompt": f"p{i}", "docker_image": f"img_{i}"}

    fleet = FakeFleet()
    iterator = sandbox_utils.PrewarmDatasetIterator(
        _gen(), fleet=fleet, num_generations=2, batch_size=2, lazy_initial=True
    )
    self.assertEmpty(dataset_iterated)
    self.assertEmpty(fleet.warm_calls)
    self.assertFalse(iterator._initial_primed)

    iterator.prime_initial()
    self.assertTrue(iterator._initial_primed)
    self.assertNotEmpty(dataset_iterated)
    self.assertNotEmpty(fleet.warm_calls)
    self.assertEqual(next(iterator)["prompt"], "p0")
    iterator.close()


if __name__ == "__main__":
  absltest.main()

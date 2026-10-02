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
from unittest import mock
from absl.testing import absltest
import numpy as np
from examples.deepswe import sandbox_utils
from examples.deepswe import swe_env
from examples.deepswe import template


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

  def test_init_global_fleet_starts_initial_warmpools(self):
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
        mock_fleet.warm_images.assert_called_once_with(
            ["test-image:v1"],
            replicas_override=4,
            wait=False,
        )



class TemplateAndLifecycleTest(absltest.TestCase):

  def test_parse_cpu_to_millicores(self):
    self.assertEqual(template.parse_cpu_to_millicores("2"), 2000)
    self.assertEqual(template.parse_cpu_to_millicores("2.5"), 2500)
    self.assertEqual(template.parse_cpu_to_millicores("2500m"), 2500)
    self.assertEqual(template.parse_cpu_to_millicores("500m"), 500)
    self.assertEqual(template.parse_cpu_to_millicores("invalid"), 0)

  def test_parse_memory_to_bytes(self):
    self.assertEqual(template.parse_memory_to_bytes("4Gi"), 4 * 1024**3)
    self.assertEqual(template.parse_memory_to_bytes("512Mi"), 512 * 1024**2)
    self.assertEqual(template.parse_memory_to_bytes("2G"), 2 * 10**9)
    self.assertEqual(template.parse_memory_to_bytes("1000M"), 1000 * 10**6)
    self.assertEqual(template.parse_memory_to_bytes("1024"), 1024)
    self.assertEqual(template.parse_memory_to_bytes("invalid"), 0)

  def test_template_cpu_limits_exceed_requests(self):
    with mock.patch.dict(
        os.environ,
        {
            "SANDBOX_CPU": "4000m",
            "SANDBOX_CPU_LIMIT": "2",
            "SANDBOX_MEMORY": "16Gi",
            "SANDBOX_MEMORY_LIMIT": "8Gi",
            "SANDBOX_ACTIVE_DEADLINE_SECONDS": "3600",
        },
    ):
      tmpl = template.get_openhands_pod_template()
      self.assertEqual(tmpl.extra_pod_spec["activeDeadlineSeconds"], 3600)
      container = tmpl.extra_pod_spec["containers"][0]
      # Limits must be adjusted so limits >= requests
      self.assertEqual(container["resources"]["limits"]["cpu"], "4000m")
      self.assertEqual(tmpl.resources.cpu, "4000m")
      self.assertEqual(container["resources"]["limits"]["memory"], "16Gi")
      self.assertEqual(tmpl.resources.memory, "16Gi")

  def test_r2egym_template(self):
    with mock.patch.dict(
        os.environ,
        {
            "SANDBOX_CPU": "1000m",
            "SANDBOX_CPU_LIMIT": "2",
            "SANDBOX_ACTIVE_DEADLINE_SECONDS": "1800",
        },
    ):
      tmpl = template.get_r2egym_pod_template()
      self.assertEqual(tmpl.extra_pod_spec["activeDeadlineSeconds"], 1800)
      container = tmpl.extra_pod_spec["containers"][0]
      self.assertEqual(container["resources"]["limits"]["cpu"], "2")
      self.assertEqual(tmpl.resources.cpu, "1000m")

  def test_get_template(self):
    t_openhands = template.get_template("openhands")
    self.assertIsNotNone(t_openhands)
    self.assertIsNotNone(t_openhands.keepalive_command)
    t_r2egym = template.get_template("r2egym")
    self.assertIsNotNone(t_r2egym)
    t_sweagent = template.get_template("sweagent")
    self.assertIsNotNone(t_sweagent)

  def test_configure_claim_lifecycle(self):
    mock_handle = mock.MagicMock()
    mock_handle.claim_name = "claim-123"
    mock_cluster = mock.MagicMock()
    mock_cluster.namespace = "test-ns"
    mock_handle._cluster = mock_cluster

    swe_env.configure_claim_lifecycle(mock_handle, ttl_seconds=120)

    mock_cluster.custom_api.patch_namespaced_custom_object.assert_called_once_with(
        group="extensions.agents.x-k8s.io",
        version="v1beta1",
        namespace="test-ns",
        plural="sandboxclaims",
        name="claim-123",
        body={
            "spec": {
                "lifecycle": {
                    "shutdownPolicy": "Delete",
                    "ttlSecondsAfterFinished": 120,
                }
            }
        },
    )

  def test_cleanup_k8s_sandbox_handle(self):
    mock_handle = mock.MagicMock()
    mock_handle.claim_name = "claim-abc"
    mock_handle.sandbox_id = "sandbox-xyz"
    mock_cluster = mock.MagicMock()
    mock_cluster.namespace = "test-ns"
    mock_handle._cluster = mock_cluster

    swe_env.cleanup_k8s_sandbox_handle(mock_handle)

    mock_handle.sandbox.terminate.assert_called_once()
    mock_cluster.resources.delete_claim.assert_called_once_with("claim-abc")
    mock_cluster.resources.delete_sandbox.assert_called_once_with("sandbox-xyz")
    self.assertEqual(
        mock_cluster.custom_api.delete_namespaced_custom_object.call_count, 2
    )

  def test_swe_env_close_resilient_to_underlying_env_exception(self):
    mock_fleet = mock.MagicMock()
    mock_handle = mock.MagicMock()
    mock_handle.claim_name = "claim-fail"
    mock_handle.sandbox_id = "sandbox-fail"

    mock_env = mock.MagicMock()
    mock_env.close.side_effect = RuntimeError("Failed closing container")

    env = swe_env.SWEEnv(
        entry={"instance_id": "test__inst-1", "docker_image": "test:img"},
        use_agent_sandbox=False,
    )
    env.env = mock_env
    env.handle = mock_handle
    env.fleet = mock_fleet

    # close() must not raise and must execute fleet and handle cleanup
    env.close()

    mock_env.close.assert_called_once()
    mock_fleet.release.assert_called_once_with(mock_handle)
    mock_handle.sandbox.terminate.assert_called_once()
    self.assertIsNone(env.handle)
    self.assertIsNone(env.env)

  def test_swe_env_context_manager(self):
    mock_fleet = mock.MagicMock()
    mock_handle = mock.MagicMock()
    with swe_env.SWEEnv(
        entry={"instance_id": "test__inst-2", "docker_image": "test:img"},
        use_agent_sandbox=False,
    ) as env:
      env.handle = mock_handle
      env.fleet = mock_fleet

    mock_fleet.release.assert_called_once_with(mock_handle)
    self.assertIsNone(env.handle)


class TrajectoryCollectEngineLifecycleTest(absltest.TestCase):

  def test_collect_closes_env_on_reset_error(self):
    import asyncio
    import types

    class MockPackage(mock.MagicMock):
      __path__ = []
      __spec__ = mock.MagicMock()

    tunix_mod = types.ModuleType("tunix")
    tunix_mod.__path__ = ["/usr/local/google/home/atwigg/work/tunix/tunix"]
    mock_modules = {
        "tunix": tunix_mod,
        "flax": MockPackage(),
        "jax": MockPackage(),
        "jax.numpy": MockPackage(),
        "jax.lax": MockPackage(),
        "jax.sharding": MockPackage(),
        "optax": MockPackage(),
        "jaxtyping": MockPackage(),
        "etils": MockPackage(),
        "etils.epath": MockPackage(),
        "sentencepiece": MockPackage(),
        "orbax": MockPackage(),
        "orbax.checkpoint": MockPackage(),
        "metrax": MockPackage(),
        "metrax.logging": MockPackage(),
    }
    with mock.patch.dict("sys.modules", mock_modules):
      from tunix.rl.agentic.trajectory.trajectory_collect_engine import TrajectoryCollectEngine

      mock_agent = mock.MagicMock()
      mock_env = mock.MagicMock()
      mock_env.max_steps = 1
      mock_env.reset.side_effect = RuntimeError("reset failed!")
      engine = TrajectoryCollectEngine(
          mock_agent, mock_env, model_call=mock.MagicMock()
      )
      with self.assertRaises(RuntimeError):
        asyncio.run(engine.collect())
      mock_env.close.assert_called_once()

  def test_collect_closes_env_on_step_error(self):
    import asyncio
    import types

    class MockPackage(mock.MagicMock):
      __path__ = []
      __spec__ = mock.MagicMock()

    tunix_mod = types.ModuleType("tunix")
    tunix_mod.__path__ = ["/usr/local/google/home/atwigg/work/tunix/tunix"]
    mock_modules = {
        "tunix": tunix_mod,
        "flax": MockPackage(),
        "jax": MockPackage(),
        "jax.numpy": MockPackage(),
        "jax.lax": MockPackage(),
        "jax.sharding": MockPackage(),
        "optax": MockPackage(),
        "jaxtyping": MockPackage(),
        "etils": MockPackage(),
        "etils.epath": MockPackage(),
        "sentencepiece": MockPackage(),
        "orbax": MockPackage(),
        "orbax.checkpoint": MockPackage(),
        "metrax": MockPackage(),
        "metrax.logging": MockPackage(),
    }
    with mock.patch.dict("sys.modules", mock_modules):
      from tunix.rl.agentic.trajectory.trajectory_collect_engine import TrajectoryCollectEngine

      mock_agent = mock.MagicMock()
      mock_env = mock.MagicMock()
      mock_env.max_steps = 5
      mock_env.reset.return_value = ("initial_obs", {})
      mock_env.step.side_effect = RuntimeError("step failed!")
      engine = TrajectoryCollectEngine(
          mock_agent, mock_env, model_call=mock.MagicMock()
      )
      with self.assertRaises(RuntimeError):
        asyncio.run(engine.collect())
      mock_env.close.assert_called_once()


if __name__ == "__main__":
  absltest.main()

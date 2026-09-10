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

"""Unit tests for PrewarmDatasetIterator in swe_env."""

from absl.testing import absltest
import numpy as np
from examples.deepswe import swe_env


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


class PrewarmDatasetIteratorTest(absltest.TestCase):

  def test_two_queue_batch_prewarming(self):
    fleet = FakeFleet()
    dataset = [
        {"prompt": "p0", "docker_image": "img_A"},
        {"prompt": "p1", "docker_image": "img_A"},
        {"prompt": "p2", "docker_image": "img_B"},
        {"prompt": "p3", "docker_image": "img_B"},
        {"prompt": "p4", "docker_image": "img_C"},
    ]
    iterator = swe_env.PrewarmDatasetIterator(
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
    # Dict is unchanged: both queues are still maintained. Fleet untouched.
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
    # current_batch becomes next_batch (p2, p3 / img_B: 2).
    # next_batch refilled with p4 (img_C: 1).
    # _previous_batch_counts retains img_A: 2 so the in-flight batch stays warm.
    # Dict updated: {"img_A": 2, "img_B": 2, "img_C": 1}.
    # After dict updated, fleet interacted: img_C warmed to 4. img_A stays warm!
    item2 = next(iterator)
    self.assertEqual(item2["prompt"], "p2")
    self.assertNotIn("img_A", fleet.unwarm_calls)
    self.assertEqual(fleet.active_pools, {"img_A": 8, "img_B": 8, "img_C": 4})

    # 5. Step 3: pop p3 (img_B). current_batch has p3 popped. Fleet untouched.
    item3 = next(iterator)
    self.assertEqual(item3["prompt"], "p3")
    self.assertEqual(fleet.active_pools, {"img_A": 8, "img_B": 8, "img_C": 4})

    # 6. Step 4: current_batch was empty -> batch transition!
    # _previous_batch_counts becomes img_B: 2.
    # current_batch becomes next_batch (p4 / img_C: 1).
    # next_batch is empty (dataset exhausted).
    # Dict updated: {"img_B": 2, "img_C": 1}.
    # Fleet interacted: img_A is retired and unwarmed!
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
    iterator = swe_env.PrewarmDatasetIterator(
        dataset,
        fleet=fleet,
        num_generations=4,
        batch_size=2,
    )

    batch0 = next(iterator)
    # Ensure item was returned in its exact original batched list format
    self.assertIsInstance(batch0, list)
    self.assertLen(batch0, 2)
    self.assertEqual(batch0[0]["prompt"], "b0_0")

    batch1 = next(iterator)
    self.assertIsInstance(batch1, list)
    self.assertLen(batch1, 2)
    self.assertEqual(batch1[0]["prompt"], "b1_0")

    with self.assertRaises(StopIteration):
      next(iterator)

    # All pools should be unwarmed after close
    iterator.close()
    self.assertEqual(fleet.active_pools, {})

  def test_nested_metadata_image_extraction(self):
    fleet = FakeFleet()
    dataset = [
        {
            "prompt": "p0",
            "metadata": {
                "env_config": {
                    "entry": {"docker_image": "nested_docker_image"}
                }
            },
        }
    ]
    iterator = swe_env.PrewarmDatasetIterator(
        dataset, fleet=fleet, num_generations=2, batch_size=1
    )
    self.assertIn("nested_docker_image", fleet.active_pools)
    item = next(iterator)
    self.assertEqual(item["prompt"], "p0")

  def test_numpy_array_docker_image_extraction(self):
    fleet = FakeFleet()
    dataset = [
        {
            "prompt": np.array(["p0", "p1"]),
            "docker_image": np.array(["array_img_A", "array_img_A"]),
        }
    ]
    iterator = swe_env.PrewarmDatasetIterator(
        dataset, fleet=fleet, num_generations=3, batch_size=2
    )
    # 2 samples * 3 num_generations = 6 replicas
    self.assertEqual(fleet.active_pools.get("array_img_A"), 6)
    item = next(iterator)
    self.assertIsInstance(item["docker_image"], np.ndarray)

  def test_items_without_docker_image(self):
    fleet = FakeFleet()
    dataset = [
        {"prompt": "text_only_0"},
        {"prompt": "text_only_1"},
    ]
    iterator = swe_env.PrewarmDatasetIterator(
        dataset, fleet=fleet, num_generations=4, batch_size=2
    )
    # Should not attempt to warm any fake/default image on Kubernetes
    self.assertEqual(fleet.warm_calls, [])
    self.assertEqual(fleet.active_pools, {})

    item0 = next(iterator)
    self.assertEqual(item0["prompt"], "text_only_0")
    item1 = next(iterator)
    self.assertEqual(item1["prompt"], "text_only_1")
    with self.assertRaises(StopIteration):
      next(iterator)

  def test_max_warmpool_replicas_capping(self):
    fleet = FakeFleet()
    dataset = [
        {"prompt": f"p{i}", "docker_image": "capped_img"} for i in range(4)
    ]
    # batch_size=4, num_generations=8 -> uncapped demand = 32.
    # max_warmpool_replicas=10 -> should be capped at 10.
    iterator = swe_env.PrewarmDatasetIterator(
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
    iterator = swe_env.PrewarmDatasetIterator(
        [], fleet=fleet, num_generations=4, batch_size=2
    )
    with self.assertRaises(StopIteration):
      next(iterator)
    self.assertEqual(fleet.active_pools, {})


if __name__ == "__main__":
  absltest.main()

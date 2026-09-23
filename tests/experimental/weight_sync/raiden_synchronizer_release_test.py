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

"""CPU unit tests for RaidenSynchronizer.release_buffers() lifecycle and invariants."""

import gc
import sys
from unittest import mock

from absl.testing import absltest
import jax.numpy as jnp
from tunix.experimental.weight_sync import raiden_synchronizer


class _WeakFakeWeightSynchronizer:
  """Fake native WeightSynchronizer that does not pin Python array references."""

  def __init__(self, arrays, **kwargs):
    self.num_bound = len(arrays)
    self.kwargs = dict(kwargs)
    self.local_port = 12345
    self.listener_port = 23456
    self.num_shards = 2
    self.rebind_count = 0

  def bind_weights(self, arrays):
    self.num_bound = len(arrays)
    self.rebind_count += 1


class _FakeWsLib:

  def __init__(self):
    self.instances = []

  def WeightSynchronizer(self, arrays, **kwargs):  # pylint: disable=invalid-name
    inst = _WeakFakeWeightSynchronizer(arrays, **kwargs)
    self.instances.append(inst)
    return inst


class RaidenSynchronizerReleaseTest(absltest.TestCase):

  def test_release_buffers_collects_arrays_and_verifies_checksum_ordering(self):
    fake_lib = _FakeWsLib()
    with mock.patch.object(
        raiden_synchronizer, "_get_ws_lib", return_value=fake_lib
    ):
      sync = raiden_synchronizer.RaidenSynchronizer("trainer")
      w0 = jnp.array([[1.5, -2.5], [3.0, 4.0]], dtype=jnp.float32)
      w1 = jnp.array([5.0, -6.0, 7.0], dtype=jnp.float32)
      base_refs = (sys.getrefcount(w0), sys.getrefcount(w1))
      tree1 = {
          "layers_0": {"w": w0},
          "layers_1": {"w": w1},
      }
      sync.bind(tree1)
      del tree1
      gc.collect()

      n_before = len(sync.arrays)
      self.assertEqual(n_before, 2)
      # sync.arrays holds a live reference to both w0 and w1.
      self.assertGreater(sys.getrefcount(w0), base_refs[0])
      self.assertGreater(sys.getrefcount(w1), base_refs[1])

      # 1. Checksums BEFORE release_buffers() must observe all bound tensors.
      pre_cs = sync.checksums()
      self.assertEqual(pre_cs["__tensor_count__"], n_before)
      self.assertAlmostEqual(pre_cs["__grand_total__"], 29.0, places=5)

      # 2. release_buffers() drops staged arrays/names so refcounts return to baseline.
      released = sync.release_buffers()
      gc.collect()
      self.assertEqual(released, n_before)
      self.assertEmpty(sync.arrays)
      self.assertEmpty(sync.names)
      self.assertFalse(sync.bound)
      self.assertEqual((sys.getrefcount(w0), sys.getrefcount(w1)), base_refs)

      # 3. Checksums AFTER release_buffers() returns zeroed counts/totals.
      post_cs = sync.checksums()
      self.assertEqual(post_cs["__tensor_count__"], 0)
      self.assertEqual(post_cs["__grand_total__"], 0.0)

  def test_coordinator_abort_reentry_preserves_transport_state(self):
    fake_lib = _FakeWsLib()
    with mock.patch.object(
        raiden_synchronizer, "_get_ws_lib", return_value=fake_lib
    ):
      sync = raiden_synchronizer.RaidenSynchronizer("trainer")
      tree1 = {"layers_0": {"w": jnp.array([1.0, 2.0], dtype=jnp.float32)}}
      sync.bind(tree1)

      orig_sync = sync._sync
      sentinel_ips = ["10.0.0.1:12345"]
      sentinel_mesh = object()
      sentinel_shard_idx = jnp.array([0], dtype=jnp.int32)
      sync._ips = list(sentinel_ips)
      sync._ffi_mesh = sentinel_mesh
      sync._ffi_shard_idx = sentinel_shard_idx

      # Simulate weight_sync_coordinator.py:1429 finally block on abort.
      self.assertEqual(sync.release_buffers(), 1)
      self.assertIs(sync._sync, orig_sync)
      self.assertEqual(sync._ips, sentinel_ips)
      self.assertIs(sync._ffi_mesh, sentinel_mesh)
      self.assertIs(sync._ffi_shard_idx, sentinel_shard_idx)

      # Re-binding tree2 reuses existing transport (_sync.bind_weights) without re-init.
      tree2 = {
          "layers_0": {"w": jnp.array([10.0, 20.0], dtype=jnp.float32)},
          "layers_1": {"w": jnp.array([30.0, 40.0], dtype=jnp.float32)},
      }
      sync.bind(tree2)
      self.assertEqual(len(fake_lib.instances), 1)
      self.assertIs(sync._sync, orig_sync)
      self.assertEqual(orig_sync.rebind_count, 1)
      self.assertEqual(sync._ips, sentinel_ips)
      self.assertIs(sync._ffi_mesh, sentinel_mesh)
      self.assertIs(sync._ffi_shard_idx, sentinel_shard_idx)
      self.assertEqual(sync.checksums()["__grand_total__"], 100.0)


if __name__ == "__main__":
  absltest.main()

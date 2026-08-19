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

"""Tests for RaidenSynchronizer."""

import numpy as np

from absl.testing import absltest
import jax
import jax.numpy as jnp

from tunix.experimental.worker import raiden_synchronizer


class _FakeWeightSynchronizer:
  """Records ctor kwargs and calls in place of the native wheel object."""

  def __init__(self, arrays, **kwargs):
    self.arrays = list(arrays)
    self.kwargs = dict(kwargs)
    self.local_port = 12345
    self.listener_port = 23456
    self.num_shards = 2
    self.bound_with = None

  def d2h(self):
    pass

  def h2d(self):
    pass

  def bind_weights(self, arrays):
    self.bound_with = list(arrays)

  def get_metrics(self):
    return {"total_tiled_bytes": 0}


class _FakeWsLib:

  def __init__(self):
    self.instances = []

  def WeightSynchronizer(self, arrays, **kwargs):  # pylint: disable=invalid-name
    ws = _FakeWeightSynchronizer(arrays, **kwargs)
    self.instances.append(ws)
    return ws


class _NoDevicesArray:
  """Array-like leaf without .devices(); must be dropped by the census."""

  shape = (2,)
  dtype = np.dtype(np.float32)
  ndim = 1


class _ProxyDevice:
  platform = "proxy"


class _ProxyArray:
  """Array-like leaf living on a proxy device; must be dropped."""

  shape = (2,)
  dtype = np.dtype(np.float32)
  ndim = 1

  def devices(self):
    return {_ProxyDevice()}


class RaidenSynchronizerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._orig_ws_lib = raiden_synchronizer._ws_lib
    self.ws_lib = _FakeWsLib()
    raiden_synchronizer._ws_lib = self.ws_lib

  def tearDown(self):
    raiden_synchronizer._ws_lib = self._orig_ws_lib
    super().tearDown()

  def _state(self):
    return {
        "w1": jnp.ones((2, 4), jnp.float32),
        "w2": jnp.arange(3, dtype=jnp.float32),
    }

  def test_ctor_passes_safety_kwargs(self):
    sync = raiden_synchronizer.RaidenSynchronizer("rollout", self._state())
    ws = self.ws_lib.instances[0]
    self.assertIs(ws.kwargs["unsafe_skip_buffer_lock"], True)
    self.assertIs(ws.kwargs["auto_h2d"], False)
    self.assertEqual(ws.kwargs["parallelism"], 4)
    self.assertEqual(ws.kwargs["local_port"], 0)
    self.assertEqual(ws.kwargs["listener_port"], 0)
    self.assertEqual(ws.arrays, sync.arrays)

  def test_census_drops_leaves_without_devices(self):
    names, arrays = raiden_synchronizer._filter_bindable(
        ["good", "bad"], [jnp.ones((2,)), _NoDevicesArray()]
    )
    self.assertEqual(names, ["good"])
    self.assertLen(arrays, 1)

  def test_census_drops_non_cpu_tpu_platforms(self):
    names, _ = raiden_synchronizer._filter_bindable(
        ["good", "proxy"], [jnp.ones((2,)), _ProxyArray()]
    )
    self.assertEqual(names, ["good"])

  def test_census_drops_extended_dtype_key_arrays(self):
    key = jax.random.key(0)
    names, _ = raiden_synchronizer._filter_bindable(
        ["good", "key"], [jnp.ones((2,)), key]
    )
    self.assertEqual(names, ["good"])

  def test_checksums_grand_total(self):
    sync = raiden_synchronizer.RaidenSynchronizer("rollout", self._state())
    sums = sync.checksums()
    self.assertEqual(sums["__grand_total__"], 8.0 + 3.0)
    self.assertLen(sums, 3)  # two sampled tensors + grand total

  def test_work_unit_metadata_shards_and_addresses(self):
    sync = raiden_synchronizer.RaidenSynchronizer(
        "rollout", self._state(), bind_ip="1.2.3.4"
    )
    md = sync.work_unit_metadata()
    self.assertEqual(md.unit.job_name, "rollout")
    self.assertEqual(md.shards, ("1.2.3.4:12345",) * 2)
    self.assertEqual(md.control_plane_rpc_address, "1.2.3.4:23456")
    self.assertLen(md.variables, 2)
    self.assertEqual([v.layer_idx for v in md.variables], [0, 1])
    for v in md.variables:
      # Single-device arrays: every dim replicated, single-axis specs only.
      self.assertTrue(all(m == 1 for m in v.mesh_shape))
      self.assertTrue(all("," not in s for s in v.sharding_spec))

  def test_work_unit_metadata_without_wheel(self):
    raiden_synchronizer._ws_lib = None
    sync = raiden_synchronizer.RaidenSynchronizer("rollout", self._state())
    self.assertFalse(sync.active)
    md = sync.work_unit_metadata()
    self.assertEqual(md.shards, ())

  def test_rebind_filters_and_rebinds(self):
    sync = raiden_synchronizer.RaidenSynchronizer("trainer", self._state())
    ws = self.ws_lib.instances[0]
    sync.rebind({"w1": jnp.zeros((2, 2)), "poison": _ProxyArray()})
    self.assertEqual(sync.names, ["['w1']"])
    self.assertLen(ws.bound_with, 1)


if __name__ == "__main__":
  absltest.main()

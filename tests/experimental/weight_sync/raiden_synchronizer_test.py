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

from unittest import mock

import numpy as np

from absl.testing import absltest
import jax
import jax.numpy as jnp

from tunix.experimental.weight_sync import raiden_synchronizer


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


class _FakeSocket:

  def __init__(self, result):
    self._result = result
    self.closed = False

  def connect(self, probe):
    del probe
    if isinstance(self._result, Exception):
      raise self._result

  def getsockname(self):
    return (self._result, 0)

  def close(self):
    self.closed = True


class LocalIpTest(absltest.TestCase):

  def _patch_sockets(self, *results):
    made = []

    def factory(family, kind):
      del family, kind
      sock = _FakeSocket(results[len(made)])
      made.append(sock)
      return sock

    patcher = mock.patch.object(
        raiden_synchronizer.socket, "socket", factory
    )
    patcher.start()
    self.addCleanup(patcher.stop)
    return made

  def test_returns_the_ipv4_address(self):
    made = self._patch_sockets("10.0.0.7")
    self.assertEqual(raiden_synchronizer.local_ip(), "10.0.0.7")
    self.assertTrue(made[0].closed)

  def test_brackets_an_ipv6_address(self):
    self._patch_sockets(OSError("no ipv4"), "fe80::1")
    self.assertEqual(raiden_synchronizer.local_ip(), "[fe80::1]")

  def test_falls_back_to_localhost(self):
    made = self._patch_sockets(OSError("no ipv4"), OSError("no ipv6"))
    self.assertEqual(raiden_synchronizer.local_ip(), "localhost")
    self.assertLen(made, 2)


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

  def test_census_keeps_proxy_platforms_for_ffi(self):
    names, _ = raiden_synchronizer._filter_bindable(
        ["good", "proxy"],
        [jnp.ones((2,)), _ProxyArray()],
        allow_proxy=True,
    )
    self.assertEqual(names, ["good", "proxy"])

  def test_census_keeps_bfloat16_weights(self):
    names, _ = raiden_synchronizer._filter_bindable(
        ["w"], [jnp.ones((2,), jnp.bfloat16)]
    )
    self.assertEqual(names, ["w"])

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

  def test_tensor_metadata_clamps_overlong_spec(self):
    class _WideSpecSharding:
      spec = ("tp", "fsdp", "extra")

    class _Arr:
      shape = (8,)
      ndim = 1
      dtype = np.dtype(np.float32)
      sharding = _WideSpecSharding()

    md = raiden_synchronizer._tensor_metadata("w", _Arr(), 0)
    self.assertEqual(md.sharding_spec, ("tp",))
    self.assertLen(md.mesh_shape, 1)

  def test_work_unit_metadata_reports_the_array_mesh(self):
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("data",))
    sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec()
    )
    arr = jax.device_put(jnp.ones((2, 4), jnp.float32), sharding)
    sync = raiden_synchronizer.RaidenSynchronizer("trainer", {"w": arr})
    md = sync.work_unit_metadata()
    self.assertEqual(md.mesh_axes, ("data",))
    self.assertEqual(md.mesh_shape, (1,))

  def test_work_unit_metadata_falls_back_without_a_mesh(self):
    sync = raiden_synchronizer.RaidenSynchronizer("trainer", self._state())
    md = sync.work_unit_metadata()
    self.assertEqual(md.mesh_axes, ("fsdp",))
    self.assertEqual(md.mesh_shape, (1,))

  def test_work_unit_metadata_without_wheel(self):
    raiden_synchronizer._ws_lib = None
    sync = raiden_synchronizer.RaidenSynchronizer("rollout", self._state())
    self.assertFalse(sync.active)
    md = sync.work_unit_metadata()
    self.assertEqual(md.shards, ())

  def test_bind_filters_on_rebind(self):
    sync = raiden_synchronizer.RaidenSynchronizer("trainer", self._state())
    ws = self.ws_lib.instances[0]
    sync.bind({"w1": jnp.zeros((2, 2)), "poison": _ProxyArray()})
    self.assertEqual(sync.names, ["['w1']"])
    self.assertLen(ws.bound_with, 1)

  def test_bind_creates_once_then_rebinds(self):
    sync = raiden_synchronizer.RaidenSynchronizer("trainer", self._state())
    ws = self.ws_lib.instances[0]
    sync.bind({"w1": jnp.zeros((2, 2))})
    self.assertLen(self.ws_lib.instances, 1)
    self.assertLen(ws.bound_with, 1)

  def test_deferred_bind(self):
    sync = raiden_synchronizer.RaidenSynchronizer("trainer")
    self.assertFalse(sync.bound)
    self.assertEmpty(self.ws_lib.instances)
    sync.bind(self._state())
    self.assertTrue(sync.bound)
    self.assertLen(self.ws_lib.instances, 1)

  def test_calls_before_bind_raise(self):
    sync = raiden_synchronizer.RaidenSynchronizer("trainer")
    with self.assertRaisesRegex(RuntimeError, "bind"):
      sync.d2h()
    with self.assertRaisesRegex(RuntimeError, "bind"):
      sync.h2d()

  def test_worker_index_stamps_replica_id(self):
    sync = raiden_synchronizer.RaidenSynchronizer(
        "rollout", self._state(), worker_index=3
    )
    self.assertEqual(sync.work_unit_metadata().unit.job_replica_id, "3")
    base = raiden_synchronizer.RaidenSynchronizer("rollout", self._state())
    self.assertEqual(base.work_unit_metadata().unit.job_replica_id, "")

  def test_defaults_to_ffi_under_proxy(self):
    with mock.patch.dict("os.environ", {"JAX_PLATFORMS": "proxy,cpu"}):
      sync = raiden_synchronizer.RaidenSynchronizer("trainer")
    self.assertTrue(sync.use_ffi)

  def test_defaults_to_non_ffi_without_proxy(self):
    with mock.patch.dict("os.environ", {}, clear=True):
      sync = raiden_synchronizer.RaidenSynchronizer("trainer")
    self.assertFalse(sync.use_ffi)

  def test_ffi_source_routes_through_d2h_init(self):
    sync = raiden_synchronizer.RaidenSynchronizer("trainer", use_ffi=True)
    with mock.patch.object(
        sync, "_init_ffi_transport", autospec=True
    ) as init_ffi:
      sync.bind(self._state())
      init_ffi.assert_not_called()
      sync.d2h()
    init_ffi.assert_called_once_with(execute_d2h=True)

  def test_ffi_destination_init_runs_at_bind(self):
    sync = raiden_synchronizer.RaidenSynchronizer(
        "rollout", use_ffi=True, auto_h2d=True
    )
    with mock.patch.object(
        sync, "_init_ffi_transport", autospec=True
    ) as init_ffi:
      sync.bind(self._state())
    init_ffi.assert_called_once_with(execute_d2h=False)

  def test_ffi_destination_h2d_routes_through_multi_h2d(self):
    sync = raiden_synchronizer.RaidenSynchronizer(
        "rollout", use_ffi=True, auto_h2d=True
    )
    sync.names, sync.arrays = raiden_synchronizer.flatten_weights(self._state())
    with mock.patch.object(sync, "_ffi_h2d", autospec=True) as ffi_h2d:
      sync.h2d()
    ffi_h2d.assert_called_once_with()

  def test_ffi_compute_on_compat_accepts_out_memory_spaces(self):
    compute_on_mod = raiden_synchronizer.jax._src.compute_on
    original = compute_on_mod.compute_on
    self.addCleanup(setattr, compute_on_mod, "compute_on", original)

    raiden_synchronizer._ensure_ffi_compute_on_compat()

    decorator = compute_on_mod.compute_on(
        compute_type="device_host",
        out_memory_spaces=jax.memory.Space.Device,
    )
    self.assertTrue(callable(decorator))


if __name__ == "__main__":
  absltest.main()
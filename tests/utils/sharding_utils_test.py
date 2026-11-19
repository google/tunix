# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for Flax sharding setup and device placement."""

from unittest import mock

from absl.testing import absltest
import flax
import jax
import numpy as np
from tunix.utils import sharding_utils


class ShardingUtilsTest(absltest.TestCase):

  def test_setup_sharding_environment_disables_flax_always_shard_variable(self):
    if hasattr(flax.config, "flax_always_shard_variable"):
      self.addCleanup(
          flax.config.update,
          "flax_always_shard_variable",
          flax.config.flax_always_shard_variable,
      )
      flax.config.update("flax_always_shard_variable", True)
      self.assertTrue(flax.config.flax_always_shard_variable)
    sharding_utils.setup_sharding_environment()
    if hasattr(flax.config, "flax_always_shard_variable"):
      self.assertFalse(flax.config.flax_always_shard_variable)

  def test_place_and_shard_arrays_on_local_and_multihost_targets(self):
    device = jax.local_devices()[0]
    arr = np.arange(4, dtype=np.float32)
    placed = sharding_utils.place_array_on_target(arr, device)
    np.testing.assert_array_equal(np.asarray(placed), arr)
    self.assertIs(
        sharding_utils.place_array_on_target(placed, placed.sharding), placed
    )

    non_addressable_target = mock.create_autospec(
        jax.sharding.NamedSharding, instance=True
    )
    non_addressable_target.is_fully_addressable = False
    non_addressable_arr = mock.create_autospec(jax.Array, instance=True)
    non_addressable_arr.sharding = placed.sharding
    non_addressable_arr.is_fully_addressable = False
    with self.assertRaisesRegex(
        ValueError, "non-fully-addressable multi-host"
    ):
      sharding_utils.place_array_on_target(
          non_addressable_arr, non_addressable_target
      )

    with mock.patch.object(
        jax,
        "make_array_from_callback",
        autospec=True,
        side_effect=lambda shape, target, cb: jax.numpy.asarray(
            cb(tuple(slice(0, d) for d in shape))
        ),
    ) as mock_from_cb:
      mh_batch = sharding_utils.place_arrays_on_targets(
          [arr], [non_addressable_target]
      )
      self.assertLen(mh_batch, 1)
      mock_from_cb.assert_called_once()
      np.testing.assert_array_equal(np.asarray(mh_batch[0]), arr)

    batch = sharding_utils.place_arrays_on_targets([arr], [device])
    self.assertLen(batch, 1)
    np.testing.assert_array_equal(np.asarray(batch[0]), arr)
    self.assertEqual(sharding_utils.place_arrays_on_targets([], []), [])

    sharded = sharding_utils.shard(arr, (None,))
    np.testing.assert_array_equal(np.asarray(sharded), arr)

    mesh = jax.sharding.Mesh(np.array([device]), ("data",))
    fake_tpu = mock.create_autospec(jax.Device, instance=True)
    fake_tpu.platform = "tpu"
    with (
        mesh,
        mock.patch.object(
            jax, "devices", autospec=True, return_value=[fake_tpu]
        ),
        mock.patch.object(
            sharding_utils,
            "place_array_on_target",
            autospec=True,
            wraps=sharding_utils.place_array_on_target,
        ) as mock_place,
        mock.patch.object(
            jax.lax,
            "with_sharding_constraint",
            autospec=True,
            side_effect=lambda x, _: jax.numpy.asarray(x),
        ) as mock_constraint,
    ):
      eager_out = sharding_utils.shard(arr, ("data",), eager=True)
      lazy_out = sharding_utils.shard(arr, ("data",), eager=False)
      mock_place.assert_called_once()
      mock_constraint.assert_called_once()
      np.testing.assert_array_equal(np.asarray(eager_out), arr)
      np.testing.assert_array_equal(np.asarray(lazy_out), arr)

    with self.assertRaises(ValueError):
      sharding_utils.place_arrays_on_targets([arr], [])


if __name__ == "__main__":
  absltest.main()

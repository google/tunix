# Copyright 2025 Google LLC
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

from absl.testing import absltest
from unittest import mock
import jax.numpy as jnp
import numpy as np
from tunix.sft import utils


class UtilsTest(absltest.TestCase):

    def test_pathways_hbm_usage_skips_pinned_host_arrays(self):
        class _FakeDevice:
            def __init__(self, name):
                self.name = name

            def __repr__(self):
                return self.name

        class _FakeData:
            def __init__(self, device, nbytes):
                self.device = device
                self.nbytes = nbytes

        class _FakeShard:
            def __init__(self, data):
                self.data = data

        class _FakeSharding:
            def __init__(self, memory_kind, device_set):
                self.memory_kind = memory_kind
                self.device_set = device_set

        class _FakeArray:
            def __init__(self, sharding, shards):
                self.sharding = sharding
                self.addressable_shards = shards

            def is_deleted(self):
                return False

        device0 = _FakeDevice("device0")
        device1 = _FakeDevice("device1")
        shared_device_buffer = _FakeData(device0, 10)
        live_arrays = [
                _FakeArray(
                        _FakeSharding("device", {device0, device1}),
                        [_FakeShard(shared_device_buffer)],
                ),
                _FakeArray(
                        _FakeSharding("pinned_host", {device0, device1}),
                        [_FakeShard(_FakeData(device0, 1000))],
                ),
                _FakeArray(
                        _FakeSharding("device", {device0, device1}),
                        [_FakeShard(shared_device_buffer)],
                ),
        ]

        with mock.patch.object(utils.jax, "live_arrays", return_value=live_arrays):
            stats = utils._pathways_hbm_usage_gb([device0, device1])

        self.assertEqual(stats, [(10, None), (0, None)])

    def test_live_array_usage_tracks_pinned_host_separately(self):
        class _FakeDevice:
            def __init__(self, name):
                self.name = name

            def __repr__(self):
                return self.name

        class _FakeData:
            def __init__(self, device, nbytes):
                self.device = device
                self.nbytes = nbytes

        class _FakeShard:
            def __init__(self, data):
                self.data = data

        class _FakeSharding:
            def __init__(self, memory_kind, device_set):
                self.memory_kind = memory_kind
                self.device_set = device_set

        class _FakeArray:
            def __init__(self, sharding, shards):
                self.sharding = sharding
                self.addressable_shards = shards

            def is_deleted(self):
                return False

        device0 = _FakeDevice("device0")
        device1 = _FakeDevice("device1")
        shared_device_buffer = _FakeData(device0, 10)
        shared_pinned_buffer = _FakeData(device1, 20)
        live_arrays = [
                _FakeArray(
                        _FakeSharding("device", {device0, device1}),
                        [_FakeShard(shared_device_buffer)],
                ),
                _FakeArray(
                        _FakeSharding("pinned_host", {device0, device1}),
                        [_FakeShard(shared_pinned_buffer)],
                ),
                _FakeArray(
                        _FakeSharding("pinned_host", {device0, device1}),
                        [_FakeShard(shared_pinned_buffer)],
                ),
        ]

        with mock.patch.object(utils.jax, "live_arrays", return_value=live_arrays):
            stats = utils._live_array_usage_by_memory_kind([device0, device1])

        self.assertEqual(stats["device"], [(10, None), (0, None)])
        self.assertEqual(stats["pinned_host"], [(0, None), (20, None)])

    def test_show_hbm_usage_logs_device_and_pinned_host_separately(self):
        device0 = mock.Mock()
        device0.__str__ = mock.Mock(return_value="device0")
        device0.__repr__ = mock.Mock(return_value="device0")

        with mock.patch.object(utils.google_utils, "pathways_available", return_value=True), \
                mock.patch.object(utils.jax, "devices", return_value=[device0]), \
                mock.patch.object(
                        utils,
                        "_live_array_usage_by_memory_kind",
                        return_value={
                                "device": [(1024, None)],
                                "pinned_host": [(2048, None)],
                        },
                ), \
                mock.patch.object(utils.gc, "collect"), \
                mock.patch.object(utils.logging, "info") as mock_info:
            utils.show_hbm_usage("test")

        logged_lines = [call.args[0] % call.args[1:] for call in mock_info.call_args_list]
        self.assertIn(
                "test - Using Pathways compatible HBM stats collector", logged_lines
        )
        self.assertIn("Device HBM: using 1.0 KiB on device0", logged_lines)
        self.assertIn("Pinned host: using 2.0 KiB on device0", logged_lines)

    def test_make_causal_attn_mask(self):
        input_mask = jnp.array([
            [True, True, True, True],
            [True, True, True, False],
            [False, True, True, False],
        ])
        attn_mask = utils.make_causal_attn_mask(input_mask)
        expected_value = jnp.array([
            [
                [True, False, False, False],
                [True, True, False, False],
                [True, True, True, False],
                [True, True, True, True],
            ],
            [
                [True, False, False, False],
                [True, True, False, False],
                [True, True, True, False],
                [True, True, True, False],
            ],
            [
                [False, False, False, False],
                [False, True, False, False],
                [False, True, True, False],
                [False, True, True, False],
            ],
        ])
        np.testing.assert_allclose(attn_mask, expected_value)

    def test_build_positions_from_mask(self):
        input_mask = jnp.array(
            [[1, 1, 1, 1], [0, 1, 1, 1], [1, 1, 1, 0], [0, 1, 1, 0]]
        )
        positions = utils.build_positions_from_mask(input_mask)
        expected_value = jnp.array([
            [0, 1, 2, 3],
            [0, 0, 1, 2],
            [0, 1, 2, 2],
            [0, 0, 1, 1],
        ])
        np.testing.assert_array_equal(positions, expected_value)


if __name__ == '__main__':
  absltest.main()

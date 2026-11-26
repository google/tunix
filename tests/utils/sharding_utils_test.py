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

"""Tests for multi-host gather and broadcast helpers in sharding_utils."""

import json
from unittest import mock

from absl.testing import absltest
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np
from tunix.utils import sharding_utils


def _sharded_mock() -> mock.Mock:
  arr = mock.Mock(spec=jax.Array)
  arr.is_fully_addressable = False
  return arr


class MultihostGatherTest(absltest.TestCase):

  def test_gather_tensor_to_host_single_controller_and_multihost_modes(self):
    arr_jax = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    arr_np = np.arange(4, dtype=np.int32)
    with mock.patch.object(
        jax, 'process_count', return_value=1
    ), mock.patch.object(multihost_utils, 'process_allgather') as mock_ag:
      np.testing.assert_array_equal(
          sharding_utils.gather_tensor_to_host(arr_jax), np.asarray(arr_jax)
      )
      np.testing.assert_array_equal(
          sharding_utils.gather_tensor_to_host(arr_np), arr_np
      )
      mock_ag.assert_not_called()

    sharded = _sharded_mock()
    gathered_val = np.full((2, 3), 7.0, dtype=np.float32)
    with mock.patch.object(
        jax, 'process_count', return_value=2
    ), mock.patch.object(
        multihost_utils, 'process_allgather', return_value=gathered_val
    ) as mock_ag, mock.patch.object(
        jax, 'device_get', wraps=jax.device_get
    ) as mock_dg:
      out_sharded = sharding_utils.gather_tensor_to_host(sharded)
      np.testing.assert_array_equal(out_sharded, gathered_val)
      mock_ag.assert_called_once_with(sharded, tiled=True)
      mock_dg.assert_not_called()

      out_local_jax = sharding_utils.gather_tensor_to_host(arr_jax)
      out_local_np = sharding_utils.gather_tensor_to_host(arr_np)
      np.testing.assert_array_equal(out_local_jax, np.asarray(arr_jax))
      np.testing.assert_array_equal(out_local_np, arr_np)
      mock_ag.assert_called_once()

  def test_gather_tensors_to_host0_materializes_only_on_host0(self):
    sharded = _sharded_mock()
    local = np.arange(3, dtype=np.float32)
    gathered = np.ones((2,), np.float32)
    tensors = {'z': sharded, 'a': local}
    with mock.patch.object(
        jax, 'process_count', return_value=2
    ), mock.patch.object(
        multihost_utils, 'process_allgather', return_value=gathered
    ) as mock_ag:
      host0 = sharding_utils.gather_tensors_to_host0(tensors, is_host0=True)
      self.assertEqual(list(host0), ['a', 'z'])
      np.testing.assert_array_equal(host0['a'], local)
      np.testing.assert_array_equal(host0['z'], gathered)

      mock_ag.reset_mock()
      with mock.patch.object(jax, 'device_get') as mock_dg:
        worker = sharding_utils.gather_tensors_to_host0(tensors, is_host0=False)
      self.assertEqual(worker, {})
      mock_ag.assert_called_once_with(sharded, tiled=True)
      mock_dg.assert_not_called()


class BroadcastHost0JsonTest(absltest.TestCase):

  def test_single_process_round_trips_without_collectives(self):
    payload = {'paths': ['a', 'b'], 'checksums': {'w': 1.5}}
    with mock.patch.object(multihost_utils, 'broadcast_one_to_all') as mock_bc:
      self.assertEqual(sharding_utils.broadcast_host0_json(payload), payload)
    mock_bc.assert_not_called()

  def test_multiprocess_worker_receives_host0_payload(self):
    payload = {'error': 'OSError: disk full'}
    buf = np.frombuffer(json.dumps(payload).encode('utf-8'), dtype=np.uint8)
    replies = iter([np.int32(buf.size), buf])
    sources = []

    def fake_broadcast(x, is_source):
      sources.append((is_source, np.asarray(x).shape))
      return next(replies)

    with mock.patch.object(
        jax, 'process_count', return_value=2
    ), mock.patch.object(
        jax, 'process_index', return_value=1
    ), mock.patch.object(
        multihost_utils, 'broadcast_one_to_all', side_effect=fake_broadcast
    ):
      self.assertEqual(sharding_utils.broadcast_host0_json({'x': 1}), payload)
    self.assertEqual(sources, [(False, ()), (False, (buf.size,))])


if __name__ == '__main__':
  absltest.main()

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

"""Unit tests for RaidenDestinationWeightSyncMixin."""

import unittest
from unittest import mock

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
from tunix.common import configs
from tunix.dist_rl.rollout import raiden_weight_sync_mixin
from tunix.dist_rl.rollout import sampler as base_sampler_lib


class _BareAdapter(raiden_weight_sync_mixin.RaidenDestinationWeightSyncMixin):

  def __init__(self):
    self.server_id = "bare"
    self.config = configs.RolloutConfig()
    self.enable_raiden = False
    self.raiden_sync_delegate = None


class _MockAdapter(raiden_weight_sync_mixin.RaidenDestinationWeightSyncMixin):

  def __init__(self, enable_raiden=True, partial_rollout=False):
    self.server_id = "test_server"
    self.config = configs.RolloutConfig(partial_rollout=partial_rollout)
    self.enable_raiden = enable_raiden
    self.raiden_sync_delegate = mock.AsyncMock()
    self.raiden_sync_delegate.is_bounded = mock.MagicMock(return_value=True)
    self._transformer_state = {"param": 1}
    self.update_params = mock.MagicMock()
    self.pause = mock.AsyncMock(return_value=True)
    self.resume = mock.AsyncMock(return_value=True)

  @property
  def transformer_state(self):
    return self._transformer_state


class _SelfHostedAdapter(
    raiden_weight_sync_mixin.RaidenDestinationWeightSyncMixin
):
  """Holds its own weights and has no `sampler`, like `VanillaSampler`."""

  def __init__(self, enable_raiden=True):
    self.server_id = "self_hosted"
    self.config = configs.RolloutConfig()
    self.enable_raiden = enable_raiden
    self.raiden_sync_delegate = mock.AsyncMock()
    self.raiden_sync_delegate.is_bounded = mock.MagicMock(return_value=False)
    self._transformer_state = {"param": 2}
    self.update_params = mock.MagicMock()

  @property
  def transformer_state(self):
    return self._transformer_state


class RaidenDestinationWeightSyncMixinTest(unittest.IsolatedAsyncioTestCase):

  async def test_get_weight_sync_metadata_raiden(self):
    adapter = _MockAdapter(enable_raiden=True)
    adapter.raiden_sync_delegate.get_weight_sync_metadata.return_value = [
        "meta"
    ]
    res = await adapter.get_weight_sync_metadata()
    self.assertEqual(res, ["meta"])

  async def test_get_weight_sync_metadata_disabled_raises(self):
    adapter = _MockAdapter(enable_raiden=False)
    with self.assertRaises(NotImplementedError):
      await adapter.get_weight_sync_metadata()

  async def test_bind_weight_sync_already_bounded(self):
    adapter = _MockAdapter(enable_raiden=True)
    adapter.raiden_sync_delegate.is_bounded.return_value = True
    res = await adapter.bind_weight_sync()
    self.assertTrue(res)

  async def test_bind_weight_sync_unbounded_calls_delegate(self):
    adapter = _MockAdapter(enable_raiden=True)
    adapter.raiden_sync_delegate.is_bounded.return_value = False
    adapter.raiden_sync_delegate.bind_weight_sync.return_value = True
    res = await adapter.bind_weight_sync()
    self.assertTrue(res)
    adapter.raiden_sync_delegate.bind_weight_sync.assert_awaited_once()

  async def test_pre_weight_sync_delegates(self):
    adapter = _MockAdapter(enable_raiden=True)
    adapter.raiden_sync_delegate.pre_weight_sync.return_value = True
    res = await adapter.pre_weight_sync(base_sampler_lib.WeightSyncRequest())
    self.assertTrue(res)
    adapter.raiden_sync_delegate.pre_weight_sync.assert_awaited_once()

  async def test_weight_sync_fallback_calls_update_params(self):
    adapter = _MockAdapter(enable_raiden=False)
    sync_req = mock.MagicMock(weights={"w": 1})
    res = await adapter.weight_sync(sync_request=sync_req)
    self.assertIsNone(res)
    adapter.update_params.assert_called_once_with({"w": 1})

  async def test_abort_weight_sync_delegates(self):
    adapter = _MockAdapter(enable_raiden=True)
    adapter.raiden_sync_delegate.abort_weight_sync = mock.AsyncMock(
        return_value=True
    )
    res = await adapter.abort_weight_sync(base_sampler_lib.WeightSyncRequest())
    self.assertTrue(res)
    adapter.raiden_sync_delegate.abort_weight_sync.assert_awaited_once()

  async def test_pre_weight_sync_waits_and_clears_cache_when_not_partial(self):
    adapter = _MockAdapter(enable_raiden=True)
    req = base_sampler_lib.WeightSyncRequest()
    await adapter.pre_weight_sync(req)
    adapter.pause.assert_awaited_once_with(mode="wait", clear_cache=True)
    adapter.raiden_sync_delegate.pre_weight_sync.assert_awaited_once_with(
        sync_request=req
    )

  async def test_pre_weight_sync_keeps_cache_when_partial(self):
    adapter = _MockAdapter(enable_raiden=True, partial_rollout=True)
    req = base_sampler_lib.WeightSyncRequest()
    await adapter.pre_weight_sync(req)
    adapter.pause.assert_awaited_once_with(mode="keep", clear_cache=False)
    adapter.raiden_sync_delegate.pre_weight_sync.assert_awaited_once_with(
        sync_request=req
    )

  async def test_pre_weight_sync_unbound_raises_before_pausing(self):
    adapter = _MockAdapter(enable_raiden=True)
    adapter.raiden_sync_delegate.is_bounded.return_value = False
    with self.assertRaises(RuntimeError):
      await adapter.pre_weight_sync(base_sampler_lib.WeightSyncRequest())
    adapter.pause.assert_not_awaited()

  async def test_post_weight_sync_resumes_when_delegate_raises(self):
    adapter = _MockAdapter(enable_raiden=True)
    adapter.raiden_sync_delegate.post_weight_sync.side_effect = RuntimeError(
        "post failed"
    )
    with self.assertRaisesRegex(RuntimeError, "post failed"):
      await adapter.post_weight_sync(base_sampler_lib.WeightSyncRequest())
    adapter.resume.assert_awaited_once()

  async def test_abort_weight_sync_resumes_when_delegate_raises(self):
    adapter = _MockAdapter(enable_raiden=True)
    adapter.raiden_sync_delegate.abort_weight_sync.side_effect = RuntimeError(
        "abort failed"
    )
    with self.assertRaisesRegex(RuntimeError, "abort failed"):
      await adapter.abort_weight_sync(base_sampler_lib.WeightSyncRequest())
    adapter.resume.assert_awaited_once()

  async def test_get_weight_sync_status_reports_delegate_round(self):
    adapter = _MockAdapter(enable_raiden=True)
    adapter.raiden_sync_delegate.get_weight_sync_status = mock.MagicMock(
        return_value={"phase": "prepared"}
    )
    self.assertEqual(
        await adapter.get_weight_sync_status(), {"phase": "prepared"}
    )

  async def test_get_weight_sync_status_is_empty_without_raiden(self):
    adapter = _MockAdapter(enable_raiden=False)
    self.assertEqual(await adapter.get_weight_sync_status(), {})

  async def test_bind_weight_sync_binds_own_state_without_a_sampler(self):
    adapter = _SelfHostedAdapter(enable_raiden=True)
    await adapter.bind_weight_sync()
    adapter.raiden_sync_delegate.bind_weight_sync.assert_awaited_once_with(
        sync_request=None, state={"param": 2}, sampler=adapter
    )

  async def test_weight_sync_fallback_updates_own_params_without_a_sampler(
      self,
  ):
    adapter = _SelfHostedAdapter(enable_raiden=False)
    sync_req = mock.MagicMock(weights={"w": 2})
    self.assertIsNone(await adapter.weight_sync(sync_request=sync_req))
    adapter.update_params.assert_called_once_with({"w": 2})

  def test_get_target_state_maps_nnx_variables_to_shape_dtype_structs(self):
    adapter = _MockAdapter(enable_raiden=False)
    adapter._transformer_state = {
        "w": nnx.Param(jnp.ones((2, 3), dtype=jnp.float32))
    }
    target = adapter.get_target_state()
    self.assertIsInstance(target["w"], nnx.Param)
    self.assertEqual(
        target["w"].value,
        jax.ShapeDtypeStruct(shape=(2, 3), dtype=jnp.float32),
    )

  def test_default_stubs_raise_not_implemented_error(self):
    adapter = _BareAdapter()
    with self.assertRaises(NotImplementedError):
      _ = adapter.transformer_state
    with self.assertRaises(NotImplementedError):
      adapter.update_params({"w": 1})
    with self.assertRaises(NotImplementedError):
      adapter.delete_cache()
    with self.assertRaises(NotImplementedError):
      adapter.reset_prefix_cache()
    with self.assertRaises(NotImplementedError):
      adapter.reinitialize_cache()
    with self.assertRaises(NotImplementedError):
      adapter.refresh_state_leaves()


if __name__ == "__main__":
  absltest.main()

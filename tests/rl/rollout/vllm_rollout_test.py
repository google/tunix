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

"""Tests for the partial-rollout weight sync hooks of VllmRollout."""

# The tests inspect the mocked sampler that VllmRollout wraps.
# pylint: disable=protected-access

from unittest import mock

from absl.testing import absltest
from tunix.generate import mappings
from tunix.generate import vllm_sampler
from tunix.rl.rollout import base_rollout
from tunix.rl.rollout import vllm_rollout


class VllmRolloutPartialRolloutTest(absltest.TestCase):

  def _rollout(self, **config_kwargs):
    """Returns a VllmRollout on a mocked sampler, and that sampler."""
    config_kwargs.setdefault("rollout_vllm_server_mode", True)
    config = base_rollout.RolloutConfig(
        rollout_mapping_config=mock.Mock(spec=mappings.MappingConfig),
        **config_kwargs,
    )
    patcher = mock.patch.object(vllm_sampler, "VllmSampler")
    self.addCleanup(patcher.stop)
    patcher.start()
    with mock.patch("flax.nnx.state", return_value=mock.Mock()):
      rollout = vllm_rollout.VllmRollout(
          model=mock.Mock(),
          tokenizer=mock.Mock(),
          cache_config_or_size=1024,
          mesh=mock.Mock(),
          rollout_config=config,
      )
    return rollout, rollout._sampler

  def test_keep_policy_keeps_kv_across_the_sync(self):
    rollout, sampler = self._rollout(partial_rollout=True)
    self.assertTrue(rollout.supports_partial_rollout_sync())
    rollout.pause_for_weight_sync()
    sampler.pause.assert_called_once_with("keep", clear_cache=False)
    rollout.update_params({"w": 1})
    sampler.update_params.assert_called_once_with(
        {"w": 1}, None, preserve_active_kv_cache=True
    )
    rollout.resume_after_weight_sync()
    sampler.resume.assert_called_once()

  def test_recompute_policy_preempts_and_frees_kv(self):
    rollout, sampler = self._rollout(
        partial_rollout=True,
        partial_rollout_kv_policy="recompute",
        rollout_vllm_async_scheduling=False,
    )
    rollout.pause_for_weight_sync()
    sampler.pause.assert_called_once_with("keep", clear_cache=True)
    rollout.update_params({"w": 1})
    sampler.update_params.assert_called_once_with(
        {"w": 1}, None, preserve_active_kv_cache=False
    )

  def test_recompute_policy_rejects_async_scheduling(self):
    with self.assertRaisesRegex(ValueError, "async scheduling"):
      self._rollout(
          partial_rollout=True,
          partial_rollout_kv_policy="recompute",
          rollout_vllm_async_scheduling=True,
      )

  def test_partial_rollout_needs_server_mode(self):
    with self.assertRaisesRegex(ValueError, "rollout_vllm_server_mode"):
      self._rollout(partial_rollout=True, rollout_vllm_server_mode=False)

  def test_without_partial_rollout(self):
    rollout, sampler = self._rollout()
    self.assertFalse(rollout.supports_partial_rollout_sync())
    with self.assertRaisesRegex(RuntimeError, "requires partial_rollout"):
      rollout.pause_for_weight_sync()
    rollout.update_params({"w": 1})
    sampler.update_params.assert_called_once_with(
        {"w": 1}, None, preserve_active_kv_cache=False
    )


if __name__ == "__main__":
  absltest.main()

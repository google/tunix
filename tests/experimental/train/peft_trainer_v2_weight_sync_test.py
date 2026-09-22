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

"""Tests for PeftTrainer V2 weight sync staging."""

import os
import types
from unittest import mock

from absl.testing import absltest
from flax import nnx
import jax.numpy as jnp

from tunix.experimental.train import peft_trainer_v2
from tunix.experimental.weight_sync import raiden_synchronizer
from tunix.generate import mappings as mappings_lib


class _FakeSynchronizer:

  def __init__(self, job_name, state=None, **kwargs):
    self.job_name = job_name
    self.state = state
    self.is_proxy = "proxy" in os.environ.get("JAX_PLATFORMS", "")
    self.kwargs = kwargs
    self.bound_state = None
    self.d2h_calls = 0
    self.bound = False

  def bind(self, state):
    self.bound_state = state
    self.bound = True

  def d2h(self):
    self.d2h_calls += 1

  def work_unit_metadata(self):
    return {"unit": self.job_name}

  def metrics(self):
    return {}

  def checksums(self):
    return {}


class _TinyModel(nnx.Module):

  def __init__(self):
    self.w = nnx.Param(jnp.ones((2, 2)))


class WeightSyncStagingTest(absltest.TestCase):

  def _fake_trainer(self):
    return types.SimpleNamespace(
        model=_TinyModel(),
        config=types.SimpleNamespace(),
        _target_state=None,
        _sampler_type="inprocess_vllm",
        _rollout_tp_size=1,
        _weight_sync_worker=None,
    )

  def test_prepare_stages_and_returns_metadata(self):
    fake = self._fake_trainer()
    with mock.patch.object(raiden_synchronizer, "RaidenSynchronizer", _FakeSynchronizer):
      md = peft_trainer_v2.PeftTrainer.prepare_weight_sync(fake)
    worker = fake._weight_sync_worker
    self.assertEqual(md, [{"unit": "trainer"}])
    self.assertEqual(worker.d2h_calls, 1)
    self.assertIsNotNone(worker.bound_state)

  def test_prepare_reuses_the_worker(self):
    fake = self._fake_trainer()
    with mock.patch.object(raiden_synchronizer, "RaidenSynchronizer", _FakeSynchronizer):
      peft_trainer_v2.PeftTrainer.prepare_weight_sync(fake)
      first = fake._weight_sync_worker
      peft_trainer_v2.PeftTrainer.prepare_weight_sync(fake)
    self.assertIs(fake._weight_sync_worker, first)

  def test_prepare_uses_ffi_under_proxy(self):
    fake = self._fake_trainer()
    with mock.patch.object(raiden_synchronizer, "RaidenSynchronizer", _FakeSynchronizer):
      with mock.patch.dict(os.environ, {"JAX_PLATFORMS": "proxy,cpu"}):
        peft_trainer_v2.PeftTrainer.prepare_weight_sync(fake)
    self.assertTrue(fake._weight_sync_worker.is_proxy)

  def test_prepare_preprocesses_and_passes_rollout_tp_to_mapping(self):
    fake = self._fake_trainer()
    fake._target_state = mock.sentinel.target_state
    fake._rollout_tp_size = 4
    fake.config = types.SimpleNamespace(
        mapping_config=mappings_lib.MappingConfig(
            to_hf_mappings={"w": ("weight", (None, None))},
            to_hf_hook_fns={"w": mock.sentinel.hook},
            to_hf_transpose_keys={"w": (1, 0)},
            preprocess_src_state=mock.Mock(
                return_value=mock.sentinel.preprocessed_state
            ),
        ),
    )

    with mock.patch.object(
        raiden_synchronizer, "RaidenSynchronizer", _FakeSynchronizer
    ), mock.patch(
        "tunix.generate.utils.transfer_state_with_mappings",
        return_value=mock.sentinel.converted_state,
    ) as transfer:
      peft_trainer_v2.PeftTrainer.prepare_weight_sync(fake)

    fake.config.mapping_config.preprocess_src_state.assert_called_once()
    transfer.assert_called_once_with(
        src_state=mock.sentinel.preprocessed_state,
        dst_state=mock.sentinel.target_state,
        key_mappings={"w": ("weight", (None, None))},
        key_mapping_hook_fns={"w": mock.sentinel.hook},
        transpose_keys={"w": (1, 0)},
        reshard_fn=None,
        rollout_engine="vllm_jax",
        tp_size=4,
    )
    self.assertIs(
        fake._weight_sync_worker.bound_state, mock.sentinel.converted_state
    )

  def test_release_without_prepare_is_a_no_op(self):
    fake = self._fake_trainer()
    self.assertTrue(peft_trainer_v2.PeftTrainer.release_weight_sync(fake))


if __name__ == "__main__":
  absltest.main()

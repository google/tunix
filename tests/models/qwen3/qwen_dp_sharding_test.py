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

"""Tests for Qwen3 replicated-DP sharding metadata."""

from absl.testing import absltest

from tunix.models.qwen3 import model


class QwenDPShardingTest(absltest.TestCase):

  def test_parameters_are_replicated_over_dp(self):
    config = model.ShardingConfig.get_data_parallel_sharding()
    parameter_specs = (
        config.emb_vd,
        config.emb_dv,
        config.q_weight_dnh,
        config.kv_weight_dnh,
        config.o_weight_nhd,
        config.ffw_weight_df,
        config.ffw_weight_fd,
        config.rms_norm_weight,
        config.score_weight_d1,
        config.exp_weight_edf,
        config.exp_weight_efd,
    )
    self.assertNotIn('dp', repr(parameter_specs))
    self.assertEqual(config.act_btd[0], 'dp')
    self.assertEqual(config.act_btf[0], 'dp')
    self.assertEqual(config.act_btnh[0], 'dp')

  def test_rejects_ambiguous_data_axis(self):
    for axis in ('', 'fsdp', 'tp'):
      with self.subTest(axis=axis):
        with self.assertRaises(ValueError):
          model.ShardingConfig.get_data_parallel_sharding(axis)


if __name__ == '__main__':
  absltest.main()

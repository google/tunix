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

from unittest import mock
from absl.testing import absltest
from tunix.utils import maxtext_utils


class MaxTextUtilsTest(absltest.TestCase):

  def test_build_maxtext_config_args_and_single_init(self):
    mock_pyconfig = mock.MagicMock()
    mock_engine = mock.MagicMock()
    mock_mutils = mock.MagicMock()

    mock_cfg = mock.MagicMock()
    mock_cfg.base_moe_mlp_dim = 2048
    mock_cfg.raw_data_dict = {}
    mock_pyconfig.initialize.return_value = mock_cfg
    mock_pyconfig.__file__ = "/fake/maxtext/configs/pyconfig.py"

    with mock.patch.object(
        maxtext_utils,
        "maxtext_modules",
        return_value=(mock_pyconfig, mock_engine, mock_mutils),
    ), mock.patch("os.path.exists", return_value=True):
      cfg = maxtext_utils.build_maxtext_config(
          model_name="gemma2-9b",
          worker_id="worker-0",
          train_micro_batch_size=8,
          mesh_fsdp=2,
          mesh_tp=4,
          mesh_expert=1,
          num_devices=8,
          base_num_kv_heads=8,
          rollout_mesh_tp=4,
          prefuse_moe_weights=True,
          use_weight_converter=False,
      )

      # Ensure pyconfig.initialize was called exactly ONCE
      mock_pyconfig.initialize.assert_called_once()
      argv = mock_pyconfig.initialize.call_args[0][0]

      self.assertIn("model_name=gemma2-9b", argv)
      self.assertIn("base_num_kv_heads=8", argv)
      self.assertIn("ici_tensor_parallelism=4", argv)
      self.assertIn("ici_fsdp_parallelism=2", argv)
      self.assertIn("prefuse_moe_weights=True", argv)
      self.assertIn("use_weight_converter=False", argv)
      self.assertIn("rollout_tensor_parallelism=4", argv)
      self.assertEqual(cfg, mock_cfg)

  def test_build_maxtext_config_auto_padded_moe_mlp_dim(self):
    mock_pyconfig = mock.MagicMock()
    mock_cfg = mock.MagicMock()
    mock_cfg.base_moe_mlp_dim = 2048
    mock_cfg.raw_data_dict = {}
    mock_pyconfig.initialize.return_value = mock_cfg
    mock_pyconfig.__file__ = "/fake/maxtext/configs/pyconfig.py"

    mock_compute = mock.MagicMock(return_value=2304)

    with mock.patch.object(
        maxtext_utils,
        "maxtext_modules",
        return_value=(mock_pyconfig, mock.MagicMock(), mock.MagicMock()),
    ), mock.patch("os.path.exists", return_value=True), mock.patch.dict(
        "sys.modules",
        {"maxtext.integration.vllm.moe_padding": mock.MagicMock(
            compute_padded_moe_mlp_dim=mock_compute
        )},
    ):
      cfg = maxtext_utils.build_maxtext_config(
          model_name="moe-test",
          rollout_mesh_tp=4,
          padded_moe_mlp_dim=0,
      )
      mock_compute.assert_called_once_with(2048, 4)
      self.assertEqual(cfg.padded_base_moe_mlp_dim, 2304)
      self.assertEqual(cfg.raw_data_dict["padded_base_moe_mlp_dim"], 2304)

  def test_build_maxtext_config_batch_size_divisibility(self):
    with self.assertRaises(ValueError):
      maxtext_utils.build_maxtext_config(
          model_name="gemma2-9b",
          train_micro_batch_size=5,
          mesh_fsdp=2,
      )


if __name__ == "__main__":
  absltest.main()

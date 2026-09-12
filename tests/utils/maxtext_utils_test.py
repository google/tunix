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

try:
  from absl.testing import absltest
except ImportError:
  import unittest as absltest

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
    mock_cfg.padded_base_moe_mlp_dim = 2304
    mock_pyconfig.initialize.return_value = mock_cfg
    mock_pyconfig.__file__ = "/fake/maxtext/configs/pyconfig.py"

    mock_compute = mock.MagicMock(return_value=2304)

    with mock.patch.object(
        maxtext_utils,
        "maxtext_modules",
        return_value=(mock_pyconfig, mock.MagicMock(), mock.MagicMock()),
    ), mock.patch("os.path.exists", return_value=True), mock.patch(
        "builtins.open",
        mock.mock_open(read_data="base_moe_mlp_dim: 2048\n"),
    ), mock.patch.dict(
        "sys.modules",
        {
            "maxtext.integration.vllm.convert_utils": mock.MagicMock(
                compute_padded_moe_mlp_dim=mock_compute
            )
        },
    ):
      cfg = maxtext_utils.build_maxtext_config(
          model_name="moe-test",
          moe_mlp_tp_size=4,
          padded_moe_mlp_dim=0,
      )
      mock_compute.assert_called_once_with(2048, 4)
      argv = mock_pyconfig.initialize.call_args[0][0]
      self.assertIn("padded_base_moe_mlp_dim=2304", argv)

  def test_derived_quantities_matrix(self):
    cases = [
        # (name, tp, ep, dp, attn_dp, exp_kv_tp, exp_moe_tp, exp_pad, exp_kv_heads)
        ("c1_baseline", 2, 1, 2, 1, 2, 2, 512, 2),
        ("c2_kv_replicated_ep2", 2, 2, 1, 1, 4, 2, 512, 4),
        ("c3_moe_doubled_tp4", 4, 1, 1, 1, 4, 4, 1024, 4),
        ("c4_attn_dp2_t6_fix", 2, 1, 1, 2, 2, 4, 1024, 2),
        ("c5_pure_dp", 1, 1, 4, 1, 1, 1, 512, 2),
        ("c6_large_scale", 4, 2, 1, 2, 8, 8, 2048, 8),
    ]
    mock_pyconfig = mock.MagicMock()
    mock_engine = mock.MagicMock()
    mock_mutils = mock.MagicMock()
    mock_pyconfig.__file__ = "/fake/maxtext/configs/pyconfig.py"

    def mock_compute_padded_moe_mlp_dim(
        hidden_size, moe_mlp_tp_size, num_lanes=128
    ):
      min_required = 2 * num_lanes * moe_mlp_tp_size
      if (hidden_size // moe_mlp_tp_size) % (2 * num_lanes) != 0:
        return (
            (max(hidden_size, min_required) + min_required - 1) // min_required
        ) * min_required
      return hidden_size

    for (
        name,
        tp,
        ep,
        dp,
        attn_dp,
        exp_kv_tp,
        exp_moe_tp,
        exp_pad,
        exp_kv_heads,
    ) in cases:
      mock_pyconfig.reset_mock()
      mock_cfg = mock.MagicMock()
      mock_pyconfig.initialize.return_value = mock_cfg

      kv_tp_size = tp * ep
      moe_mlp_tp_size = tp * attn_dp

      self.assertEqual(kv_tp_size, exp_kv_tp, f"{name}: kv_tp_size mismatch")
      self.assertEqual(
          moe_mlp_tp_size, exp_moe_tp, f"{name}: moe_mlp_tp_size mismatch"
      )

      with mock.patch.object(
          maxtext_utils,
          "maxtext_modules",
          return_value=(mock_pyconfig, mock_engine, mock_mutils),
      ), mock.patch("os.path.exists", return_value=True), mock.patch(
          "builtins.open",
          mock.mock_open(
              read_data="base_moe_mlp_dim: 512\nbase_num_kv_heads: 2\n"
          ),
      ), mock.patch.dict(
          "sys.modules",
          {
              "maxtext.integration.vllm.convert_utils": mock.MagicMock(
                  compute_padded_moe_mlp_dim=mock_compute_padded_moe_mlp_dim
              )
          },
      ):
        maxtext_utils.build_maxtext_config(
            model_name="qwen3-moe",
            worker_id="worker-0",
            train_micro_batch_size=8,
            mesh_fsdp=1,
            mesh_tp=tp,
            mesh_expert=ep,
            num_devices=8,
            base_num_kv_heads=2,
            kv_tp_size=kv_tp_size,
            moe_mlp_tp_size=moe_mlp_tp_size,
        )
        mock_pyconfig.initialize.assert_called_once()
        argv = mock_pyconfig.initialize.call_args[0][0]
        self.assertIn(
            f"padded_base_moe_mlp_dim={exp_pad}",
            argv,
            f"{name}: padded_base_moe_mlp_dim mismatch in argv",
        )
        self.assertIn(
            f"base_num_kv_heads={exp_kv_heads}",
            argv,
            f"{name}: base_num_kv_heads mismatch in argv",
        )

  def test_indivisible_kv_heads_fails_fast(self):
    # tp=3, ep=1, base_num_kv_heads=2 -> 3 % 2 != 0 -> must raise ValueError
    mock_pyconfig = mock.MagicMock()
    mock_pyconfig.__file__ = "/fake/maxtext/configs/pyconfig.py"
    with mock.patch.object(
        maxtext_utils,
        "maxtext_modules",
        return_value=(mock_pyconfig, mock.MagicMock(), mock.MagicMock()),
    ), mock.patch("os.path.exists", return_value=True):
      with self.assertRaisesRegex(ValueError, "must be cleanly divisible"):
        maxtext_utils.build_maxtext_config(
            model_name="qwen3-test",
            base_num_kv_heads=2,
            kv_tp_size=3,
            moe_mlp_tp_size=1,
        )

  def test_build_maxtext_config_batch_size_divisibility(self):
    mock_pyconfig = mock.MagicMock()
    with mock.patch.object(
        maxtext_utils,
        "maxtext_modules",
        return_value=(mock_pyconfig, mock.MagicMock(), mock.MagicMock()),
    ):
      with self.assertRaises(ValueError):
        maxtext_utils.build_maxtext_config(
            model_name="gemma2-9b",
            train_micro_batch_size=5,
            mesh_fsdp=2,
        )

  def test_build_maxtext_config_range_validations(self):
    mock_pyconfig = mock.MagicMock()
    with mock.patch.object(
        maxtext_utils,
        "maxtext_modules",
        return_value=(mock_pyconfig, mock.MagicMock(), mock.MagicMock()),
    ):
      with self.assertRaisesRegex(
          ValueError, "padded_moe_mlp_dim must be non-negative"
      ):
        maxtext_utils.build_maxtext_config("gemma2-9b", padded_moe_mlp_dim=-1)

      with self.assertRaisesRegex(
          ValueError, "base_num_kv_heads must be non-negative"
      ):
        maxtext_utils.build_maxtext_config("gemma2-9b", base_num_kv_heads=-2)

      with self.assertRaisesRegex(
          ValueError, "kv_tp_size must be non-negative"
      ):
        maxtext_utils.build_maxtext_config("gemma2-9b", kv_tp_size=-1)

      with self.assertRaisesRegex(
          ValueError, "moe_mlp_tp_size must be non-negative"
      ):
        maxtext_utils.build_maxtext_config("gemma2-9b", moe_mlp_tp_size=-1)

      with self.assertRaisesRegex(
          ValueError, "rollout_mesh_tp must be non-negative"
      ):
        maxtext_utils.build_maxtext_config("gemma2-9b", rollout_mesh_tp=-4)

  def test_build_maxtext_config_auto_padding_failure_raises_runtime_error(self):
    mock_pyconfig = mock.MagicMock()
    mock_cfg = mock.MagicMock()
    mock_cfg.base_moe_mlp_dim = 2048
    mock_pyconfig.initialize.return_value = mock_cfg
    mock_pyconfig.__file__ = "/fake/maxtext/configs/pyconfig.py"

    with mock.patch.object(
        maxtext_utils,
        "maxtext_modules",
        return_value=(mock_pyconfig, mock.MagicMock(), mock.MagicMock()),
    ), mock.patch("os.path.exists", return_value=True), mock.patch(
        "builtins.open",
        mock.mock_open(read_data="base_moe_mlp_dim: 2048\n"),
    ), mock.patch.dict(
        "sys.modules",
        {
            "maxtext.integration.vllm.convert_utils": mock.MagicMock(
                compute_padded_moe_mlp_dim=mock.MagicMock(
                    side_effect=ValueError("Padding failure")
                )
            )
        },
    ):
      with self.assertRaisesRegex(
          RuntimeError, "Failed to auto-compute padded_base_moe_mlp_dim"
      ):
        maxtext_utils.build_maxtext_config(
            model_name="moe-test",
            moe_mlp_tp_size=4,
            padded_moe_mlp_dim=0,
        )

  def test_kv_tp_size_missing_base_heads_raises_value_error(self):
    mock_pyconfig = mock.MagicMock()
    mock_pyconfig.__file__ = "/fake/maxtext/configs/pyconfig.py"
    with mock.patch.object(
        maxtext_utils,
        "maxtext_modules",
        return_value=(mock_pyconfig, mock.MagicMock(), mock.MagicMock()),
    ), mock.patch(
        "os.path.exists", side_effect=lambda p: str(p).endswith("base.yml")
    ):
      with self.assertRaisesRegex(ValueError, "requires base_num_kv_heads > 0"):
        maxtext_utils.build_maxtext_config(
            model_name="qwen3-test",
            base_num_kv_heads=0,
            kv_tp_size=4,
        )

  def test_single_file_load_for_kv_heads_and_moe_dim(self):
    mock_pyconfig = mock.MagicMock()
    mock_cfg = mock.MagicMock()
    mock_pyconfig.initialize.return_value = mock_cfg
    mock_pyconfig.__file__ = "/fake/maxtext/configs/pyconfig.py"
    mock_compute = mock.MagicMock(return_value=2048)

    mock_file = mock.mock_open(
        read_data="base_num_kv_heads: 4\nbase_moe_mlp_dim: 1024\n"
    )
    with mock.patch.object(
        maxtext_utils,
        "maxtext_modules",
        return_value=(mock_pyconfig, mock.MagicMock(), mock.MagicMock()),
    ), mock.patch("os.path.exists", return_value=True), mock.patch(
        "builtins.open", mock_file
    ), mock.patch.dict(
        "sys.modules",
        {
            "maxtext.integration.vllm.convert_utils": mock.MagicMock(
                compute_padded_moe_mlp_dim=mock_compute
            )
        },
    ), self.assertLogs(
        level="INFO"
    ) as logs:
      maxtext_utils.build_maxtext_config(
          model_name="moe-test",
          rollout_mesh_tp=4,  # Overrides kv_tp_size=4 and moe_mlp_tp_size=4
      )
      # Assert the YAML was opened only once
      mock_file.assert_called_once()
      # Assert overrides were logged
      log_text = "\n".join(logs.output)
      self.assertIn(
          "Overriding kv_tp_size from 0 to rollout_mesh_tp=4", log_text
      )
      self.assertIn(
          "Overriding moe_mlp_tp_size from 0 to rollout_mesh_tp=4", log_text
      )

  def test_auto_padding_import_error_logs_warning(self):
    mock_pyconfig = mock.MagicMock()
    mock_cfg = mock.MagicMock()
    mock_pyconfig.initialize.return_value = mock_cfg
    mock_pyconfig.__file__ = "/fake/maxtext/configs/pyconfig.py"

    with mock.patch.object(
        maxtext_utils,
        "maxtext_modules",
        return_value=(mock_pyconfig, mock.MagicMock(), mock.MagicMock()),
    ), mock.patch("os.path.exists", return_value=True), mock.patch(
        "builtins.open",
        mock.mock_open(
            read_data="base_moe_mlp_dim: 2048\nbase_num_kv_heads: 2\n"
        ),
    ), mock.patch.dict(
        "sys.modules",
        {"maxtext.integration.vllm.convert_utils": None},
    ), self.assertLogs(
        level="WARNING"
    ) as logs:
      maxtext_utils.build_maxtext_config(
          model_name="moe-test",
          base_num_kv_heads=2,
          moe_mlp_tp_size=4,
      )
      log_text = "\n".join(logs.output)
      self.assertIn("Could not import compute_padded_moe_mlp_dim", log_text)
      self.assertIn("Skipping automatic MoE dimension padding", log_text)

  def _build_config_argv(self, **kwargs):
    mock_pyconfig = mock.MagicMock()
    mock_pyconfig.initialize.return_value = mock.MagicMock()
    mock_pyconfig.__file__ = "/fake/maxtext/configs/pyconfig.py"

    with mock.patch.object(
        maxtext_utils,
        "maxtext_modules",
        return_value=(mock_pyconfig, mock.MagicMock(), mock.MagicMock()),
    ), mock.patch("os.path.exists", return_value=True):
      maxtext_utils.build_maxtext_config(model_name="gemma2-9b", **kwargs)
    return mock_pyconfig.initialize.call_args[0][0]

  def test_checkpoint_save_interval_zero_disables_saving(self):
    # MaxText config validation requires enable_checkpointing=True when
    # load_parameters_path is set. Saving is disabled post-initialization.
    argv_with_load = self._build_config_argv(
        load_parameters_path="gs://bucket/ckpt",
        checkpointing_options=mock.MagicMock(
            save_interval_steps=0, max_to_keep=10
        ),
    )
    self.assertIn("enable_checkpointing=True", argv_with_load)
    self.assertIn("load_parameters_path=gs://bucket/ckpt", argv_with_load)

    # Without load_parameters_path, enable_checkpointing=False is passed directly.
    argv_no_load = self._build_config_argv(
        checkpointing_options=mock.MagicMock(
            save_interval_steps=0, max_to_keep=10
        ),
    )
    self.assertIn("enable_checkpointing=False", argv_no_load)

  def test_checkpoint_save_interval_positive_enables_saving(self):
    argv = self._build_config_argv(
        checkpointing_options=mock.MagicMock(
            save_interval_steps=5, max_to_keep=3
        ),
    )
    self.assertIn("enable_checkpointing=True", argv)
    self.assertIn("checkpoint_period=5", argv)
    self.assertIn("max_num_checkpoints_to_keep=3", argv)

  def test_checkpoint_save_interval_negative_raises(self):
    with self.assertRaisesRegex(ValueError, "must be non-negative"):
      self._build_config_argv(
          checkpointing_options=mock.MagicMock(
              save_interval_steps=-1, max_to_keep=3
          ),
      )


if __name__ == "__main__":
  absltest.main()

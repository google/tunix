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

import os
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
      self.assertIn("attention=dot_product", argv)
      self.assertEqual(cfg, mock_cfg)

  def test_build_maxtext_config_attention_and_remat_and_lr(self):
    mock_pyconfig = mock.MagicMock()
    mock_engine = mock.MagicMock()
    mock_mutils = mock.MagicMock()

    mock_cfg = mock.MagicMock()
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
          attention="flash",
          remat_policy="full",
          learning_rate_final_fraction=1.0,
      )
      mock_pyconfig.initialize.assert_called_once()
      argv = mock_pyconfig.initialize.call_args[0][0]
      self.assertIn("attention=flash", argv)
      self.assertIn("remat_policy=full", argv)
      self.assertIn("learning_rate_final_fraction=1.0", argv)
      self.assertEqual(cfg, mock_cfg)

  def test_build_maxtext_config_skip_step_on_spikes_and_nan(self):
    mock_pyconfig = mock.MagicMock()
    mock_engine = mock.MagicMock()
    mock_mutils = mock.MagicMock()

    mock_cfg = mock.MagicMock()
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
          skip_step_on_spikes=True,
          skip_step_on_nan=True,
          skip_step_interval=64,
          skip_step_scaling_factor=4.5,
      )
      mock_pyconfig.initialize.assert_called_once()
      argv = mock_pyconfig.initialize.call_args[0][0]
      self.assertIn("skip_step_on_spikes=True", argv)
      self.assertIn("skip_step_on_nan=True", argv)
      self.assertIn("skip_step_interval=64", argv)
      self.assertIn("skip_step_scaling_factor=4.5", argv)
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

      with self.assertRaisesRegex(
          ValueError, "max_seq_token_per_tpu must be non-negative"
      ):
        maxtext_utils.build_maxtext_config(
            "gemma2-9b", max_seq_token_per_tpu=-1
        )

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

  def test_max_seq_token_per_tpu_raises_max_target_length(self):
    # A packed row holds several trajectories, so it is wider than any single
    # one. max_prompt+max_response describes one trajectory, and everything
    # MaxText derives from max_target_length would be computed for that width.
    argv = self._build_config_argv(
        max_prompt_length=512,
        max_response_length=1024,
        max_seq_token_per_tpu=4096,
    )
    self.assertIn("max_target_length=4096", argv)

  def test_max_seq_token_per_tpu_equal_to_floor_is_legal(self):
    # One maximal trajectory per row is the smallest legal packing budget.
    argv = self._build_config_argv(
        max_prompt_length=512,
        max_response_length=1024,
        max_seq_token_per_tpu=1536,
    )
    self.assertIn("max_target_length=1536", argv)

  def test_max_seq_token_per_tpu_below_floor_raises(self):
    # rl_utils.validate_packing_budget rejects this when the learner builds its
    # assembler. Fail here instead, before the mesh and the model are built.
    with self.assertRaisesRegex(
        ValueError, "max_seq_token_per_tpu=1024 is smaller than the longest"
    ):
      self._build_config_argv(
          max_prompt_length=512,
          max_response_length=1024,
          max_seq_token_per_tpu=1024,
      )

  def test_max_seq_token_per_tpu_unset_keeps_default(self):
    argv = self._build_config_argv(
        max_prompt_length=512, max_response_length=1024
    )
    self.assertIn("max_target_length=1536", argv)

  def test_max_seq_token_per_tpu_none_keeps_default(self):
    argv = self._build_config_argv(
        max_prompt_length=512,
        max_response_length=1024,
        max_seq_token_per_tpu=None,
    )
    self.assertIn("max_target_length=1536", argv)

  def test_max_seq_token_per_tpu_negative_raises(self):
    with self.assertRaisesRegex(
        ValueError, "max_seq_token_per_tpu must be non-negative"
    ):
      self._build_config_argv(
          max_seq_token_per_tpu=-1,
      )

  def test_checkpoint_save_interval_zero_keeps_restore_but_disables_saving(
      self,
  ):
    # `save_interval_steps=0` means "never save". MaxText restores
    # `load_parameters_path` through its own `ocp.Checkpointer`, so warm start
    # still works with `enable_checkpointing=False`.
    argv = self._build_config_argv(
        load_parameters_path="gs://bucket/ckpt",
        checkpointing_options=mock.MagicMock(
            save_interval_steps=0, max_to_keep=10
        ),
    )
    self.assertIn("enable_checkpointing=False", argv)
    self.assertNotIn("enable_checkpointing=True", argv)
    self.assertIn("load_parameters_path=gs://bucket/ckpt", argv)

  def test_checkpoint_save_interval_zero_without_restore_disables_saving(self):
    argv = self._build_config_argv(
        load_parameters_path=None,
        checkpointing_options=mock.MagicMock(
            save_interval_steps=0, max_to_keep=10
        ),
    )
    self.assertIn("enable_checkpointing=False", argv)
    self.assertNotIn("enable_checkpointing=True", argv)

  def test_ckpt_d2h_concurrent_gb_override(self):
    with mock.patch.dict("os.environ", {"CKPT_D2H_CONCURRENT_GB": "32"}):
      argv = self._build_config_argv()
    self.assertIn("checkpoint_storage_device_host_concurrent_gb=32", argv)

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

  def test_trainable_parameters_mask_forwarded_to_config(self):
    mask = '["^(?!.*routed_experts/gate/kernel).*"]'
    argv = self._build_config_argv(trainable_parameters_mask=mask)
    self.assertIn(f"trainable_parameters_mask={mask}", argv)

  def test_trainable_parameters_mask_none_not_in_config(self):
    argv = self._build_config_argv(trainable_parameters_mask=None)
    self.assertFalse(any("trainable_parameters_mask=" in arg for arg in argv))

  def test_optimizer_overrides_forwarded_to_config(self):
    argv = self._build_config_argv(
        adam_b1=0.9,
        adam_b2=0.999,
        adam_eps=1e-8,
        adam_weight_decay=0.0,
        gradient_clipping_threshold=0.125,
    )
    self.assertIn("adam_b1=0.9", argv)
    self.assertIn("adam_b2=0.999", argv)
    self.assertIn("adam_eps=1e-08", argv)
    # 0.0 is a value, not "unset": it must still override base.yml's 0.1.
    self.assertIn("adam_weight_decay=0.0", argv)
    self.assertIn("gradient_clipping_threshold=0.125", argv)

  def test_optimizer_overrides_none_leave_argv_unchanged(self):
    default_argv = self._build_config_argv()
    none_argv = self._build_config_argv(
        adam_b1=None,
        adam_b2=None,
        adam_eps=None,
        adam_weight_decay=None,
        gradient_clipping_threshold=None,
    )
    self.assertEqual(none_argv, default_argv)
    self.assertFalse(
        any(
            arg.startswith(("adam_", "gradient_clipping_threshold="))
            for arg in default_argv
        )
    )

  def test_maxtext_extra_flags_override_optimizer_overrides(self):
    with mock.patch.dict("os.environ", {"MAXTEXT_EXTRA_FLAGS": "adam_b2=0.98"}):
      argv = self._build_config_argv(adam_b2=0.999)
    # pyconfig keeps the last occurrence of a key.
    self.assertGreater(argv.index("adam_b2=0.98"), argv.index("adam_b2=0.999"))

  def test_attention_default_dot_product(self):
    argv = self._build_config_argv()
    self.assertIn("attention=dot_product", argv)

  def test_attention_explicit_arg(self):
    argv = self._build_config_argv(attention="flash")
    self.assertIn("attention=flash", argv)

  def test_attention_env_var(self):
    with mock.patch.dict("os.environ", {"TRAINER_MAXTEXT_ATTENTION": "flash"}):
      argv = self._build_config_argv()
    self.assertIn("attention=flash", argv)

  def test_pathways_persistence_requires_gs_output_dir_when_saving_enabled(self):
    with mock.patch.dict("os.environ", {"ENABLE_PATHWAYS_PERSISTENCE": "1"}):
      with self.assertRaisesRegex(
          ValueError, "requires a gs:// base_output_directory"
      ):
        self._build_config_argv(
            base_output_directory="artifacts/math_gsm8k_dist/maxtext",
            checkpointing_options=mock.MagicMock(
                save_interval_steps=1, max_to_keep=2
            ),
        )

  def test_pathways_persistence_with_gs_output_dir_sets_ocdbt_and_zarr3_false(
      self,
  ):
    with mock.patch.dict(
        "os.environ",
        {
            "ENABLE_PATHWAYS_PERSISTENCE": "1",
        },
    ):
      argv = self._build_config_argv(
          base_output_directory="gs://yixuannwang-maxtext-dataset/trellis/0921",
          checkpointing_options=mock.MagicMock(
              save_interval_steps=1, max_to_keep=2
          ),
      )
    self.assertIn(
        "base_output_directory=gs://yixuannwang-maxtext-dataset/trellis/0921",
        argv,
    )
    self.assertIn("checkpoint_storage_use_ocdbt=false", argv)
    self.assertIn("checkpoint_storage_use_zarr3=false", argv)

  def test_pathways_persistence_allows_local_output_dir_when_save_interval_zero(
      self,
  ):
    with mock.patch.dict(
        "os.environ",
        {"ENABLE_PATHWAYS_PERSISTENCE": "1", "CHECKPOINT_ASYNC": "false"},
    ):
      argv = self._build_config_argv(
          base_output_directory="artifacts/math_gsm8k_dist/maxtext",
          load_parameters_path="gs://bucket/ckpt",
          checkpointing_options=None,
      )
    self.assertIn("checkpoint_storage_use_ocdbt=false", argv)
    self.assertIn("checkpoint_storage_use_zarr3=false", argv)
    self.assertIn("async_checkpointing=false", argv)

  def test_pathways_checkpointing_impl_defaults_to_persistence(self):
    with mock.patch.dict("os.environ", {"ENABLE_PATHWAYS_PERSISTENCE": "1"}, clear=False):
      os.environ.pop("PATHWAYS_CHECKPOINTING_IMPL", None)
      argv = self._build_config_argv(
          base_output_directory="gs://bucket/out",
          checkpointing_options=mock.MagicMock(save_interval_steps=1, max_to_keep=2),
      )
    self.assertIn("pathways_checkpointing_impl=persistence", argv)

  def test_colocated_python_requires_sidecar_image(self):
    """Requesting colocated with no sidecar is exactly the silent NO_DISPATCHER OOM."""
    with mock.patch.dict(
        "os.environ",
        {
            "ENABLE_PATHWAYS_PERSISTENCE": "1",
            "PATHWAYS_CHECKPOINTING_IMPL": "colocated_python",
        },
        clear=False,
    ):
      os.environ.pop("COLOCATED_PYTHON_SIDECAR_IMAGE", None)
      with self.assertRaisesRegex(ValueError, r"requires COLOCATED_PYTHON_SIDECAR_IMAGE"):
        self._build_config_argv(
            base_output_directory="gs://bucket/out",
            checkpointing_options=mock.MagicMock(save_interval_steps=1, max_to_keep=2),
        )

  def test_colocated_python_emits_impl_and_keeps_layout_identical(self):
    with mock.patch.dict(
        "os.environ",
        {
            "ENABLE_PATHWAYS_PERSISTENCE": "1",
            "PATHWAYS_CHECKPOINTING_IMPL": "colocated_python",
            "COLOCATED_PYTHON_SIDECAR_IMAGE": "us-docker.pkg.dev/proj/repo/sidecar:tag",
        },
        clear=False,
    ):
      argv = self._build_config_argv(
          base_output_directory="gs://bucket/out",
          checkpointing_options=mock.MagicMock(save_interval_steps=1, max_to_keep=2),
      )
    self.assertIn("pathways_checkpointing_impl=colocated_python", argv)
    # D3: identical on-disk layout in both modes keeps cross-mode restore safe.
    self.assertIn("checkpoint_storage_use_ocdbt=false", argv)
    self.assertIn("checkpoint_storage_use_zarr3=false", argv)

  def test_unknown_pathways_checkpointing_impl_raises(self):
    with mock.patch.dict(
        "os.environ",
        {
            "ENABLE_PATHWAYS_PERSISTENCE": "1",
            "PATHWAYS_CHECKPOINTING_IMPL": "remote_python",
        },
        clear=False,
    ):
      with self.assertRaisesRegex(ValueError, r"PATHWAYS_CHECKPOINTING_IMPL='remote_python'"):
        self._build_config_argv(
            base_output_directory="gs://bucket/out",
            checkpointing_options=mock.MagicMock(save_interval_steps=1, max_to_keep=2),
        )

  def test_impl_not_emitted_without_pathways_persistence(self):
    """The selector is meaningless unless Pathways checkpointing is on."""
    with mock.patch.dict(
        "os.environ",
        {"PATHWAYS_CHECKPOINTING_IMPL": "colocated_python"},
        clear=False,
    ):
      os.environ.pop("ENABLE_PATHWAYS_PERSISTENCE", None)
      argv = self._build_config_argv(
          base_output_directory="gs://bucket/out",
          checkpointing_options=mock.MagicMock(save_interval_steps=1, max_to_keep=2),
      )
    self.assertNotIn("pathways_checkpointing_impl=colocated_python", argv)

  def test_default_emits_mu_dtype_float32_and_float32_gate_logits_true(self):
    with mock.patch.dict("os.environ", {}, clear=False):
      os.environ.pop("FLOAT32_GATE_LOGITS", None)
      os.environ.pop("FLOAT32_LOGITS", None)
      argv = self._build_config_argv()
    self.assertIn("mu_dtype=float32", argv)
    self.assertIn("float32_gate_logits=True", argv)
    self.assertFalse(any(arg.startswith("float32_logits=") for arg in argv))

  def test_float32_gate_logits_and_logits_env_and_args(self):
    with mock.patch.dict(
        "os.environ",
        {"FLOAT32_GATE_LOGITS": "false", "FLOAT32_LOGITS": "true"},
        clear=False,
    ):
      argv = self._build_config_argv()
      self.assertIn("float32_gate_logits=False", argv)
      self.assertIn("float32_logits=True", argv)

      # Explicit arguments override environment variables.
      argv_override = self._build_config_argv(
          float32_gate_logits=True, float32_logits=False
      )
      self.assertIn("float32_gate_logits=True", argv_override)
      self.assertIn("float32_logits=False", argv_override)

  def test_build_vllm_maxtext_additional_config_float32_gate_logits(self):
    with mock.patch.dict("os.environ", {}, clear=False):
      os.environ.pop("FLOAT32_GATE_LOGITS", None)
      os.environ.pop("FLOAT32_LOGITS", None)
      cfg = maxtext_utils.build_vllm_maxtext_additional_config("qwen3.5-35b-a3b")
    mt_cfg = cfg["maxtext_config"]
    self.assertEqual(mt_cfg["weight_dtype"], "bfloat16")
    self.assertTrue(mt_cfg["float32_gate_logits"])
    self.assertNotIn("float32_logits", mt_cfg)

    with mock.patch.dict(
        "os.environ",
        {"FLOAT32_GATE_LOGITS": "false", "FLOAT32_LOGITS": "true"},
        clear=False,
    ):
      cfg_env = maxtext_utils.build_vllm_maxtext_additional_config(
          "qwen3.5-35b-a3b"
      )
    self.assertFalse(cfg_env["maxtext_config"]["float32_gate_logits"])
    self.assertTrue(cfg_env["maxtext_config"]["float32_logits"])

  def test_fp32_master_optimizer_preserves_bf16_params_and_accumulates_small_updates(
      self,
  ):
    from flax import nnx
    import jax
    import jax.numpy as jnp
    import optax

    class ToyModel(nnx.Module):

      def __init__(self):
        self.bf16_w = nnx.Param(jnp.ones((4,), dtype=jnp.bfloat16))
        self.f32_gate = nnx.Param(jnp.ones((4,), dtype=jnp.float32))

    model = ToyModel()
    tx = optax.adamw(learning_rate=1e-6, b1=0.9, b2=0.999, weight_decay=0.0)
    opt_cls = maxtext_utils._build_fp32_master_optimizer_cls(nnx)
    optimizer = opt_cls(model, tx, wrt=nnx.Param)

    # Master weights and Optax moments are initialized in float32 while model
    # weights retain their original dtypes. Parameters already in float32 store
    # None in master_params to avoid duplicate buffers and PJRT donation aliasing.
    self.assertEqual(model.bf16_w[...].dtype, jnp.bfloat16)
    self.assertEqual(model.f32_gate[...].dtype, jnp.float32)
    self.assertEqual(
        optimizer.opt_state["master_params"]["bf16_w"][...].dtype, jnp.float32
    )
    self.assertIsNone(
        optimizer.opt_state["master_params"]["f32_gate"].get_value()
    )
    leaves = jax.tree.leaves(optimizer.opt_state["inner"])
    for leaf in leaves:
      if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating):
        self.assertEqual(leaf.dtype, jnp.float32)

    grads = nnx.state(model, nnx.Param)
    grads = jax.tree.map(lambda x: jnp.full_like(x, 0.1, dtype=jnp.float32), grads)

    # One step at lr=1e-6 updates FP32 master weights and FP32 model params by ~1e-6.
    optimizer.update(model, grads)
    master_delta_1 = float(
        jnp.max(
            jnp.abs(optimizer.opt_state["master_params"]["bf16_w"][...] - 1.0)
        )
    )
    f32_gate_delta_1 = float(jnp.max(jnp.abs(model.f32_gate[...] - 1.0)))
    self.assertGreater(master_delta_1, 5e-7)
    self.assertGreater(f32_gate_delta_1, 5e-7)
    self.assertIsNone(
        optimizer.opt_state["master_params"]["f32_gate"].get_value()
    )
    self.assertEqual(model.bf16_w[...].dtype, jnp.bfloat16)
    self.assertEqual(model.f32_gate[...].dtype, jnp.float32)

  def test_fp32_master_optimizer_mirrors_is_skipped(self):
    from flax import nnx
    import jax.numpy as jnp
    import optax

    class ToyModel(nnx.Module):

      def __init__(self):
        self.w = nnx.Param(jnp.ones((2,), dtype=jnp.bfloat16))

    def init_fn(params):
      del params
      return {"is_skipped": jnp.asarray(False)}

    def update_fn(updates, state, params=None):
      del state, params
      return updates, {"is_skipped": jnp.asarray(True)}

    tx = optax.GradientTransformation(init_fn, update_fn)
    opt_cls = maxtext_utils._build_fp32_master_optimizer_cls(nnx)
    model = ToyModel()
    optimizer = opt_cls(model, tx, wrt=nnx.Param)
    pure_before = nnx.to_pure_dict(nnx.state(optimizer))["opt_state"]
    self.assertFalse(bool(pure_before["is_skipped"]))
    grads = nnx.state(model, nnx.Param)
    optimizer.update(model, grads)
    pure_after = nnx.to_pure_dict(nnx.state(optimizer))["opt_state"]
    self.assertTrue(bool(pure_after["is_skipped"]))

  def test_create_maxtext_engine_uses_fp32_master_and_clears_direct_target_dtype(
      self,
  ):
    mock_pyconfig = mock.MagicMock()
    mock_mutils = mock.MagicMock()

    class FakeOptimizer:
      pass

    class FakeNNX:
      Optimizer = FakeOptimizer
      Param = object

    captured = {}

    class FakeEngine:

      def __init__(
          self,
          config,
          mesh=None,
          wrap_with_tunix_adapter=True,
          tokenizer_pad_id=0,
      ):
        del mesh, wrap_with_tunix_adapter, tokenizer_pad_id
        self._config = config
        self.model = type("TunixMaxTextAdapter", (), {})()
        self._weight_converter = mock.MagicMock()
        self._weight_converter._direct = mock.MagicMock()
        self._weight_converter._direct.target_dtype = "bfloat16"
        self._init_state()

      def _init_state(self):
        captured["optimizer_cls_during_init"] = FakeNNX.Optimizer

    mock_engine_mod = mock.MagicMock()
    mock_engine_mod.nnx = FakeNNX
    mock_engine_mod.MaxTextTrainingEngine = FakeEngine

    mock_cfg = mock.MagicMock()
    mock_cfg.weight_dtype = "bfloat16"
    mock_cfg.float32_gate_logits = True
    mock_cfg.checkpoint_dir = "/tmp/ckpts"
    mock_mesh = mock.MagicMock()

    with mock.patch.object(
        maxtext_utils,
        "maxtext_modules",
        return_value=(mock_pyconfig, mock_engine_mod, mock_mutils),
    ):
      engine = maxtext_utils.create_maxtext_engine(
          mock_cfg, mock_mesh, log_shapes=False
      )

    self.assertIsNot(captured["optimizer_cls_during_init"], FakeOptimizer)
    self.assertEqual(FakeNNX.Optimizer, FakeOptimizer)
    self.assertIsNone(engine._weight_converter._direct.target_dtype)
    self.assertEqual(engine.checkpoint_dir, "/tmp/ckpts")

  def test_load_and_convert_scanned_checkpoint(self):
    scanned_cfg = mock.sentinel.scanned_cfg
    scanned_model = mock.sentinel.scanned_model
    scanned_state = mock.sentinel.scanned_state
    target_state = mock.sentinel.target_state
    converted_inner = {"decoder": {"layers_0": "weights"}}

    pyconfig_mod = mock.MagicMock(initialize=mock.Mock(return_value=scanned_cfg))
    model_creation_mod = mock.MagicMock(
        from_pretrained=mock.Mock(
            return_value=(scanned_model, mock.sentinel.mesh)
        )
    )
    converter_inst = mock.MagicMock(
        convert=mock.Mock(return_value={"model": converted_inner})
    )
    converter_cls = mock.Mock(return_value=converter_inst)
    nnx_mod = mock.MagicMock(
        Param=mock.sentinel.Param,
        state=mock.Mock(return_value=scanned_state),
    )
    jax_mod = mock.MagicMock(
        devices=mock.Mock(return_value=["d0", "d1"]),
        clear_caches=mock.Mock(),
    )
    mock_sampler = mock.MagicMock()
    mock_sampler.vllm_sampler.transformer_state = target_state

    with mock.patch.dict(
        "sys.modules",
        {
            "flax": mock.MagicMock(nnx=nnx_mod),
            "flax.nnx": nnx_mod,
            "jax": jax_mod,
            "maxtext.common.common_types": mock.MagicMock(
                MODEL_MODE_AUTOREGRESSIVE="autoregressive"
            ),
            "maxtext.configs": mock.MagicMock(pyconfig=pyconfig_mod),
            "maxtext.integration.vllm.weight_converter": mock.MagicMock(
                MaxTextToMaxTextConverter=converter_cls
            ),
            "maxtext.utils": mock.MagicMock(
                model_creation_utils=model_creation_mod
            ),
            "maxtext.utils.globals": mock.MagicMock(
                MAXTEXT_CONFIGS_DIR="/maxtext/configs"
            ),
        },
    ):
      maxtext_utils.load_and_convert_scanned_checkpoint(
          path="/ckpt/0/items",
          sampler=mock_sampler,
          mesh_tp=2,
          ckpt_prefuse_moe=False,
          maxtext_config_overrides={"model_name": "qwen3.5-35b-a3b"},
      )

    pyconfig_mod.initialize.assert_called_once()
    init_args, init_kwargs = pyconfig_mod.initialize.call_args
    self.assertEqual(init_args[0], ["", "/maxtext/configs/base.yml"])
    self.assertTrue(init_kwargs["scan_layers"])
    self.assertEqual(init_kwargs["model_name"], "qwen3.5-35b-a3b")
    self.assertEqual(init_kwargs["load_parameters_path"], "/ckpt/0/items")
    model_creation_mod.from_pretrained.assert_called_once_with(
        scanned_cfg,
        devices=["d0", "d1"],
        model_mode="autoregressive",
    )
    converter_cls.assert_called_once_with(
        config=scanned_cfg,
        tp=2,
        prefuse_moe_weights=True,
        target_dtype=None,
    )
    converter_inst.convert.assert_called_once_with(
        scanned_state, target_state=target_state
    )
    mock_sampler.vllm_sampler.update_params.assert_called_once_with(
        converted_inner
    )

  def test_load_and_convert_scanned_checkpoint_exec_moe_keeps_prefused_wi_alive(
      self,
  ):
    class FakeArray:

      def __init__(self, shape):
        self.shape = shape
        self.deleted = False

      def __getitem__(self, item):
        return FakeArray((self.shape[0], self.shape[1] // 2))

      def delete(self):
        self.deleted = True

    prefused_wi = FakeArray((4, 8))
    out_arr = FakeArray((4, 8))
    prefused_wi_deleted_during_exec = None

    def fake_exec_group(group, src_flat, tgt_flat):
      nonlocal prefused_wi_deleted_during_exec
      prefused_wi_deleted_during_exec = prefused_wi.deleted
      return [("out_key", out_arr)]

    orig_converter = mock.MagicMock()
    orig_converter._build_plan = mock.MagicMock()
    orig_converter._execute_group = fake_exec_group

    def fake_convert(scanned_state, target_state=None):
      src_flat = {("decoder", "layers", 0, "wi"): prefused_wi}
      orig_converter._build_plan(src_flat, {})
      group = mock.MagicMock(
          op="fuse_moe",
          source_keys=[
              ("decoder", "layers", 0, "wi_0"),
              ("decoder", "layers", 0, "wi_1"),
          ],
      )
      orig_converter._execute_group(group, src_flat, {})
      return {"model": {}}

    orig_converter.convert = fake_convert
    converter_cls = mock.Mock(return_value=orig_converter)

    scanned_cfg = mock.sentinel.scanned_cfg
    pyconfig_mod = mock.MagicMock(initialize=mock.Mock(return_value=scanned_cfg))
    model_creation_mod = mock.MagicMock(
        from_pretrained=mock.Mock(
            return_value=(mock.sentinel.model, mock.sentinel.mesh)
        )
    )
    nnx_mod = mock.MagicMock(
        Param=mock.sentinel.Param,
        state=mock.Mock(return_value=mock.sentinel.scanned_state),
    )
    jax_mod = mock.MagicMock(
        Array=FakeArray,
        devices=mock.Mock(return_value=["d0"]),
        clear_caches=mock.Mock(),
        block_until_ready=mock.Mock(),
        tree_util=None,
    )
    mock_sampler = mock.MagicMock()

    with mock.patch.dict(
        "sys.modules",
        {
            "flax": mock.MagicMock(nnx=nnx_mod),
            "flax.nnx": nnx_mod,
            "jax": jax_mod,
            "maxtext.common.common_types": mock.MagicMock(
                MODEL_MODE_AUTOREGRESSIVE="autoregressive"
            ),
            "maxtext.configs": mock.MagicMock(pyconfig=pyconfig_mod),
            "maxtext.integration.vllm.weight_converter": mock.MagicMock(
                MaxTextToMaxTextConverter=converter_cls
            ),
            "maxtext.utils": mock.MagicMock(
                model_creation_utils=model_creation_mod
            ),
            "maxtext.utils.globals": mock.MagicMock(
                MAXTEXT_CONFIGS_DIR="/maxtext/configs"
            ),
        },
    ):
      maxtext_utils.load_and_convert_scanned_checkpoint(
          path="/ckpt/0/items",
          sampler=mock_sampler,
          mesh_tp=2,
          ckpt_prefuse_moe=True,
      )

    self.assertFalse(prefused_wi_deleted_during_exec)
    self.assertTrue(prefused_wi.deleted)


if __name__ == "__main__":
  absltest.main()



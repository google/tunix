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

import types
from typing import NamedTuple
from unittest import mock

from absl.testing import absltest
from flax import nnx
from flax import traverse_util
import jax
from jax import numpy as jnp
import numpy as np
from tunix.utils import maxtext_utils

# MaxText optimizer config keys. `AdamWOptions` sets all but opt_type, which
# stays at MaxText's default (adamw) so env overrides of it keep working.
_OPTIMIZER_KEYS = ("opt_type=", "adam_", "gradient_clipping_threshold=")


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
      self.assertIn("dtype=bfloat16", argv)
      self.assertIn("weight_dtype=float32", argv)
      self.assertIn("rollout_weight_dtype=bfloat16", argv)
      self.assertIn("grad_dtype=float32", argv)
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

    mock_convert_utils = mock.MagicMock(compute_padded_moe_mlp_dim=mock_compute)
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
            "maxtext.src.maxtext.integration.vllm.convert_utils": (
                mock_convert_utils
            ),
            "maxtext.integration.vllm.convert_utils": mock_convert_utils,
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

      mock_convert_utils = mock.MagicMock(
          compute_padded_moe_mlp_dim=mock_compute_padded_moe_mlp_dim
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
              "maxtext.src.maxtext.integration.vllm.convert_utils": (
                  mock_convert_utils
              ),
              "maxtext.integration.vllm.convert_utils": mock_convert_utils,
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
        self.assertIn(
            "override_model_config=True",
            argv,
            f"{name}: override_model_config=True missing in argv",
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

  def test_batch_size_must_divide_fsdp_times_expert(self):
    # MaxText shards the batch over fsdp and expert together: 6 rows divide
    # fsdp=2 but cannot be split across fsdp=2 x expert=2 = 4 batch shards.
    with self.assertRaisesRegex(
        ValueError, r"mesh_fsdp \* mesh_expert = 2 \* 2 = 4"
    ):
      self._build_config_argv(
          train_micro_batch_size=6,
          mesh_fsdp=2,
          mesh_tp=2,
          mesh_expert=2,
          num_devices=8,
      )
    argv = self._build_config_argv(
        train_micro_batch_size=8,
        mesh_fsdp=2,
        mesh_tp=2,
        mesh_expert=2,
        num_devices=8,
    )
    self.assertIn("per_device_batch_size=1.0", argv)
    self.assertIn("ici_expert_parallelism=2", argv)

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

    mock_convert_utils = mock.MagicMock(
        compute_padded_moe_mlp_dim=mock.MagicMock(
            side_effect=ValueError("Padding failure")
        )
    )
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
            "maxtext.src.maxtext.integration.vllm.convert_utils": (
                mock_convert_utils
            ),
            "maxtext.integration.vllm.convert_utils": mock_convert_utils,
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
    mock_convert_utils = mock.MagicMock(compute_padded_moe_mlp_dim=mock_compute)
    with mock.patch.object(
        maxtext_utils,
        "maxtext_modules",
        return_value=(mock_pyconfig, mock.MagicMock(), mock.MagicMock()),
    ), mock.patch("os.path.exists", return_value=True), mock.patch(
        "builtins.open", mock_file
    ), mock.patch.dict(
        "sys.modules",
        {
            "maxtext.src.maxtext.integration.vllm.convert_utils": (
                mock_convert_utils
            ),
            "maxtext.integration.vllm.convert_utils": mock_convert_utils,
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
        {
            "maxtext.src.maxtext.integration.vllm.convert_utils": None,
            "maxtext.integration.vllm.convert_utils": None,
        },
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

  def test_max_seq_token_per_tpu_below_floor_is_ignored(self):
    argv = self._build_config_argv(
        max_prompt_length=512,
        max_response_length=1024,
        max_seq_token_per_tpu=1024,
    )
    self.assertIn("max_target_length=1536", argv)

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

  def test_adamw_options_override_maxtext_optimizer_defaults(self):
    argv = self._build_config_argv(
        adamw_options=maxtext_utils.AdamWOptions(
            b1=0.8,
            b2=0.999,
            eps=1e-6,
            weight_decay=0.0,
            gradient_clipping_threshold=0.125,
        )
    )
    # Exactly once each: MaxText's CLI parsing keeps the last duplicate.
    self.assertEqual(
        [a for a in argv if a.startswith(_OPTIMIZER_KEYS)],
        [
            "adam_b1=0.8",
            "adam_b2=0.999",
            "adam_eps=1e-06",
            "adam_weight_decay=0.0",
            "gradient_clipping_threshold=0.125",
        ],
    )

  def test_without_adamw_options_keeps_maxtext_optimizer_defaults(self):
    argv = self._build_config_argv()
    self.assertEmpty([a for a in argv if a.startswith(_OPTIMIZER_KEYS)])

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

  def test_device_host_concurrent_gb_sets_checkpoint_d2h_cap(self):
    argv = self._build_config_argv(device_host_concurrent_gb=32)
    self.assertIn("checkpoint_storage_device_host_concurrent_gb=32", argv)
    # Without it MaxText keeps its own cap; the environment is not read.
    with mock.patch.dict("os.environ", {"CKPT_D2H_CONCURRENT_GB": "32"}):
      argv = self._build_config_argv()
    self.assertEmpty([
        a
        for a in argv
        if a.startswith("checkpoint_storage_device_host_concurrent_gb=")
    ])
    with self.assertRaisesRegex(TypeError, "must be an int or None"):
      self._build_config_argv(device_host_concurrent_gb=True)
    with self.assertRaisesRegex(ValueError, "must be positive"):
      self._build_config_argv(device_host_concurrent_gb=0)

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

  def test_max_seq_token_per_tpu_ignores_train_micro_batch_size(self):
    argv = self._build_config_argv(
        train_micro_batch_size=1,
        mesh_fsdp=2,
        mesh_tp=4,
        mesh_expert=1,
        num_devices=8,
        max_prompt_length=512,
        max_response_length=512,
        max_seq_token_per_tpu=4096,
    )
    self.assertIn("per_device_batch_size=0.25", argv)
    self.assertIn("max_target_length=4096", argv)

  def test_max_seq_token_per_tpu_includes_mesh_expert_in_per_device_batch_size(
      self,
  ):
    argv = self._build_config_argv(
        train_micro_batch_size=1,
        mesh_fsdp=2,
        mesh_tp=2,
        mesh_expert=2,
        num_devices=8,
        max_prompt_length=512,
        max_response_length=512,
        max_seq_token_per_tpu=4096,
    )
    self.assertIn("per_device_batch_size=0.5", argv)
    self.assertIn("max_target_length=4096", argv)

  def test_build_maxtext_config_profiling_options_enables_xplane_and_tpu_options(
      self,
  ):
    argv = self._build_config_argv(
        profiling_options=maxtext_utils.ProfilerOptions(
            skip_first_n_steps=8,
            profiler_steps=3,
            profiler_period=-1,
        ),
    )
    self.assertIn("profiler=xplane", argv)
    self.assertIn("profiler_steps=3", argv)
    self.assertIn("skip_first_n_steps_for_profiler=8", argv)
    self.assertIn("profile_periodically_period=-1", argv)
    self.assertIn("upload_all_profiler_results=false", argv)
    self.assertIn("enable_tpu_profiling_options=true", argv)
    self.assertIn("tpu_num_chips_to_profile_per_task=1", argv)
    self.assertIn("tpu_num_sparse_cores_to_trace=1", argv)
    self.assertIn("tpu_num_sparse_core_tiles_to_trace=1", argv)

  def test_build_maxtext_config_profiling_options_custom_tpu_and_upload_flags(
      self,
  ):
    argv = self._build_config_argv(
        profiling_options=maxtext_utils.ProfilerOptions(
            skip_first_n_steps=5,
            profiler_steps=6,
            profiler_period=10,
            enable_tpu_profiling_options=True,
            tpu_num_chips_to_profile_per_task=2,
            tpu_num_sparse_cores_to_trace=2,
            tpu_num_sparse_core_tiles_to_trace=16,
            upload_all_profiler_results=True,
        ),
    )
    self.assertIn("profiler=xplane", argv)
    self.assertIn("profiler_steps=6", argv)
    self.assertIn("skip_first_n_steps_for_profiler=5", argv)
    self.assertIn("profile_periodically_period=10", argv)
    self.assertIn("upload_all_profiler_results=true", argv)
    self.assertIn("enable_tpu_profiling_options=true", argv)
    self.assertIn("tpu_num_chips_to_profile_per_task=2", argv)
    self.assertIn("tpu_num_sparse_cores_to_trace=2", argv)
    self.assertIn("tpu_num_sparse_core_tiles_to_trace=16", argv)

    argv_disabled_tpu = self._build_config_argv(
        profiling_options=maxtext_utils.ProfilerOptions(
            skip_first_n_steps=1,
            profiler_steps=2,
            enable_tpu_profiling_options=False,
        ),
    )
    self.assertIn("profiler=xplane", argv_disabled_tpu)
    self.assertIn("enable_tpu_profiling_options=false", argv_disabled_tpu)
    self.assertEmpty([a for a in argv_disabled_tpu if a.startswith("tpu_num_")])

  def test_build_maxtext_config_profiling_options_none_disables_profiler(self):
    argv = self._build_config_argv(profiling_options=None)
    self.assertIn("profiler=''", argv)
    self.assertIn("profiler_steps=0", argv)
    self.assertIn("skip_first_n_steps_for_profiler=-1", argv)
    self.assertNotIn("profiler=xplane", argv)

  def test_build_maxtext_config_extra_flags_passthrough_and_validation(self):
    with self.assertLogs(level="INFO") as logs:
      argv = self._build_config_argv(
          extra_flags=("use_ring_of_experts=true", "sa_block_q=512")
      )
    # Appended last, so they override the arguments built before them.
    self.assertEqual(argv[-2:], ["use_ring_of_experts=true", "sa_block_q=512"])
    # Log tooling parses this line.
    self.assertIn(
        "MAXTEXT_EXTRA_FLAGS adding 2 flag(s): ['use_ring_of_experts=true',"
        " 'sa_block_q=512']",
        "\n".join(logs.output),
    )

    with self.assertRaisesRegex(
        ValueError, "extra_flags entries must be key=value, got: invalid_flag"
    ):
      self._build_config_argv(
          extra_flags=["use_ring_of_experts=true", "invalid_flag"]
      )

    # The environment is not read.
    with mock.patch.dict(
        "os.environ", {"MAXTEXT_EXTRA_FLAGS": "use_ring_of_experts=true"}
    ):
      argv = self._build_config_argv()
    self.assertNotIn("use_ring_of_experts=true", argv)

  def test_vllm_additional_config_returns_routed_experts_only_when_asked(self):
    default = maxtext_utils.build_vllm_maxtext_additional_config(
        "qwen3-30b-a3b"
    )
    self.assertNotIn("return_routed_experts", default["maxtext_config"])

    enabled = maxtext_utils.build_vllm_maxtext_additional_config(
        "qwen3-30b-a3b", return_routed_experts=True
    )
    self.assertIs(enabled["maxtext_config"]["return_routed_experts"], True)

  def test_weight_dtype_override_and_rollout_weight_dtype_parity(self):
    argv = self._build_config_argv(weight_dtype="bfloat16")
    self.assertIn("weight_dtype=bfloat16", argv)
    self.assertIn(
        f"rollout_weight_dtype={maxtext_utils.DEFAULT_ROLLOUT_WEIGHT_DTYPE}",
        argv,
    )
    vllm_cfg = maxtext_utils.build_vllm_maxtext_additional_config(
        model_name="gemma2-9b"
    )
    self.assertEqual(
        vllm_cfg["maxtext_config"]["weight_dtype"],
        maxtext_utils.DEFAULT_ROLLOUT_WEIGHT_DTYPE,
    )

  def test_build_maxtext_config_defaults_omit_optional_knobs(self):
    argv = self._build_config_argv()
    self.assertIn("allow_split_physical_axes=True", argv)
    self.assertIn("attention=dot_product", argv)
    self.assertIn("ici_context_parallelism=1", argv)
    self.assertNotIn("context_parallel_load_balance=False", argv)
    self.assertNotIn("override_model_config=True", argv)

  def test_build_maxtext_config_context_parallelism_disables_load_balance(self):
    argv = self._build_config_argv(mesh_context=2)
    self.assertIn("ici_context_parallelism=2", argv)
    self.assertIn("context_parallel_load_balance=False", argv)

  def test_build_maxtext_config_base_num_kv_heads_enables_override_model_config(
      self,
  ):
    argv = self._build_config_argv(base_num_kv_heads=8)
    self.assertIn("base_num_kv_heads=8", argv)
    self.assertIn("override_model_config=True", argv)

  def test_build_maxtext_config_yaml_kv_heads_override_only_when_replicated(
      self,
  ):
    mock_pyconfig = mock.MagicMock()
    mock_pyconfig.initialize.return_value = mock.MagicMock()
    mock_pyconfig.__file__ = "/fake/maxtext/configs/pyconfig.py"

    with mock.patch.object(
        maxtext_utils,
        "maxtext_modules",
        return_value=(mock_pyconfig, mock.MagicMock(), mock.MagicMock()),
    ), mock.patch("os.path.exists", return_value=True), mock.patch(
        "builtins.open",
        mock.mock_open(read_data="base_num_kv_heads: 4\n"),
    ):
      # kv_tp_size == YAML heads (4): no replication, so neither
      # base_num_kv_heads nor override_model_config is emitted.
      maxtext_utils.build_maxtext_config(
          model_name="qwen3-test",
          base_num_kv_heads=0,
          kv_tp_size=4,
      )
      argv_no_rep = mock_pyconfig.initialize.call_args[0][0]
      self.assertEmpty(
          [a for a in argv_no_rep if a.startswith("base_num_kv_heads=")]
      )
      self.assertNotIn("override_model_config=True", argv_no_rep)

      # kv_tp_size > YAML heads (8 > 4): KV replication occurs, so both
      # base_num_kv_heads=8 and override_model_config=True are emitted.
      mock_pyconfig.reset_mock()
      maxtext_utils.build_maxtext_config(
          model_name="qwen3-test",
          base_num_kv_heads=0,
          kv_tp_size=8,
      )
      argv_rep = mock_pyconfig.initialize.call_args[0][0]
      self.assertIn("base_num_kv_heads=8", argv_rep)
      self.assertIn("override_model_config=True", argv_rep)

  def test_build_maxtext_config_explicit_attention_override(self):
    argv = self._build_config_argv(attention="flash")
    self.assertIn("attention=flash", argv)
    self.assertNotIn("attention=dot_product", argv)

  def test_build_maxtext_config_mesh_context_validation(self):
    with self.assertRaisesRegex(ValueError, "mesh_context must be positive"):
      self._build_config_argv(mesh_context=0)

  def test_vllm_additional_config_scan_layers_and_float32_logit_knobs(self):
    default_cfg = maxtext_utils.build_vllm_maxtext_additional_config(
        "qwen3.5-35b-a3b"
    )["maxtext_config"]
    self.assertIs(default_cfg["scan_layers"], False)
    self.assertNotIn("float32_gate_logits", default_cfg)
    self.assertNotIn("float32_logits", default_cfg)

    explicit_cfg = maxtext_utils.build_vllm_maxtext_additional_config(
        "qwen3.5-35b-a3b",
        float32_gate_logits=True,
        float32_logits=False,
    )["maxtext_config"]
    self.assertIs(explicit_cfg["float32_gate_logits"], True)
    self.assertIs(explicit_cfg["float32_logits"], False)

  def test_create_maxtext_engine_enters_mesh_and_logical_axis_rules(self):
    from flax.core import spmd  # pylint: disable=g-import-not-at-top

    expected_rules = (("batch", ("data", "fsdp")), ("embed", "tensor"))
    mock_cfg = mock.MagicMock()
    mock_cfg.logical_axis_rules = expected_rules
    mock_mesh = mock.MagicMock()
    captured = {}

    class TunixMaxTextAdapter:
      pass

    def fake_engine_ctor(
        cfg, *, mesh, wrap_with_tunix_adapter, tokenizer_pad_id
    ):
      captured["cfg"] = cfg
      captured["mesh"] = mesh
      captured["wrap_with_tunix_adapter"] = wrap_with_tunix_adapter
      captured["tokenizer_pad_id"] = tokenizer_pad_id
      captured["active_rules"] = spmd.get_logical_axis_rules()
      captured["mesh_entered"] = mock_mesh.__enter__.call_count
      captured["mesh_exited"] = mock_mesh.__exit__.call_count
      engine = mock.MagicMock()
      engine.model = TunixMaxTextAdapter()
      return engine

    mock_engine_mod = mock.MagicMock()
    mock_engine_mod.MaxTextTrainingEngine.side_effect = fake_engine_ctor

    with mock.patch.object(
        maxtext_utils,
        "maxtext_modules",
        return_value=(mock.MagicMock(), mock_engine_mod, mock.MagicMock()),
    ):
      engine = maxtext_utils.create_maxtext_engine(
          mock_cfg,
          mock_mesh,
          tokenizer_pad_id=42,
          wrap_with_tunix_adapter=True,
          log_shapes=False,
      )

    self.assertIsInstance(engine.model, TunixMaxTextAdapter)
    self.assertEqual(captured["cfg"], mock_cfg)
    self.assertEqual(captured["mesh"], mock_mesh)
    self.assertIs(captured["wrap_with_tunix_adapter"], True)
    self.assertEqual(captured["tokenizer_pad_id"], 42)
    self.assertEqual(captured["active_rules"], expected_rules)
    self.assertEqual(captured["mesh_entered"], 1)
    self.assertEqual(captured["mesh_exited"], 0)
    mock_mesh.__exit__.assert_called_once()
    self.assertEqual(spmd.get_logical_axis_rules(), ())

  # TODO(tunix-dev): [legacy-maxtext-a2cf3bc8b] Replace the three legacy-pin
  # compatibility tests below with a direct assertion that
  # `build_maxtext_config()` unconditionally emits
  # `rollout_weight_dtype=bfloat16` once pinned MaxText images predating
  # `rollout_weight_dtype` are retired.
  def test_build_maxtext_config_omits_rollout_weight_dtype_when_unsupported(
      self,
  ):
    mock_pyconfig = mock.MagicMock()
    mock_cfg = mock.MagicMock()
    mock_pyconfig.__file__ = "/fake/maxtext/configs/pyconfig.py"
    mock_pyconfig.initialize.return_value = mock_cfg
    mock_pyconfig.types.MaxTextConfig.model_fields = {"model_name": object()}

    with mock.patch.object(
        maxtext_utils,
        "maxtext_modules",
        return_value=(mock_pyconfig, mock.MagicMock(), mock.MagicMock()),
    ), mock.patch("os.path.exists", return_value=True):
      result = maxtext_utils.build_maxtext_config("qwen3.5-35b-a3b")

    self.assertIs(result, mock_cfg)
    mock_pyconfig.initialize.assert_called_once()
    legacy_argv = mock_pyconfig.initialize.call_args[0][0]
    self.assertFalse(
        any(arg.startswith("rollout_weight_dtype=") for arg in legacy_argv)
    )

    mock_pyconfig.reset_mock()
    mock_pyconfig.types.MaxTextConfig.model_fields = {
        "model_name": object(),
        "rollout_weight_dtype": object(),
    }
    mock_pyconfig.initialize.side_effect = ValueError(
        "invalid M_ROLLOUT_WEIGHT_DTYPE"
    )
    with mock.patch.object(
        maxtext_utils,
        "maxtext_modules",
        return_value=(mock_pyconfig, mock.MagicMock(), mock.MagicMock()),
    ), mock.patch("os.path.exists", return_value=True):
      with self.assertRaisesRegex(ValueError, "invalid M_ROLLOUT_WEIGHT_DTYPE"):
        maxtext_utils.build_maxtext_config("qwen3.5-35b-a3b")
    mock_pyconfig.initialize.assert_called_once()
    supported_argv = mock_pyconfig.initialize.call_args[0][0]
    self.assertIn("rollout_weight_dtype=bfloat16", supported_argv)

  def test_create_maxtext_engine_target_free_shim_legacy_pinned_maxtext(self):
    class TunixMaxTextAdapter:
      pass

    class LegacyDirectConverter:
      """Simulates pinned MaxText predating `rollout_weight_dtype`."""

      def __init__(self):
        self.target_dtype = "float32"

      def _resolve_target_dtype(self):
        return jnp.dtype(self.target_dtype)

      def _execute_group_target_free(self, group, src_flat):
        key = group.source_keys[0]
        raw_val = src_flat.get(
            key, jax.ShapeDtypeStruct((4, 4), jnp.float32)
        )
        if getattr(group, "quantize_to_fp8", False):
          return [(key, raw_val)]
        path_str = ".".join(str(p) for p in key)
        target_dt = self._resolve_target_dtype()
        if "gate" in path_str or "router" in path_str:
          return [(key, raw_val)]
        if hasattr(raw_val, "dtype") and jnp.issubdtype(
            raw_val.dtype, jnp.floating
        ):
          if isinstance(raw_val, jax.ShapeDtypeStruct):
            return [(key, jax.ShapeDtypeStruct(raw_val.shape, target_dt))]
          return [(key, raw_val.astype(target_dt))]
        return [(key, raw_val)]

    def _make_group(*path, quantize_to_fp8=False):
      return mock.MagicMock(source_keys=[path], quantize_to_fp8=quantize_to_fp8)

    # Older pinned MaxText container images (e.g. built for the `a2cf3bc8b` 35B
    # recipe): `get_weight_dtype` keeps norm/gate/GDN/logits_dense modules in
    # FP32 when `float32_gate_logits=True`, and `_execute_group_target_free`
    # returns uncast source leaves.
    def _legacy_get_weight_dtype(config, module_name):
      if getattr(config, "float32_gate_logits", False) and module_name in (
          "norm",
          "gate",
          "shared_expert_gate",
          "A_log",
          "dt_bias",
          "conv1d",
          "logits_dense",
      ):
        return "float32"
      return config.weight_dtype

    mock_cfg = mock.MagicMock()
    mock_cfg.logical_axis_rules = ()
    mock_cfg.float32_gate_logits = True
    mock_cfg.rollout_weight_dtype = None
    direct_legacy = LegacyDirectConverter()
    engine_legacy = mock.MagicMock()
    engine_legacy.model = TunixMaxTextAdapter()
    engine_legacy._weight_converter._direct = direct_legacy
    mock_engine_mod = mock.MagicMock()
    mock_engine_mod.MaxTextTrainingEngine.return_value = engine_legacy

    with mock.patch.object(
        maxtext_utils,
        "maxtext_modules",
        return_value=(mock.MagicMock(), mock_engine_mod, mock.MagicMock()),
    ), mock.patch.object(
        maxtext_utils,
        "_maxtext_get_weight_dtype",
        return_value=_legacy_get_weight_dtype,
    ):
      maxtext_utils.create_maxtext_engine(
          mock_cfg,
          mock.MagicMock(),
          wrap_with_tunix_adapter=True,
          log_shapes=False,
      )

    fp32_paths = (
        ("decoder", "layers", "mlp", "gate", "kernel"),
        ("decoder", "layers", "mlp", "gate", "bias"),
        ("decoder", "layers", "input_layernorm", "scale"),
        ("decoder", "layers", "post_attention_layernorm", "scale"),
        ("decoder", "layers", "linear_attention", "norm", "scale"),
        ("decoder", "layers", "linear_attention", "A_log"),
        ("decoder", "layers", "linear_attention", "conv1d", "kernel"),
        ("decoder", "layers", "linear_attention", "dt_bias"),
        ("decoder", "layers", "mlp", "shared_expert_gate", "kernel"),
        ("decoder", "logits_dense", "kernel"),
    )
    for path in fp32_paths:
      with self.subTest(fp32_path=path):
        out = direct_legacy._execute_group_target_free(_make_group(*path), {})
        self.assertEqual(out[0][1].dtype, jnp.float32)

    bf16_paths = (
        ("token_embedder", "embedding"),
        ("decoder", "decoder_norm", "scale"),
        ("decoder", "layers", "self_attention", "query_norm", "scale"),
        ("decoder", "layers", "mlp", "wi_0"),
    )
    for path in bf16_paths:
      with self.subTest(bf16_path=path):
        out = direct_legacy._execute_group_target_free(_make_group(*path), {})
        self.assertEqual(out[0][1].dtype, jnp.bfloat16)

    # Live `jax.Array` floating leaves exercise the `.astype` branch, while
    # non-floating `int32` leaves and `quantize_to_fp8=True` groups stay intact.
    wi_key = ("decoder", "layers", "mlp", "wi_0")
    gate_key = ("decoder", "layers", "mlp", "gate", "kernel")
    step_key = ("decoder", "layers", "router_gate", "step_count")
    fp8_key = ("decoder", "layers", "mlp", "wo")
    live_wi_out = direct_legacy._execute_group_target_free(
        _make_group(*wi_key),
        {wi_key: jnp.ones((2, 2), dtype=jnp.float32)},
    )
    self.assertIsInstance(live_wi_out[0][1], jax.Array)
    self.assertEqual(live_wi_out[0][1].dtype, jnp.bfloat16)

    live_gate_out = direct_legacy._execute_group_target_free(
        _make_group(*gate_key),
        {gate_key: jnp.ones((2, 2), dtype=jnp.float32)},
    )
    self.assertEqual(live_gate_out[0][1].dtype, jnp.float32)

    for int_leaf in (
        jax.ShapeDtypeStruct((1,), jnp.int32),
        jnp.ones((1,), dtype=jnp.int32),
    ):
      int_out = direct_legacy._execute_group_target_free(
          _make_group(*step_key), {step_key: int_leaf}
      )
      self.assertEqual(int_out[0][1].dtype, jnp.int32)

    fp8_leaf = jax.ShapeDtypeStruct((2, 2), jnp.float8_e4m3fn)
    fp8_out = direct_legacy._execute_group_target_free(
        _make_group(*fp8_key, quantize_to_fp8=True), {fp8_key: fp8_leaf}
    )
    self.assertEqual(fp8_out[0][1].dtype, jnp.float8_e4m3fn)

    # Pinned MaxText with `float32_gate_logits=False`: `fp32_modules` is empty,
    # yet the shim must still install to override pinned MaxText's unconditional
    # `"gate"`/`"router"` raw-FP32 retention so gate weights come out BF16.
    mock_cfg_no_fp32_gates = mock.MagicMock()
    mock_cfg_no_fp32_gates.logical_axis_rules = ()
    mock_cfg_no_fp32_gates.float32_gate_logits = False
    mock_cfg_no_fp32_gates.rollout_weight_dtype = None
    direct_legacy_bf16 = LegacyDirectConverter()
    engine_legacy_bf16 = mock.MagicMock()
    engine_legacy_bf16.model = TunixMaxTextAdapter()
    engine_legacy_bf16._weight_converter._direct = direct_legacy_bf16
    mock_engine_mod.MaxTextTrainingEngine.return_value = engine_legacy_bf16
    with mock.patch.object(
        maxtext_utils,
        "maxtext_modules",
        return_value=(mock.MagicMock(), mock_engine_mod, mock.MagicMock()),
    ), mock.patch.object(
        maxtext_utils,
        "_maxtext_get_weight_dtype",
        return_value=_legacy_get_weight_dtype,
    ):
      maxtext_utils.create_maxtext_engine(
          mock_cfg_no_fp32_gates,
          mock.MagicMock(),
          wrap_with_tunix_adapter=True,
          log_shapes=False,
      )
    self.assertIn("_execute_group_target_free", direct_legacy_bf16.__dict__)
    for gate_path in (
        ("decoder", "layers", "mlp", "routed_experts", "gate", "kernel"),
        ("decoder", "layers", "mlp", "shared_expert_gate", "kernel"),
    ):
      gate_out = direct_legacy_bf16._execute_group_target_free(
          _make_group(*gate_path), {}
      )
      self.assertEqual(gate_out[0][1].dtype, jnp.bfloat16)

  def test_create_maxtext_engine_target_free_noop_on_upstream_main(self):
    class TunixMaxTextAdapter:
      pass

    class MainDirectConverter:
      """Simulates upstream MaxText `main` with `_resolve_target_dtype`."""

      def __init__(self):
        self.target_dtype = jnp.dtype("bfloat16")
        self.seen_target_dtypes = []

      def _resolve_target_dtype(self):
        return jnp.dtype(self.target_dtype)

      def _execute_group_target_free(self, group, src_flat):
        del src_flat
        resolved = self._resolve_target_dtype()
        self.seen_target_dtypes.append(resolved)
        return [(group.source_keys[0], jax.ShapeDtypeStruct((4, 4), resolved))]

    def _make_group(*path):
      return mock.MagicMock(source_keys=[path], quantize_to_fp8=False)

    mock_cfg_main = mock.MagicMock()
    mock_cfg_main.logical_axis_rules = ()
    mock_cfg_main.float32_gate_logits = True
    mock_cfg_main.routed_bias = True
    mock_cfg_main.rollout_weight_dtype = jnp.dtype("bfloat16")
    direct_main = MainDirectConverter()
    engine_main = mock.MagicMock()
    engine_main.model = TunixMaxTextAdapter()
    engine_main._weight_converter._direct = direct_main
    mock_engine_mod = mock.MagicMock()
    mock_engine_mod.MaxTextTrainingEngine.return_value = engine_main

    with mock.patch.object(
        maxtext_utils,
        "maxtext_modules",
        return_value=(mock.MagicMock(), mock_engine_mod, mock.MagicMock()),
    ):
      maxtext_utils.create_maxtext_engine(
          mock_cfg_main,
          mock.MagicMock(),
          wrap_with_tunix_adapter=True,
          log_shapes=False,
      )

    # Strict no-op on upstream `main`: method is not wrapped on the instance,
    # and all modules (including `gate.bias` when `routed_bias=True`) stay BF16.
    self.assertNotIn("_execute_group_target_free", direct_main.__dict__)
    for path in (
        ("decoder", "layers", "mlp", "gate", "kernel"),
        ("decoder", "layers", "post_attention_layernorm", "scale"),
        ("decoder", "layers", "mlp", "gate", "bias"),
    ):
      out = direct_main._execute_group_target_free(_make_group(*path), {})
      self.assertEqual(out[0][1].dtype, jnp.bfloat16)
    self.assertEqual(
        direct_main.seen_target_dtypes,
        [jnp.dtype("bfloat16")] * 3,
    )

    # Warn if `_direct` exists without a callable `_execute_group_target_free`.
    engine_missing_hook = mock.MagicMock()
    engine_missing_hook.model = TunixMaxTextAdapter()
    engine_missing_hook._weight_converter._direct = object()
    mock_engine_mod.MaxTextTrainingEngine.return_value = engine_missing_hook
    with mock.patch.object(
        maxtext_utils,
        "maxtext_modules",
        return_value=(mock.MagicMock(), mock_engine_mod, mock.MagicMock()),
    ), self.assertLogs(level="WARNING") as warn_logs:
      maxtext_utils.create_maxtext_engine(
          mock_cfg_main,
          mock.MagicMock(),
          wrap_with_tunix_adapter=True,
          log_shapes=False,
      )
    self.assertIn(
        "no callable '_execute_group_target_free'", "\n".join(warn_logs.output)
    )


class _Group(NamedTuple):
  op: str
  source_keys: tuple[tuple[str, ...], ...]
  targets: tuple[tuple[str, ...], ...]
  local_index: int | None = None


_WO_GROUP = _Group(
    "slice", (("layers", "wo"),), (("layers_0", "wo"), ("layers_1", "wo"))
)
_WI_GROUP = _Group(
    "fuse_moe",
    (("layers", "wi_0"), ("layers", "wi_1")),
    (("layers_0", "wi"), ("layers_1", "wi")),
)
_SHARED_LOCAL_GROUPS = (
    _Group(
        "slice",
        (("layers", "local_layers", "wo"),),
        (("layers_0", "wo"), ("layers_2", "wo")),
        local_index=0,
    ),
    _Group(
        "slice",
        (("layers", "local_layers", "wo"),),
        (("layers_1", "wo"), ("layers_3", "wo")),
        local_index=1,
    ),
    _Group(
        "fuse_moe",
        (
            ("layers", "local_layers", "wi_0"),
            ("layers", "local_layers", "wi_1"),
        ),
        (("layers_0", "wi"), ("layers_2", "wi")),
        local_index=0,
    ),
    _Group(
        "fuse_moe",
        (
            ("layers", "local_layers", "wi_0"),
            ("layers", "local_layers", "wi_1"),
        ),
        (("layers_1", "wi"), ("layers_3", "wi")),
        local_index=1,
    ),
)


class _FakeConverter:
  """Stands in for MaxText's `MaxTextToMaxTextConverter`.

  `convert` plans with `_build_plan` and runs each group through
  `_execute_group`, the two hooks the loader wraps.
  """

  instances: list["_FakeConverter"] = []
  groups: tuple[_Group, ...] = (_WO_GROUP, _WI_GROUP)
  # A source array whose liveness is recorded when a fuse group runs.
  watched: jax.Array | None = None

  def __init__(self, **kwargs):
    self.kwargs = kwargs
    self.planned_sources = []
    self.watched_alive_during_fuse = None
    self.fuse_inputs = []
    _FakeConverter.instances.append(self)

  def _build_plan(self, src_flat, tgt_flat, skip_paths):
    del tgt_flat, skip_paths
    self.planned_sources = sorted(src_flat)
    return self.groups

  def _execute_group(self, group, src_flat, tgt_flat):
    del tgt_flat
    raw_sources = [src_flat[key] for key in group.source_keys]
    if group.op == "fuse_moe":
      self.fuse_inputs.append(tuple(raw_sources))
      if self.watched is not None:
        alive = not self.watched.is_deleted()
        self.watched_alive_during_fuse = (
            alive
            if self.watched_alive_during_fuse is None
            else self.watched_alive_during_fuse and alive
        )
    sources = (
        [arr[:, group.local_index] for arr in raw_sources]
        if group.local_index is not None
        else raw_sources
    )
    if group.op == "fuse_moe":
      fused = jnp.concatenate(sources, axis=-1)
      return [(key, fused[i]) for i, key in enumerate(group.targets)]
    if group.op == "raise":
      raise ValueError("conversion failed")
    return [(key, sources[0][i]) for i, key in enumerate(group.targets)]

  def convert(self, src_pytree, target_state):
    src_flat = traverse_util.flatten_dict(src_pytree.to_pure_dict())
    tgt_flat = traverse_util.flatten_dict(target_state.to_pure_dict())
    result = {}
    for group in self._build_plan(src_flat, tgt_flat, frozenset()):
      result.update(self._execute_group(group, dict(src_flat), tgt_flat))
    nested = traverse_util.unflatten_dict(result)
    return {"model": jax.tree_util.tree_map(nnx.Param, nested)}


class _ScannedLayers(nnx.Module):

  def __init__(self, prefused: bool, shape_prefix: tuple[int, ...] = (2,)):
    n = 2 if shape_prefix == (2,) else shape_prefix[0] * shape_prefix[1]
    if prefused:
      self.wi = nnx.Param(
          jnp.arange(float(n * 32)).reshape(*shape_prefix, 4, 8)
      )
    elif shape_prefix == (2,):
      self.wi_0 = nnx.Param(jnp.full((2, 4, 4), 1.0))
      self.wi_1 = nnx.Param(jnp.full((2, 4, 4), 2.0))
    else:
      self.wi_0 = nnx.Param(
          jnp.arange(float(n * 16)).reshape(*shape_prefix, 4, 4)
      )
      self.wi_1 = nnx.Param(
          (100.0 + jnp.arange(float(n * 16))).reshape(*shape_prefix, 4, 4)
      )
    self.wo = nnx.Param(jnp.arange(float(n * 16)).reshape(*shape_prefix, 4, 4))


class _InhomogeneousLayers(nnx.Module):

  def __init__(self, prefused: bool, num_slots: int = 2):
    self.local_layers = _ScannedLayers(prefused, shape_prefix=(2, num_slots))


class _ScannedModel(nnx.Module):

  def __init__(self, prefused: bool, num_slots: int | None = None):
    self.layers = (
        _ScannedLayers(prefused)
        if num_slots is None
        else _InhomogeneousLayers(prefused, num_slots)
    )


class _FakeTarget:
  """In-process vLLM weights with one non-parameter leaf."""

  def __init__(self, free_on_update: bool = False, num_layers: int = 2):
    layer = lambda: {
        "wi": nnx.Param(jnp.zeros((4, 8))),
        "wo": nnx.Param(jnp.zeros((4, 4))),
    }
    self.state = nnx.State({
        **{f"layers_{i}": layer() for i in range(num_layers)},
        "stat": nnx.BatchStat(jnp.zeros((3,))),
    })
    self.free_on_update = free_on_update
    self.calls = []

  @property
  def transformer_state(self):
    return self.state

  def delete_cache(self):
    self.calls.append("delete_cache")

  def update_params(self, updated_weights):
    self.calls.append("update_params")
    self.state = nnx.State({**updated_weights, "stat": self.state["stat"]})
    if self.free_on_update:
      jax.tree_util.tree_leaves(updated_weights["layers_0"]["wo"])[0].delete()


class LoadAndConvertScannedCheckpointTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    _FakeConverter.instances = []

  def _load(self, model, target, converter_cls=_FakeConverter, **kwargs):
    pyconfig = mock.MagicMock()
    pyconfig.initialize.return_value = mock.sentinel.scanned_cfg
    creation = mock.MagicMock()
    creation.from_pretrained.return_value = (model, mock.sentinel.mesh)
    modules = {
        "common.common_types": types.SimpleNamespace(
            MODEL_MODE_AUTOREGRESSIVE="autoregressive"
        ),
        "configs": types.SimpleNamespace(pyconfig=pyconfig),
        "integration.vllm.weight_converter": types.SimpleNamespace(
            MaxTextToMaxTextConverter=converter_cls
        ),
        "utils": types.SimpleNamespace(model_creation_utils=creation),
        "utils.globals": types.SimpleNamespace(
            MAXTEXT_CONFIGS_DIR="/maxtext/configs"
        ),
    }
    args = dict(
        mesh_tp=2,
        ckpt_prefuse_moe=False,
        fp8_moe=False,
        maxtext_config_overrides={"model_name": "qwen3-test"},
    )
    args.update(kwargs)
    with mock.patch.dict(
        "sys.modules",
        {
            prefix + name: module
            for prefix in ("maxtext.", "maxtext.src.maxtext.")
            for name, module in modules.items()
        },
    ):
      maxtext_utils.load_and_convert_scanned_checkpoint(
          "/ckpt/items", target, **args
      )
    return pyconfig, creation

  def test_loads_scanned_checkpoint_and_frees_sources(self):
    model = _ScannedModel(prefused=False)
    scanned_wo = nnx.state(model).to_pure_dict()["layers"]["wo"]
    target = _FakeTarget()
    placeholder = target.state.to_pure_dict()["layers_0"]["wo"]
    stat = target.state.to_pure_dict()["stat"]

    with mock.patch.object(jax, "clear_caches") as clear_caches:
      pyconfig, creation = self._load(model, target)

    clear_caches.assert_not_called()
    init_args, init_kwargs = pyconfig.initialize.call_args
    self.assertEqual(init_args, (["", "/maxtext/configs/base.yml"],))
    self.assertEqual(init_kwargs["model_name"], "qwen3-test")
    self.assertEqual(init_kwargs["load_parameters_path"], "/ckpt/items")
    self.assertTrue(init_kwargs["scan_layers"])
    self.assertFalse(init_kwargs["prefuse_moe_weights"])
    creation.from_pretrained.assert_called_once_with(
        mock.sentinel.scanned_cfg,
        devices=jax.devices(),
        model_mode="autoregressive",
    )
    (converter,) = _FakeConverter.instances
    self.assertEqual(
        converter.kwargs,
        dict(
            config=mock.sentinel.scanned_cfg,
            tp=2,
            prefuse_moe_weights=True,
            target_dtype=None,
            fp8_moe=False,
        ),
    )
    self.assertIn(("layers", "wi_0"), converter.planned_sources)
    self.assertEqual(target.calls, ["delete_cache", "update_params"])
    self.assertTrue(placeholder.is_deleted())
    self.assertTrue(scanned_wo.is_deleted())
    self.assertFalse(stat.is_deleted())
    loaded = target.state.to_pure_dict()
    self.assertEqual(loaded["layers_1"]["wo"][0, 0], 16.0)
    self.assertEqual(loaded["layers_0"]["wi"][0, 7], 2.0)

  def test_prefused_moe_wi_stays_alive_until_its_group_ran(self):
    model = _ScannedModel(prefused=True)
    prefused_wi = nnx.state(model).to_pure_dict()["layers"]["wi"]
    target = _FakeTarget()
    with mock.patch.object(_FakeConverter, "watched", prefused_wi):
      self._load(model, target, ckpt_prefuse_moe=True)

    (converter,) = _FakeConverter.instances
    self.assertNotIn(("layers", "wi"), converter.planned_sources)
    self.assertTrue(converter.watched_alive_during_fuse)
    self.assertTrue(prefused_wi.is_deleted())
    loaded = target.state.to_pure_dict()
    # Halves [0:4] and [4:8] of layer 1 are re-fused in order.
    self.assertEqual(loaded["layers_1"]["wi"][0, 0], 32.0)
    self.assertEqual(loaded["layers_1"]["wi"][0, 4], 36.0)

  def test_shared_local_layers_sources_stay_alive_across_local_indices(self):
    model = _ScannedModel(prefused=False, num_slots=2)
    local = nnx.state(model).to_pure_dict()["layers"]["local_layers"]
    scanned_wo = local["wo"]
    scanned_wi_0, scanned_wi_1 = local["wi_0"], local["wi_1"]
    expected_wo = jax.device_get(scanned_wo)
    expected_wi = np.concatenate(
        [jax.device_get(scanned_wi_0), jax.device_get(scanned_wi_1)], axis=-1
    )
    target = _FakeTarget(num_layers=4)

    with (
        mock.patch.object(_FakeConverter, "groups", _SHARED_LOCAL_GROUPS),
        mock.patch.object(_FakeConverter, "watched", scanned_wi_0),
    ):
      self._load(model, target, ckpt_prefuse_moe=False)

    (converter,) = _FakeConverter.instances
    self.assertTrue(converter.watched_alive_during_fuse)
    self.assertTrue(scanned_wo.is_deleted())
    self.assertTrue(scanned_wi_0.is_deleted())
    self.assertTrue(scanned_wi_1.is_deleted())
    loaded = target.state.to_pure_dict()
    for block in range(2):
      for slot in range(2):
        layer = loaded[f"layers_{block * 2 + slot}"]
        np.testing.assert_array_equal(layer["wo"], expected_wo[block, slot])
        np.testing.assert_array_equal(layer["wi"], expected_wi[block, slot])

  def test_prefused_moe_wi_shared_across_local_indices(self):
    model = _ScannedModel(prefused=True, num_slots=2)
    local = nnx.state(model).to_pure_dict()["layers"]["local_layers"]
    prefused_wi, scanned_wo = local["wi"], local["wo"]
    expected_wi = jax.device_get(prefused_wi)
    expected_wo = jax.device_get(scanned_wo)
    target = _FakeTarget(num_layers=4)

    with (
        mock.patch.object(_FakeConverter, "groups", _SHARED_LOCAL_GROUPS),
        mock.patch.object(_FakeConverter, "watched", prefused_wi),
    ):
      self._load(model, target, ckpt_prefuse_moe=True)

    (converter,) = _FakeConverter.instances
    self.assertNotIn(
        ("layers", "local_layers", "wi"), converter.planned_sources
    )
    self.assertTrue(converter.watched_alive_during_fuse)
    self.assertLen(converter.fuse_inputs, 2)
    self.assertTrue(prefused_wi.is_deleted())
    self.assertTrue(scanned_wo.is_deleted())
    for sliced_pair in converter.fuse_inputs:
      for sliced_half in sliced_pair:
        self.assertTrue(sliced_half.is_deleted())
    loaded = target.state.to_pure_dict()
    for block in range(2):
      for slot in range(2):
        layer = loaded[f"layers_{block * 2 + slot}"]
        np.testing.assert_array_equal(layer["wo"], expected_wo[block, slot])
        np.testing.assert_array_equal(layer["wi"], expected_wi[block, slot])

  def test_uses_native_free_sources_when_converter_supports_it(self):
    class _NativeFreeConverter(_FakeConverter):

      def convert(self, src_pytree, target_state=None, *, free_sources=False):
        self.free_sources = free_sources
        self.hooks_unpatched = (
            "_build_plan" not in self.__dict__
            and "_execute_group" not in self.__dict__
        )
        out = super().convert(src_pytree, target_state)
        if free_sources:
          for leaf in jax.tree_util.tree_leaves(src_pytree):
            if isinstance(leaf, jax.Array) and not leaf.is_deleted():
              leaf.delete()
        return out

    model = _ScannedModel(prefused=True)
    prefused_wi = nnx.state(model).to_pure_dict()["layers"]["wi"]
    target = _FakeTarget()
    native_wi_group = _Group(
        "slice", (("layers", "wi"),), (("layers_0", "wi"), ("layers_1", "wi"))
    )
    with mock.patch.object(
        _NativeFreeConverter, "groups", (_WO_GROUP, native_wi_group)
    ):
      self._load(
          model,
          target,
          converter_cls=_NativeFreeConverter,
          ckpt_prefuse_moe=True,
      )

    (converter,) = _FakeConverter.instances
    self.assertTrue(converter.free_sources)
    self.assertTrue(converter.hooks_unpatched)
    self.assertIn(("layers", "wi"), converter.planned_sources)
    self.assertTrue(prefused_wi.is_deleted())

  def test_fp8_moe_reaches_the_converter(self):
    self._load(_ScannedModel(prefused=False), _FakeTarget(), fp8_moe=True)
    self.assertTrue(_FakeConverter.instances[0].kwargs["fp8_moe"])

  def test_freed_vllm_weight_fails_the_load(self):
    target = _FakeTarget(free_on_update=True)
    with self.assertRaisesRegex(RuntimeError, "1 vLLM weight"):
      self._load(_ScannedModel(prefused=False), target)

  def test_converter_hooks_are_restored_when_conversion_fails(self):
    target = _FakeTarget()
    with mock.patch.object(
        _FakeConverter,
        "groups",
        (_Group("raise", (("layers", "wo"),), (("layers_0", "wo"),)),),
    ):
      with self.assertRaisesRegex(ValueError, "conversion failed"):
        self._load(_ScannedModel(prefused=True), target, ckpt_prefuse_moe=True)
    (converter,) = _FakeConverter.instances
    self.assertIs(
        converter._execute_group.__func__, _FakeConverter._execute_group
    )
    self.assertIs(converter._build_plan.__func__, _FakeConverter._build_plan)
    self.assertEqual(target.calls, ["delete_cache"])


if __name__ == "__main__":
  absltest.main()

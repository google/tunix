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

"""Unit tests for TorchTitan configuration validation."""

from __future__ import annotations

from absl.testing import absltest
from tunix.dist_rl.launch.common import resolve_config
from tunix.experimental.torchtitan import config_validation


class ConfigValidationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.base_overrides = [
        "execution_targets.trainer=torch:tpu",
        "actor_model_config.dtype=bfloat16",
        "actor_model_config.mesh.shape=(8,1)",
        "actor_model_config.mesh.axis_names=('fsdp','tp')",
        "weight_sync_config.mode=basic",
        "batch_size=4",
        "rl_training_config.mini_batch_size=2",
        "rl_training_config.train_micro_batch_size=8",
    ]

  def test_valid_config_passes(self):
    hp = resolve_config.initialize(self.base_overrides)
    config_validation.validate_torchtitan_config(hp)

  def test_reject_execution_targets_mismatch(self):
    # execution_targets.trainer='torch:tpu' rejects
    # actor_model_config.model_source='maxtext'.
    hp_maxtext = resolve_config.initialize([
        "execution_targets.trainer=torch:tpu",
        "actor_model_config.model_source=maxtext",
        "actor_model_config.model_path=/tmp/ckpt",
        "actor_model_config.kwargs.maxtext_model_name=qwen3-1.7b",
    ])
    with self.assertRaisesRegex(
        ValueError,
        r"actor_model_config\.model_source != 'maxtext'",
    ):
      config_validation.validate_torchtitan_config(hp_maxtext)

    # execution_targets.trainer='jax:tpu' rejected by validate_torchtitan_config
    hp_jax = resolve_config.initialize([])
    with self.assertRaisesRegex(
        ValueError,
        "requires execution_targets.trainer='torch:tpu'",
    ):
      config_validation.validate_torchtitan_config(hp_jax)

  def test_reject_non_1d_fsdp_mesh(self):
    with self.assertRaisesRegex(ValueError, "requires 1D FSDP"):
      hp = resolve_config.initialize([
          *self.base_overrides,
          "actor_model_config.mesh.shape=(4,2)",
          "actor_model_config.mesh.axis_names=('fsdp','tp')",
      ])
      config_validation.validate_torchtitan_config(hp)

  def test_reject_sequence_packing(self):
    with self.assertRaisesRegex(
        ValueError, "does not support sequence packing"
    ):
      hp = resolve_config.initialize([
          *self.base_overrides,
          "rl_training_config.max_seq_tokens_per_device=1024",
      ])
      config_validation.validate_torchtitan_config(hp)

  def test_reject_non_divisible_micro_batch_size(self):
    with self.assertRaisesRegex(ValueError, "must be divisible by"):
      hp = resolve_config.initialize([
          *self.base_overrides,
          "rl_training_config.train_micro_batch_size=6",
      ])
      config_validation.validate_torchtitan_config(hp)

  def test_reject_unsupported_actor_dtype(self):
    with self.assertRaisesRegex(
        ValueError, r"must be one of \('bfloat16', 'float32'\)"
    ):
      hp = resolve_config.initialize([
          *self.base_overrides,
          "actor_model_config.dtype=int32",
      ])
      config_validation.validate_torchtitan_config(hp)

  def test_weight_sync_modes(self):
    for mode in ("basic", "none", "noop"):
      hp = resolve_config.initialize([
          *self.base_overrides,
          f"weight_sync_config.mode={mode}",
      ])
      config_validation.validate_torchtitan_config(hp)

    # mode='fs' is accepted by resolve_config but rejected by
    # validate_torchtitan_config.
    with self.assertRaisesRegex(
        ValueError, r"requires weight_sync_config\.mode to be one of"
    ):
      hp = resolve_config.initialize([
          *self.base_overrides,
          "weight_sync_config.mode=fs",
      ])
      config_validation.validate_torchtitan_config(hp)

    # mode='raiden' with torch:tpu trainer is rejected during resolve_config
    with self.assertRaisesRegex(
        ValueError,
        r"weight_sync_mode='raiden' requires both trainer and rollout"
        r" execution targets to be 'jax:tpu'",
    ):
      resolve_config.initialize([
          *self.base_overrides,
          "weight_sync_config.mode=raiden",
      ])

  def test_allow_fp32_master_with_bf16_export_and_vllm_under_basic(self):
    hp = resolve_config.initialize([
        *self.base_overrides,
        "actor_model_config.load_dtype=float32",
        "actor_model_config.dtype=bfloat16",
        "vllm_config.dtype=bfloat16",
        "weight_sync_config.mode=basic",
    ])
    config_validation.validate_torchtitan_config(hp)

  def test_reject_vllm_dtype_mismatch_under_basic(self):
    for actor_export_dtype, vllm_dtype in (
        ("float32", "bfloat16"),
        ("bfloat16", "auto"),
    ):
      with self.assertRaisesRegex(
          ValueError, r"export dtype.*must match explicit vllm_config\.dtype"
      ):
        hp = resolve_config.initialize([
            *self.base_overrides,
            "actor_model_config.load_dtype=float32",
            f"actor_model_config.dtype={actor_export_dtype}",
            f"vllm_config.dtype={vllm_dtype}",
            "weight_sync_config.mode=basic",
        ])
        config_validation.validate_torchtitan_config(hp)

  def test_reject_non_empty_remat_config(self):
    hp_valid = resolve_config.initialize([
        *self.base_overrides,
        "actor_model_config.remat_config=none",
    ])
    config_validation.validate_torchtitan_config(hp_valid)

    for unsupported_remat in ("decoder", "block"):
      with self.assertRaisesRegex(ValueError, "remat_config"):
        hp = resolve_config.initialize([
            *self.base_overrides,
            f"actor_model_config.remat_config={unsupported_remat}",
        ])
        config_validation.validate_torchtitan_config(hp)

  def test_optimizer_validation(self):
    with self.assertRaisesRegex(
        ValueError, "TorchTitan trainer only supports AdamW optimizer"
    ):
      hp = resolve_config.initialize([
          *self.base_overrides,
          "rl_training_config.actor_optimizer_config.opt_type=sgd",
      ])
      config_validation.validate_torchtitan_config(hp)

    with self.assertRaisesRegex(
        ValueError, "Unsupported schedule_type for TorchTitan: bad_schedule"
    ):
      hp = resolve_config.initialize([
          *self.base_overrides,
          "rl_training_config.actor_optimizer_config.schedule_type=bad_schedule",
      ])
      config_validation.validate_torchtitan_config(hp)

    with self.assertRaisesRegex(
        ValueError, "Unsupported opt_chain_type for TorchTitan: bad_chain"
    ):
      hp = resolve_config.initialize([
          *self.base_overrides,
          "rl_training_config.actor_optimizer_config.opt_chain_type=bad_chain",
      ])
      config_validation.validate_torchtitan_config(hp)

    with self.assertRaisesRegex(
        ValueError, "clip_by_global_norm requires chain_kwargs.max_norm > 0"
    ):
      hp = resolve_config.initialize([
          *self.base_overrides,
          "rl_training_config.actor_optimizer_config.chain_kwargs.max_norm=0",
      ])
      config_validation.validate_torchtitan_config(hp)

    sched_override = (
        "rl_training_config.actor_optimizer_config.schedule_type="
        "warmup_cosine_decay_schedule"
    )
    with self.assertRaisesRegex(
        ValueError, "warmup_cosine_decay_schedule requires"
    ):
      hp = resolve_config.initialize([
          *self.base_overrides,
          sched_override,
          "rl_training_config.actor_optimizer_config.decay_steps=0",
      ])
      config_validation.validate_torchtitan_config(hp)

    with self.assertRaisesRegex(
        ValueError, "warmup_cosine_decay_schedule requires non-null init_value"
    ):
      hp = resolve_config.initialize([
          *self.base_overrides,
          sched_override,
          "rl_training_config.actor_optimizer_config.init_value=null",
          "rl_training_config.actor_optimizer_config.decay_steps=100",
      ])
      config_validation.validate_torchtitan_config(hp)

    with self.assertRaisesRegex(
        ValueError, "warmup_cosine_decay_schedule requires warmup_steps >= 0"
    ):
      hp = resolve_config.initialize([
          *self.base_overrides,
          sched_override,
          "rl_training_config.actor_optimizer_config.init_value=1e-6",
          "rl_training_config.actor_optimizer_config.end_value=1e-7",
          "rl_training_config.actor_optimizer_config.decay_steps=100",
          "rl_training_config.actor_optimizer_config.warmup_steps=-1",
      ])
      config_validation.validate_torchtitan_config(hp)

  def test_reject_fused_agreement_in_loss(self):
    with self.assertRaisesRegex(
        ValueError, "fused in-loss sampler-trainer agreement"
    ):
      hp = resolve_config.initialize([
          *self.base_overrides,
          "agentic_grpo_config.num_iterations=1",
          "batch_size=4",
          "rl_training_config.mini_batch_size=4",
      ])
      config_validation.validate_torchtitan_config(hp)


if __name__ == "__main__":
  absltest.main()

import copy

from absl.testing import absltest
from absl.testing import parameterized
import optax
from tunix.common import configs
from tunix.common import datatypes


class CacheGeometryTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name="zero_num_kv_heads", field="num_kv_heads", value=0),
      dict(testcase_name="zero_head_dim", field="head_dim", value=0),
      dict(testcase_name="zero_window_size", field="window_size", value=0),
      dict(
          testcase_name="negative_window_size", field="window_size", value=-8
      ),
  )
  def test_non_positive_field_raises(self, field, value):
    kwargs = dict(num_kv_heads=2, head_dim=16, window_size=8)
    kwargs[field] = value
    with self.assertRaisesRegex(ValueError, rf"{field} must be positive"):
      datatypes.CacheGeometry(**kwargs)


class ConfigsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.actor_optimizer = optax.adam(learning_rate=0.001)

  def test_valid_rl_training_config(self):
    # Should not raise any errors
    config = configs.RLTrainingConfig(
        eval_every_n_steps=100,
        actor_optimizer=self.actor_optimizer,
        mini_batch_size=16,
        train_micro_batch_size=4,
        rollout_micro_batch_size=8,
    )
    self.assertEqual(config.gradient_accumulation_steps, 4)

  def test_batch_size_config(self):
    cfg = configs.RLTrainingConfig(
        actor_optimizer=self.actor_optimizer,
        critic_optimizer=None,
        mini_batch_size=8,
        train_micro_batch_size=4,
        eval_every_n_steps=1,
    )
    self.assertEqual(cfg.gradient_accumulation_steps, 2)

    cfg = configs.RLTrainingConfig(
        actor_optimizer=self.actor_optimizer,
        eval_every_n_steps=1,
    )
    self.assertIsNone(cfg.gradient_accumulation_steps)

    for mini_batch_size, train_micro_batch_size in zip(
        [8, -8, None], [3, 4, 4]
    ):
      with self.assertRaises(ValueError):
        configs.RLTrainingConfig(
            actor_optimizer=self.actor_optimizer,
            critic_optimizer=None,
            mini_batch_size=mini_batch_size,
            train_micro_batch_size=train_micro_batch_size,
            eval_every_n_steps=1,
        )

  def test_is_positive_integer_validation(self):
    # Test negative integer
    with self.assertRaisesRegex(
        ValueError, "mini_batch_size must be a positive integer. Got: -1"
    ):
      configs.RLTrainingConfig(
          eval_every_n_steps=100,
          actor_optimizer=self.actor_optimizer,
          mini_batch_size=-1,
      )

    # Test float
    with self.assertRaisesRegex(
        ValueError,
        "train_micro_batch_size must be a positive integer. Got: 4.5",
    ):
      configs.RLTrainingConfig(
          eval_every_n_steps=100,
          actor_optimizer=self.actor_optimizer,
          train_micro_batch_size=4.5,
      )

    # Test zero
    with self.assertRaisesRegex(
        ValueError,
        "rollout_micro_batch_size must be a positive integer. Got: 0",
    ):
      configs.RLTrainingConfig(
          eval_every_n_steps=100,
          actor_optimizer=self.actor_optimizer,
          rollout_micro_batch_size=0,
      )

  def test_check_divisibility_validation(self):
    with self.assertRaisesRegex(
        ValueError,
        "self.mini_batch_size=10 must be a multiple of"
        " self.train_micro_batch_size=3.",
    ):
      configs.RLTrainingConfig(
          eval_every_n_steps=100,
          actor_optimizer=self.actor_optimizer,
          mini_batch_size=10,
          train_micro_batch_size=3,
      )

  def test_train_micro_batch_size_requires_mini_batch_size(self):
    with self.assertRaisesRegex(
        ValueError,
        "`mini_batch_size` must be set when `train_micro_batch_size` is set.",
    ):
      configs.RLTrainingConfig(
          eval_every_n_steps=100,
          actor_optimizer=self.actor_optimizer,
          train_micro_batch_size=4,
      )

  def test_gradient_accumulation_steps_must_be_none(self):
    with self.assertRaisesRegex(
        ValueError, "gradient_accumulation_steps should be None"
    ):
      configs.RLTrainingConfig(
          actor_optimizer=self.actor_optimizer,
          eval_every_n_steps=1,
          gradient_accumulation_steps=4,
      )

  def test_sequence_packing_skips_train_micro_batch_size_divisibility(self):
    cfg = configs.RLTrainingConfig(
        eval_every_n_steps=1,
        actor_optimizer=self.actor_optimizer,
        mini_batch_size=6,
        train_micro_batch_size=4,
        max_seq_token_per_tpu=4096,
    )
    self.assertTrue(cfg.sequence_packing_enabled)
    self.assertIsNone(cfg.gradient_accumulation_steps)

  def test_is_sequence_packing_enabled(self):
    cfg_disabled = configs.TrainingConfig(eval_every_n_steps=1)
    self.assertFalse(cfg_disabled.sequence_packing_enabled)
    self.assertFalse(configs.is_sequence_packing_enabled(cfg_disabled))
    self.assertFalse(configs.is_sequence_packing_enabled(object()))

    cfg_zero = configs.TrainingConfig(
        eval_every_n_steps=1, max_seq_tokens_per_device=0
    )
    self.assertFalse(cfg_zero.sequence_packing_enabled)
    self.assertFalse(configs.is_sequence_packing_enabled(cfg_zero))
    self.assertIsNone(cfg_zero.max_seq_tokens_per_device)
    self.assertIsNone(cfg_zero.max_seq_token_per_tpu)

    cfg_device_with_zero_tpu = configs.TrainingConfig(
        eval_every_n_steps=1,
        max_seq_tokens_per_device=2048,
        max_seq_token_per_tpu=0,
    )
    self.assertEqual(cfg_device_with_zero_tpu.max_seq_tokens_per_device, 2048)
    self.assertEqual(cfg_device_with_zero_tpu.max_seq_token_per_tpu, 2048)

    cfg_device = configs.RLTrainingConfig(
        eval_every_n_steps=1,
        actor_optimizer=self.actor_optimizer,
        mini_batch_size=6,
        train_micro_batch_size=4,
        max_seq_tokens_per_device=2048,
    )
    self.assertTrue(cfg_device.sequence_packing_enabled)
    self.assertEqual(cfg_device.max_seq_token_per_tpu, 2048)
    self.assertNotIn(
        "__class__", configs.RLTrainingConfig.__post_init__.__code__.co_freevars
    )
    cfg_copied = copy.deepcopy(cfg_device)
    self.assertEqual(cfg_copied.max_seq_tokens_per_device, 2048)
    self.assertEqual(cfg_copied.max_seq_token_per_tpu, 2048)

    with self.assertRaisesRegex(ValueError, "non-negative integer or None"):
      configs.TrainingConfig(
          eval_every_n_steps=1,
          max_seq_tokens_per_device=-1,
      )

    with self.assertRaisesRegex(ValueError, "must match when both are set"):
      configs.TrainingConfig(
          eval_every_n_steps=1,
          max_seq_tokens_per_device=2048,
          max_seq_token_per_tpu=4096,
      )

  def test_rollout_config_num_logprobs_validation(self):
    self.assertEqual(configs.RolloutConfig().num_logprobs, 1)
    self.assertEqual(configs.RolloutConfig(num_logprobs=32).num_logprobs, 32)
    with self.assertRaisesRegex(ValueError, "num_logprobs must be >= 1"):
      configs.RolloutConfig(num_logprobs=0)

  def test_rl_profile_config_defaults_and_validation(self):
    cfg = configs.RLProfileConfig()
    self.assertEqual(cfg.start_invocation, 2)
    self.assertEqual(cfg.num_invocations, 1)
    self.assertEqual(cfg.timeout_secs, 60.0)
    self.assertFalse(cfg.enable_continuous_profiling)

    cfg_custom = configs.RLProfileConfig(
        timeout_secs=45.0,
        enable_continuous_profiling=True,
    )
    self.assertEqual(cfg_custom.timeout_secs, 45.0)
    self.assertTrue(cfg_custom.enable_continuous_profiling)

    with self.assertRaisesRegex(ValueError, "start_invocation must be >= 0"):
      configs.RLProfileConfig(start_invocation=-1)

    with self.assertRaisesRegex(ValueError, "num_invocations must be > 0"):
      configs.RLProfileConfig(num_invocations=0)

    with self.assertRaisesRegex(ValueError, "num_invocations must be > 0"):
      configs.RLProfileConfig(num_invocations=-2)

    with self.assertRaisesRegex(ValueError, "timeout_secs must be > 0"):
      configs.RLProfileConfig(timeout_secs=0)

    with self.assertRaisesRegex(ValueError, "timeout_secs must be > 0"):
      configs.RLProfileConfig(timeout_secs=-5.0)

    with self.assertRaisesRegex(ValueError, "timeout_secs must be > 0"):
      configs.RLProfileConfig(timeout_secs=float("nan"))

  def test_rollout_config_validation(self):
    cfg = configs.RolloutConfig(
        max_response_length=256,
        episode_timeout=120,
        weight_sync_mode="raiden",
    )
    self.assertEqual(cfg.max_response_length, 256)
    self.assertEqual(cfg.episode_timeout, 120.0)
    self.assertIsInstance(cfg.episode_timeout, float)
    self.assertEqual(cfg.weight_sync_mode, "raiden")

    # Positional arguments should be rejected (kw_only=True).
    with self.assertRaises(TypeError):
      configs.RolloutConfig(64)  # pytype: disable=wrong-arg-count

    for bad_max_resp in (0, -1, True, 1.5):
      with self.assertRaisesRegex(
          ValueError, "max_response_length must be a positive integer"
      ):
        configs.RolloutConfig(max_response_length=bad_max_resp)  # pytype: disable=wrong-arg-types

    for bad_timeout in (
        0,
        -1.0,
        True,
        False,
        float("nan"),
        float("inf"),
        "60",
    ):
      with self.assertRaisesRegex(
          ValueError, "episode_timeout must be a positive finite number"
      ):
        configs.RolloutConfig(episode_timeout=bad_timeout)  # pytype: disable=wrong-arg-types

    for valid_mode in (
        "none",
        "noop",
        "raiden",
        "in_memory",
        "fs",
        "nccl",
    ):
      self.assertEqual(
          configs.RolloutConfig(weight_sync_mode=valid_mode).weight_sync_mode,
          valid_mode,
      )

    for bad_mode in ("fallback", "raden", "RAIDEN", "invalid", ""):
      with self.assertRaisesRegex(
          ValueError, "weight_sync_mode must be one of"
      ):
        configs.RolloutConfig(weight_sync_mode=bad_mode)

  def test_rollout_config_validates_sampling_settings(self):
    configs.RolloutConfig(temperature=0.0, top_p=None, top_k=-1)
    configs.RolloutConfig(temperature=1.5, top_p=1.0, top_k=0)
    for bad_temperature in (-0.1, float("nan"), float("inf")):
      with self.assertRaisesRegex(ValueError, "temperature must be"):
        configs.RolloutConfig(temperature=bad_temperature)
    for bad_top_p in (0.0, -0.5, 1.1):
      with self.assertRaisesRegex(ValueError, "top_p must be"):
        configs.RolloutConfig(top_p=bad_top_p)
    with self.assertRaisesRegex(ValueError, "top_k must be"):
      configs.RolloutConfig(top_k=-2)


if __name__ == "__main__":
  absltest.main()

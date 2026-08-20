from absl.testing import absltest
import optax
from tunix.common import configs


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


if __name__ == "__main__":
  absltest.main()

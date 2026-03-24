import sys

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from tunix.rl.rollout import base_rollout
from tunix.rl.rollout import mock_rollout


class MockRolloutTest(absltest.TestCase):

  def test_generate(self):
    m = mock_rollout.MockRollout(
        min_time=0.01,
        max_time=0.02,
        vocab_size=100,
        pad_id=0,
        eos_id=1,
    )

    rc = base_rollout.RolloutConfig(
        max_prompt_length=10, max_tokens_to_generate=15
    )

    out = m.generate(["prompt 1", "prompt 2"], rollout_config=rc)

    self.assertLen(out.text, 2)
    self.assertLen(out.logits, 2)
    self.assertLen(out.tokens, 2)
    self.assertEqual(out.left_padded_prompt_tokens.shape, (2, 10))


if __name__ == "__main__":
  absltest.main()

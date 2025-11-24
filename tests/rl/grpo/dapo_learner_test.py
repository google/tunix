# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax.numpy as jnp
from tunix.rl import function_registry as fr
from tunix.rl.grpo import dapo_learner as dapo_lib
from tunix.rl.grpo import grpo_learner as grpo_lib
from tunix.tests import test_common as tc


class DAPOlearnerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_model = mock.MagicMock()
    self.pad_id = 0
    self.eos_id = 1

    # Common data shapes
    self.batch_size = 2
    self.seq_len = 4
    self.prompt_ids = jnp.zeros(
        (self.batch_size, self.seq_len), dtype=jnp.int32
    )
    self.completion_ids = jnp.ones(
        (self.batch_size, self.seq_len), dtype=jnp.int32
    )
    self.completion_mask = jnp.array(
        [[1, 1, 1, 0], [1, 1, 0, 0]], dtype=jnp.float32
    )
    self.advantages = jnp.array([0.5, -0.2], dtype=jnp.float32)
    self.ref_per_token_logps = (
        jnp.ones_like(self.completion_ids, dtype=jnp.float32) * -0.2
    )
    self.old_per_token_logps = (
        jnp.ones_like(self.completion_ids, dtype=jnp.float32) * -0.15
    )

  def create_train_example(self):
    example = mock.MagicMock()
    example.prompt_ids = self.prompt_ids
    example.completion_ids = self.completion_ids
    example.completion_mask = self.completion_mask
    example.advantages = self.advantages
    example.ref_per_token_logps = self.ref_per_token_logps
    example.old_per_token_logps = self.old_per_token_logps
    return example

  def test_diff_loss(self):
    dapo_config = dapo_lib.DAPOConfig()
    grpo_config = grpo_lib.GRPOConfig()

    dapo_loss_fn_impl = fr.default_registry.get(
        "policy_loss_fn", dapo_config.policy_loss_fn
    )
    grpo_loss_fn_impl = fr.default_registry.get(
        "policy_loss_fn", grpo_config.policy_loss_fn
    )

    # Test that the functions is same
    self.assertEqual(dapo_loss_fn_impl, grpo_loss_fn_impl)

    # Create the same input for both functions
    train_example = self.create_train_example()
    pad_id = self.pad_id
    eos_id = self.eos_id
    vocab = tc.MockVocab()
    model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(0),
    )

    # Call DAPO loss function
    dapo_loss, dapo_aux = dapo_loss_fn_impl(
        model, train_example, dapo_config, pad_id, eos_id
    )

    # Call GRPO loss function
    grpo_loss, grpo_aux = grpo_loss_fn_impl(
        model, train_example, grpo_config, pad_id, eos_id
    )

    # Assert that the loss values are different
    self.assertNotEqual(
        dapo_loss.item(),
        grpo_loss.item(),
        msg=(
            "DAPO and GRPO loss values should be different for the same input"
            " due to different configurations and potentially different"
            " logic."
        ),
    )

    self.assertIn("kl", dapo_aux)
    self.assertIn("kl", grpo_aux)
    self.assertNotEqual(
        dapo_aux["kl"], grpo_aux["kl"]
    )  # Expected as beta differs


if __name__ == "__main__":
  absltest.main()

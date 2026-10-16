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

"""Unit tests for BEFT (Bias-Elevation Fine-Tuning)."""

import os
import tempfile

from absl.testing import absltest
from flax import nnx
from flax.nnx import filterlib
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tunix.generate import sampler
from tunix.sft import checkpoint_manager
from tunix.sft import peft_trainer
from tunix.sft import utils
from tunix.sft.peft.beft import apply_beft_to_model
from tunix.sft.peft.beft import BEFTConfig
from tunix.sft.peft.beft import BEFTLinear
from tunix.sft.peft.beft import BEFTParam
from tunix.sft.peft.beft import unwrap_beft_from_model
from tunix.tests import test_common as tc

# CPU environment setup to simulate multi device env.
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'
jax.config.update('jax_default_matmul_precision', 'highest')


def dummy_gen_model_input_fn(x: peft_trainer.TrainingInput):
  return {
      'input_tokens': x.input_tokens,
      'input_mask': x.input_mask,
      'positions': jnp.arange(x.input_tokens.shape[1]),
      'attention_mask': jnp.ones_like(x.input_tokens),
  }


def dummy_datasets(batch_size: int, repeat: int = 1):
  dummy_input = np.arange(128).reshape((-1, batch_size, 16))
  return [
      peft_trainer.TrainingInput(
          input_tokens=x, input_mask=jnp.ones(x.shape, dtype=jnp.int32)
      )
      for x in dummy_input
  ] * repeat


class BeftTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    try:
      self.temp_path = self.create_tempdir().full_path
    except Exception:
      self.temp_path = tempfile.TemporaryDirectory().name

    self.train_ds = dummy_datasets(batch_size=4, repeat=2)
    self.eval_ds = dummy_datasets(batch_size=4, repeat=1)

  def test_beft_linear_forward_and_properties(self):
    rngs = nnx.Rngs(0)
    base_linear = nnx.Linear(
        in_features=8,
        out_features=16,
        use_bias=False,
        rngs=rngs,
    )
    beft_linear = BEFTLinear(base_linear)

    # Check attribute delegation
    self.assertEqual(beft_linear.in_features, 8)
    self.assertEqual(beft_linear.out_features, 16)
    self.assertIsInstance(beft_linear.bias, BEFTParam)
    self.assertEqual(beft_linear.bias.value.shape, (16,))

    # Check forward pass
    x = jnp.ones((2, 8))
    base_out = base_linear(x)
    beft_out = beft_linear(x)
    np.testing.assert_allclose(base_out, beft_out, atol=1e-5)

    # Mutate bias and check that output shifts by bias
    beft_linear.bias.value = beft_linear.bias.value + 2.0
    shifted_out = beft_linear(x)
    np.testing.assert_allclose(shifted_out, base_out + 2.0, atol=1e-5)

  def test_beft_merge(self):
    rngs = nnx.Rngs(0)
    base_linear = nnx.Linear(
        in_features=8,
        out_features=16,
        use_bias=True,
        rngs=rngs,
    )
    beft_linear = BEFTLinear(base_linear)
    beft_linear.bias.value = jnp.full((16,), 3.0)
    original_bias = jnp.copy(base_linear.bias.value)

    beft_linear.merge()
    np.testing.assert_allclose(
        base_linear.bias.value, original_bias + 3.0, atol=1e-5
    )

  def test_apply_beft_to_model_and_unwrap(self):
    rngs = nnx.Rngs(0)
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=rngs)

    self.assertFalse(utils.is_beft_enabled(model))
    self.assertFalse(utils.is_peft_enabled(model))
    self.assertIsNone(utils.get_peft_param_type(model))

    apply_beft_to_model(model, module_path=r".*w1|.*w2")

    self.assertTrue(utils.is_beft_enabled(model))
    self.assertTrue(utils.is_peft_enabled(model))
    self.assertEqual(utils.get_peft_param_type(model), BEFTParam)

    # Test forward pass with BEFT applied
    dummy_input = tc.get_dummy_inputs_for_lora_toy_transformer_tests()
    logits, _ = model(**dummy_input)
    self.assertEqual(logits.shape[-1], 256)

    # Unwrap model
    unwrap_beft_from_model(model)
    self.assertFalse(utils.is_beft_enabled(model))

  def test_peft_trainer_with_beft(self):
    rngs = nnx.Rngs(0)
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=rngs)
    apply_beft_to_model(model, module_path=r".*w1|.*w2")

    optimizer = optax.inject_hyperparams(optax.adamw)(learning_rate=1e-3)
    config = peft_trainer.TrainingConfig(
        eval_every_n_steps=2,
        max_steps=10,
        checkpoint_root_directory=f"{self.temp_path}/beft_test/checkpoints",
    )

    original_base_params = jax.tree.map(
        jnp.copy, nnx.state(model, filterlib.Not(BEFTParam))
    )
    original_beft_params = jax.tree.map(
        jnp.copy, nnx.state(model, BEFTParam)
    )

    trainer = peft_trainer.PeftTrainer(model, optimizer, config)
    trainer = trainer.with_gen_model_input_fn(dummy_gen_model_input_fn)
    trainer.train(self.train_ds, self.eval_ds, cache_nnx_graph=True)

    trained_base_params = nnx.state(model, filterlib.Not(BEFTParam))
    trained_beft_params = nnx.state(model, BEFTParam)

    # 1. Base weights MUST remain strictly UNCHANGED
    jax.tree.map_with_path(
        tc.assert_equal, original_base_params, trained_base_params
    )

    # 2. BEFT trainable parameters MUST be UPDATED
    jax.tree.map_with_path(
        tc.assert_not_equal, original_beft_params, trained_beft_params
    )

  def test_beft_checkpoint_save_and_restore(self):
    rngs = nnx.Rngs(0)
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=rngs)
    apply_beft_to_model(model, module_path=r".*w1|.*w2")

    optimizer = nnx.Optimizer(
        model, optax.adamw(learning_rate=1e-3), wrt=BEFTParam
    )
    ckpt_dir = f"{self.temp_path}/beft_ckpt_test"
    ckpt_mgr = checkpoint_manager.CheckpointManager(root_directory=ckpt_dir)

    # Mutate BEFT params to simulate training
    for _, val in nnx.iter_graph(model):
      if isinstance(val, BEFTParam):
        val.value = val.value + 5.0

    expected_beft_state = nnx.clone(nnx.state(model, BEFTParam))

    saved = ckpt_mgr.save(
        step=1,
        model=model,
        optimizer=optimizer,
        param_type=BEFTParam,
        force=True,
    )
    self.assertTrue(saved)

    # Create fresh model with BEFT applied
    new_model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    apply_beft_to_model(new_model, module_path=r".*w1|.*w2")
    new_optimizer = nnx.Optimizer(
        new_model, optax.adamw(learning_rate=1e-3), wrt=BEFTParam
    )

    restored_step, _ = ckpt_mgr.maybe_restore(
        new_model,
        new_optimizer,
        restore_only_lora_params=True,
        param_type=BEFTParam,
    )
    self.assertEqual(restored_step, 1)

    restored_beft_state = nnx.state(new_model, BEFTParam)
    jax.tree.map_with_path(
        tc.assert_equal, expected_beft_state, restored_beft_state
    )

  def test_sampler_update_params_with_beft(self):
    rngs = nnx.Rngs(0)
    model = tc.ToyTransformer(config=tc.ModelConfig(), rngs=rngs)
    apply_beft_to_model(model, module_path=r".*w1|.*w2")

    vocab = tc.MockVocab()
    s = sampler.Sampler(
        transformer=model,
        tokenizer=vocab,
        cache_config=None,
    )

    # Update BEFT state
    new_beft_state = jax.tree.map(
        lambda x: x + 1.0, nnx.state(model, BEFTParam)
    )
    s.update_params(new_beft_state, (BEFTParam,))

    current_beft_state = nnx.state(model, BEFTParam)
    jax.tree.map_with_path(
        tc.assert_equal, new_beft_state, current_beft_state
    )


if __name__ == '__main__':
  absltest.main()

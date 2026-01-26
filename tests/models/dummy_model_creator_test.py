import unittest

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp

from tunix.models.dummy_model_creator import create_dummy_model
from tunix.models.llama3 import model as llama3_model
from tunix.models.qwen3 import model as qwen3_model
from tunix.tests import test_common


class DummyModelCreatorTest(absltest.TestCase):

  def _test_dummy_model_creation(self, *, model_name, model_class, config_fn, mesh_config):
    config = config_fn()
    mesh = jax.make_mesh(*mesh_config, axis_types=(jax.sharding.AxisType.Auto,) * len(mesh_config[0]))

    with mesh:
      # pylint: disable=broad-exception-caught
      try:
        model = create_dummy_model(
            model_class,
            config,
            mesh=mesh,
            dtype=jnp.bfloat16,
            random_seed=0,
        )
      except Exception as e:
        self.fail(f"create_dummy_model failed for {model_name}: {e}")

      self.assertIsNotNone(model)

      # Validate nnx.display works
      try:
        nnx.display(model)
        display_success = True
      except Exception as e:
        self.fail(f"nnx.display failed for {model_name}: {e}")
      self.assertTrue(display_success, f"{model_name} model display should succeed")

      # Minimal forward to ensure parameters are materialized and callable
      inputs = model.get_model_input()
      try:
        logits, _ = model(**inputs)
      except TypeError:
        # If the model does not accept kwargs, fall back to positional ordering
        if 'last_tokens' in inputs:
          logits, _ = model(inputs['last_tokens'], inputs['positions'], inputs['cache'], inputs['attention_mask'])
        else:
          logits, _ = model(inputs['input_tokens'], inputs['positions'], inputs['cache'], inputs['attention_mask'])

      # Shape checks: [B, T, V]
      if 'last_tokens' in inputs:
        bsz, seqlen = inputs['last_tokens'].shape
      else:
        bsz, seqlen = inputs['input_tokens'].shape

      if hasattr(config, 'vocab_size'):
        vocab = config.vocab_size
      else:
        vocab = config.num_embed

      self.assertEqual(logits.shape[0], bsz, f"{model_name}: batch dimension mismatch")
      self.assertEqual(logits.shape[1], seqlen, f"{model_name}: seq length mismatch")
      self.assertEqual(logits.shape[2], vocab, f"{model_name}: vocab size mismatch")

  def test_llama3p2_1b(self):
    self._test_dummy_model_creation(
        model_name="llama3p2_1b",
        model_class=llama3_model.Llama3,
        config_fn=llama3_model.ModelConfig.llama3p2_1b,
        mesh_config=[(1, len(jax.devices())), ("fsdp", "tp")],
    )

  def test_qwen3_0p6b(self):
    self._test_dummy_model_creation(
        model_name="qwen3_0p6b",
        model_class=qwen3_model.Qwen3,
        config_fn=qwen3_model.ModelConfig.qwen3_0p6b,
        mesh_config=[(1, len(jax.devices())), ("fsdp", "tp")],
    )

if __name__ == "__main__":
  # Check if running in Jupyter/IPython environment
  if test_common.is_running_in_colab():
    # Running in Jupyter/IPython - run tests directly to avoid SystemExit
    suite = unittest.TestLoader().loadTestsFromTestCase(DummyModelCreatorTest)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
  else:
    # Running as a script - use absltest.main()
    absltest.main()

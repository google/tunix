import dataclasses
from unittest import mock
from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from tunix.generate import continuous_sampler as sampler_lib
from tunix.generate import engine as engine_lib
from tunix.generate import scheduler
from tunix.tests import test_common as tc

class ContinuousSamplerTest(absltest.TestCase):

  def setUp(self):
    self.vocab = tc.MockVocab()
    self.transformer = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=self.vocab.GetPieceSize()),
        rngs=nnx.Rngs(42),
    )
    self.cache_config = sampler_lib.CacheConfig(page_size=2, max_num_seqs=8)
    
  def test_initialization(self):
    engine = engine_lib.LLMEngine(
        transformer=self.transformer,
        tokenizer=self.vocab,
        cache_config=self.cache_config,
    )
    
    self.assertIsNotNone(engine.scheduler)
    self.assertIsNotNone(engine.cache_manager)
    self.assertEqual(engine.scheduler.page_size, 2)
    self.assertEqual(engine.cache_manager.available_hbm_pages, engine.cache_manager.hbm_page_manager.total_num_pages)

  def test_unified_step_creates_valid_arrays(self):
    engine = engine_lib.LLMEngine(
        transformer=self.transformer,
        tokenizer=self.vocab,
        cache_config=self.cache_config,
    )
    
    # We mock _compiled_step_fn to intercept the arrays and verify the continuous structure
    mock_step_fn = mock.MagicMock()
    # Mock return matches: next_tokens, hbm_pm
    mock_step_fn.return_value = (jnp.array([4, 5], dtype=jnp.int32), engine.sampler.hbm_pm if hasattr(engine.sampler, 'hbm_pm') else engine.hbm_pm)
    engine.sampler._compiled_step_fn = mock_step_fn
    
    engine.add_request("req_1", [1, 2, 3])
    engine.add_request("req_2", [7, 8])
    
    self.assertTrue(engine.has_unfinished_requests())
    engine.step()
    
    mock_step_fn.assert_called_once()
    kwargs = mock_step_fn.call_args[1]
    
    # Distribution should map identically to continuous chunk boundaries
    np.testing.assert_array_equal(kwargs['distribution'], jnp.array([0, 2, 2]))
    # Tokens should match the exact flat list for prefills
    np.testing.assert_array_equal(kwargs['tokens'], jnp.array([1, 2, 3, 7, 8]))
    
    # Verify outputs are recorded
    self.assertIn("req_1", engine.generated_tokens)
    self.assertIn("req_2", engine.generated_tokens)
    self.assertEqual(engine.generated_tokens["req_1"], [4])
    self.assertEqual(engine.generated_tokens["req_2"], [5])

  def test_end_to_end_mocked_generation_loop(self):
    engine = engine_lib.LLMEngine(
        transformer=self.transformer,
        tokenizer=self.vocab,
        cache_config=self.cache_config,
    )
    
    engine.add_request("req_1", [1, 2])
    
    eos = self.vocab.eos_id()
    fake_outputs = [
      jnp.array([10], dtype=jnp.int32), 
      jnp.array([11], dtype=jnp.int32), 
      jnp.array([eos], dtype=jnp.int32)
    ]
    
    class FakeStepFn:
        def __init__(self):
            self.calls = 0
            
        def __call__(self, state, tokens, positions, cache, distribution, seq_lens, soft_cap=None, **kwargs):
            out = fake_outputs[self.calls]
            self.calls += 1
            return out, cache
            
    engine.sampler._compiled_step_fn = FakeStepFn()
    
    steps = 0
    while engine.has_unfinished_requests() and steps < 10:
        engine.step()
        steps += 1
        
    self.assertEqual(steps, 3)  # Reached EOS!
    self.assertEqual(engine.generated_tokens["req_1"], [10, 11, eos])
    
  def test_simultaneous_decode_and_prefill_distribution(self):
    engine = engine_lib.LLMEngine(
        transformer=self.transformer,
        tokenizer=self.vocab,
        cache_config=self.cache_config,
    )
    
    engine.add_request("req_old", [1, 2])
    
    mock_step_fn = mock.MagicMock()
    mock_step_fn.return_value = (jnp.array([3], dtype=jnp.int32), engine.hbm_pm)
    engine.sampler._compiled_step_fn = mock_step_fn
    
    engine.step()
    
    engine.add_request("req_new", [4, 5, 6])
    
    mock_step_fn.reset_mock()
    mock_step_fn.return_value = (jnp.array([9, 10], dtype=jnp.int32), engine.hbm_pm)
    
    engine.step()
    
    mock_step_fn.assert_called_once()
    kwargs = mock_step_fn.call_args[1]
    
    np.testing.assert_array_equal(kwargs['distribution'], jnp.array([1, 2, 2]))
    np.testing.assert_array_equal(kwargs['tokens'], jnp.array([3, 4, 5, 6]))
    np.testing.assert_array_equal(kwargs['seq_lens'], jnp.array([1, 3]))

if __name__ == '__main__':
  absltest.main()

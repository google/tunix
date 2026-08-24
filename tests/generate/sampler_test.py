import unittest
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized
from flax import nnx

from tunix.generate import cache_manager as cm_lib
from tunix.generate import sampler_v2
from tunix.tests import test_common as tc

class VanillaSamplerTest(parameterized.TestCase):
  def test_sample_step(self):
    vocab = tc.MockVocab()
    transformer = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(42),
    )
    sampler = sampler_v2.VanillaSampler(
        transformer=transformer,
        tokenizer=vocab,
        cache_config=sampler_v2.CacheConfig(),
    )
    cm = cm_lib.init_cache_manager(
        cache_config=sampler.cache_config,
        model_config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
        kv_dtype=jnp.float32,
    )
    pm = cm.page_manager

    # Batch size 2
    # Seq 0: prefill length 3
    # Seq 1: decode length 1 (total length 4)
    tokens = np.array([4, 5, 6, 7], dtype=np.int32) 
    active_seq_lens = np.array([3, 1], dtype=np.int32)
    seq_lens = np.array([3, 4], dtype=np.int32)
    distribution = np.array([1, 2, 2], dtype=np.int32)

    gen_tokens, logits, logp, updated_pm = sampler.sample_step(
        cache=pm,
        seq_lens=seq_lens,
        tokens=tokens,
        active_seq_lens=active_seq_lens,
        distribution=distribution,
        static_token_capacity=4,
        temperature=0.0,
    )

    self.assertEqual(gen_tokens.shape, (2,))
    self.assertEqual(logits.shape, (2, vocab.GetPieceSize()))
    
  def test_forbidden_tokens(self):
    vocab = tc.MockVocab()
    transformer = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(42),
    )
    sampler = sampler_v2.VanillaSampler(
        transformer=transformer,
        tokenizer=vocab,
        cache_config=sampler_v2.CacheConfig(),
    )
    cm = cm_lib.init_cache_manager(
        cache_config=sampler.cache_config,
        model_config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
        kv_dtype=jnp.float32,
    )
    
    tokens = np.array([4, 5], dtype=np.int32) 
    active_seq_lens = np.array([2], dtype=np.int32)
    seq_lens = np.array([2], dtype=np.int32)
    distribution = np.array([0, 1, 1], dtype=np.int32)
    
    gen, logits, logp, pm = sampler.sample_step(
        cache=cm.page_manager,
        seq_lens=seq_lens,
        tokens=tokens,
        active_seq_lens=active_seq_lens,
        distribution=distribution,
        static_token_capacity=2,
        temperature=0.0,
        forbidden_token_ids=[1]
    )
    
    self.assertEqual(logits[0, 1], -np.inf)

if __name__ == '__main__':
  unittest.main()

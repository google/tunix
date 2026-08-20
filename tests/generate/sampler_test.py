# Copyright 2025 Google LLC

import dataclasses
from unittest import mock
from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

# Use the new continuous sampler
from tunix.generate import continous_sampler as sampler_lib
from tunix.tests import test_common as tc


@dataclasses.dataclass(kw_only=True)
class ModelConfigWithDtype(tc.ModelConfig):
  dtype: jax.numpy.dtype = jax.numpy.bfloat16


class ContinuousSamplerTest(absltest.TestCase):

  def test_generation_pipeline(self):
    vocab = tc.MockVocab()
    transformer = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=vocab.GetPieceSize()),
        rngs=nnx.Rngs(42),
    )

    sampler = sampler_lib.VanillaSampler(
        transformer=transformer,
        tokenizer=vocab,
        cache_config=sampler_lib.CacheConfig(batch_size=2),
    )

    sampler.add_request("req_1", [1, 2, 3])
    sampler.add_request("req_2", [4, 5])
    
    steps = 0
    while sampler.has_unfinished_requests():
        sampler.step()
        steps += 1
        if steps > 10:
            break
            
    self.assertIn("req_1", sampler.generated_tokens)
    self.assertIn("req_2", sampler.generated_tokens)
    self.assertGreater(len(sampler.generated_tokens["req_1"]), 0)
    self.assertGreater(len(sampler.generated_tokens["req_2"]), 0)

if __name__ == '__main__':
  absltest.main()

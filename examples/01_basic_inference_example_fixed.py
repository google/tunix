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

"""
Fixed Basic Inference Example for Tunix.

This script demonstrates how to correctly initialize a Tunix model (using dummy weights)
and use the Sampler class for text generation.
"""

from flax import nnx
import jax
from tunix import Tokenizer
from tunix.generate import sampler
from tunix.models import dummy_model_creator
from tunix.models.gemma import model as gemma_model


def main():
  # 1. Initialize Tokenizer
  # Note: In a real scenario, you would point this to your tokenizer file.
  # For this example, we assume the default behavior or a mock if needed.
  print("Initializing Tokenizer...")
  try:
    tokenizer = Tokenizer()
  except Exception as e:
    print(f"Warning: Could not initialize real tokenizer ({e}). Using mock.")

    class MockTokenizer:

      def encode(self, s):
        return [1, 2, 3]  # Dummy tokens

      def decode(self, t):
        return "dummy output"

      def pad_id(self):
        return 0

      def bos_id(self):
        return 1

      def eos_id(self):
        return 2

    tokenizer = MockTokenizer()

  # 2. Initialize Model Configuration and Dummy Model
  print("Initializing Dummy Model...")
  # Using Gemma 2B config for this example
  config = gemma_model.ModelConfig.gemma_2b()

  # Create a dummy model with random weights
  # We use eval_shape/lazy initialization where possible to avoid OOM on small machines
  # but dummy_model_creator creates actual arrays.
  model = dummy_model_creator.create_dummy_model(
      gemma_model.Transformer, config, dtype=jax.numpy.float32
  )

  # 3. Initialize Sampler
  print("Initializing Sampler...")
  # Sampler needs the model, tokenizer, and cache config
  cache_config = sampler.CacheConfig(
      cache_size=1024,  # Max sequence length
      num_layers=config.num_layers,
      num_kv_heads=config.num_kv_heads,
      head_dim=config.head_dim,
  )

  inference_sampler = sampler.Sampler(
      transformer=model, tokenizer=tokenizer, cache_config=cache_config
  )

  # 4. Run Generation
  prompts = [
      "If I have 3 apples and eat 1, how many remain?",
      "Write a short story about a robot.",
  ]

  print("\n" + "=" * 50)
  print("Starting Generation...")

  # The Sampler class is callable directly
  output = inference_sampler(
      input_strings=prompts,
      max_generation_steps=50,
      temperature=0.7,
      echo=True,  # Include prompt in output
  )

  # 5. Print Results
  for i, text in enumerate(output.text):
    print(f"\nPrompt {i+1}: {prompts[i]}")
    print(f"Generated: {text}")


if __name__ == "__main__":
  main()

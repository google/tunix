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

"""Mock rollout worker."""

import random
import time
from typing import Any, Optional, Tuple

import jax
from jax import numpy as jnp
import jaxtyping
import numpy as np
from tunix.rl.rollout import base_rollout

_DUMMY_WORDS = [
    "mock",
    "test",
    "token",
    "rollout",
    "random",
    "data",
    "output",
    "engine",
]


class MockRollout(base_rollout.BaseRollout):
  """Mock rollout worker."""

  def __init__(
      self,
      model: Any | None = None,
      tokenizer: Any | None = None,
      vocab_size: int | None = None,
      pad_id: int | None = None,
      eos_id: int | None = None,
      **kwargs,
  ):
    self._model = model
    self._tokenizer = tokenizer
    self._vocab_size = vocab_size if vocab_size is not None else 32000
    self._pad_id = pad_id if pad_id is not None else 0
    self._eos_id = eos_id if eos_id is not None else 1

  def generate(
      self,
      prompts: list[str],
      rollout_config: base_rollout.RolloutConfig,
      **kwargs,
  ) -> base_rollout.RolloutOutput:
    """Generates random samples and simulates time delay."""
    rng = random.Random()
    np_rng = np.random.default_rng()
    if rollout_config.seed is not None:
      seed_val = int(
          rollout_config.seed.item()
          if isinstance(rollout_config.seed, jax.Array)
          else rollout_config.seed
      )
      rng = random.Random(seed_val)
      np_rng = np.random.default_rng(seed_val)

    min_generation_time = rollout_config.rollout_mock_min_generation_time
    max_generation_time = rollout_config.rollout_mock_max_generation_time

    sleep_time = rng.uniform(min_generation_time, max_generation_time)
    time.sleep(sleep_time)

    batch_size = len(prompts)
    max_tokens = rollout_config.max_tokens_to_generate
    # Fallback to at least 1 token if max_tokens is less than 1
    max_tokens = max(1, max_tokens)

    texts = []
    logits_list = []
    tokens_list = []

    for _ in range(batch_size):
      target_length = rng.randint(1, max_tokens)
      chosen_words = rng.choices(_DUMMY_WORDS, k=target_length)
      text = " ".join(chosen_words)

      if self._tokenizer is not None and hasattr(self._tokenizer, "encode"):
        try:
          tokens = np.array(self._tokenizer.encode(text), dtype=np.int32)
          if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
          elif len(tokens) == 0:
            tokens = np_rng.integers(
                0, self._vocab_size, size=(1,), dtype=np.int32
            )
          length = len(tokens)
          if hasattr(self._tokenizer, "decode"):
            text = self._tokenizer.decode(tokens.tolist())
        except Exception:  # pylint: disable=broad-except
          length = target_length
          tokens = np_rng.integers(
              0, self._vocab_size, size=(length,), dtype=np.int32
          )
      else:
        length = target_length
        tokens = np_rng.integers(
            0, self._vocab_size, size=(length,), dtype=np.int32
        )

      tokens_list.append(tokens)
      texts.append(text)

      # Mock logits
      logits = np.zeros((length, self._vocab_size), dtype=np.float16)
      logits_list.append(logits)

    left_padded_prompt_tokens = np.zeros(
        (batch_size, rollout_config.max_prompt_length), dtype=np.int32
    )

    return base_rollout.RolloutOutput(
        text=texts,
        logits=logits_list,
        tokens=tokens_list,
        left_padded_prompt_tokens=left_padded_prompt_tokens,
        logprobs=None,
    )

  def get_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      completion_mask: jax.Array | None = None,
  ) -> jax.Array:
    """Returns mock per-token log probabilities."""
    batch_size, length = completion_tokens.shape
    return np.zeros((batch_size, length), dtype=np.float32)

  def update_params(
      self,
      params: jaxtyping.PyTree,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ) -> None:
    """Mock update params."""
    pass

  def pad_id(self) -> int:
    if self._tokenizer is not None and hasattr(self._tokenizer, "pad_id"):
      return self._tokenizer.pad_id()
    return self._pad_id

  def eos_id(self) -> int:
    if self._tokenizer is not None and hasattr(self._tokenizer, "eos_id"):
      return self._tokenizer.eos_id()
    return self._eos_id

  def model(self) -> Any:
    return self._model

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

from collections.abc import Sequence
import random
import time
from typing import Any

from absl import logging
import jax
import jaxtyping
import numpy as np
from tunix.rl.rollout import base_rollout

_DUMMY_WORDS = (
    "mock",
    "test",
    "token",
    "rollout",
    "random",
    "data",
    "output",
    "engine",
)


class MockRollout(base_rollout.BaseRollout):
  """Mock rollout worker for testing RL pipelines.

  This engine simulates the behavior of a real LLM rollout worker (like vLLM
  or SG-Lang) without requiring an actual model, weights, or accelerator
  resources.

  The mock rollout is particularly useful for testing and benchmarking
  performance, especially on the training and communication side, without
  needing huge TPU clusters. For example, it can be used for benchmarking
  new optimizations, or collecting mock perf metric traces at scale.

  Behaviors mocked:
    * Text Generation: Produces sequences of random dummy words.
    * Tokenization: Uses the provided tokenizer to encode/decode the dummy text,
      or falls back to generating random token IDs if no tokenizer is provided.
    * Latency: Simulates inference delay by sleeping for a random duration
      between `min_generation_time` and
      `max_generation_time`.
    * Tensors (Logits/Logprobs/Logps): Returns zero-filled numpy arrays of the
      correct shapes to simulate model outputs while keeping memory on the host.
    * Parameter Updates: `update_params` is a no-op.
    * Reproducibility: Fully supports seeding via `RolloutConfig.seed` for
      deterministic testing.

  Kwargs:
    min_generation_time (float): Minimum simulated generation delay in seconds
      (default: 1.0).
    max_generation_time (float): Maximum simulated generation delay in seconds
      (default: 10.0).
  """

  def __init__(
      self,
      rollout_actor: Any | None = None,
      tokenizer: Any | None = None,
      vocab_size: int | None = None,
      pad_id: int | None = None,
      eos_id: int | None = None,
      rollout_config: base_rollout.RolloutConfig | None = None,
      **kwargs,
  ):
    self._model = rollout_actor
    self._tokenizer = tokenizer
    self._vocab_size = vocab_size if vocab_size is not None else 32_000
    self._pad_id = pad_id if pad_id is not None else 0
    self._eos_id = eos_id if eos_id is not None else 1
    self._min_generation_time = kwargs.get("min_generation_time", 1.0)
    self._max_generation_time = kwargs.get("max_generation_time", 10.0)

    seed_val = None
    if rollout_config is not None and rollout_config.seed is not None:
      seed_val = int(
          rollout_config.seed.item()
          if isinstance(rollout_config.seed, jax.Array)
          else rollout_config.seed
      )

    if seed_val is not None:
      self._rng = random.Random(seed_val)
      self._np_rng = np.random.default_rng(seed_val)
    else:
      self._rng = random.Random()
      self._np_rng = np.random.default_rng()

  def _encode_text(self, text: str) -> np.ndarray | None:
    """Attempts to encode text using the tokenizer, returning None on failure."""
    if self._tokenizer is not None and hasattr(self._tokenizer, "encode"):
      try:
        return np.array(self._tokenizer.encode(text), dtype=np.int32)
      except Exception as e:  # pylint: disable=broad-except
        logging.log_every_n(
            logging.WARNING, "Tokenization failed in mock_rollout: %s", 100, e
        )
    return None

  def generate(
      self,
      prompts: str | Sequence[str],
      rollout_config: base_rollout.RolloutConfig,
      **kwargs,
  ) -> base_rollout.RolloutOutput:
    """Generates random samples and simulates time delay.

    Args:
      prompts: A list of text prompts for generation.
      rollout_config: Configuration settings for generation and mock behavior.
      **kwargs: Additional generation arguments.

    Returns:
      A RolloutOutput containing the mock generated texts, tokens, and tensors.
    """
    if isinstance(prompts, str):
      prompts = [prompts]

    rng = self._rng
    np_rng = self._np_rng

    min_generation_time = self._min_generation_time
    max_generation_time = self._max_generation_time

    sleep_time = rng.uniform(min_generation_time, max_generation_time)
    time.sleep(sleep_time)

    batch_size = len(prompts)
    max_tokens = rollout_config.max_tokens_to_generate
    # Fallback to at least 1 token if max_tokens is less than 1
    max_tokens = max(1, max_tokens)

    texts = []
    logits_list = []
    tokens_list = []

    left_padded_prompt_tokens = np.full(
        (batch_size, rollout_config.max_prompt_length),
        self.pad_id(),
        dtype=np.int32,
    )

    for i in range(batch_size):
      prompt = prompts[i]
      target_length = rng.randint(1, max_tokens)
      chosen_words = rng.choices(_DUMMY_WORDS, k=target_length)
      text = " ".join(chosen_words)

      # 1. Tokenize the prompt for left_padded_prompt_tokens
      prompt_tokens = self._encode_text(prompt)
      if prompt_tokens is not None:
        if len(prompt_tokens) > rollout_config.max_prompt_length:
          # Truncate to fit, keeping the suffix for left-padding
          prompt_tokens = prompt_tokens[-rollout_config.max_prompt_length :]

        start_idx = rollout_config.max_prompt_length - len(prompt_tokens)
        left_padded_prompt_tokens[i, start_idx:] = prompt_tokens

      # 2. Tokenize the generated completion
      tokens = self._encode_text(text)
      if tokens is not None:
        if len(tokens) > max_tokens:
          tokens = tokens[:max_tokens]
        elif len(tokens) == 0:
          tokens = np_rng.integers(
              0, self._vocab_size, size=(1,), dtype=np.int32
          )

        length = len(tokens)
        if hasattr(self._tokenizer, "decode"):
          text = self._tokenizer.decode(tokens.tolist())
      else:
        length = target_length
        tokens = np_rng.integers(
            0, self._vocab_size, size=(length,), dtype=np.int32
        )

      tokens_list.append(tokens)
      texts.append(text)

      logits = np.zeros((length, self._vocab_size), dtype=np.float16)
      logits_list.append(logits)

    if rollout_config.return_logprobs:
      logprobs_list = [np.zeros(len(t), dtype=np.float32) for t in tokens_list]
    else:
      logprobs_list = None

    return base_rollout.RolloutOutput(
        text=texts,
        logits=logits_list,
        tokens=tokens_list,
        left_padded_prompt_tokens=left_padded_prompt_tokens,
        logprobs=logprobs_list,
    )

  def get_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      completion_mask: jax.Array | None = None,
  ) -> jax.Array:
    """Returns mock per-token log probabilities.

    Args:
      prompt_tokens: The tokens of the input prompts.
      completion_tokens: The generated completion tokens.
      completion_mask: An optional mask indicating valid completion tokens.

    Returns:
      A zero-filled array of shape (batch_size, length) representing mock
      log probabilities.
    """
    batch_size, length = completion_tokens.shape
    # Use numpy to keep it on host memory.
    return np.zeros((batch_size, length), dtype=np.float32)

  def update_params(
      self,
      params: jaxtyping.PyTree,
      filter_types: tuple[Any, ...] | None = None,
  ) -> None:
    """Mock update params.

    Args:
      params: A PyTree of parameters to update.
      filter_types: Optional types to filter which parameters to update.
    """
    pass

  def pad_id(self) -> int:
    if self._tokenizer is not None and hasattr(self._tokenizer, "pad_id"):
      pad_id_attr = self._tokenizer.pad_id
      return pad_id_attr() if callable(pad_id_attr) else pad_id_attr
    return self._pad_id

  def eos_id(self) -> int:
    if self._tokenizer is not None and hasattr(self._tokenizer, "eos_id"):
      eos_id_attr = self._tokenizer.eos_id
      return eos_id_attr() if callable(eos_id_attr) else eos_id_attr
    return self._eos_id

  def model(self) -> Any:
    return self._model

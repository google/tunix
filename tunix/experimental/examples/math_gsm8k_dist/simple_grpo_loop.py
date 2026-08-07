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

"""A thin single-turn GRPO learner built on the RLOrchestrator primitive API.

This is a Layer-3 loop: it composes only `RLOrchestrator` primitives --
generate,
compute_advantages, assemble_train_example, train_step, sync_weights -- and
knows
nothing about where compute runs (cluster's choice) or the algorithm internals
(adapter's choice). It exists to demonstrate that the primitive API is
sufficient
to express a learning loop and is pluggable: swap the cluster to distribute,
swap
the adapter to change algorithm.

It is deliberately minimal -- single-turn (one completion per generation, no
multi-turn episodes, no async producer/consumer, no micro-batch grad-accum). The
full agentic loop remains the faithful, production learner; this is the
reference
that the primitive surface is complete.
"""

from typing import Any, Callable, Iterable

import jax.numpy as jnp
from tunix.experimental.examples.math_gsm8k_dist import rl_orchestrator as rl_orchestrator_lib


class SimpleGRPOLoop:
  """Minimal GRPO training loop over an `RLOrchestrator`."""

  def __init__(
      self,
      orchestrator: rl_orchestrator_lib.RLOrchestrator,
      *,
      reward_fn: Callable[..., list[float]],
      tokenizer: Any,
      num_generations: int,
      max_prompt_length: int,
      max_response_length: int,
      pad_id: int,
      sync_weights: bool = False,
  ):
    self._orch = orchestrator
    self._reward_fn = reward_fn
    self._tokenizer = tokenizer
    self._num_generations = num_generations
    self._max_prompt_length = max_prompt_length
    self._max_response_length = max_response_length
    self._pad_id = pad_id
    self._sync_weights = sync_weights
    # Wire the algorithm's loss onto the trainer once, up front.
    self._orch.configure_trainer()

  def train(self, prompts: Iterable[str]) -> None:
    for prompt in prompts:
      self.train_step(prompt)

  def train_step(self, prompt: str) -> None:
    group_prompts = [prompt] * self._num_generations

    # 1. Generate a group of completions.
    rollout = self._orch.generate(group_prompts)
    completions = rollout.text
    completion_tokens = rollout.tokens
    prompt_tokens = [
        self._tokenizer.encode(prompt) for _ in range(self._num_generations)
    ]

    # 2. Reward -> 3. group-relative advantage.
    rewards = self._reward_fn(prompts=group_prompts, completions=completions)
    advantages = self._orch.compute_advantages(
        jnp.asarray(rewards, dtype=jnp.float32),
        num_generations=self._num_generations,
    )

    # 4. Assemble a train example, then 5. train, 6. (optionally) sync.
    batch = self._orch.assemble_train_example(
        prompt_tokens,
        completion_tokens,
        advantages,
        max_prompt_length=self._max_prompt_length,
        max_response_length=self._max_response_length,
        pad_id=self._pad_id,
    )
    self._orch.train_step([batch])
    if self._sync_weights:
      self._orch.sync_weights()
    self._orch.global_steps += 1

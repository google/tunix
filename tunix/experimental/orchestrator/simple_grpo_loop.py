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

This is a Layer-3 loop: it composes only `RLOrchestrator` primitives -- generate,
compute_advantages, assemble_train_example, train_step, sync_weights -- and knows
nothing about where compute runs (cluster's choice) or the algorithm internals
(adapter's choice). It exists to demonstrate that the primitive API is sufficient
to express a learning loop and is pluggable: swap the cluster to distribute, swap
the adapter to change algorithm.

It is deliberately minimal -- single-turn (one completion per generation, no
multi-turn episodes, no async producer/consumer, no micro-batch grad-accum). The
full agentic loop remains the faithful, production learner; this is the reference
that the primitive surface is complete.
"""

import asyncio
from typing import Any, Callable, Iterable

import jax.numpy as jnp

from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import group_gate
from tunix.experimental.orchestrator import rl_orchestrator as rl_orchestrator_lib


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
      rollout_pool: Any = None,
  ):
    """Initializes the loop.

    Args:
      orchestrator: Supplies the compute primitives and the algorithm.
      reward_fn: Scores completions.
      tokenizer: Encodes prompts on the whole-batch path.
      num_generations: Completions per prompt, i.e. the group size.
      max_prompt_length: Padding width for prompts.
      max_response_length: Padding width for completions.
      pad_id: Padding token id.
      sync_weights: Whether to sync weights after each step.
      rollout_pool: Opt-in. When given, generation goes one request per
        trajectory across a pool of rollout workers instead of one batched
        call to a single worker. Off by default: the pooled path is
        single-turn only, and it changes which worker produced what.

    Raises:
      ValueError: If a pool is combined with weight syncing. Installing new
        weights across several rollout workers needs a versioned protocol
        that acknowledges each replica; without it some workers would keep
        generating from stale weights and nothing would say so. Pin the
        version instead until that lands.
    """
    if rollout_pool is not None and sync_weights:
      raise ValueError(
          "Weight syncing across a rollout pool is not supported yet: there"
          " is no protocol that confirms every worker installed the new"
          " version, so some would silently keep generating from old"
          " weights. Run the pool at a pinned version, or use the"
          " single-worker path."
      )
    self._rollout_pool = rollout_pool
    self._async_loop: Any = None
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
    if self._rollout_pool is not None:
      generated = self._generate_pooled(prompt)
      if generated is None:
        # The group did not come back whole. Training on what survived would
        # be a different update, not a smaller one, so the step is skipped.
        return
      prompt_tokens, completion_tokens, completions = generated
    else:
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

  def _run_async(self, coro: Any) -> Any:
    """Runs pooled generation on this loop's own, reused event loop.

    One loop for the life of the run, because the pool's admission lock and
    its in-process transports bind to whichever loop first touched them; a
    fresh loop per step would find them owned by a loop that no longer exists.
    """
    if self._async_loop is None:
      self._async_loop = asyncio.new_event_loop()
    return self._async_loop.run_until_complete(coro)

  def close(self) -> None:
    """Releases the event loop used for pooled generation."""
    if self._async_loop is not None:
      self._async_loop.close()
      self._async_loop = None

  def _generate_pooled(self, prompt: str):
    """Generates one group across the pool, or None if it did not come whole.

    Returns:
      `(prompt_tokens, completion_tokens, completion_texts)` for a complete,
      successful group, else None. Tokens come back as the worker produced
      them, so nothing is re-encoded here and the trained tokens are the ones
      that were sampled.
    """
    group_id = str(self._orch.global_steps)
    requests = [
        datatypes.RolloutRequest(
            request_id=f"{group_id}-{index}",
            prompt={"prompts": prompt},
            prompt_id=group_id,
            group_id=group_id,
        )
        for index in range(self._num_generations)
    ]

    responses = self._run_async(self._rollout_pool.generate(requests))
    gated = group_gate.gate_groups(
        requests,
        responses,
        group_size=self._num_generations,
        tokenizer=self._tokenizer,
    )
    if gated.dropped:
      group_gate.log_dropped(gated.dropped)
      return None

    items = gated.complete[group_id]
    return (
        [item.traj["prompt_tokens"] for item in items],
        [item.traj["conversation_tokens"] for item in items],
        [_completion_text(item) for item in items],
    )


def _completion_text(item: Any) -> str:
  conversation = item.traj.get("conversation_text") or []
  return next(
      (m["content"] for m in conversation if m["role"] == "assistant"), ""
  )

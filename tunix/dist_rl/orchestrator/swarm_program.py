# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Synchronous shared-policy swarm program using the native dist_rl trainer.

Each complete episode produces a variable number of call rows. This is a
sibling of StandardRLProgram, whose trajectory cardinality is unchanged.
The first transport is local; no RPC episode collector is claimed here.
"""

import asyncio
import collections
from collections.abc import Awaitable, Callable, Sequence
import dataclasses
from typing import Protocol

import numpy as np
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.orchestrator import batch_assembly
from tunix.dist_rl.orchestrator import rl_program
from tunix.dist_rl.swarm import credit
from tunix.dist_rl.swarm import trace
from tunix.rl import algorithm_config


@dataclasses.dataclass(frozen=True, kw_only=True)
class SwarmConfig:
  group_size: int = 8
  micro_batch_size: int = 4
  max_prompt_length: int = 8192
  max_response_length: int = 4096
  pad_id: int = 0
  eos_id: int = 0
  normalization: str = "role"
  invalid_action_penalty: float = 0.1
  max_logprob_error: float = 0.005

  def __post_init__(self):
    if (
        self.group_size < 2
        or min(
            self.micro_batch_size,
            self.max_prompt_length,
            self.max_response_length,
        )
        < 1
    ):
      raise ValueError(
          "Require group_size >= 2 and positive batch/token budgets."
      )
    if self.normalization not in ("role", "episode"):
      raise ValueError("Unknown advantage normalization.")
    if not np.isfinite(self.max_logprob_error) or self.max_logprob_error < 0:
      raise ValueError("max_logprob_error must be finite and nonnegative.")


def shared_grpo_config(group_size: int = 8) -> algorithm_config.GRPOConfig:
  """Dr. MAS token-mean dual-clip objective; no reference KL or entropy term."""
  config = algorithm_config.GRPOConfig(
      num_generations=group_size,
      beta=0.0,
      epsilon=0.2,
      epsilon_high=0.2,
      epsilon_c=3.0,
      loss_agg_mode="token-mean",
      use_rollout_logps=True,
  )
  config.temperature = 1.0
  return config


class SwarmEngine(Protocol):
  """Minimal transport boundary; implementations own trainer and weight sync."""

  def configure(
      self, algo_config: algorithm_config.GRPOConfig, config: SwarmConfig
  ) -> None:
    ...

  async def prepare(self) -> int:
    ...

  async def score(self, payload: datatypes.RLTrainerPayload) -> np.ndarray:
    ...

  async def train(
      self, payload: datatypes.RLTrainerPayload, *, apply_optimizer: bool
  ) -> object:
    ...

  async def publish(self) -> int:
    ...


class SwarmRLProgram(rl_program.RLProgram):
  """Collect complete groups, compute credit, score, train once, then publish."""

  def __init__(
      self,
      collect: Callable[[int, int], Awaitable[Sequence[trace.SwarmEpisode]]],
      config: SwarmConfig,
  ):
    super().__init__()
    self.collect = collect
    self.config = config
    self.history: list[dict] = []
    self._failed = False

  def run(self, engine: SwarmEngine, *, num_steps: int = 2, **kwargs) -> None:
    if kwargs:
      raise TypeError(f"Unknown run options: {tuple(kwargs)}")
    asyncio.run(self.run_async(engine, num_steps=num_steps))

  async def run_async(self, engine: SwarmEngine, *, num_steps: int = 2) -> None:
    if self._failed:
      raise RuntimeError("Restart a failed swarm run from a known model state.")
    if num_steps < 1 or self._is_running:
      raise ValueError(
          "num_steps must be positive and the program must be idle."
      )
    self._is_running = True
    try:
      config = self.config
      engine.configure(shared_grpo_config(config.group_size), config)
      self.policy_version = await engine.prepare()
      for _ in range(num_steps):
        episodes = tuple(await self.collect(self.policy_version, self.step))
        counts = collections.Counter(e.task_id for e in episodes)
        if not counts or any(n != config.group_size for n in counts.values()):
          raise ValueError(
              "Collect complete groups before computing advantages."
          )
        if any(
            c.policy_version != self.policy_version
            for e in episodes
            for c in e.calls
        ):
          raise ValueError("Collection returned a stale policy version.")
        credited = credit.build_credit(
            episodes,
            normalization=config.normalization,
            invalid_action_penalty=config.invalid_action_penalty,
        )
        for row in credited.payloads:
          if (
              row.prompt_ids.shape[-1] > config.max_prompt_length
              or row.completion_ids.shape[-1] > config.max_response_length
          ):
            raise ValueError(
                "A call exceeds the configured training token budget."
            )
        if not any(credited.advantages):
          # AdamW can change parameters even with zero gradients. A no-signal
          # round consumes data without advancing optimizer or policy version.
          self._step += 1
          rewards = [e.reward for e in episodes]
          self.last_step_result = rl_program.RLStepResult(
              step=self.step,
              policy_version=self.policy_version,
              num_rollouts=len(episodes),
              num_microbatches=0,
              reward_mean=float(np.mean(rewards)),
              reward_std=float(np.std(rewards)),
              train_result="zero_signal",
          )
          self.history.append({
              "step": self.step,
              "policy_version": self.policy_version,
              "episodes": len(episodes),
              "calls": len(credited.payloads),
              "reward_mean": float(np.mean(rewards)),
              "optimizer_applied": False,
              "skip_reason": "zero_signal",
          })
          continue
        # All variable call rows belong to ONE optimizer boundary. Do not use
        # group_size here: a team attempt may contain two or three calls.
        assembler = batch_assembly.PaddedBatchAssembler(
            batch_size=config.micro_batch_size,
            max_prompt_length=config.max_prompt_length,
            max_response_length=config.max_response_length,
            pad_id=config.pad_id,
            num_generations=len(credited.payloads),
            mini_batch_size=1,
        )
        batches = list(assembler.feed(credited.payloads))
        # Snapshot the actor's old logps for ALL batches before any update,
        # following upstream's actor recomputation, not a changing PPO baseline.
        scored = []
        max_error = 0.0
        for batch in batches:
          payload = batch.payload
          old = np.asarray(await engine.score(payload), dtype=np.float32)
          active = payload.completion_mask > 0
          if (
              old.shape != payload.completion_ids.shape
              or not np.isfinite(old[active]).all()
          ):
            raise ValueError(
                "Actor scoring returned invalid or misaligned logprobs."
            )
          error = float(
              np.max(np.abs(old[active] - payload.old_per_token_logps[active]))
          )
          max_error = max(max_error, error)
          if error > config.max_logprob_error:
            raise ValueError(f"Sampler/actor logprob mismatch: {error:.6g}")
          scored.append(dataclasses.replace(payload, old_per_token_logps=old))
        result = None
        for payload, batch in zip(scored, batches, strict=True):
          result = await engine.train(
              payload, apply_optimizer=batch.is_final_batch
          )
        version = await engine.publish()
        if version != self.policy_version + 1:
          raise ValueError(
              "Publishing must advance the shared policy exactly once."
          )
        self.policy_version = version
        self._step += 1
        self.last_step_result = rl_program.RLStepResult(
            step=self.step,
            policy_version=version,
            num_rollouts=len(episodes),
            num_microbatches=len(batches),
            reward_mean=float(np.mean([e.reward for e in episodes])),
            reward_std=float(np.std([e.reward for e in episodes])),
            advantage_mean=float(np.mean(credited.advantages)),
            advantage_std=float(np.std(credited.advantages)),
            train_result=result,
        )
        self.history.append({
            "step": self.step,
            "policy_version": version,
            "optimizer_applied": True,
            "episodes": len(episodes),
            "calls": len(credited.payloads),
            "active_tokens": int(
                sum(np.sum(p.completion_mask) for p in scored)
            ),
            "reward_mean": self.last_step_result.reward_mean,
            "max_logprob_error": max_error,
            "role_advantages": {
                role: {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "calls": len(values),
                }
                for role in set(credited.roles)
                for values in [[
                    a
                    for a, r in zip(
                        credited.advantages, credited.roles, strict=True
                    )
                    if r == role
                ]]
            },
        })
    except BaseException:
      self._failed = True
      raise
    finally:
      self._is_running = False

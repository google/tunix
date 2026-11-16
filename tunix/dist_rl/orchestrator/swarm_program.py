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

from absl import logging
import numpy as np
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.orchestrator import batch_assembly
from tunix.dist_rl.orchestrator import rl_program
from tunix.dist_rl.swarm import credit
from tunix.dist_rl.swarm import trace
from tunix.rl import algorithm_config

LOSS_REDUCTIONS = ("global_token_mean", "microbatch_token_mean")


@dataclasses.dataclass(frozen=True, kw_only=True)
class SwarmConfig:
  """Round settings shared by the program and its engine.

  `loss_reduction` chooses how token losses are averaged into one update:
  "global_token_mean" divides the batch-wide token sum by the batch-wide token
  count; "microbatch_token_mean" averages the token-mean of every group of
  `loss_microbatch_rows` rows, as the pinned verl `dp_actor` does with
  `ppo_micro_batch_size_per_gpu` rows per microbatch. The loss groups are
  independent of the device `micro_batch_size`.
  """

  group_size: int = 8
  micro_batch_size: int = 4
  max_prompt_length: int = 8192
  max_response_length: int = 4096
  pad_id: int = 0
  eos_id: int = 0
  normalization: str = "role"
  invalid_action_penalty: float = 0.1
  max_logprob_error: float = 0.005
  loss_reduction: str = "global_token_mean"
  loss_microbatch_rows: int = 4
  sampler_is: str | None = None
  sampler_is_threshold: float = 2.0
  epsilon_high: float = 0.2
  allow_variable_group_sizes: bool = False

  def __post_init__(self):
    if (
        self.group_size < 2
        or min(
            self.micro_batch_size,
            self.max_prompt_length,
            self.max_response_length,
            self.loss_microbatch_rows,
        )
        < 1
    ):
      raise ValueError(
          "Require group_size >= 2 and positive batch/token budgets."
      )
    if self.normalization not in ("role", "episode"):
      raise ValueError("Unknown advantage normalization.")
    if self.loss_reduction not in LOSS_REDUCTIONS:
      raise ValueError(f"loss_reduction must be one of {LOSS_REDUCTIONS}.")
    if self.sampler_is not in (None, "token"):
      raise ValueError("sampler_is must be None or 'token'.")
    if not np.isfinite(self.max_logprob_error) or self.max_logprob_error < 0:
      raise ValueError("max_logprob_error must be finite and nonnegative.")
    if (
        not np.isfinite(self.sampler_is_threshold)
        or self.sampler_is_threshold <= 0
    ):
      raise ValueError("sampler_is_threshold must be finite and positive.")
    if not np.isfinite(self.epsilon_high) or self.epsilon_high <= 0:
      raise ValueError("epsilon_high must be finite and positive.")


def shared_grpo_config(
    group_size: int = 8,
    *,
    sampler_is: str | None = None,
    sampler_is_threshold: float = 2.0,
    epsilon_high: float = 0.2,
) -> algorithm_config.GRPOConfig:
  """Dr. MAS / R2Orch token-mean dual-clip objective; no reference KL term."""
  config = algorithm_config.GRPOConfig(
      num_generations=group_size,
      beta=0.0,
      epsilon=0.2,
      epsilon_high=epsilon_high,
      epsilon_c=3.0,
      loss_agg_mode="token-mean",
      use_rollout_logps=True,
      sampler_is=sampler_is,
      sampler_is_threshold=sampler_is_threshold,
  )
  config.temperature = 1.0
  return config


def microbatch_groups(
    num_rows: int, rows_per_microbatch: int, seed: int
) -> np.ndarray:
  """Assigns rows to loss microbatches of `rows_per_microbatch` rows.

  verl balances rows across data-parallel ranks before splitting each rank's
  rows into microbatches, so a microbatch mixes tasks and roles. A permutation
  seeded by the round reproduces that mixing deterministically, also on resume.

  Args:
    num_rows: Number of rows with at least one active token.
    rows_per_microbatch: Rows per loss microbatch.
    seed: Seed for the row permutation.

  Returns:
    The microbatch index of every row.
  """
  order = np.random.default_rng(seed).permutation(num_rows)
  groups = np.empty(num_rows, dtype=np.int64)
  groups[order] = np.arange(num_rows) // rows_per_microbatch
  return groups


def microbatch_token_mean_weights(
    token_counts: np.ndarray, groups: np.ndarray
) -> np.ndarray:
  """Row weights that turn one batch token-mean into a microbatch mean.

  With weight `w_r = N / (G * N_g)` on every token of row `r`, the batch
  token-mean `sum_r w_r L_r / N` equals `(1 / G) sum_g L_g / N_g`: `L` is a
  summed token loss, `N` the batch token count, `G` the number of microbatches
  and `N_g` the token count of row `r`'s microbatch `g`.

  Args:
    token_counts: Active tokens of every row; all positive.
    groups: Microbatch index of every row.

  Returns:
    The weight of every row.
  """
  counts = np.asarray(token_counts, dtype=np.float64)
  group_tokens = np.bincount(groups, weights=counts)
  return counts.sum() / (len(group_tokens) * group_tokens[groups])


def weight_microbatches(
    payloads: Sequence[datatypes.RLTrainerPayload],
    rows_per_microbatch: int,
    seed: int,
) -> list[datatypes.RLTrainerPayload]:
  """Scales advantages so the native token-mean equals verl's microbatch mean.

  Every PPO-clip and dual-clip branch is linear in the advantage, so scaling a
  row's advantage by a positive weight scales its token losses and gradient by
  exactly that weight. Rows without active tokens keep weight 0.

  Args:
    payloads: Scored device microbatches of one optimizer boundary.
    rows_per_microbatch: Rows per loss microbatch.
    seed: Seed for the row permutation.

  Returns:
    The payloads with weighted advantages.
  """
  counts = np.concatenate(
      [np.sum(np.asarray(p.completion_mask) > 0, axis=-1) for p in payloads]
  )
  active = counts > 0
  weights = np.zeros(counts.shape, dtype=np.float64)
  weights[active] = microbatch_token_mean_weights(
      counts[active],
      microbatch_groups(int(active.sum()), rows_per_microbatch, seed),
  )
  weighted = []
  start = 0
  for payload in payloads:
    advantages = np.asarray(payload.advantages, dtype=np.float32)
    scale = weights[start : start + advantages.shape[0]].astype(np.float32)
    start += advantages.shape[0]
    weighted.append(
        dataclasses.replace(
            payload,
            advantages=advantages
            * (scale if advantages.ndim == 1 else scale[:, None]),
        )
    )
  return weighted


_TRAINER_METRIC_KEYS = (
    "loss",
    "grad_norm",
    "entropy",
    "ppo_kl",
    "pg_clipfrac",
    "pg_clipfrac_lower",
    "is_ratio/max",
    "log_ratio/abs_mean",
    "sampler_is/weight_mean",
    "sampler_is/weight_min",
    "sampler_is/frac_clipped_at_threshold",
    "sampler_trainer/logp_diff_mean",
    "sampler_trainer/logp_diff_max",
)


def _trainer_metrics(result: object) -> dict[str, float]:
  """Extracts JSON-safe step scalars from a `TrainerWorker.update()` result."""
  buf = result[1] if isinstance(result, tuple) and len(result) == 2 else result
  if not isinstance(buf, datatypes.MetricsBuffer):
    return {}
  out = {
      k: float(np.asarray(buf.scalar_metrics[k]))
      for k in _TRAINER_METRIC_KEYS
      if k in buf.scalar_metrics
  }
  for k in _TRAINER_METRIC_KEYS:
    if k in buf.weighted_metrics:
      m = buf.weighted_metrics[k]
      denom = max(float(np.asarray(m.denominator)), float(m.min_denom or 0.0))
      out[k] = float(np.asarray(m.unreduced_sum)) / denom if denom > 0 else 0.0
  return out


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


async def _score_microbatches(
    engine: SwarmEngine,
    batches: Sequence[batch_assembly.AssembledBatch],
    config: SwarmConfig,
    step: int,
) -> tuple[list[datatypes.RLTrainerPayload], float, float]:
  """Validates sampler/actor logprob agreement and prepares training payloads."""
  scored = []
  max_error = 0.0
  diff_sum = 0.0
  diff_count = 0
  for batch in batches:
    payload = batch.payload
    old = np.asarray(await engine.score(payload), dtype=np.float32)
    active = payload.completion_mask > 0
    if (
        old.shape != payload.completion_ids.shape
        or not np.isfinite(old[active]).all()
    ):
      raise ValueError("Actor scoring returned invalid or misaligned logprobs.")
    abs_diff = np.abs(old[active] - payload.old_per_token_logps[active])
    error = float(np.max(abs_diff))
    max_error = max(max_error, error)
    diff_sum += float(np.sum(abs_diff))
    diff_count += int(abs_diff.size)
    if error > config.max_logprob_error:
      raise ValueError(
          f"Sampler/actor logprob mismatch: max={error:.6g},"
          f" mean={diff_sum / max(diff_count, 1):.6g}"
      )
    scored.append(
        payload
        if config.sampler_is is not None
        else dataclasses.replace(payload, old_per_token_logps=old)
    )
  mean_error = diff_sum / max(diff_count, 1)
  logging.info(
      "Step %d scored %d microbatches: max_logprob_error=%.6g,"
      " mean_logprob_error=%.6g.",
      step,
      len(scored),
      max_error,
      mean_error,
  )
  return scored, max_error, mean_error


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

  def restore_state(self, *, step: int, history: Sequence[dict]) -> None:
    """Continues after a committed round boundary restored by the caller.

    The caller restores the trainer checkpoint and the engine's published
    version; `run_async` then reads that version back through `prepare()`.

    Args:
      step: Number of rounds completed at the restored boundary.
      history: Per-round records completed before that boundary.
    """
    if self._is_running or self._failed or step < 0:
      raise ValueError("Restore an idle, healthy program to a valid step.")
    self._step = step
    self.history = list(history)

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
      engine.configure(
          shared_grpo_config(
              config.group_size,
              sampler_is=config.sampler_is,
              sampler_is_threshold=config.sampler_is_threshold,
              epsilon_high=config.epsilon_high,
          ),
          config,
      )
      self.policy_version = await engine.prepare()
      for _ in range(num_steps):
        episodes = tuple(await self.collect(self.policy_version, self.step))
        counts = collections.Counter(e.task_id for e in episodes)
        bad_group = (
            any(n < 2 for n in counts.values())
            if config.allow_variable_group_sizes
            else any(n != config.group_size for n in counts.values())
        )
        if not counts or bad_group:
          raise ValueError(
              "Collect complete groups before computing advantages."
          )
        if any(
            c.policy_version != self.policy_version
            for e in episodes
            for c in e.calls
            if c.trainable
        ):
          raise ValueError("Collection returned a stale policy version.")
        frozen_calls = sum(
            not c.trainable for e in episodes for c in e.calls
        )
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
              "frozen_calls": frozen_calls,
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
        logging.info(
            "Step %d collected %d episodes (%d calls, %d microbatches).",
            self.step + 1,
            len(episodes),
            len(credited.payloads),
            len(batches),
        )
        # Snapshot the actor's old logps for ALL batches before any update,
        # following upstream's actor recomputation, not a changing PPO baseline.
        scored, max_error, mean_error = await _score_microbatches(
            engine, batches, config, self.step + 1
        )
        if config.loss_reduction == "microbatch_token_mean":
          scored = weight_microbatches(
              scored, config.loss_microbatch_rows, seed=self.step
          )
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
            "frozen_calls": frozen_calls,
            "active_tokens": int(
                sum(np.sum(p.completion_mask) for p in scored)
            ),
            "reward_mean": self.last_step_result.reward_mean,
            "max_logprob_error": max_error,
            "mean_logprob_error": mean_error,
            "trainer_metrics": _trainer_metrics(result),
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

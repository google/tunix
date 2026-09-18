# Copyright 2026 Google LLC
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

"""Trajectory Collector Engine wrapping TrajectoryCollectEngine with pause/resume/cancel control."""

from typing import Any, Collection, List, Mapping, Sequence
import zlib
from absl import logging
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.rollout import sampler as sampler_lib
from tunix.experimental.rollout import vanilla_sampler_adapter
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.trajectory import trajectory_collect_engine as rl_collect_engine
from tunix.rl.rollout import base_rollout

_DEFAULT_EPISODE_TIMEOUT_SECS: float = 600.0


def generate_vanilla_rollout_seed(
    prompt_id: str | int,
    group_index: int = 0,
) -> int:
  """Generates a deterministic rollout seed for vanilla samplers.

  Computes `seed = (crc32(prompt_id) & 0x7FFFFFFF + group_index) & 0x7FFFFFFF`.

  Args:
    prompt_id: Prompt identifier string or integer (e.g. 'prompt_0', 42,
      'gsm8k_q1').
    group_index: The index of the rollout within its group (group_index >= 0).

  Returns:
    A deterministic 31-bit non-negative integer seed for the rollout request.
  """
  prompt_hash = zlib.crc32(str(prompt_id).encode("utf-8")) & 0x7FFFFFFF
  return (prompt_hash + group_index) & 0x7FFFFFFF


def _build_prompt(chat_parser: Any, chat_completions: Any) -> Any:
  """Vanilla samplers take a string; parse chat messages when needed."""
  if chat_parser and not isinstance(chat_completions, str):
    return chat_parser.parse(
        chat_completions, add_generation_prompt=True, is_first_msg=True
    )
  return chat_completions


def response_budget_facts(
    response_tokens: Sequence[int] | np.ndarray,
    max_response_length: int,
    eos_ids: Collection[int],
) -> tuple[int, bool]:
  """Computes `(raw_length, clipped)` for a generated response token sequence.

  A response is considered truncated (`clipped = True`) when its token length
  reaches or exceeds `max_response_length` without stopping on a configured EOS
  token (`int(response_tokens[-1]) in eos_ids`).

  TODO(tunix-dev): Move to a shared module and unify with
  `GRPOLearner._process_results` in `tunix/rl/agentic/agentic_grpo_learner.py`
  once agentic GRPO plumbs configured stop sets (`RolloutConfig.eos_tokens`).

  Args:
    response_tokens: Sequence or 1D array of response token IDs (including
      environment/tool turns).
    max_response_length: Maximum allowed response tokens for this rollout.
    eos_ids: Set of valid stop token IDs configured for the sampler.

  Returns:
    A tuple `(raw_length, clipped)` where `raw_length` is clamped to
    `max_response_length`.
  """
  if max_response_length <= 0:
    raise ValueError(
        f"max_response_length must be positive, got {max_response_length}"
    )
  raw_length = len(response_tokens)
  stopped_on_eos = (
      0 < raw_length <= max_response_length
      and int(response_tokens[-1]) in eos_ids
  )
  clipped = raw_length >= max_response_length and not stopped_on_eos
  return min(raw_length, max_response_length), clipped


class TrajectoryCollectorEngine:
  """Wrapper around TrajectoryCollectEngine providing lifecycle controls and Trajectory conversion."""

  def __init__(
      self,
      traj_id: str,
      request: datatypes.RolloutRequest,
      sampler: sampler_lib.Sampler,
      env_client: Any,
      agent: Any,
      tokenizer: Any,
      chat_parser: Any,
      eos_ids: Collection[int] | None = None,
  ):
    if (
        sampler is None
        or env_client is None
        or agent is None
        or tokenizer is None
        or chat_parser is None
    ):
      raise ValueError(
          "TrajectoryCollectorEngine requires valid sampler, env_client, agent,"
          " tokenizer, and chat_parser arguments (none can be None)."
      )
    self.traj_id = traj_id
    self.request = request
    self.sampler = sampler
    self.env = env_client
    self.agent = agent
    self.tokenizer = tokenizer
    self.chat_parser = chat_parser
    self.is_paused: bool = False
    self.is_cancelled: bool = False
    self.is_done: bool = False
    self.max_response_length = request.max_response_length
    # The stop set the sampler was configured with, which is what decides
    # whether a rollout ended on its own. Defined at the recipe level via
    # `RolloutConfig.eos_tokens` (e.g. `<|im_end|>` for Qwen chat models) rather
    # than forced from the base tokenizer.
    self.eos_ids = frozenset(int(token_id) for token_id in (eos_ids or ()))
    metadata = request.metadata or {}
    timeout = metadata.get("episode_timeout")
    self.episode_timeout = float(
        timeout if timeout is not None else _DEFAULT_EPISODE_TIMEOUT_SECS
    )
    if self.episode_timeout <= 0:
      raise ValueError("episode_timeout must be positive.")
    overlong_filter = metadata.get("overlong_filter")
    if overlong_filter is None:
      self.overlong_filter = False
    elif isinstance(overlong_filter, bool):
      self.overlong_filter = overlong_filter
    else:
      raise TypeError(
          "overlong_filter must be a boolean, got"
          f" {type(overlong_filter).__name__}: {overlong_filter!r}."
      )

  async def run_episode(self) -> agent_types.TrajectoryItem:
    """Executes multi-turn agentic rollout episode and returns TrajectoryItem."""
    # Note: model_call is an async coroutine callback invoked directly by
    # TrajectoryCollectEngine on the asyncio event loop without blocking
    # threads.
    async def model_call(
        chat_completions, env=None, max_generation_steps=None, **kwargs
    ):
      del env
      generation_kwargs = dict(self.request.generation_kwargs)
      # NB: extra kwargs can be passed in from trajectory_collect_engine.
      generation_kwargs.update(kwargs)
      request_max_generation_steps = generation_kwargs.pop(
          "max_generation_steps", None
      )
      req_max_tokens = (
          request_max_generation_steps
          if request_max_generation_steps is not None
          else generation_kwargs.get("max_tokens")
      )

      if max_generation_steps is not None and req_max_tokens is not None:
        effective_max_tokens = min(max_generation_steps, req_max_tokens)
      elif max_generation_steps is not None:
        effective_max_tokens = max_generation_steps
      elif req_max_tokens is not None:
        effective_max_tokens = req_max_tokens
      else:
        raise ValueError(
            "TrajectoryCollectorEngine requires"
            " request.max_response_length, request.generation_kwargs"
            " ('max_generation_steps' or 'max_tokens'), or the model_call"
            " callback to specify max_generation_steps."
        )

      generation_kwargs["max_tokens"] = effective_max_tokens

      seed = generation_kwargs.get("seed", None)
      if isinstance(
          self.sampler, vanilla_sampler_adapter.VanillaSamplerAdapter
      ):
        # TODO(tunix-dev): make vanilla sampler stateful with internal RNG key
        if seed is None and self.request.prompt_id is not None:
          seed = generate_vanilla_rollout_seed(
              self.request.prompt_id, self.request.group_index
          )
        if seed is None:
          raise ValueError(
              "Vanilla sampler requires a seed or valid prompt_id to generate"
              " diverse rollouts, but got seed=None."
          )

      sampling_params = sampler_lib.SamplingParams(
          max_tokens=effective_max_tokens,
          temperature=generation_kwargs.get("temperature", 0.0),
          top_p=generation_kwargs.get("top_p", None),
          top_k=generation_kwargs.get("top_k", None),
          seed=seed,
          return_logprobs=generation_kwargs.get("return_logprobs", False),
          return_routed_experts=generation_kwargs.get(
              "return_routed_experts",
              os.environ.get("ENABLE_ROUTER_REPLAY", "1") != "0",
          ),
          routed_experts_prompt_start=generation_kwargs.get(
              "routed_experts_prompt_start", 0
          ),
      )
      sampling_req = sampler_lib.SamplingRequest(
          request_id=self.traj_id,
          prompt=_build_prompt(self.chat_parser, chat_completions),
          sampling_params=sampling_params,
      )
      res = await self.sampler.sample(sampling_req, **generation_kwargs)
      text = res if isinstance(res, str) else getattr(res, "text", str(res))
      tokens = getattr(res, "token_ids", np.array([], dtype=np.int32))
      logprobs = getattr(res, "logprobs", None)
      routed_experts = getattr(res, "routed_experts", None)
      prompt_tokens = np.asarray(
          getattr(res, "prompt_token_ids", np.array([], dtype=np.int32)),
          dtype=np.int32,
      ).reshape(-1)
      if prompt_tokens.size:
        prompt_tokens = prompt_tokens.reshape(1, -1)
      else:
        prompt_tokens = np.array([[0]], dtype=np.int32)

      return base_rollout.RolloutOutput(
          text=[text],
          logits=None,
          tokens=[tokens],
          left_padded_prompt_tokens=prompt_tokens,
          logprobs=[logprobs] if logprobs is not None else None,
          routed_experts=[routed_experts] if routed_experts is not None else None,
      )

    if not self.agent or not self.env:
      raise RuntimeError(
          "RolloutCollector requires valid registered agent and env instances"
          " to run an episode."
      )
    inner_engine = rl_collect_engine.TrajectoryCollectEngine(
        agent=self.agent,
        env=self.env,
        model_call=model_call,  # pyrefly: ignore[bad-argument-type]
        tokenizer=self.tokenizer,
        chat_parser=self.chat_parser,
        max_response_length=self.max_response_length,
        timeout=self.episode_timeout,
        overlong_filter=self.overlong_filter,
    )
    rl_traj = await inner_engine.collect(mode="Token")
    self.is_done = True
    return self._convert_to_trajectory(rl_traj)

  def _annotate_response_budget(
      self, rl_traj: Mapping[str, Any], metadata: dict[str, Any]
  ) -> None:
    """Records whether this rollout was truncated by the response budget.

    Written here rather than by the consumer because only the producer knows
    the budget actually enforced: `max_response_length` is read per request,
    and `DistributedRLEngine` lets a dataset item override the default.

    Sets two keys in `metadata`, both computed by `response_budget_facts`:
      * `clipped`: reached the budget without stopping on a configured EOS.
      * `raw_length`: response tokens including env/tool turns, clamped to the
        budget (the rLLM/VERL `response_length` convention).

    Both are left unset when no budget was enforced, no EOS id is known, or
    there is no token stream, so consumers can tell "not truncated" from
    "unknown". Each of those skips warns once, because an absent metric is
    otherwise indistinguishable from a zero one on a dashboard. An empty stream
    is annotated rather than skipped: it is a rollout that produced nothing,
    and keeps its place in the group denominator.

    TODO(tunix-dev): Prefer `TrajectoryStatus.MAX_CONTEXT_LIMIT_REACHED`, which
    already reaches the consumer via `traj["status"]`, over this last-token
    heuristic. Deferred because that status is not EOS-aware and is evaluated
    per turn, so adopting it would shift the reported metric. Note
    `finish_reason` exists only on the sampler responses here; agentic has
    never had such a field.

    Args:
      rl_traj: Token-mode trajectory mapping, read-only.
      metadata: TrajectoryItem metadata dictionary to annotate in place.
    """
    skipped = "Not annotating clipped/raw_length for rollout %s: %s."
    if self.max_response_length is None:
      logging.log_first_n(
          logging.WARNING,
          skipped,
          1,
          self.traj_id,
          "the request carries no max_response_length, so no budget was"
          " enforced",
      )
      return
    if self.max_response_length <= 0:
      # A non-positive budget is a misconfiguration rather than "no budget":
      # every rollout would score as having reached it.
      logging.log_first_n(
          logging.WARNING,
          skipped,
          1,
          self.traj_id,
          f"max_response_length is {self.max_response_length!r}, which is"
          " not a usable budget",
      )
      return
    if not self.eos_ids:
      logging.log_first_n(
          logging.WARNING,
          skipped,
          1,
          self.traj_id,
          "no eos_tokens are configured in RolloutConfig, so termination"
          " cannot be detected",
      )
      return
    tokens = rl_traj.get("conversation_tokens")
    if tokens is None:
      logging.log_first_n(
          logging.WARNING,
          skipped,
          1,
          self.traj_id,
          "the trajectory carries no conversation_tokens",
      )
      return

    raw_length, clipped = response_budget_facts(
        tokens, self.max_response_length, self.eos_ids
    )
    metadata["raw_length"] = raw_length
    metadata["clipped"] = clipped

  def _convert_to_trajectory(
      self, rl_traj: dict[str, Any]
  ) -> agent_types.TrajectoryItem:
    """Converts internal Token-mode rollout trajectory to agent_types.TrajectoryItem."""
    if not isinstance(rl_traj, dict):
      raise TypeError(
          f"Expected rl_traj to be a dict, got {type(rl_traj).__name__}"
      )

    # Metadata carries only request-scoped context. Everything about the
    # episode itself stays on `traj`, which is the single source of truth and
    # matches how `agentic_grpo_learner` consumes rollouts. Mirroring episode
    # fields here previously let the copy drift from the original: the mirrored
    # reward was read instead of the real one, and the mirrored text held the
    # whole conversation rather than the model's answer.
    metadata = dict(self.request.metadata or {})
    metadata["prompt_id"] = self.request.prompt_id
    metadata["group_index"] = self.request.group_index
    metadata["status"] = rl_traj.get("status", "")
    policy_version = getattr(
        self.request,
        "target_policy_version",
        rl_traj.get("policy_version", 0),
    )
    metadata["policy_version"] = int(policy_version or 0)

    self._annotate_response_budget(rl_traj, metadata)

    return agent_types.TrajectoryItem(
        prompt_id=self.request.prompt_id,
        group_index=self.request.group_index,
        start_step=0,
        traj=rl_traj,
        metadata=metadata,
    )

  def pause(self) -> None:
    self.is_paused = True

  def resume(self) -> None:
    self.is_paused = False

  def cancel(self) -> None:
    self.is_cancelled = True
    self.is_done = True

  def get_accumulated_token_ids(self) -> List[int]:
    """Returns token IDs of historical turns for Raiden KV-cache transfer."""
    return []

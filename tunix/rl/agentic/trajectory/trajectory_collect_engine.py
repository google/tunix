# Copyright 2025 Google LLC
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

"""Engine for collecting trajectories from agent-environment interactions.

This module defines the `TrajectoryCollectEngine`, which facilitates the
asynchronous collection of rollouts by managing the interaction loop between
an LLM-based agent and an environment. It supports single and concurrent
multi-pair trajectory collection.
"""

import asyncio
import copy
import inspect
import json
import time
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set, Tuple

from absl import logging
import numpy as np
from tunix.experimental.trajectory import converter as converter_lib
from tunix.experimental.trajectory import store as store_lib
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.generate import utils as generate_utils
from tunix.perf.experimental import constants as perf_constants
from tunix.perf.experimental import tracer as perf_tracer_v2
from tunix.rl.agentic import utils
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.agents import base_agent
from tunix.rl.agentic.environments import base_environment
from tunix.rl.rollout import base_rollout

BaseTaskEnv = base_environment.BaseTaskEnv
ConversationAgentBase = base_agent.ConversationAgentBase


class TrajectoryCollectEngine:
  """Asynchronous trajectory collection engine for agent-env interactions.

  This engine orchestrates complete rollout episodes by managing the interaction
  loop between LLM-based agents and environments. It handles model inference,
  environment stepping, reward computation, and trajectory storage with support
  for concurrent multi-pair execution and streaming results.

  The engine implements the standard RL rollout pattern: reset → step* → final
  reward computation → return calculation, while providing flexible callback
  integration for custom model calls and reward functions.
  """

  def __init__(
      self,
      agent: ConversationAgentBase,
      env: BaseTaskEnv,
      *,
      model_call: Callable[..., base_rollout.RolloutOutput],
      model_call_kwargs: Optional[Dict[str, Any]] = None,
      gamma: float = 1.0,
      max_response_length: Optional[int] = None,
      timeout: float = 600.0,
      tokenizer=None,
      chat_parser=None,
      filter_statuses: Optional[Set[agent_types.TrajectoryStatus]] = None,
      overlong_filter: bool = False,
      perf_v2: Optional[perf_tracer_v2.Tracer] = None,
      exact_token_continuity: bool = False,
      policy_version: Optional[int] = None,
      trajectory_store: Optional[store_lib.TrajectoryWriter] = None,
      metadata: Optional[trajectory_lib.TrajectoryMetadata] = None,
  ):
    """Initialize the trajectory collection engine.

    Args:
        agent (ConversationAgentBase): The agent that will interact with the
          environment
        env (BaseTaskEnv): The environment providing tasks and feedback
        model_call (Callable): Function that takes chat completions as first
          argument with optional kwargs and returns model response string.
          Handles the actual LLM inference.
        model_call_kwargs (Optional[Dict[str, Any]]): Optional kwargs to pass to
          model_call.
        gamma (float): Discount factor for MC reward calculation (1.0 = no
          discounting).
        max_response_length (Optional[int]): Maximum number of context tokens to
          use before forced termination.
        timeout (float): Maximum episode duration in seconds before timeout
          termination
        tokenizer: Optional tokenizer for converting messages to token IDs. This
          is required if we want to track down token counts.
        chat_parser: Optional chat parser for formatting messages
        filter_statuses (Set[TrajectoryStatus]): A set of statuses that are
          masked out for overlong filtering.
        overlong_filter: Whether to filter overlong trajectories.
        perf_v2 (Optional[perf_tracer_v2.Tracer]): Optional performance tracer
          to use for performance measurements. Defaults to a no-op tracer.
        exact_token_continuity: Preserve recorded token history on later turns.
          Requires a token-aware model_call, tokenizer, and parser.
        policy_version: Optional policy version integer to pass down for
          trajectory and performance tracing.
        trajectory_store: Optional TrajectoryWriter to write trajectory steps
          to.
        metadata: Optional TrajectoryMetadata for the current episode.
    """
    self.agent = agent
    self.env = env
    self.model_call = model_call
    self.policy_version = policy_version
    self.trajectory_store = trajectory_store
    self.metadata = metadata
    self.final_reward_fn = None
    self.model_call_kwargs = model_call_kwargs or {}
    if exact_token_continuity and (tokenizer is None or chat_parser is None):
      raise ValueError("exact_token_continuity requires a tokenizer and parser")
    self.exact_token_continuity = exact_token_continuity
    self._exact_chat_history = None
    self.perf_v2 = (
        perf_v2 if perf_v2 is not None else perf_tracer_v2.NoopTracer()
    )
    self.max_steps = getattr(self.env, "max_steps", 1)
    self.gamma = gamma
    self.max_response_length = max_response_length
    self._response_token_count = 0
    self.timeout = timeout

    # Tokenizer utilities for stepwise tokenization
    self.tokenizer = tokenizer
    self.chat_parser = chat_parser
    self._start_ts: float = 0.0
    self._logged_clip_reasons: Set[str] = set()
    self.filter_statuses = filter_statuses or {
        agent_types.TrajectoryStatus.MAX_STEPS_REACHED,
        agent_types.TrajectoryStatus.MAX_CONTEXT_LIMIT_REACHED,
        agent_types.TrajectoryStatus.TIMEOUT,
        agent_types.TrajectoryStatus.ENV_TIMEOUT,
    }

    self.overlong_filter = overlong_filter
    self.perf_v2 = perf_v2 or perf_tracer_v2.NoopTracer()
    self._cumulative_prompt_tokens: int = 0
    self._current_step_initial_routed_experts: Optional[np.ndarray] = None
    self.env_time = {
        "reset_latency": 0.0,  # Wall-clock time (Total real-world time elapsed)
        "step_latency": [],  # List of per-step wall-clock times, ordered by step index
        "close_latency": 0.0,  # Wall-clock time (Total real-world time elapsed)
    }
    self.reward_time = {
        "reward_latency": (
            0.0
        ),  # Wall-clock time (Total real-world time elapsed)
    }

    if self.max_response_length is not None and not (
        self.tokenizer and self.chat_parser
    ):
      logging.warning(
          "max_response_length is set to %d, but no tokenizer or chat_parser is"
          " provided. response length limits will not be enforced.",
          self.max_response_length,
      )

  async def _run_with_timing(
      self, func: Callable[..., Any], *args, timeout: Optional[float] = None
  ) -> Tuple[Any, float]:
    """Runs a sync function in an executor and returns (result, wall_time).

    Args:
      func: Synchronous callable to run in the executor.
      *args: Positional arguments forwarded to func.
      timeout: Optional deadline in seconds. If provided and the executor future
        does not finish in time, asyncio.TimeoutError is re-raised to the
        caller.

    Returns:
      A tuple of (result, wall_time).

    Raises:
      asyncio.TimeoutError: When timeout is not None and is exceeded.
    """
    loop = asyncio.get_running_loop()
    wall_start = time.perf_counter()

    fut = loop.run_in_executor(None, func, *args)
    if timeout is not None:
      result = await asyncio.wait_for(fut, timeout=timeout)
    else:
      result = await fut

    wall_delta = time.perf_counter() - wall_start
    return result, wall_delta

  def _log_trajectory_clip(self, reason: str) -> None:
    """Logs the reason a trajectory was clipped."""
    if reason in self._logged_clip_reasons:
      return
    self._logged_clip_reasons.add(reason)
    logging.warning("%s trajectory clipped: %s", self._debug_prefix, reason)

  def _finalize_terminal_step_routing(self) -> None:
    """Pad the terminal step's unrouted last token with UNSET_ROUTED_EXPERT.

    In autoregressive sampling, vLLM routes P + G - 1 tokens; the G-th token
    is sampled at the end of decode and only passes through an MoE layer if a
    subsequent turn prefills it. On episode termination, the final step's last
    token was never forwarded through a subsequent prefill. In causal loss,
    targets are rolled by -1 and targets_segmentation is 0 at the final token,
    so padding with UNSET_ROUTED_EXPERT has zero effect on training loss.
    """
    if not self.agent.trajectory.steps:
      return
    final_step = self.agent.trajectory.steps[-1]
    if (
        final_step.assistant_routed_experts is not None
        and final_step.assistant_tokens is not None
        and len(final_step.assistant_routed_experts)
        < len(final_step.assistant_tokens)
    ):
      missing = len(final_step.assistant_tokens) - len(
          final_step.assistant_routed_experts
      )
      pad = np.full(
          (missing,) + final_step.assistant_routed_experts.shape[1:],
          agent_types.UNSET_ROUTED_EXPERT,
          dtype=np.int16,
      )
      final_step.assistant_routed_experts = np.concatenate(
          [final_step.assistant_routed_experts, pad], axis=0
      )

  def sync_trajectory_metadata(
      self,
      status: Optional[str | agent_types.TrajectoryStatus] = None,
      policy_version: Optional[int] = None,
  ) -> None:
    """Syncs metadata status and agent trajectory timing/reward in place."""
    if self.metadata is not None:
      effective_policy_version = (
          policy_version if policy_version is not None else self.policy_version
      )
      converter_lib.update_trajectory_metadata(
          metadata=self.metadata,
          agent=self.agent,
          policy_version=effective_policy_version,
          status=status,
          env_time=self.env_time,
          reward_time=self.reward_time,
      )

  def _record_task_step(self) -> None:
    """Writes the initial task step (step 0) live to trajectory_store."""
    if self.trajectory_store is None or self.metadata is None:
      return
    self.sync_trajectory_metadata(policy_version=self.policy_version)
    task = self.agent.trajectory.task or getattr(self.env, "task", None)
    task_step = converter_lib.create_task_step(task)
    if task_step is not None:
      self.trajectory_store.add_step(task_step, self.metadata)

  def _record_agent_step(
      self,
      step: Optional[agent_types.Step],
      step_idx: Optional[int] = None,
  ) -> None:
    """Writes or updates an agent turn step (2*i + 1) in trajectory_store."""
    if self.trajectory_store is None or self.metadata is None or step is None:
      return
    if step_idx is None:
      step_idx = max(0, len(self.agent.trajectory.steps) - 1)
    agent_step = converter_lib.create_agent_step(
        step,
        tunix_step_id=step_idx,
        policy_version=self.policy_version,
    )
    if agent_step is not None:
      self.sync_trajectory_metadata(policy_version=self.policy_version)
      self.trajectory_store.add_step(agent_step, self.metadata)

  def _record_env_step(
      self,
      step: Optional[agent_types.Step],
      step_idx: Optional[int] = None,
  ) -> None:
    """Writes or updates an environment turn step (2*i + 2) in trajectory_store."""
    if self.trajectory_store is None or self.metadata is None or step is None:
      return
    if step_idx is None:
      step_idx = max(0, len(self.agent.trajectory.steps) - 1)
    env_step = converter_lib.create_env_step(step, tunix_step_id=step_idx)
    if env_step is not None:
      self.sync_trajectory_metadata()
      self.trajectory_store.add_step(env_step, self.metadata)

  def _finalize_assistant_step(
      self,
      cur_step: Optional[agent_types.Step],
      rollout_output: base_rollout.RolloutOutput,
  ) -> None:
    """Populates assistant tokens, masks, logprobs, and routed experts on cur_step."""
    if cur_step is None:
      return
    if rollout_output.logprobs is not None:
      cur_step.logprobs = rollout_output.logprobs[0]
    if self._current_step_initial_routed_experts is not None:
      cur_step.assistant_routed_experts = (
          self._current_step_initial_routed_experts
      )
    if self.tokenizer and self.chat_parser and rollout_output.tokens:
      assistant_message, _ = utils.get_recent_assistant_user_messages(
          self.agent.chat_completions
      )
      if assistant_message:
        cur_step.assistant_tokens, n_append = (
            self.chat_parser.update_assistant_end_tokens(
                rollout_output.tokens[0]
            )
        )
        if self.exact_token_continuity:
          cur_step.assistant_tokens = utils.assistant_with_suffix(
              rollout_output.tokens[0], cur_step.assistant_tokens, n_append
          )
        cur_step.assistant_masks = np.concatenate(
            [
                np.ones(len(rollout_output.tokens[0]), dtype=np.int32),
                np.zeros(n_append, dtype=np.int32),
            ],
            axis=0,
        )
        if cur_step.logprobs is not None:
          cur_step.logprobs = np.concatenate(
              [cur_step.logprobs, np.zeros(n_append, dtype=np.float32)],
              axis=0,
          )

  async def collect(self, mode: str = "Conversation") -> Any:
    """Execute a complete rollout episode and return the resulting trajectory.

    Orchestrates the full interaction sequence: environment reset, iterative
    agent-environment steps, final reward computation, Monte Carlo return
    calculation, and resource cleanup.

    Args:
        mode (str): Output format. Options:
          - "Trajectory": return full Trajectory object.
          - "Token": return flattened tokenized dict for training.
          - "Steps": return stepwise tokenized data only.
          - "Conversation": return raw conversation messages (default).

    Returns:
        Trajectory | dict | list: Depending on mode.
    """  # fmt: skip
    try:
      await self._reset()

      self.agent.trajectory.status = agent_types.TrajectoryStatus.RUNNING
      self._logged_clip_reasons.clear()
      self._record_task_step()

      while True:
        if len(self.agent.trajectory.steps) >= self.max_steps:
          self.agent.trajectory.status = (
              agent_types.TrajectoryStatus.MAX_STEPS_REACHED
          )
          self._log_trajectory_clip("MAX_STEPS_REACHED")
          break

        done = await self._one_step()

        if done:
          if (
              self.agent.trajectory.status
              == agent_types.TrajectoryStatus.RUNNING
          ):
            self.agent.trajectory.status = (
                agent_types.TrajectoryStatus.SUCCEEDED
            )
          break

      self._finalize_terminal_step_routing()

      masked_out = (
          self.overlong_filter
          and self.agent.trajectory.status in self.filter_statuses
      )
      try:
        if not masked_out:
          await self._append_final_reward()
        self.compute_mc_reward()
        self.compute_trajectory_reward()
      finally:
        await self._close()
        for i, step in enumerate(self.agent.trajectory.steps):
          self._record_agent_step(step, step_idx=i)
    except asyncio.TimeoutError:
      self.agent.trajectory.status = agent_types.TrajectoryStatus.TIMEOUT
      raise
    except (asyncio.CancelledError, Exception):
      self.agent.trajectory.status = agent_types.TrajectoryStatus.FAILED
      raise
    finally:
      if self.trajectory_store is not None and self.metadata is not None:
        self.sync_trajectory_metadata(policy_version=self.policy_version)
        self.trajectory_store.update_metadata(self.metadata)

    if mode not in ["Trajectory", "Steps", "Token", "Conversation"]:
      raise ValueError(
          f"Unsupported mode: {mode}, currently supported modes: "
          f" {['Trajectory', 'Steps', 'Token', 'Conversation']}",
      )

    if mode == "Trajectory":
      self.agent.trajectory.env_time = self.env_time  # pyrefly: ignore[bad-assignment]
      self.agent.trajectory.reward_time = self.reward_time
      return self.agent.trajectory
    elif mode == "Steps":
      return [
          {
              "assistant_text": getattr(step, "model_response", ""),
              "env_text": getattr(step, "observation", ""),
              "done": getattr(step, "done", False),
              "assistant_tokens": getattr(step, "assistant_tokens", []),
              "assistant_masks": getattr(step, "assistant_masks", []),
              "env_tokens": getattr(step, "env_tokens", []),
              "env_masks": getattr(step, "env_masks", []),
              "assistant_routed_experts": getattr(
                  step, "assistant_routed_experts", None
              ),
              "env_routed_experts": getattr(step, "env_routed_experts", None),
              "reward": step.reward,
              "mc_return": step.mc_return,
              "env_time": self.env_time,
              "reward_time": self.reward_time,
          }
          for step in self.agent.trajectory.steps
      ]
    elif mode == "Token":
      # flatten all steps into single batch dict
      conversation_tokens, conversation_masks, logprobs = [], [], []
      routed_experts = []
      prompt_tokens = getattr(self.agent.trajectory, "prompt_tokens", [])
      has_routed_experts = getattr(
          self.agent.trajectory, "prompt_routed_experts", None
      ) is not None or any(
          getattr(step, "assistant_routed_experts", None) is not None
          or getattr(step, "env_routed_experts", None) is not None
          for step in self.agent.trajectory.steps
      )

      for idx, step in enumerate(self.agent.trajectory.steps):
        # Keep tokens/masks/logprobs/routed_experts appended in lockstep.
        assistant_tokens = getattr(step, "assistant_tokens", None)
        env_tokens = getattr(step, "env_tokens", None)
        step_logprobs = getattr(step, "logprobs", None)
        step_routed = getattr(step, "assistant_routed_experts", None)
        step_env_routed = getattr(step, "env_routed_experts", None)
        if assistant_tokens is not None:
          conversation_tokens.append(assistant_tokens)
          conversation_masks.append(step.assistant_masks)
          if step_logprobs is not None:
            assert len(step_logprobs) == len(assistant_tokens), (
                f"Logprobs length {len(step_logprobs)} does not match assistant"
                f" tokens length {len(assistant_tokens)}"
            )
            logprobs.append(step_logprobs)
          else:
            logprobs.append(np.zeros(len(assistant_tokens)))
          if has_routed_experts:
            if step_routed is None:
              raise ValueError(
                  f"Step {idx} has assistant_tokens (len"
                  f" {len(assistant_tokens)}) but missing"
                  " assistant_routed_experts while routed_experts is active."
              )
            if len(step_routed) != len(assistant_tokens):
              raise ValueError(
                  f"Step {idx} assistant_routed_experts length"
                  f" {len(step_routed)} does not match assistant_tokens length"
                  f" {len(assistant_tokens)}."
              )
            routed_experts.append(np.asarray(step_routed, dtype=np.int16))
        if env_tokens is not None:
          conversation_tokens.append(env_tokens)
          conversation_masks.append(step.env_masks)
          logprobs.append(np.zeros(len(env_tokens)))
          if has_routed_experts:
            if step_env_routed is None:
              raise ValueError(
                  f"Step {idx} has env_tokens (len {len(env_tokens)}) but"
                  " missing env_routed_experts while routed_experts is active."
              )
            if len(step_env_routed) != len(env_tokens):
              raise ValueError(
                  f"Step {idx} env_routed_experts length"
                  f" {len(step_env_routed)} does not match env_tokens length"
                  f" {len(env_tokens)}."
              )
            routed_experts.append(np.asarray(step_env_routed, dtype=np.int16))

      conversation_tokens = [
          np.asarray(tokens)
          for tokens in conversation_tokens
          if len(tokens) > 0
      ]
      conversation_masks = [
          np.asarray(masks) for masks in conversation_masks if len(masks) > 0
      ]
      logprobs = [
          np.asarray(step_logprobs)
          for step_logprobs in logprobs
          if len(step_logprobs) > 0
      ]
      conversation_masks = (
          np.concatenate(conversation_masks, axis=0)
          if conversation_masks
          else np.array([], dtype=np.int32)
      )
      conversation_tokens = (
          np.concatenate(conversation_tokens, axis=0)
          if conversation_tokens
          else np.array([], dtype=np.int32)
      )
      final_masks = (
          np.zeros_like(conversation_masks)
          if masked_out
          else conversation_masks
      )

      final_routed_experts = None
      if has_routed_experts:
        prompt_routed = getattr(
            self.agent.trajectory, "prompt_routed_experts", None
        )
        sample_arr = next(
            iter(routed_experts),
            prompt_routed,
        )
        sample_shape = (
            sample_arr.shape[1:] if sample_arr is not None else (0, 0)
        )
        conv_routed = (
            np.concatenate(routed_experts, axis=0)
            if routed_experts
            else np.zeros((0,) + sample_shape, dtype=np.int16)
        )
        prompt_len = (
            (self.agent.trajectory.prompt_length or 0)
            if self.exact_token_continuity
            else (len(prompt_tokens) if prompt_tokens is not None else 0)
        )
        if prompt_len > 0:
          if prompt_routed is None:
            raise ValueError(
                f"Trajectory has prompt_tokens (len {prompt_len}) but missing"
                " prompt_routed_experts while routed_experts is active."
            )
          if getattr(prompt_routed, "shape", (0,))[0] != prompt_len:
            raise ValueError(
                "prompt_routed_experts shape"
                f" {getattr(prompt_routed, 'shape', None)} does not match"
                f" prompt_tokens length {prompt_len}."
            )
          prompt_routed_arr = np.asarray(prompt_routed, dtype=np.int16)
        else:
          prompt_routed_arr = np.zeros((0,) + sample_shape, dtype=np.int16)

        final_routed_experts = np.concatenate(
            [prompt_routed_arr, conv_routed], axis=0
        )

      policy_version = self.policy_version
      if (
          policy_version is None
          and hasattr(self.env, "task")
          and isinstance(self.env.task, dict)
      ):
        policy_version = self.env.task.get("policy_version")

      result = {
          "conversation_text": self.agent.chat_completions,
          "prompt_tokens": prompt_tokens,
          "conversation_tokens": conversation_tokens,
          "conversation_masks": final_masks,
          "status": self.agent.trajectory.status.name,
          "trajectory_reward": self.agent.trajectory.reward,
          "env_time": self.env_time,
          "reward_time": self.reward_time,
          "old_logprobs": (
              np.concatenate(logprobs, axis=0) if logprobs else None
          ),
          "routed_experts": final_routed_experts,
          "policy_version": policy_version,
          "original_input": self.agent.trajectory.task,
          "group_id": self.env.extra_kwargs.get("group_id"),
      }
      if self.agent.trajectory.prompt_length is not None:
        # Set only by exact token continuity; lets training unpad by length.
        result["prompt_length"] = self.agent.trajectory.prompt_length
      return result
    elif mode == "Conversation":
      # return raw conversation history
      return self.agent.chat_completions

  @staticmethod
  async def collect_multiple(
      pairs: List[Tuple[ConversationAgentBase, BaseTaskEnv]],
      *,
      model_call: Callable[..., base_rollout.RolloutOutput],
      gamma: float = 1.0,
      timeout: float = 30.0,
      max_response_length: Optional[int] = None,
      mode: str = "Trajectory",
      filter_statuses: Optional[Set[agent_types.TrajectoryStatus]] = None,
      overlong_filter: bool = True,
      perf_v2: Optional[perf_tracer_v2.Tracer] = None,
  ) -> AsyncGenerator[Tuple[int, Any], None]:
    """Execute multiple agent-environment pairs concurrently.

    Runs multiple rollouts in parallel and yields completed trajectories
    as they finish, enabling efficient batch processing with streaming
    results. Useful for distributed training or large-scale evaluation.

    Args:
        pairs (List[Tuple[ConversationAgentBase, BaseTaskEnv]]): List of (agent,
          environment) pairs
        model_call (Callable): Shared model inference function for all pairs
        gamma (float): Discount factor for return calculation
        timeout (float): Per-episode timeout in seconds
        max_response_length (Optional[int]): Maximum context limit per episode
        mode (str): Output format. See `collect` method for options.
        filter_statuses (Optional[Set[TrajectoryStatus]]): A set of statuses
          that are masked out for filtering.
        overlong_filter (bool): Whether to filter overlong trajectories.
        perf_v2 (Optional[perf_tracer_v2.Tracer]): Optional performance tracer
          to use for performance measurements.

    Yields:
        Tuple[int, Any]: `(pair_index, result)`. The type of `result`
          depends on the `mode` argument. See the `collect` method for details.
    """

    async def _run_one(i: int, agent: ConversationAgentBase, env: BaseTaskEnv):
      """Execute a single agent-env pair with the given configuration."""
      engine = TrajectoryCollectEngine(
          agent,
          env,
          model_call=model_call,
          gamma=gamma,
          max_response_length=max_response_length,
          timeout=timeout,
          filter_statuses=filter_statuses,
          overlong_filter=overlong_filter,
          perf_v2=perf_v2,
      )
      traj = await engine.collect(mode=mode)
      return i, traj

    # Launch all pairs concurrently and yield results as they complete
    tasks = [_run_one(i, agent, env) for i, (agent, env) in enumerate(pairs)]
    for coro in asyncio.as_completed(tasks):
      yield await coro

  async def _reset(self):
    """Resets the environment and agent at the beginning of a new episode.

    This involves calling the environment's reset method, updating the agent's
    state, and optionally tokenizing the initial prompt messages.
    """
    logging.debug("%s env.reset starting", self._debug_prefix)
    (obs, info), wall_time = await self._run_with_timing(self.env.reset)
    logging.debug(
        "%s env.reset done in %.1fs",
        self._debug_prefix,
        wall_time,
    )

    self.env_time["reset_latency"] += wall_time
    self.final_reward_fn = (
        self.env.final_reward_fn
        if hasattr(self.env, "final_reward_fn")
        else None
    )
    self.agent.reset()
    self._start_ts = time.perf_counter()
    self._response_token_count = 0
    self._cumulative_prompt_tokens = 0
    self._current_step_initial_routed_experts = None
    self.agent.update_from_env(
        observation=obs,
        reward=0.0,
        done=False,
        info=self._rollout_state_info(info),
    )

    if (
        self.tokenizer is not None
        and self.chat_parser is not None
        and not self.exact_token_continuity
    ):
      # Get the current messages (usually System + User)
      init_messages = self.agent.chat_completions
      prompt_tokens, _ = utils.tokenize_and_generate_masks(
          init_messages,
          tokenizer=self.tokenizer,
          parser=self.chat_parser,
          contains_first_msg=True,
          contains_generation_msg=True,
      )
      self.agent.trajectory.prompt_tokens = prompt_tokens  # pyrefly: ignore[missing-attribute]
    if self.exact_token_continuity:
      self._exact_chat_history = copy.deepcopy(self.agent.chat_completions)

  def _record_exact_turn(self, cur_step, *, terminal: bool) -> None:
    """Closes the recorded turn and checks the agent only appended messages.

    Raises:
      ValueError: the agent rewrote earlier chat history, which would silently
        desynchronize text from the recorded ids.
    """
    messages = self.agent.chat_completions
    previous = self._exact_chat_history
    if previous is not None and messages[: len(previous)] != previous:
      raise ValueError("agent rewrote previously recorded chat history")
    if terminal and cur_step is not None:
      cur_step.done = True
    self._exact_chat_history = copy.deepcopy(messages)

  @property
  def _debug_prefix(self) -> str:
    """Returns a consistent log prefix with step_idx, pair_index, and group_id."""
    extra = getattr(self.env, "extra_kwargs", {}) or {}
    step_idx = len(self.agent.trajectory.steps)
    pair_index = extra.get("pair_index")
    group_id = extra.get("group_id")
    return (
        f"[step_idx={step_idx}, pair_index={pair_index}, group_id={group_id}]"
    )

  def _rollout_state_info(
      self, info: Optional[Dict[str, Any]] = None
  ) -> Dict[str, Any]:
    """Adds engine-managed rollout metadata for agents."""
    enriched_info = dict(info or {})
    enriched_info["max_steps"] = self.max_steps
    enriched_info["cur_tokens"] = self._response_token_count
    return enriched_info

  def _get_perf_tags(self) -> Dict[str, Any]:
    """Extracts performance tracing tags from the environment."""
    tags = {}
    if hasattr(self.env, "extra_kwargs"):
      group_id = self.env.extra_kwargs.get("group_id")
      if group_id is not None:
        tags[perf_constants.GROUP_ID] = group_id
      pair_index = self.env.extra_kwargs.get("pair_index")
      if pair_index is not None:
        tags[perf_constants.PAIR_INDEX] = pair_index
    if self.policy_version is not None:
      tags[perf_constants.STEP] = self.policy_version
    elif hasattr(self.env, "task") and isinstance(self.env.task, dict):
      policy_version = self.env.task.get("policy_version")
      if policy_version is not None:
        tags[perf_constants.STEP] = policy_version
    return tags

  def _check_and_set_context_limit_reached(self) -> bool:
    """Returns True and updates trajectory status if response budget is exhausted."""
    if (
        self.max_response_length is not None
        and self._response_token_count >= self.max_response_length
    ):
      self.agent.trajectory.status = (
          agent_types.TrajectoryStatus.MAX_CONTEXT_LIMIT_REACHED
      )
      self._log_trajectory_clip("MAX_CONTEXT_LIMIT_REACHED")
      return True
    return False

  async def _one_step(self) -> bool:
    """Executes a single step and returns the Step object and Done status.

    This involves calling the model, updating the agent with the response,
    stepping the environment with the agent's action, and updating the agent
    with the environment's feedback.

    Returns:
        bool: True if the episode is done (either by environment or timeout),
          False otherwise.
    """
    if self._check_and_set_context_limit_reached():
      return True
    max_generation_steps = (
        self.max_response_length - self._response_token_count
        if self.max_response_length is not None
        else None
    )
    logging.debug("%s model_call starting", self._debug_prefix)

    chat_input = self.agent.chat_completions
    call_kwargs = dict(self.model_call_kwargs)
    if self.exact_token_continuity and self.agent.trajectory.steps:
      chat_input = None  # later turn: recorded ids, not re-encoded text
      call_kwargs["prompt_token_ids"] = utils.continuation_prompt_tokens(
          self.agent.trajectory
      )

    model_call_fn = self.model_call
    is_async = inspect.iscoroutinefunction(model_call_fn) or (
        hasattr(model_call_fn, "__call__")
        and inspect.iscoroutinefunction(  # pytype: disable=not-supported-yet
            getattr(model_call_fn, "__call__")
        )
    )
    if self._cumulative_prompt_tokens > 0:
      call_kwargs["routed_experts_prompt_start"] = (
          self._cumulative_prompt_tokens
      )

    if is_async:
      try:
        rollout_output = await model_call_fn(  # pytype: disable=bad-return-type
            chat_input,
            self.env,
            max_generation_steps=max_generation_steps,
            **call_kwargs,
        )
      except Exception as e:
        logging.exception("Caught exception inside async model_call: %s", e)
        raise
    else:

      def _safe_model_call():
        try:
          return model_call_fn(
              chat_input,
              self.env,
              max_generation_steps=max_generation_steps,
              **call_kwargs,
          )
        except Exception as e:
          logging.exception("Caught exception inside model_call: %s", e)
          raise

      rollout_output = await asyncio.get_running_loop().run_in_executor(
          None,
          _safe_model_call,
      )
    logging.debug("%s model_call done", self._debug_prefix)

    if self.exact_token_continuity:
      if not self.agent.trajectory.steps:
        # The owned first-turn prompt; later turns replay exactly these ids.
        self.agent.trajectory.prompt_tokens = (  # pyrefly: ignore[missing-attribute]
            rollout_output.left_padded_prompt_tokens[0]
        )
        self.agent.trajectory.prompt_length = int(
            rollout_output.prompt_lengths[0]
        )
      else:
        echoed = generate_utils.unpad_prompt(
            rollout_output.left_padded_prompt_tokens[0],
            rollout_output.prompt_lengths[0],
        )
        if not np.array_equal(echoed, call_kwargs["prompt_token_ids"]):
          raise ValueError("later-turn prompt differs from recorded history")

    self._current_step_initial_routed_experts = None
    if (
        not self.agent.trajectory.steps
        and rollout_output.routed_experts
        and rollout_output.routed_experts[0] is not None
    ):
      init_routed = np.asarray(rollout_output.routed_experts[0], dtype=np.int16)
      prompt_len = (
          (self.agent.trajectory.prompt_length or 0)
          if self.exact_token_continuity
          else (
              len(self.agent.trajectory.prompt_tokens)
              if getattr(self.agent.trajectory, "prompt_tokens", None)
              is not None
              else 0
          )
      )
      self.agent.trajectory.prompt_routed_experts = (  # pyrefly: ignore[missing-attribute]
          init_routed[:prompt_len]
      )
      self._cumulative_prompt_tokens = init_routed.shape[0]
      self._current_step_initial_routed_experts = init_routed[prompt_len:]
    elif (
        self.agent.trajectory.steps
        and rollout_output.routed_experts
        and rollout_output.routed_experts[0] is not None
    ):
      delta_routed = np.asarray(
          rollout_output.routed_experts[0], dtype=np.int16
      )
      prev_step = self.agent.trajectory.steps[-1]
      needed_asst = 0
      if (
          prev_step.assistant_tokens is not None
          and prev_step.assistant_routed_experts is not None
      ):
        needed_asst = max(
            0,
            len(prev_step.assistant_tokens)
            - len(prev_step.assistant_routed_experts),
        )
        if needed_asst > 0:
          if len(delta_routed) < needed_asst:
            raise ValueError(
                f"Insufficient delta_routed length {len(delta_routed)} to"
                f" stitch {needed_asst} trailing assistant tokens at step"
                f" {len(self.agent.trajectory.steps) - 1}."
            )
          prev_step.assistant_routed_experts = np.concatenate(
              [prev_step.assistant_routed_experts, delta_routed[:needed_asst]],
              axis=0,
          )
      num_env = (
          len(prev_step.env_tokens) if prev_step.env_tokens is not None else 0
      )
      if num_env > 0:
        prev_step.env_routed_experts = delta_routed[
            needed_asst : needed_asst + num_env
        ]
        if len(prev_step.env_routed_experts) != num_env:
          raise ValueError(
              "Mismatch between captured env_routed_experts length "
              f"{len(prev_step.env_routed_experts)} and env_tokens length "
              f"{num_env} at step {len(self.agent.trajectory.steps) - 1}."
          )
      self._current_step_initial_routed_experts = delta_routed[
          needed_asst + num_env :
      ]
      self._cumulative_prompt_tokens += delta_routed.shape[0]

    if rollout_output.tokens:
      self._response_token_count += len(rollout_output.tokens[0])

    action = self.agent.update_from_model(rollout_output.text[0]).action
    cur_step = self.agent.get_current_step()
    self._finalize_assistant_step(cur_step, rollout_output)
    self._record_agent_step(cur_step)

    logging.debug(
        "%s Agent Action:\n%s",
        self._debug_prefix,
        json.dumps(action, default=str, indent=2),
    )
    if action is None:
      logging.warning(
          "Agent returned None action, using empty action list as fallback"
      )
      action = []

    step_idx = len(self.agent.trajectory.steps)
    remaining_time = self.timeout - (time.perf_counter() - self._start_ts)

    tags = self._get_perf_tags()
    if not self._check_and_set_context_limit_reached():
      try:
        with self.perf_v2.span(
            perf_constants.ENVIRONMENT,
            tags=tags,
        ):
          (obs, rew, done, info), wall_time = await self._run_with_timing(
              self.env.step, action, timeout=remaining_time
          )
      except asyncio.TimeoutError:
        self.agent.trajectory.status = agent_types.TrajectoryStatus.ENV_TIMEOUT
        self._log_trajectory_clip("ENV_TIMEOUT")
        if step_idx == 0:
          logging.error(
              "%s env.step hung at step 0 (first action) and was killed after"
              " %.1f s remaining timeout. This trajectory produced no usable"
              " data. Consider investigating the environment.",
              self._debug_prefix,
              remaining_time,
          )
        else:
          logging.error(
              "%s env.step hung at step %d and was killed after %.1f s"
              " remaining timeout.",
              self._debug_prefix,
              step_idx,
              remaining_time,
          )
        cur_step = self.agent.get_current_step()
        if cur_step is not None:
          cur_step.done = True
        self._record_env_step(cur_step)
        return True

      self.env_time["step_latency"].append(wall_time)

      logging.debug(
          "%s Env Observation (Rew: %s, Done: %s):\n%s",
          self._debug_prefix,
          rew,
          done,
          json.dumps(obs, default=str, indent=2),
      )
      logging.debug(
          "%s Env Info:\n%s",
          self._debug_prefix,
          json.dumps(info, default=str, indent=2),
      )
      self.agent.update_from_env(obs, rew, done, self._rollout_state_info(info))
    else:
      done = True
      if cur_step is not None:
        cur_step.done = True

    step_timed_out = time.perf_counter() - self._start_ts > self.timeout
    if cur_step is not None and self.tokenizer and self.chat_parser:
      _, env_messages = utils.get_recent_assistant_user_messages(
          self.agent.chat_completions
      )

      # Environment tokens/masks
      # Terminal-step environment messages are not appended to the response
      # token stream when the step ends the trajectory.
      if env_messages and not done and not step_timed_out:
        e_tokens, e_masks = utils.tokenize_and_generate_masks(
            env_messages,
            tokenizer=self.tokenizer,
            parser=self.chat_parser,
            contains_first_msg=False,
            contains_generation_msg=True,
        )
        cur_step.env_tokens = np.array(e_tokens)
        cur_step.env_masks = np.array(e_masks)
        self._response_token_count += len(e_tokens)

    if self.exact_token_continuity:
      self._record_exact_turn(cur_step, terminal=done or step_timed_out)

    if step_timed_out:
      self.agent.trajectory.status = agent_types.TrajectoryStatus.TIMEOUT
      logging.warning("Episode timed out after %d seconds.", self.timeout)
      self._log_trajectory_clip("TIMEOUT")
      self.agent.get_current_step().done = True  # pyrefly: ignore[missing-attribute]
      done = True

    self._record_env_step(cur_step)

    return done

  async def _append_final_reward(self):
    """Compute and add final reward to the last step of the episode.

    Applies the final reward function (if provided) to the episode's
    final response and adds it to the last step's reward. This enables
    additional reward signals based on overall episode performance.
    """
    last_step = self.agent.get_current_step()
    if (
        last_step is None
        or self.final_reward_fn is None
        or not callable(self.final_reward_fn)
    ):
      # Skip reward computation in trajectory collection if no reward function
      # is provided or no step is taken.
      logging.debug("%s Final reward function is skipped", self._debug_prefix)
      return
    final_reward, wall_time = await self._run_with_timing(self.final_reward_fn)

    self.reward_time["reward_latency"] += wall_time
    last_step.reward += final_reward
    self._record_env_step(last_step)
    logging.debug(
        "%s Final reward computed: %s", self._debug_prefix, final_reward
    )

  def compute_trajectory_reward(self):
    """Computes and stores the total reward for the trajectory.

    The trajectory reward is the undiscounted sum of rewards from all steps and
    is stored in `trajectory.reward`.

    Returns:
        The updated trajectory with the `reward` attribute populated.
    """
    trajectory = self.agent.trajectory
    if not trajectory:
      return None
    trajectory.reward = float(
        np.sum(np.array([s.reward for s in trajectory.steps]))
    )
    return trajectory

  def compute_mc_reward(self):
    """Compute Monte Carlo rewards for all steps in the trajectory.

    Calculates discounted rewards working backwards from the final step.
    Each step's Monte Carlo reward (return) is its immediate reward plus the
    discounted reward of subsequent steps. The result is stored in
    `step.mc_return`.
    """
    trajectory = self.agent.trajectory
    g = 0.0
    for step in reversed(trajectory.steps):
      g = step.reward + self.gamma * g
      step.mc_return = g

  async def _close(self):
    """Clean up resources by closing the environment.

    Ensures proper cleanup of environment resources such as network
    connections, file handles, or external processes.
    """
    logging.debug("%s Closing environment.", self._debug_prefix)
    try:
      _, wall_time = await self._run_with_timing(self.env.close, timeout=150.0)
      self.env_time["close_latency"] += wall_time
    except asyncio.TimeoutError:
      logging.error(
          "%s env.close() timed out after 150s — executor thread may be"
          " leaked. This will starve the thread pool over time.",
          self._debug_prefix,
      )
    finally:
      for k, v in self.env_time.items():
        logging.debug("%s k=%s v=%s", self._debug_prefix, k, v)
      for k, v in self.reward_time.items():
        logging.debug("%s k=%s v=%s", self._debug_prefix, k, v)
    logging.debug("%s Environment closed.", self._debug_prefix)

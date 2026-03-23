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
import time
from typing import Any, AsyncGenerator, Callable, Concatenate, Dict, List, Optional, ParamSpec, Set, Tuple

from absl import logging
import numpy as np
from tunix.rl.agentic import utils
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.agents import base_agent
from tunix.rl.agentic.environments import base_environment
from tunix.rl.agentic.rewards import reward_types
from tunix.rl.rollout import base_rollout

P = ParamSpec("P")

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
      model_call: Callable[
          Concatenate[Dict[str, str], P], base_rollout.RolloutOutput
      ],
      model_call_kwargs: Optional[Dict[str, Any]] = None,
      final_reward_fn: Optional[
          Callable[[Dict[str, Any], str], reward_types.RewardOutput]
      ] = None,
      gamma: float = 1.0,
      max_context_limit: Optional[int] = None,
      timeout: float = 600.0,
      tokenizer=None,
      chat_parser=None,
      valid_statuses: Optional[Set[agent_types.TrajectoryStatus]] = None,
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
        final_reward_fn (Optional[Callable]): Optional function to compute
          additional reward at episode end. Takes (task, response) and returns
          float. Defaults to zero if not provided.
        gamma (float): Discount factor for MC reward calculation (1.0 = no
          discounting).
        max_context_limit (Optional[int]): Maximum number of context tokens to
          use before forced termination.
        timeout (float): Maximum episode duration in seconds before timeout
          termination
        tokenizer: Optional tokenizer for converting messages to token IDs. This
          is required if we want to track down `max_context_tokens`.
        chat_parser: Optional chat parser for formatting messages
        valid_statuses (Set[TrajectoryStatus]): A set of statuses that are
          considered not "penalized" for reward computation.
    """
    self.agent = agent
    self.env = env
    self.model_call = model_call
    self.final_reward_fn = final_reward_fn or (
        lambda *_: reward_types.RewardOutput(reward=0.0)
    )
    self.model_call_kwargs = model_call_kwargs or {}
    self.max_steps = getattr(self.env, "max_steps", 1)
    self.gamma = gamma
    self.max_context_limit = max_context_limit
    self.timeout = timeout

    # Tokenizer utilities for stepwise tokenization
    self.tokenizer = tokenizer
    self.chat_parser = chat_parser
    self._start_ts: float = 0.0
    self.valid_statuses = valid_statuses or {
        agent_types.TrajectoryStatus.SUCCEEDED
    }

    if self.max_context_limit and not (self.tokenizer and self.chat_parser):
      logging.warning(
          "max_context_limit is set to %d, but no tokenizer or chat_parser is"
          " provided. Context limits will not be enforced.",
          self.max_context_limit,
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
    await self._reset()

    # Initial Prompt Cost
    current_token_count = 0
    if (
        hasattr(self.agent.trajectory, "prompt_tokens")
        and self.agent.trajectory.prompt_tokens
    ):
      current_token_count += len(self.agent.trajectory.prompt_tokens)

    self.agent.trajectory.status = agent_types.TrajectoryStatus.RUNNING

    while True:
      if len(self.agent.trajectory.steps) >= self.max_steps:
        self.agent.trajectory.status = (
            agent_types.TrajectoryStatus.MAX_STEPS_REACHED
        )
        break

      done = await self._one_step()
      current_step = self.agent.get_current_step()

      if current_step:
        if getattr(current_step, "assistant_tokens", None) is not None:
          current_token_count += len(current_step.assistant_tokens)
        if getattr(current_step, "env_tokens", None) is not None:
          current_token_count += len(current_step.env_tokens)

        if (
            self.max_context_limit is not None
            and current_token_count >= self.max_context_limit
        ):
          self.agent.trajectory.status = (
              agent_types.TrajectoryStatus.MAX_CONTEXT_LIMIT_REACHED
          )
          break

      if done:
        if self.agent.trajectory.status == agent_types.TrajectoryStatus.RUNNING:
          self.agent.trajectory.status = agent_types.TrajectoryStatus.SUCCEEDED
        break

    await self._append_final_reward()
    self.compute_mc_reward()
    self.compute_trajectory_reward()
    await self._close()

    if mode not in ["Trajectory", "Steps", "Token", "Conversation"]:
      raise ValueError(
          f"Unsupported mode: {mode}, currently supported modes: "
          f" {['Trajectory', 'Steps', 'Token', 'Conversation']}",
      )

    if mode == "Trajectory":
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
              "reward": step.reward,
              "mc_return": step.mc_return,
          }
          for step in self.agent.trajectory.steps
      ]
    elif mode == "Token":
      # flatten all steps into single batch dict
      conversation_tokens, conversation_masks, logprobs = [], [], []
      prompt_tokens = getattr(self.agent.trajectory, "prompt_tokens", [])

      for step in self.agent.trajectory.steps:
        # assistant tokens
        if getattr(step, "assistant_tokens", None) is not None:
          conversation_tokens.append(step.assistant_tokens)
          conversation_masks.append(step.assistant_masks)

        # env tokens
        if getattr(step, "env_tokens", None) is not None:
          conversation_tokens.append(step.env_tokens)
          conversation_masks.append(step.env_masks)

        # logprobs
        if getattr(step, "logprobs", None) is not None:
          assert len(step.logprobs) == len(step.assistant_tokens), (
              f"Logprobs length {len(step.logprobs)} does not match assistant"
              f" tokens length {len(step.assistant_tokens)}"
          )
          logprobs.append(step.logprobs)
          if getattr(step, "env_tokens", None) is not None:
            logprobs.append(np.zeros(len(step.env_tokens)))

      # TODO(sizhi): (b/484422277)
      is_valid = self.agent.trajectory.status in self.valid_statuses

      return {
          "conversation_text": self.agent.chat_completions,
          "prompt_tokens": prompt_tokens,
          "conversation_tokens": np.concatenate(conversation_tokens, axis=0),
          "conversation_masks": np.concatenate(conversation_masks, axis=0),
          "status": self.agent.trajectory.status.name,
          "trajectory_reward": self.agent.trajectory.reward,
          "old_logprobs": (
              np.concatenate(logprobs, axis=0) if logprobs else None
          ),
          "policy_version": self.env.task.get("policy_version"),
          "original_input": self.agent.trajectory.task,
          "group_id": self.env.extra_kwargs.get("group_id"),
      }
    elif mode == "Conversation":
      # return raw conversation history
      return self.agent.chat_completions

  @staticmethod
  async def collect_multiple(
      pairs: List[Tuple[ConversationAgentBase, BaseTaskEnv]],
      *,
      model_call: Callable[..., base_rollout.RolloutOutput],
      final_reward_fn: Optional[
          Callable[[Dict[str, Any], str], reward_types.RewardOutput]
      ] = None,
      gamma: float = 1.0,
      max_context_limit: Optional[int] = None,
      timeout: float = 30.0,
      mode: str = "Trajectory",
  ) -> AsyncGenerator[Tuple[int, Any], None]:
    """Execute multiple agent-environment pairs concurrently.

    Runs multiple rollouts in parallel and yields completed trajectories
    as they finish, enabling efficient batch processing with streaming
    results. Useful for distributed training or large-scale evaluation.

    Args:
        pairs (List[Tuple[ConversationAgentBase, BaseTaskEnv]]): List of (agent,
          environment) pairs
        model_call (Callable): Shared model inference function for all pairs
        final_reward_fn (Optional[Callable]): Shared final reward function
        gamma (float): Discount factor for return calculation
        max_context_limit (Optional[int]): Maximum context limit per episode
        timeout (float): Per-episode timeout in seconds
        mode (str): Output format. See `collect` method for options.

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
          final_reward_fn=final_reward_fn,
          gamma=gamma,
          max_context_limit=max_context_limit,
          timeout=timeout,
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
    obs, _ = await asyncio.get_event_loop().run_in_executor(
        None, self.env.reset
    )
    if hasattr(self.env, "compute_final_reward") and callable(
        self.env.compute_final_reward
    ):
      self.final_reward_fn = self.env.compute_final_reward

    self.agent.reset()
    self.agent.update_from_env(observation=obs, reward=0.0, done=False, info={})

    if self.tokenizer is not None and self.chat_parser is not None:
      # Get the current messages (usually System + User)
      init_messages = self.agent.chat_completions
      prompt_tokens, _ = utils.tokenize_and_generate_masks(
          init_messages,
          tokenizer=self.tokenizer,
          parser=self.chat_parser,
          contains_first_msg=True,
          contains_generation_msg=True,
      )
      self.agent.trajectory.prompt_tokens = prompt_tokens

    self._start_ts = time.time()

  async def _one_step(self) -> bool:
    """Executes a single step and returns the Step object and Done status.

    This involves calling the model, updating the agent with the response,
    stepping the environment with the agent's action, and updating the agent
    with the environment's feedback.

    Returns:
        bool: True if the episode is done (either by environment or timeout),
          False otherwise.
    """
    rollout_output = await asyncio.get_event_loop().run_in_executor(
        None,
        self.model_call,
        self.agent.chat_completions,
        self.env,
        **self.model_call_kwargs,
    )

    action = self.agent.update_from_model(rollout_output.text[0]).action

    if action is None:
      logging.warning(
          "Agent returned None action, using empty action list as fallback"
      )
      action = []

    obs, rew, done, info = await asyncio.get_event_loop().run_in_executor(
        None, self.env.step, action
    )

    self.agent.update_from_env(obs, rew, done, info)

    cur_step = self.agent.get_current_step()

    if cur_step is not None and rollout_output.logprobs is not None:
      cur_step.logprobs = rollout_output.logprobs[0]

    if cur_step is not None and self.tokenizer and self.chat_parser:
      assistant_message, env_messages = (
          utils.get_recent_assistant_user_messages(self.agent.chat_completions)
      )

      # Assistant tokens/masks
      if assistant_message:
        cur_step.assistant_tokens = rollout_output.tokens[0]
        cur_step.assistant_masks = np.ones_like(rollout_output.tokens[0])

      # Environment tokens/masks
      if env_messages:
        e_tokens, e_masks = utils.tokenize_and_generate_masks(
            env_messages,
            tokenizer=self.tokenizer,
            parser=self.chat_parser,
            contains_first_msg=False,
            contains_generation_msg=True,
        )
        cur_step.env_tokens = np.array(e_tokens)
        cur_step.env_masks = np.array(e_masks)

    if time.time() - self._start_ts > self.timeout:
      self.agent.trajectory.status = agent_types.TrajectoryStatus.TIMEOUT
      logging.warning("Episode timed out after %d seconds.", self.timeout)
      self.agent.get_current_step().done = True
      return True

    return done

  async def _append_final_reward(self):
    """Compute and add final reward to the last step of the episode.

    If the environment provides a ``compute_final_reward`` method (e.g.
    SWE-bench running tests inside a Docker container), it is used in
    preference to the callback-based ``final_reward_fn``.  This mirrors
    the pattern used by rllm's ``AgentExecutionEngine``.
    """
    last_step = self.agent.get_current_step()
    if last_step is None:
      return

    if hasattr(self.env, 'compute_final_reward') and callable(
        self.env.compute_final_reward
    ):
      reward = await asyncio.get_event_loop().run_in_executor(
          None, self.env.compute_final_reward
      )
      last_step.reward += reward
      return

    final_reward = await asyncio.get_event_loop().run_in_executor(
        None, self.final_reward_fn, self.env.task, last_step.model_response
    )
    last_step.reward += final_reward.reward

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
    await asyncio.get_event_loop().run_in_executor(None, self.env.close)

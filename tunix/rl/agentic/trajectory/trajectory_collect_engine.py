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
import os
import json
import asyncio
import time
import traceback
from typing import Any, AsyncGenerator, Callable, Concatenate, Dict, List, Optional, ParamSpec, Set, Tuple

from absl import logging
import numpy as np
from tunix.rl.agentic import utils
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.agents import base_agent
# from tunix.rl.agentic.agents import context_util
from tunix.rl.agentic.environments import base_environment
from tunix.rl.agentic.rewards import reward_types

P = ParamSpec("P")

BaseTaskEnv = base_environment.BaseTaskEnv
ConversationAgentBase = base_agent.ConversationAgentBase


class TrajectoryCollectEngine:
  """Asynchronous trajectory collection engine for agent-environment interactions.

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
      model_call: Callable[Concatenate[Dict[str, str], P], str],
      final_reward_fn: Optional[
          Callable[[Dict[str, Any], str], reward_types.RewardOutput]
      ] = None,
      gamma: float = 1.0,
<<<<<<< HEAD
=======
      max_steps: int = 10,
      max_context_tokens: Optional[int] = None,
>>>>>>> 05ec4fe (add deepswe train script)
      timeout: float = 600.0,
      tokenizer=None,
      chat_parser=None,
      model_call_kwargs: Optional[Dict[str, Any]] = None,
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
        final_reward_fn (Optional[Callable]): Optional function to compute
          additional reward at episode end. Takes (task, response) and returns
          float. Defaults to zero if not provided.
        gamma (float): Discount factor for return calculation (1.0 = no
          discounting).
        max_steps (int): Maximum number of interaction steps before forced
          termination
        max_context_tokens (Optional[int]): Maximum number of context tokens to
          use before forced termination.
        timeout (float): Maximum episode duration in seconds before timeout
          termination
        tokenizer: Optional tokenizer for converting messages to token IDs. This
          is required if we want to track down `max_context_tokens`.
        chat_parser: Optional chat parser for formatting messages
        model_call_kwargs: Optional kwargs to pass to model_call.
        valid_statuses (Set[TrajectoryStatus]): A set of statuses that are
          considered "valid" for training. Trajectories ending in these statuses
          will have a loss_mask of 1.0. Others (like FAILED) will be 0.0.
          Defaults to {SUCCEEDED}.
    """
    self.agent = agent
    self.env = env
    self.model_call = model_call
    self.final_reward_fn = final_reward_fn or (
        lambda *_: reward_types.RewardOutput(reward=0.0)
    )
    self.model_call_kwargs = model_call_kwargs or {}
    self.gamma = gamma
    self.max_steps = max_steps
    self.max_context_tokens = max_context_tokens
    self.timeout = timeout

    # Tokenizer utilities for stepwise tokenization
    self.tokenizer = tokenizer
    self.chat_parser = chat_parser
    self._start_ts: float = 0.0
    self.valid_statuses = valid_statuses or {
        agent_types.TrajectoryStatus.SUCCEEDED
    }

    if self.tokenizer and not getattr(self.agent, "tokenizer", None):
      logging.info("Injecting tokenizer into Agent to enable step caching.")
      self.agent.tokenizer = self.tokenizer

    if self.max_context_tokens and not self.tokenizer:
      logging.warning(
          "max_context_tokens is set to %d, but no tokenizer was provided. "
          "Context limits will not be enforced.",
          self.max_context_tokens,
      )

    def _create_failure_response(self, mode, error_msg):
      # It creates a "Empty" but VALID structure matches the requested mode
      if mode == "Token":
        return {
            "conversation_tokens": [],  # Empty list (Valid)
            "conversation_masks": [],  # Empty list (Valid)
            "loss_mask": 0.0,  # IGNORE this data (Valid)
            "status": "FAILED",  # Status flag (Valid)
            "error": error_msg,  # Debug info
        }
      elif mode == "Trajectory":
        return self.agent.Trajectory(
            status=agent_types.TrajectoryStatus.FAILED,
            failure_message=error_msg,
        )

  async def collect(self, mode: str = "Conversation") -> Any:
    """Execute a complete rollout episode and return the resulting trajectory.

    Orchestrates the full interaction sequence: environment reset, iterative
    agent-environment steps, final reward computation, Monte Carlo return
    calculation, and resource cleanup.

    Args:
        mode (str): Output format. Options: - "Trajectory": return full
          Trajectory object. - "Token": return flattened tokenized dict for
          training. - "Steps": return stepwise tokenized data only. -
          "Conversation": return raw conversation messages (default).

    Returns:
        Trajectory | dict | list: Depending on mode.
    """
    try:
      await self._reset()
      # Ensure task is captured
      if self.agent.trajectory and not self.agent.trajectory.task:
        task_data = getattr(self.env, "entry", getattr(self.env, "task", {}))
        self.agent.trajectory.task = task_data
    except Exception as e:
      logging.error(f"Reset failed: {e}")
      return self._create_failure_response(mode, str(e))

    # Tokenizer only used if cache miss
    token_func = (
        lambda s: len(self.tokenizer.encode(s)) if self.tokenizer else 0
    )

    # Initial Prompt cost
    current_token_count = 0
    if (
        hasattr(self.agent.trajectory, "prompt_tokens")
        and self.agent.trajectory.prompt_tokens
    ):
      current_token_count += len(self.agent.trajectory.prompt_tokens)

    # elif self.agent.trajectory.task and self.tokenizer:
    #   current_token_count += len(
    #       self.tokenizer.encode(
    #           context_util.safe_serialize(self.agent.trajectory.task)
    #       )
    #   )

    self.agent.trajectory.status = agent_types.TrajectoryStatus.RUNNING
    try:
      for step_idx in range(self.max_steps):
        step_log = f"\n[DEBUG] Running step {step_idx}\n"
        os.write(1, step_log.encode('utf-8'))
        current_step, done = await self._one_step()

        if self.tokenizer and current_step:
          # step_cost = context_util.count_step_tokens(
          #     current_step,
          #     tokenizer=token_func,
          # )
          # current_token_count += step_cost

          # Check if adding this step pushed us over the limit
          if (
              self.max_context_tokens is not None
              and current_token_count >= self.max_context_tokens
          ):
            self.agent.trajectory.status = (
                agent_types.TrajectoryStatus.TRUNCATED_TOKENS
            )
            break

        if done:
          if (
              self.agent.trajectory.status
              == agent_types.TrajectoryStatus.RUNNING
          ):
            self.agent.trajectory.status = (
                agent_types.TrajectoryStatus.SUCCEEDED
            )
          break

      if self.agent.trajectory.status == agent_types.TrajectoryStatus.RUNNING:
        self.agent.trajectory.status = (
            agent_types.TrajectoryStatus.TRUNCATED_STEPS
        )
    except Exception as e:
      logging.error(f"Rollout crashed: {e}")
      traceback.print_exc()
      self.agent.trajectory.status = agent_types.TrajectoryStatus.FAILED
      self.agent.trajectory.failure_message = f"{type(e).__name__}: {str(e)}"

    try:
      await self._append_final_reward()
      self.compute_mc_reward()
      self.compute_trajectory_reward()
    except Exception as e:
      logging.error(f"Reward computation failed: {e}")
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
              "conversation_tokens": (
                  getattr(step, "assistant_tokens", [])
                  + getattr(step, "env_tokens", [])
              ),
              "conversation_masks": (
                  getattr(step, "assistant_masks", [])
                  + getattr(step, "env_masks", [])
              ),
              "reward": step.reward,
              "mc_return": step.mc_return,
          }
          for step in self.agent.trajectory.steps
      ]
    elif mode == "Token":
      # flatten all steps into single batch dict
      conversation_tokens, conversation_masks = [], []
      prompt_tokens = getattr(self.agent.trajectory, "prompt_tokens", [])

      for step in self.agent.trajectory.steps:

        resp_tokens = getattr(step, "model_response_tokens", None)
        if not resp_tokens:
          resp_tokens = getattr(step, "assistant_tokens", [])

        if resp_tokens:
          conversation_tokens.extend(resp_tokens)
          conversation_masks.extend([1] * len(resp_tokens))

        obs_tokens = getattr(step, "observation_tokens", None)
        if not obs_tokens:
          obs_tokens = getattr(step, "env_tokens", [])

        if obs_tokens:
          conversation_tokens.extend(obs_tokens)
          conversation_masks.extend([0] * len(obs_tokens))

        act_tokens = getattr(step, "action_tokens", None)
        if act_tokens:
          conversation_tokens.extend(act_tokens)
          conversation_masks.extend([1] * len(act_tokens))

      is_valid = self.agent.trajectory.status in self.valid_statuses
      loss_mask = 1.0 if is_valid else 0.0

      return {
          "conversation_text": self.agent.chat_completions,
          "prompt_tokens": prompt_tokens,
          "conversation_tokens": conversation_tokens,
          "conversation_masks": conversation_masks,
          "loss_mask": loss_mask,
          "status": self.agent.trajectory.status.name,
          "trajectory_reward": self.agent.trajectory.reward,
          "policy_version": self.env.task.get("policy_version"),
          "original_input": getattr(self.agent.trajectory, "task", {}),
          "group_id": self.env.task.get("group_id"),
      }
    elif mode == "Conversation":
      # return raw conversation history
      return self.agent.chat_completions

  @staticmethod
  async def collect_multiple(
      pairs: List[Tuple[ConversationAgentBase, BaseTaskEnv]],
      *,
      model_call: Callable[..., str],
      final_reward_fn: Optional[
          Callable[[Dict[str, Any], str], reward_types.RewardOutput]
      ] = None,
      gamma: float = 1.0,
      max_steps: int = 10,
      max_context_tokens: Optional[int] = None,
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
        max_steps (int): Maximum steps per episode
        max_context_tokens (Optional[int]): Maximum context tokens per episode
        timeout (float): Per-episode timeout in seconds
        mode (str): Output format. See `collect` method for options.

    Yields:
        Tuple[int, Any]: `(pair_index, result)`. The type of `result`
          depends on the `mode` argument. See the `collect` method for details.
    """

    async def _run_one(i: int, agent: ConversationAgentBase, env: BaseTaskEnv):
      """Execute a single agent-environment pair with the given configuration."""
      engine = TrajectoryCollectEngine(
          agent,
          env,
          model_call=model_call,
          final_reward_fn=final_reward_fn,
          gamma=gamma,
          max_steps=max_steps,
          max_context_tokens=max_context_tokens,
          timeout=timeout,
      )
      traj = await engine.collect(mode=mode)
      return i, traj

    # Launch all pairs concurrently and yield results as they complete
    tasks = [_run_one(i, agent, env) for i, (agent, env) in enumerate(pairs)]
    for coro in asyncio.as_completed(tasks):
      pair_index, result = await coro
      yield pair_index, result

  async def _reset(self):
    """Resets the environment and agent at the beginning of a new episode.

    This involves calling the environment's reset method, updating the agent's
    state, and optionally tokenizing the initial prompt messages.
    """
    obs, _ = await asyncio.get_event_loop().run_in_executor(
        None, self.env.reset
    )
    self.agent.reset()
    self.agent.update_from_env(observation=obs, reward=0.0, done=False, info={})

    if self.tokenizer is not None and self.chat_parser is not None:
      init_messages = self.agent.chat_completions
      prompt_tokens, _ = utils.tokenize_and_generate_masks(
          init_messages,
          tokenizer=self.tokenizer,
          parser=self.chat_parser,
          contains_first_msg=True,
          contains_generation_msg=False,
      )
      self.agent.trajectory.prompt_tokens = prompt_tokens

    self._start_ts = time.time()

  async def _one_step(self) -> tuple[Optional[agent_types.Step], bool]:
    """Executes a single step and returns the Step object and Done status.

    This involves calling the model, updating the agent with the response,
    stepping the environment with the agent's action, and updating the agent
    with the environment's feedback.

    Returns:
        tuple[Step | None, bool]: Returns the generated Step object (or None if
        failed)
                                  and a boolean indicating if the episode is
                                  done.
    """
    resp = await asyncio.to_thread(
        self.model_call,
        self.agent.chat_completions,
        self.env,
        **self.model_call_kwargs,
    )

    # --- DEBUG LOG: MODEL RESPONSE ---
    # We use default=str to handle objects that aren't natively JSON serializable
    resp_log = f"\n[DEBUG] Model Response:\n{json.dumps(resp, default=str, indent=2)}\n"
    os.write(1, resp_log.encode('utf-8'))
    # ---------------------------------

    agent_update = self.agent.update_from_model(resp)
    action = agent_update.action

    if action is None:
      logging.warning(
          "Agent returned None action, using empty action list as fallback"
      )
      action = []

    # --- DEBUG LOG: AGENT ACTION ---
    action_log = f"\n[DEBUG] Agent Action:\n{json.dumps(action, default=str, indent=2)}\n"
    os.write(1, action_log.encode('utf-8'))
    # -------------------------------

    obs, rew, done, info = await asyncio.to_thread(self.env.step, action)


    # --- DEBUG LOG: ENV OBSERVATION & INFO ---
    # Obs and Info can be very large, so we truncate them if necessary or just log keys
    obs_log = f"\n[DEBUG] Env Observation (Rew: {rew}, Done: {done}):\n{json.dumps(obs, default=str, indent=2)}\n"
    info_log = f"\n[DEBUG] Env Info:\n{json.dumps(info, default=str, indent=2)}\n"
    
    os.write(1, obs_log.encode('utf-8'))
    os.write(1, info_log.encode('utf-8'))
    # -----------------------------------------

    self.agent.update_from_env(obs, rew, done, info)

    cur_step = self.agent.get_current_state()

    if cur_step is not None and self.tokenizer and self.chat_parser:
      assistant_message, env_messages = (
          utils.get_recent_assistant_user_messages(self.agent.chat_completions)
      )

      # Assistant tokens/masks
      if assistant_message:
        a_tokens, a_masks = utils.tokenize_and_generate_masks(
            [assistant_message],
            tokenizer=self.tokenizer,
            parser=self.chat_parser,
            contains_first_msg=False,
            contains_generation_msg=False,
        )
        cur_step.assistant_tokens = a_tokens
        cur_step.assistant_masks = a_masks

      # Environment tokens/masks
      if env_messages:
        e_tokens, e_masks = utils.tokenize_and_generate_masks(
            env_messages,
            tokenizer=self.tokenizer,
            parser=self.chat_parser,
            contains_first_msg=False,
            contains_generation_msg=False,
        )
        cur_step.env_tokens = e_tokens
        cur_step.env_masks = e_masks

    if time.time() - self._start_ts > self.timeout:
      if cur_step:
        cur_step.done = True

      self.agent.trajectory.status = (
          agent_types.TrajectoryStatus.TRUNCATED_TIMEOUT
      )
      logging.warning("Episode timed out after %d seconds.", self.timeout)

      return cur_step, True

    return cur_step, done

  async def _append_final_reward(self):
    """Compute and add final reward to the last step of the episode.

    Applies the final reward function (if provided) to the episode's
    final response and adds it to the last step's reward. This enables
    additional reward signals based on overall episode performance.
    """
    last_step = self.agent.get_current_state()
    if last_step is None:
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

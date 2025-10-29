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
import logging
import time
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple

import jax.numpy as jnp
from tunix.rl.agentic import utils
from tunix.rl.agentic.agents import base_agent
from tunix.rl.agentic.environments import base_environment
from tunix.rl.agentic.rewards import reward_types


BaseEnv = base_environment.BaseEnv
Trajectory = base_agent.Trajectory
LLMBaseAgent = base_agent.LLMBaseAgent
logger = logging.getLogger(__name__)


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
      agent: LLMBaseAgent,
      env=None,
      *,
      model_call: Callable[[list[Dict[str, str]]], str],
      final_reward_fn: Optional[Callable[[Dict[str, Any], str],
                                         reward_types.RewardOutput]] = None,
      max_steps: int = 10,
      gamma: float = 1.0,
      timeout: float = 30.0,
      tokenizer=None,
      chat_parser=None,
  ):
    """Initialize the trajectory collection engine.

    Args:
        agent (LLMBaseAgent): The agent that will interact with the environment
        env (BaseEnv): The environment providing tasks and feedback
        model_call (Callable): Function that takes chat completions and returns
          model response string. Handles the actual LLM inference.
        final_reward_fn (Optional[Callable]): Optional function to compute
          additional reward at episode end. Takes (task, response) and returns
          float. Defaults to zero if not provided.
        max_steps (int): Maximum number of interaction steps before forced
          termination
        gamma (float): Discount factor for return calculation (1.0 = no
          discounting)
        timeout (float): Maximum episode duration in seconds before timeout
          termination
        tokenizer: Optional tokenizer for converting messages to token IDs
        chat_parser: Optional chat parser for formatting messages
    """
    self.agent = agent
    self.env = env
    self.model_call = model_call
    self.final_reward_fn = final_reward_fn or (
        lambda *_: reward_types.RewardOutput(reward=0.0)
    )
    self.max_steps = max_steps
    self.gamma = gamma
    self.timeout = timeout

    # Tokenizer utilities for stepwise tokenization
    self.tokenizer = tokenizer
    self.chat_parser = chat_parser
    self._start_ts: float = 0.0

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
    await self._reset()
    for _ in range(self.max_steps):
      done = await self._one_step()
      if done:
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
        # assistant tokens
        if hasattr(step, "assistant_tokens"):
          conversation_tokens.extend(step.assistant_tokens)
          conversation_masks.extend(step.assistant_masks)

        # env tokens
        if hasattr(step, "env_tokens"):
          conversation_tokens.extend(step.env_tokens)
          conversation_masks.extend(step.env_masks)

      return {
          "conversation_text": self.agent.chat_completions,
          "prompt_tokens": prompt_tokens,
          "conversation_tokens": conversation_tokens,
          "conversation_masks": conversation_masks,
          "trajectory_reward": self.agent.trajectory.reward,
      }
    elif mode == "Conversation":
      # return raw conversation history
      return self.agent.chat_completions

  @staticmethod
  async def collect_multiple(
      pairs: List[Tuple[LLMBaseAgent, BaseEnv]],
      *,
      model_call: Callable[[list[Dict[str, str]]], str],
      final_reward_fn: Optional[Callable[[Dict[str, Any], str],
                                         reward_types.RewardOutput]] = None,
      max_steps: int = 10,
      gamma: float = 1.0,
      timeout: float = 30.0,
      mode: str = "Trajectory",
  ) -> AsyncGenerator[Tuple[int, Any], None]:
    """Execute multiple agent-environment pairs concurrently.

    Runs multiple rollouts in parallel and yields completed trajectories
    as they finish, enabling efficient batch processing with streaming
    results. Useful for distributed training or large-scale evaluation.

    Args:
        pairs (List[Tuple[LLMBaseAgent, BaseEnv]]): List of (agent, environment)
          pairs
        model_call (Callable): Shared model inference function for all pairs
        final_reward_fn (Optional[Callable]): Shared final reward function
        max_steps (int): Maximum steps per episode
        gamma (float): Discount factor for return calculation
        timeout (float): Per-episode timeout in seconds
        mode (str): Output format. See `collect` method for options.

    Yields:
        Tuple[int, Any]: `(pair_index, result)`. The type of `result`
          depends on the `mode` argument. See the `collect` method for details.
    """

    async def _run_one(i: int, agent: LLMBaseAgent, env: BaseEnv):
      """Execute a single agent-environment pair with the given configuration."""
      engine = TrajectoryCollectEngine(
          agent,
          env,
          model_call=model_call,
          final_reward_fn=final_reward_fn,
          max_steps=max_steps,
          gamma=gamma,
          timeout=timeout,
      )
      traj = await engine.collect(mode=mode)
      return i, traj

    # Launch all pairs concurrently and yield results as they complete
    tasks = [_run_one(i, a, e) for i, (a, e) in enumerate(pairs)]
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

  async def _one_step(self) -> bool:
    """Executes a single step of the agent-environment interaction.

    This involves calling the model, updating the agent with the response,
    stepping the environment with the agent's action, and updating the agent
    with the environment's feedback.

    Returns:
        bool: True if the episode is done (either by environment or timeout),
          False otherwise.
    """
    resp = await asyncio.get_event_loop().run_in_executor(
        None, self.model_call, [self.agent.chat_completions]
    )
    action = self.agent.update_from_model(resp).action

    if action is None:
      logger.warning(
          "Agent returned None action, using empty action list as fallback"
      )
      action = []

    obs, rew, done, info = await asyncio.get_event_loop().run_in_executor(
        None, self.env.step, action
    )
    self.agent.update_from_env(obs, rew, done, info)

    if self.tokenizer is not None and self.chat_parser is not None:
      cur_step = self.agent.get_current_state()
      if cur_step is not None:
        assistant_message, env_messages = (
            utils.get_recent_assistant_user_messages(
                self.agent.chat_completions
            )
        )

        # assistant tokens
        if assistant_message:
          assistant_tokens, assistant_masks = utils.tokenize_and_generate_masks(
              [assistant_message],
              tokenizer=self.tokenizer,
              parser=self.chat_parser,
              contains_first_msg=False,
              contains_generation_msg=False,
          )
          cur_step.assistant_tokens = assistant_tokens
          cur_step.assistant_masks = assistant_masks

        # env tokens
        if env_messages:
          env_tokens, env_masks = utils.tokenize_and_generate_masks(
              env_messages,
              tokenizer=self.tokenizer,
              parser=self.chat_parser,
              contains_first_msg=False,
              contains_generation_msg=False,
          )
          cur_step.env_tokens = env_tokens
          cur_step.env_masks = env_masks

    if time.time() - self._start_ts > self.timeout:
      self.agent.get_current_state().done = True
      return True
    return done

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
    trajectory_reward = jnp.sum(jnp.array([d.reward for d in trajectory.steps]))
    trajectory.reward = float(trajectory_reward)
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

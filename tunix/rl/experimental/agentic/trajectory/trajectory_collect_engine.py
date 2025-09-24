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

from tunix.rl.experimental.agentic.agents import base_agent
from tunix.rl.experimental.agentic.environments import base_environment

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
      final_reward_fn: Optional[Callable[[Dict[str, Any], str], float]] = None,
      max_steps: int = 10,
      gamma: float = 1.0,
      timeout: float = 30.0,
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
    """
    self.agent = agent
    self.env = env
    self.model_call = model_call
    self.final_reward_fn = final_reward_fn or (lambda *_: 0.0)
    self.max_steps = max_steps
    self.gamma = gamma
    self.timeout = timeout

  async def collect(self) -> Trajectory:
    """Execute a complete rollout episode and return the resulting trajectory.

    Orchestrates the full interaction sequence: environment reset, iterative
    agent-environment steps, final reward computation, Monte Carlo return
    calculation, and resource cleanup.

    Returns:
        Trajectory: Complete episode trace with all steps, rewards, and returns
    """
    await self._reset()
    for _ in range(self.max_steps):
      done = await self._one_step()
      if done:
        break
    await self._append_final_reward()
    self._fill_returns()
    await self._close()
    return self.agent.trajectory

  @staticmethod
  async def collect_multiple(
      pairs: List[Tuple[LLMBaseAgent, BaseEnv]],
      *,
      model_call: Callable[[list[Dict[str, str]]], str],
      final_reward_fn: Optional[Callable[[Dict[str, Any], str], float]] = None,
      max_steps: int = 10,
      gamma: float = 1.0,
      timeout: float = 30.0,
  ) -> AsyncGenerator[Tuple[int, Trajectory], None]:
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

    Yields:
        Tuple[int, Trajectory]: (pair_index, completed_trajectory) as episodes
        finish
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
      traj = await engine.collect()
      return i, traj

    # Launch all pairs concurrently and yield results as they complete
    tasks = [_run_one(i, a, e) for i, (a, e) in enumerate(pairs)]
    for coro in asyncio.as_completed(tasks):
      yield await coro

  async def _reset(self):
    """Initialize the episode by resetting environment and agent state.

    Resets the environment to get initial observation, clears agent state,
    and provides the initial observation to the agent. Also starts the
    episode timer for timeout tracking.
    """
    obs, _ = await asyncio.get_event_loop().run_in_executor(
        None, self.env.reset
    )
    self.agent.reset()
    self.agent.update_from_env(observation=obs, reward=0.0, done=False, info={})
    self._start_ts = time.time()

  async def _one_step(self) -> bool:
    """Execute one complete agent-environment interaction step.

    Performs the core interaction cycle: get agent's chat completions,
    call the model to generate response, parse response into action,
    execute action in environment, and update agent with results.
    Also checks for timeout conditions.

    Returns:
        bool: True if episode should terminate (done or timeout), False to
        continue
    """
    # 1) Generate model response from current conversation context
    resp = await asyncio.get_event_loop().run_in_executor(
        None, self.model_call, self.agent.chat_completions
    )
    action = self.agent.update_from_model(resp).action

    if action is None:
      logger.warning(
          "Agent returned None action, using empty action list as fallback"
      )
      action = []

    # 2) Execute action in environment and get feedback
    obs, rew, done, info = await asyncio.get_event_loop().run_in_executor(
        None, self.env.step, action
    )
    self.agent.update_from_env(obs, rew, done, info)

    # 3) Check for timeout termination
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
    add_r = await asyncio.get_event_loop().run_in_executor(
        None, self.final_reward_fn, self.env.task, last_step.model_response
    )
    last_step.reward += add_r

  def _fill_returns(self):
    """Compute Monte Carlo returns for all steps in the trajectory.

    Calculates discounted returns working backwards from the final step,
    where each step's return is its immediate reward plus the discounted
    return of subsequent steps. Sets the trajectory's total reward to
    the first step's return.
    """
    traj = self.agent.trajectory
    g = 0.0
    for step in reversed(traj.steps):
      g = step.reward + self.gamma * g
      step.mc_return = g
    traj.reward = traj.steps[0].mc_return if traj.steps else 0.0

  async def _close(self):
    """Clean up resources by closing the environment.

    Ensures proper cleanup of environment resources such as network
    connections, file handles, or external processes.
    """
    await asyncio.get_event_loop().run_in_executor(None, self.env.close)

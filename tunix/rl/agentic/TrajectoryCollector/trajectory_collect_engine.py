# tunix/rl/multi_turn/execution/trajectory_collector.py
import asyncio
import time
from typing import Any, Callable, Dict, Optional, Tuple, List, AsyncGenerator

from tunix.rl.agentic.agents.base_agent import BaseAgent, Trajectory
from tunix.rl.agentic.environments.base_environment import BaseEnv


class TrajectoryCollectEngine:
    """
    A ready-to-use rollout sampler:
      • Implements complete rollout logic (reset → step* → final reward → return)
      • Allows custom model inference and final reward via callbacks
      • Supports concurrent multi-pair rollout (streaming results like rllm)
    """

    # --------------------------------------------------
    # Constructor
    # --------------------------------------------------
    def __init__(
        self,
        agent: BaseAgent,
        env: BaseEnv,
        *,
        model_call: Callable[[list[Dict[str, str]]], str],
        final_reward_fn: Optional[Callable[[Dict, str], float]] = None,
        max_steps: int = 10,
        gamma: float = 1.0,
        timeout: float = 30.0,
    ):
        self.agent = agent
        self.env = env
        self.model_call = model_call
        self.final_reward_fn = final_reward_fn or (lambda *_: 0.0)
        self.max_steps = max_steps
        self.gamma = gamma
        self.timeout = timeout

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    async def collect(self) -> Trajectory:
        """Performs a full rollout and returns the agent's trajectory"""
        await self._reset()
        for t in range(self.max_steps):
            done = await self._one_step(t)
            if done:
                break
        await self._append_final_reward()
        self._fill_returns()
        await self._close()
        return self.agent.trajectory

    @staticmethod
    async def collect_many(
        pairs: List[Tuple[BaseAgent, BaseEnv]],
        *,
        model_call: Callable[[list[Dict[str, str]]], str],
        final_reward_fn: Optional[Callable[[Dict, str], float]] = None,
        max_steps: int = 10,
        gamma: float = 1.0,
        timeout: float = 30.0,
    ) -> AsyncGenerator[Tuple[int, Trajectory], None]:
        """
        Run multiple (agent, env) pairs concurrently.
        Yields results one by one as they complete.

        Yields:
            (index, trajectory)
        """

        async def _run_one(i: int, agent: BaseAgent, env: BaseEnv):
            engine = TrajectoryCollectEngine(
                agent, env,
                model_call=model_call,
                final_reward_fn=final_reward_fn,
                max_steps=max_steps,
                gamma=gamma,
                timeout=timeout,
            )
            traj = await engine.collect()
            return i, traj

        tasks = [_run_one(i, a, e) for i, (a, e) in enumerate(pairs)]
        for coro in asyncio.as_completed(tasks):
            yield await coro

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------
    async def _reset(self):
        obs, _ = await asyncio.get_event_loop().run_in_executor(None, self.env.reset)
        self.agent.reset()
        self.agent.update_from_env(obs, 0.0, False, {})
        self._start_ts = time.time()

    async def _one_step(self) -> bool:
        # 1) Call LLM
        resp = await asyncio.get_event_loop().run_in_executor(
            None, self.model_call, self.agent.chat_completions
        )
        action = self.agent.update_from_model(resp).action

        # 2) Step environment
        obs, rew, done, info = await asyncio.get_event_loop().run_in_executor(
            None, self.env.step, action
        )
        self.agent.update_from_env(obs, rew, done, info)

        # 3) Timeout check
        if time.time() - self._start_ts > self.timeout:
            self.agent.get_current_state().done = True
            return True
        return done

    async def _append_final_reward(self):
        last_step = self.agent.get_current_state()
        if last_step is None:
            return
        add_r = await asyncio.get_event_loop().run_in_executor(
            None, self.final_reward_fn, self.env.task, last_step.model_response
        )
        last_step.reward += add_r

    def _fill_returns(self):
        traj = self.agent.trajectory
        g = 0.0
        for step in reversed(traj.steps):
            g = step.reward + self.gamma * g
            step.mc_return = g
        traj.reward = traj.steps[0].mc_return if traj.steps else 0.0

    async def _close(self):
        await asyncio.get_event_loop().run_in_executor(None, self.env.close)

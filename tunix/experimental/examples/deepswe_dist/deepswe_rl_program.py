"""DeepSWE RL Program managing multi-turn agent interaction in Kubernetes sandboxes."""

import asyncio
import logging
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator.rl_program import RLProgram
from tunix.experimental.orchestrator.rl_program import RLStepResult
from examples.deepswe.swe_env import SWEEnv


class DeepsweRLProgram(RLProgram):
  """Orchestrates multi-turn DeepSWE episodes inside sandboxes."""

  def __init__(
      self, dataset, algo, max_steps: int = 100, max_turns: int = 10, **kwargs
  ):
    super().__init__()
    self.dataset = dataset
    self.algo = algo
    self.max_steps = max_steps
    self.max_turns = max_turns
    self.agent_kwargs = {}
    self.env_kwargs = {
        "max_steps": max_turns,
        "step_timeout": 1800,
        "backend": "kubernetes",
    }

  async def _run_agent_sandbox_loop(self, example):
    logging.info(
        "Initializing agent and env for task: %s",
        example.get("instance_id", "unknown"),
    )

    self.env = SWEEnv(example, max_steps=self.max_turns, use_agent_sandbox=True)
    self.obs = self.env.reset()

    chat_history = str(self.obs)
    done = False
    turn = 0
    current_reward = 0.0

    try:
      while not done and turn < self.max_turns:
        logging.info(
            "Turn %d/%d calling remote Rollout...", turn, self.max_turns
        )

        responses = await self.engine.generate([chat_history])
        if responses:
          item = responses[0]
          action_str = (
              item.traj
              if isinstance(item, datatypes.TrajectoryItem)
              else str(item)
          )
        else:
          action_str = ""

        obs, current_reward, done, _ = self.env.step(action_str)
        logging.info(
            "Turn %d/%d Rollout completed. Action: %s... Reward: %s",
            turn,
            self.max_turns,
            action_str[:50],
            current_reward,
        )
        chat_history += f"\nAction: {action_str}\nObs: {obs}"

        turn += 1
        if done:
          logging.info(
              "Episode done for %s at turn %d. Reward: %s",
              example.get("instance_id"),
              turn,
              current_reward,
          )
          break
    finally:
      self.env.close()

    return chat_history, current_reward

  async def _run_async(self) -> None:
    self._is_running = True
    step = 0

    while step < self.max_steps:
      try:
        example = next(self.dataset)
      except StopIteration:
        logging.info("Dataset exhausted.")
        break

      trajectory, reward = await self._run_agent_sandbox_loop(example)

      payload = datatypes.RLTrainerPayload(
          advantages=np.zeros(1), loss_mask=np.zeros((1, 1))
      )

      await self.engine.train_step(payload=payload)
      logging.info(
          "Trainer processed trajectory of length %d with reward %s",
          len(trajectory),
          reward,
      )
      logging.info("Step %d finished. Reward: %s", step, reward)
      self.last_step_result = RLStepResult(
          step=step,
          policy_version=step,
          num_rollouts=1,
          num_microbatches=1,
          reward_mean=reward,
          reward_std=0.0,
      )
      step += 1

    self._is_running = False

  def run(self, engine, train_dataset=None, num_steps=None, **kwargs):
    self.engine = engine
    if train_dataset is not None:
      self.dataset = iter(train_dataset)
    if num_steps is not None:
      self.max_steps = num_steps

    try:
      loop = asyncio.get_running_loop()
      task = loop.create_task(self._run_async())
      loop.run_until_complete(task)
    except RuntimeError:
      asyncio.run(self._run_async())


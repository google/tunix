"""Guarded trajectory collection engine for DeepSWE.

Subclasses TrajectoryCollectEngine to insert an ActionGuard between
agent.update_from_model() and env.step(). When the guard blocks an
action, a synthetic observation is injected instead of calling the
environment.
"""

import asyncio
import time

from absl import logging
from tunix.rl.agentic import utils
from tunix.rl.agentic.trajectory.trajectory_collect_engine import (
    TrajectoryCollectEngine,
)

from action_guard import ActionGuard, GuardConfig


class GuardedTrajectoryCollectEngine(TrajectoryCollectEngine):
  """TrajectoryCollectEngine with failure-aware action guard."""

  def __init__(self, *args, guard_config=None, **kwargs):
    # Pop guard_config before passing to parent (it's not a parent kwarg)
    super().__init__(*args, **kwargs)
    self.guard = ActionGuard(guard_config or GuardConfig())

  async def _reset(self):
    await super()._reset()
    self.guard.reset()

  async def _one_step(self) -> bool:
    """Single step with guard evaluation inserted before env.step()."""
    # 1. Model call
    resp = await asyncio.get_event_loop().run_in_executor(
        None,
        self.model_call,
        self.agent.chat_completions,
        self.env,
        **self.model_call_kwargs,
    )

    # 2. Parse action
    action = self.agent.update_from_model(resp).action
    if action is None:
      logging.warning(
          "Agent returned None action, using empty action list as fallback"
      )
      action = []

    # 3. Guard evaluation
    verdict = self.guard.evaluate(action)
    if verdict.blocked:
      logging.warning("Guard blocked action: %s", verdict.reason)
      obs, rew, done, info = verdict.message, 0.0, False, {
          "guard_blocked": True,
          "guard_reason": verdict.reason,
      }
    else:
      obs, rew, done, info = await asyncio.get_event_loop().run_in_executor(
          None, self.env.step, action
      )
      self.guard.record_outcome(action, str(obs))

    # 4. Update agent
    self.agent.update_from_env(obs, rew, done, info)

    # 5. Tokenization (same as parent _one_step)
    if self.tokenizer is not None and self.chat_parser is not None:
      cur_step = self.agent.get_current_state()
      if cur_step is not None:
        assistant_message, env_messages = (
            utils.get_recent_assistant_user_messages(
                self.agent.chat_completions
            )
        )

        if assistant_message:
          assistant_tokens, assistant_masks = (
              utils.tokenize_and_generate_masks(
                  [assistant_message],
                  tokenizer=self.tokenizer,
                  parser=self.chat_parser,
                  contains_first_msg=False,
                  contains_generation_msg=False,
              )
          )
          cur_step.assistant_tokens = assistant_tokens
          cur_step.assistant_masks = assistant_masks

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

    # 6. Timeout check
    if time.time() - self._start_ts > self.timeout:
      logging.warning("Episode timed out after %d seconds.", self.timeout)
      self.agent.get_current_state().done = True
      return True
    return done

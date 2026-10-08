"""SWE environment wrapper with lightweight action guarding."""

from action_guard import ActionGuard, GuardConfig
from swe_env import SWEEnv
from tunix.rl.agentic.environments.base_environment import EnvStepResult


class GuardedSWEEnv(SWEEnv):
  """SWEEnv with failure-aware action restrictions applied in step()."""

  def __init__(self, *args, guard_config=None, **kwargs):
    self.guard = ActionGuard(guard_config or GuardConfig())
    super().__init__(*args, **kwargs)

  def _initial_observation(self):
    self.guard.reset()
    return super()._initial_observation()

  def _step_impl(self, action):
    if isinstance(action, str):
      func_name, _ = self.guard._parse_action(action)  # pylint: disable=protected-access
      if not func_name:
        return EnvStepResult(
            observation=(
                "[ACTION GUARD] Your previous response did not include a"
                " valid function call. You must output exactly one tool call"
                " in the required XML format."
            ),
            reward=0.0,
            done=False,
            info={
                "guard_blocked": True,
                "guard_reason": "missing_function_call",
            },
        )

      verdict = self.guard.evaluate(action)
      if verdict.blocked:
        return EnvStepResult(
            observation=verdict.message,
            reward=0.0,
            done=False,
            info={
                "guard_blocked": True,
                "guard_reason": verdict.reason,
            },
        )

    result = super()._step_impl(action)
    if isinstance(action, str):
      self.guard.record_outcome(action, str(result.observation))
    return result

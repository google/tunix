import json
import logging
import os
import time
from typing import Any, Optional, cast
import numpy as np

try:
  import r2egym  # pytype: disable=import-error
  from r2egym.agenthub.action import Action  # pytype: disable=import-error
  from r2egym.agenthub.environment.env import EnvArgs, RepoEnv  # pytype: disable=import-error
except ImportError:
  r2egym = cast(Any, None)
  EnvArgs = cast(Any, None)
  RepoEnv = cast(Any, None)
  Action = cast(Any, None)

from tunix.rl.agentic.environments.base_environment import BaseTaskEnv, EnvStepResult


if r2egym:
  R2EGYM_PATH = os.path.dirname(r2egym.__file__)
else:
  R2EGYM_PATH = ""
# List of tools to be used in the environment.
R2EGYM_COMMAND_FILES = [
    os.path.join(R2EGYM_PATH, "agenthub/tools/r2egym/file_editor.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools/search.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools/r2egym/execute_bash.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools/finish.py"),
]

SWEAGENT_COMMAND_FILES = [
    os.path.join(R2EGYM_PATH, "agenthub/tools/str_replace_editor.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools/execute_bash.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools/submit.py"),
]


def _unpack_entry(entry: dict) -> dict:
  """Utility to clean up and unpack the dataset entry."""
  unpacked_entry = {}
  for k, v in entry.items():
    if isinstance(v, np.ndarray):
      unpacked_entry[k] = v.item()
    elif isinstance(v, list):
      if len(v) != 1:
        raise ValueError(
            f"Can only convert a list of size 1; got size {len(v)}"
        )
      unpacked_entry[k] = v[0]
    else:
      unpacked_entry[k] = v
  return unpacked_entry


class SWEEnv(BaseTaskEnv):
  """Software Engineering Environment for code-related tasks."""

  def __init__(
      self,
      entry: dict,
      group_id: int | None = None,
      pair_index: int | None = None,
      step_timeout: int = 30 * 60,
      reward_timeout: int = 30 * 60,
      backend: str = "kubernetes",
      delete_image: bool = False,
      verbose: bool = False,
      scaffold: str = "r2egym",
      max_steps: int = 1,
      sandbox_manager: Any | None = None,
  ):
    """Initialize the SWE environment.

    Args:
        entry: Dataset containing the tasks. If None, uses default dataset.
        group_id: ID of the group to which the task belongs.
        pair_index: Index of the pair to use. If None, selects a random pair.
        step_timeout: Timeout for each step in seconds.
        reward_timeout: Timeout for reward computation in seconds.
        backend: Backend to use for the environment.
        delete_image: Whether to delete the Docker image after closing.
        sandbox_manager: Optional run-owned Agent Sandbox Fleet manager. When
          absent, the existing direct R2E runtime is used unchanged.
    """
    self.entry = _unpack_entry(entry)
    prompt = self.entry.get("prompts", self.entry.get("problem_statement"))
    if not isinstance(prompt, str) or not prompt:
      raise ValueError(
          "SWEEnv entry must contain a non-empty string in 'prompts' or "
          "'problem_statement'"
      )
    self.step_timeout = step_timeout
    self.reward_timeout = reward_timeout
    self.total_steps = 0
    self.delete_image = delete_image
    self.backend = backend
    self.env = None
    self.sandbox_handle = None
    self.sandbox_manager = sandbox_manager
    self.runtime_time = {
        "sandbox_acquire_latency": 0.0,
        "sandbox_start_latency": 0.0,
        "environment_reset_latency": 0.0,
    }
    self.verbose = verbose
    self.scaffold = scaffold
    assert scaffold in [
        "r2egym",
        "sweagent",
    ], f"Invalid scaffold: {scaffold}, must be one of ['r2egym', 'sweagent']"
    if sandbox_manager is not None and scaffold != "r2egym":
      raise ValueError("Agent Sandbox Fleet admits only scaffold=r2egym")
    # ``BaseTaskEnv.task`` is the durable pre-observation task record used by
    # the trajectory collector.  R2E sandbox creation can time out before
    # ``get_task_instruction()`` gives the agent its first observation, so the
    # agent trajectory cannot be the only copy of the prompt.  Keep the
    # singleton batch shape expected by ``merge_micro_batches``; the learner
    # adds ``policy_version`` to this same record before reset.
    super().__init__(task={"prompts": [prompt]}, max_steps=max_steps)

    if not hasattr(self, "extra_kwargs"):
      self.extra_kwargs = {}

    self.extra_kwargs["group_id"] = group_id
    self.extra_kwargs["pair_index"] = pair_index

  @property
  def _debug_prefix(self) -> str:
    return (
        "[SWEEnv "
        f"group={self.extra_kwargs.get('group_id', '?')} "
        f"pair={self.extra_kwargs.get('pair_index', '?')}]"
    )

  def _initial_observation(self) -> Any:
    if not self.env:
      # Initialize environment if not created yet.
      logging.info(
          "%s creating RepoEnv backend=%s scaffold=%s step_timeout=%ss "
          "reward_timeout=%ss",
          self._debug_prefix,
          self.backend,
          self.scaffold,
          self.step_timeout,
          self.reward_timeout,
      )
      started = time.perf_counter()
      try:
        if self.sandbox_manager is None:
          env_args = EnvArgs(ds=self.entry)
          self.env = RepoEnv(
              env_args,
              backend=self.backend,
              step_timeout=self.step_timeout,
              reward_timeout=self.reward_timeout,
              verbose=self.verbose,
          )
        else:
          batch_started = self.extra_kwargs.get(
              "_trajectory_batch_started_monotonic"
          )
          self.sandbox_handle, acquire_elapsed = (
              self.sandbox_manager.acquire(
                  self.entry, batch_started_monotonic=batch_started
              )
          )
          self.runtime_time["sandbox_acquire_latency"] += acquire_elapsed
          bind_started = time.perf_counter()
          try:
            self.env = self.sandbox_manager.make_repo_env(
                self.sandbox_handle,
                command_files=R2EGYM_COMMAND_FILES,
                step_timeout=self.step_timeout,
                reward_timeout=self.reward_timeout,
                verbose=self.verbose,
            )
          except BaseException as bind_error:
            try:
              self.sandbox_manager.release(self.sandbox_handle)
            except BaseException as release_error:
              raise ExceptionGroup(
                  "Agent Sandbox bind and cleanup both failed",
                  [bind_error, release_error],
              ) from release_error
            self.sandbox_handle = None
            raise
          self.runtime_time["sandbox_start_latency"] += (
              time.perf_counter() - bind_started
          )
      except BaseException:
        if self.sandbox_manager is None:
          self.runtime_time["sandbox_start_latency"] += (
              time.perf_counter() - started
          )
        raise
      if self.sandbox_manager is None:
        runtime_time = getattr(
            getattr(self.env, "runtime", None), "_tunix_runtime_time", {}
        )
        if isinstance(runtime_time, dict):
          self.runtime_time["sandbox_acquire_latency"] += float(
              runtime_time.get("sandbox_acquire_latency", 0.0)
          )
          self.runtime_time["sandbox_start_latency"] += float(
              runtime_time.get("sandbox_start_latency", 0.0)
          )
      lifecycle_elapsed = time.perf_counter() - started
      classified_elapsed = (
          self.runtime_time["sandbox_acquire_latency"]
          + self.runtime_time["sandbox_start_latency"]
      )
      self.runtime_time["environment_reset_latency"] += max(
          0.0, lifecycle_elapsed - classified_elapsed
      )
      logging.info(
          "%s RepoEnv created in %.2fs",
          self._debug_prefix,
          time.perf_counter() - started,
      )
    else:
      logging.info("%s resetting existing RepoEnv", self._debug_prefix)
      started = time.perf_counter()
      try:
        self.env.reset()
      finally:
        self.runtime_time["environment_reset_latency"] += (
            time.perf_counter() - started
        )
    self.final_reward_fn = self.env.compute_reward
    if self.scaffold == "r2egym" and self.sandbox_manager is None:
      self.env.add_commands(R2EGYM_COMMAND_FILES)
    elif self.scaffold == "sweagent":
      self.env.add_commands(SWEAGENT_COMMAND_FILES)
    self.total_steps = 0

    # Polls docker runtime to get task instruction.
    started = time.perf_counter()
    try:
      instruction = self.env.get_task_instruction()
    finally:
      self.runtime_time["environment_reset_latency"] += (
          time.perf_counter() - started
      )
    logging.info(
        "%s get_task_instruction done in %.2fs chars=%d",
        self._debug_prefix,
        time.perf_counter() - started,
        len(str(instruction)),
    )
    return instruction

  def _step_impl(self, action: Any) -> EnvStepResult:
    if isinstance(action, str):
      action_obj = Action.from_string(action)
    else:
      action_obj = action

    if not action_obj.function_name:
      return EnvStepResult(observation="", reward=0, done=False, info={})

    # RepoEnv always returns 0 reward, must be evaluated by DockerRuntime.
    if not self.env:
      raise ValueError("Environment not initialized")
    logging.info(
        "%s env.step start step=%d function=%s",
        self._debug_prefix,
        self.total_steps,
        getattr(action_obj, "function_name", None),
    )
    started = time.perf_counter()
    obs, reward, done, info = self.env.step(action_obj)
    logging.info(
        "%s env.step done in %.2fs done=%s reward=%s chars=%d",
        self._debug_prefix,
        time.perf_counter() - started,
        done,
        reward,
        len(str(obs)),
    )

    self.total_steps += 1

    return EnvStepResult(
        observation=str(obs), reward=reward, done=done, info=info
    )

  def close(self) -> None:
    """Close the environment and clean up resources."""
    close_error = None
    repo_env = self.env
    if repo_env is not None:
      logging.info("%s closing RepoEnv", self._debug_prefix)
      try:
        repo_env.close()
      except BaseException as error:  # release still must run
        close_error = error
      finally:
        self.env = None

      # Preserve the pre-Fleet direct-R2E behavior exactly.  Fleet resources
      # are controller-owned and are deleted through sandbox_manager.release().
      if self.delete_image and self.sandbox_manager is None:
        docker_image = repo_env.runtime.docker_image
        os.system(f"docker rmi {docker_image}")

    if self.sandbox_handle is not None:
      handle = self.sandbox_handle
      self.sandbox_handle = None
      try:
        cleanup_elapsed = self.sandbox_manager.release(handle)
        logging.info(
            "%s Agent Sandbox released in %.2fs",
            self._debug_prefix,
            cleanup_elapsed,
        )
      except BaseException as error:
        if close_error is not None:
          raise ExceptionGroup(
              "R2E adapter close and Agent Sandbox release both failed",
              [close_error, error],
          ) from error
        raise

    if close_error is not None:
      raise close_error

  @staticmethod
  def from_dict(extra_info: dict | str) -> "SWEEnv":  # pyrefly: ignore[bad-override]
    """Create an environment instance from JSON configuration.

    Args:
        extra_info: Dictionary containing configuration parameters. The entire
          dict will be used as 'entry', and any keys matching __init__
          parameters will be extracted and passed.

    Returns:
        Initialized SWEEnv instance
    """
    import inspect

    if isinstance(extra_info, str):
      extra_info = json.loads(extra_info)

    sig = inspect.signature(SWEEnv.__init__)
    init_params = {}
    for param_name, param in sig.parameters.items():
      if param_name == "self":
        continue
      if param_name in extra_info:
        init_params[param_name] = extra_info[param_name]
      # else if param has default value, use the default value
    init_params["entry"] = extra_info
    return SWEEnv(**init_params)

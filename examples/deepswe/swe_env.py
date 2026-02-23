import json
import os
import traceback 
from typing import Any
from absl import logging 
import sys
import numpy as np
try:
    import r2egym
    from r2egym.agenthub.action import Action
    from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
except ImportError:
    r2egym = None
    EnvArgs = None
    RepoEnv = None
    Action = None

from tunix.rl.agentic.environments.base_environment import BaseTaskEnv, EnvStepResult

R2EGYM_PATH = os.path.dirname(r2egym.__file__)
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


class SWEEnv(BaseTaskEnv):
    """Software Engineering Environment for code-related tasks."""

    def __init__(
        self,
        entry: dict,
        group_id: int | None = None,
        pair_index: int | None = None,
        step_timeout: int = 90,
        reward_timeout: int = 300,
        backend: str = "kubernetes",
        delete_image: bool = False,
        verbose: bool = False,
        scaffold: str = "r2egym",
        max_steps: int = 1,
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
        """
        self.entry = entry
        self.step_timeout = step_timeout
        self.reward_timeout = reward_timeout
        self.total_steps = 0
        self.delete_image = delete_image
        self.backend = backend
        self.env = None
        self.verbose = verbose
        self.scaffold = scaffold
        assert scaffold in ["r2egym", "sweagent"], f"Invalid scaffold: {scaffold}, must be one of ['r2egym', 'sweagent']"
        os.write(1, f"SWEEnv is initialized with: {max_steps}\n".encode())
        super().__init__(max_steps=max_steps)

        if not hasattr(self, "extra_kwargs"):
            self.extra_kwargs = {}
            
        self.extra_kwargs["group_id"] = group_id
        self.extra_kwargs["pair_index"] = pair_index

    def _prepare_entry(self, example: dict) -> dict:
        single_example = {}
        json_keys = {'modified_files', 'relevant_files', 'modified_entity_summaries'}
        for k, v in self.entry.items():
            if k == 'prompts':
                continue
            if isinstance(v, (list, np.ndarray)) and len(v) > 0:
                val = v[0]
            else:
                val= v

            if k in json_keys:
                single_example[k] = json.loads(str(val))
            else:
                if  hasattr(val, "item"):
                    single_example[k] = val.item()
                else:
                    single_example[k] = val
        return single_example
        
    def _initial_observation(self) -> Any:
        if not self.env:
            # Initialize environment if not created yet.
            env_args = EnvArgs(ds=self._prepare_entry(self.entry))
            self.env = RepoEnv(env_args, backend=self.backend, step_timeout=self.step_timeout, reward_timeout=self.reward_timeout, verbose=self.verbose)
        else:
            self.env.reset()
        if self.scaffold == "r2egym":
            self.env.add_commands(R2EGYM_COMMAND_FILES)
        else:
            self.env.add_commands(SWEAGENT_COMMAND_FILES)
        self.total_steps = 0

        # Polls docker runtime to get task instruction.
        return self.env.get_task_instruction()

    def _step_impl(self, action: Any) -> EnvStepResult:
        os.write(1, b"SWEEnv step impl called\n")
        if isinstance(action, str):
            os.write(1, f"action string: {action}\n".encode())
            action_obj: Action = Action.from_string(action)
        else:
            action_obj = action

        if not action_obj.function_name:
            os.write(1, b"didn't find any funciton to call\n")
            return EnvStepResult(observation="", reward=0, done=False, info={})

        # RepoEnv always returns 0 reward, must be evaluated by DockerRuntime.
        os.write(1, b"calling r2e env\n")
        obs, reward, done, info = self.env.step(action_obj)

        self.total_steps += 1

        return EnvStepResult(observation=str(obs), reward=reward, done=done, info=info)

    def compute_final_reward(self) -> float:
        os.write(1, b"computing final reward\n")
        reward = self.env.compute_reward()
        os.write(1, f"final reward: {reward}\n".encode())
        return reward

    def close(self) -> None:
        """Close the environment and clean up resources."""
        if self.env is not None:
            self.env.close()

        if self.delete_image:
            docker_image = self.env.runtime.docker_image
            os.system(f"docker rmi {docker_image}")
            
    def compute_final_reward(self):
        """Run tests in the Docker container and return reward (0 or 1)."""
        return self.env.compute_reward()

    @staticmethod
    def from_dict(extra_info: dict | str) -> "SWEEnv":
        """Create an environment instance from JSON configuration.

        Args:
            extra_info: Dictionary containing configuration parameters.
                       The entire dict will be used as 'entry', and any keys
                       matching __init__ parameters will be extracted and passed.

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

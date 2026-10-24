# Copyright 2026 Google LLC
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

"""DeepSWE components for the experimental distributed GRPO recipe."""

from __future__ import annotations

from collections.abc import Iterator
import json
import logging
import os
import re
import time
from typing import Any

import numpy as np
from tunix.dist_rl.common import registry
from tunix.dist_rl.orchestrator import dataset_utils
from tunix.experimental.examples.deepswe_dist import openhands_template
from tunix.experimental.examples.deepswe_dist import openhands_utils
from tunix.experimental.examples.deepswe_dist import sandbox_utils
try:
  from tunix.oss.examples.deepswe import deepswe_data
  from tunix.oss.examples.deepswe import swe_agent
  from tunix.oss.examples.deepswe import swe_env
except ImportError:
  from examples.deepswe import deepswe_data  # pyrefly: ignore[missing-import]
  from examples.deepswe import swe_agent  # pyrefly: ignore[missing-import]
  from examples.deepswe import swe_env  # pyrefly: ignore[missing-import]
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.agents import base_agent
from tunix.rl.agentic.environments import base_environment


DEEPSWE_ENV_NAME = "deepswe_env"
DEEPSWE_AGENT_NAME = "deepswe_agent"
CODEACT_AGENT_NAME = "codeact_agent"
DEFAULT_DATASET_NAME = "R2E-Gym/R2E-Gym-Subset"


def normalize_example_value(value: Any) -> Any:
  """Converts dataset scalar wrappers into plain Python values."""
  if isinstance(value, np.ndarray):
    flat = value.reshape(-1).tolist()
    if len(flat) == 1:
      return normalize_example_value(flat[0])
    return [normalize_example_value(v) for v in flat]
  if isinstance(value, np.bytes_):
    return value.tobytes().decode("utf-8")
  if isinstance(value, bytes):
    return value.decode("utf-8")
  if isinstance(value, list):
    return [normalize_example_value(v) for v in value]
  return value


def as_text(value: Any) -> str:
  value = normalize_example_value(value)
  return value if isinstance(value, str) else str(value)


def _jsonify_lists(entry: dict[str, Any]) -> dict[str, Any]:
  """Matches the legacy DeepSWE recipe's heterogeneous dataset normalization."""
  normalized = {}
  for key, value in entry.items():
    value = normalize_example_value(value)
    normalized[key] = json.dumps(value) if isinstance(value, list) else value
  return normalized


def load_deepswe_dataset(
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_split: str = "train",
    dataset_path: str = "",
    cache_dir: str | None = None,
    shuffle: bool = True,
    seed: int = 42,
) -> Any:
  """Loads the R2E-Gym dataset used by the DeepSWE recipe."""
  if dataset_path:
    from datasets import DatasetDict  # pylint: disable=g-import-not-at-top
    from datasets import load_from_disk  # pylint: disable=g-import-not-at-top

    logging.info("Loading DeepSWE dataset from disk: %s", dataset_path)
    dataset = load_from_disk(dataset_path)
    if isinstance(dataset, DatasetDict):
      dataset = dataset[dataset_split]
    dataset = dataset.map(_jsonify_lists, keep_in_memory=True)
    if shuffle:
      dataset = dataset.shuffle(seed=seed)
    return dataset

  logging.info(
      "Loading DeepSWE dataset %s split=%s cache_dir=%s shuffle=%s seed=%d.",
      dataset_name,
      dataset_split,
      cache_dir,
      shuffle,
      seed,
  )
  return deepswe_data.create_dataset(
      dataset_name=dataset_name,
      dataset_split=dataset_split,
      cache_dir=cache_dir,
      shuffle=shuffle,
      seed=seed,
  )


def _entry_at(dataset: Any, index: int) -> dict[str, Any]:
  entry = dataset[index]
  if not isinstance(entry, dict):
    raise TypeError(f"DeepSWE dataset item must be a dict, got {type(entry)}")
  return _jsonify_lists(dict(entry))


def _problem_statement(entry: dict[str, Any]) -> str:
  for key in ("problem_statement", "prompt", "instance_id"):
    if key in entry and entry[key] is not None:
      return as_text(entry[key])
  return ""


def build_prompt_item(
    *,
    entry: dict[str, Any],
    prompt_idx: int,
    max_turns: int,
    max_response_length: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    step_timeout_secs: int,
    reward_timeout_secs: int,
    env_backend: str,
    use_agent_sandbox: bool,
    scaffold: str,
    env_verbose: bool,
    overlong_filter: bool = False,
    episode_timeout_secs: int | None = None,
    return_routed_experts: bool = False,
) -> dict[str, Any]:
  """Builds one StandardRLProgram prompt item for a DeepSWE task."""
  problem = _problem_statement(entry)
  prompt_id = as_text(
      entry["instance_id"]
      if "instance_id" in entry and entry["instance_id"]
      else f"deepswe_{prompt_idx}"
  )
  env_config = {
      "entry": entry,
      "prompt_id": prompt_id,
      "max_steps": max_turns,
      "step_timeout": step_timeout_secs,
      "reward_timeout": reward_timeout_secs,
      "backend": env_backend,
      "use_agent_sandbox": use_agent_sandbox,
      "scaffold": scaffold,
      "verbose": env_verbose,
  }
  agent_config = {"scaffold": scaffold}
  metadata: dict[str, Any] = {
      "instance_id": prompt_id,
      "problem_statement": problem,
      "env_config": env_config,
      "agent_config": agent_config,
      # Read by TrajectoryCollectorEngine to decide which terminal statuses
      # get their policy loss mask zeroed out.
      "overlong_filter": overlong_filter,
  }
  # When `episode_timeout_secs` is None, `"episode_timeout"` is omitted from
  # `metadata` so `TrajectoryCollectorEngine` uses its default
  # (`timeouts.DEFAULT_EPISODE_TIMEOUT_S`).
  if episode_timeout_secs is not None:
    metadata["episode_timeout"] = episode_timeout_secs
  return {
      "prompt": problem,
      "prompt_id": prompt_id,
      "max_turns": max_turns,
      "max_response_length": max_response_length,
      "generation_kwargs": {
          "max_generation_steps": max_response_length,
          "temperature": temperature,
          "top_p": top_p,
          "top_k": top_k,
          "return_logprobs": True,
          "return_routed_experts": return_routed_experts,
      },
      "metadata": metadata,
  }


def iter_prompt_items(
    *,
    dataset: Any,
    max_steps: int,
    batch_size: int,
    max_turns: int,
    max_response_length: int,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    step_timeout_secs: int,
    reward_timeout_secs: int,
    env_backend: str,
    use_agent_sandbox: bool,
    scaffold: str,
    env_verbose: bool,
    overlong_filter: bool = False,
    max_staleness: int = 0,
    episode_timeout_secs: int | None = None,
    return_routed_experts: bool = False,
) -> Iterator[dict[str, Any]]:
  """Yields one full-batch worth of prompt groups per RL step plus off-policy lookahead."""
  dataset_size = len(dataset)
  if dataset_size <= 0:
    raise ValueError("DeepSWE dataset is empty.")

  total_prompts = dataset_utils.total_prompt_groups(
      max_steps, batch_size, max_staleness
  )
  for prompt_idx in range(total_prompts):
    yield build_prompt_item(
        entry=_entry_at(dataset, prompt_idx % dataset_size),
        prompt_idx=prompt_idx,
        max_turns=max_turns,
        max_response_length=max_response_length,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        step_timeout_secs=step_timeout_secs,
        reward_timeout_secs=reward_timeout_secs,
        episode_timeout_secs=episode_timeout_secs,
        env_backend=env_backend,
        use_agent_sandbox=use_agent_sandbox,
        scaffold=scaffold,
        env_verbose=env_verbose,
        overlong_filter=overlong_filter,
        return_routed_experts=return_routed_experts,
    )


def multi_tool_calls_enabled() -> bool:
  """Whether CodeActAgent runs every tool call of a turn, like OpenHands."""
  value = os.getenv("OPENHANDS_MULTI_TOOL_CALLS", "false")
  return value.strip().lower() in ("1", "true", "yes")


def parse_openhands_xml_action(action_str: str) -> Any:
  """Parses an XML function call while preserving multiline code indentation."""
  swe_action_cls = openhands_utils._get_swe_action_cls()  # pylint: disable=protected-access
  fn_match = re.search(r"<function\s*=\s*([^>]+)>", action_str)
  function_name = fn_match.group(1).strip() if fn_match else ""

  pattern = r"<parameter\s*=\s*([^>]+)>(.*?)</parameter>"
  param_matches = re.findall(pattern, action_str, flags=re.DOTALL)

  params: dict[str, str] = {}
  for param_key, param_value in param_matches:
    param_key = param_key.strip()
    if param_value.startswith("\r\n"):
      param_value = param_value[2:]
    elif param_value.startswith("\n"):
      param_value = param_value[1:]
    if param_value.endswith("\r\n"):
      param_value = param_value[:-2]
    elif param_value.endswith("\n"):
      param_value = param_value[:-1]
    params[param_key] = param_value

  return swe_action_cls(function_name, params)


def parse_codeact_response(response_text: str) -> tuple[str, Any]:
  """Parses a model response in CodeAct / OpenHands format."""
  swe_action_cls = openhands_utils._get_swe_action_cls()  # pylint: disable=protected-access
  xml_pattern = re.compile(
      r"(?s)(<function\s*=\s*([^>]+)>.*?(?:</function>|$))"
  )
  tc_pattern = re.compile(r"(?s)<tool_call>\s*(.*?)\s*(?:</tool_call>|$)")

  def _first_tool_call(text_slice: str) -> re.Match[str] | None:
    for m in tc_pattern.finditer(text_slice):
      if xml_pattern.search(m.group(1)):
        return m
    return None

  def _collect_candidates(text_slice: str, offset: int):
    found = []
    tc_match = _first_tool_call(text_slice)
    if tc_match:
      found.append((0, offset + tc_match.start(), "tool_call", tc_match))

    xml_match = xml_pattern.search(text_slice)
    if xml_match:
      if not tc_match or not (
          tc_match.start() <= xml_match.start() <= tc_match.end()
      ):
        found.append((0, offset + xml_match.start(), "xml", xml_match))
    return found

  think_end = response_text.rfind("</think>")
  if think_end != -1:
    post_think_offset = think_end + len("</think>")
    candidates = _collect_candidates(
        response_text[post_think_offset:], post_think_offset
    )
    if not candidates:
      candidates = _collect_candidates(response_text, 0)
  else:
    candidates = _collect_candidates(response_text, 0)

  if candidates:
    candidates.sort(key=lambda x: (x[0], x[1]))
    _, match_start, match_type, payload = candidates[0]

    if match_type == "xml":
      thought = response_text[:match_start].strip()
      xml_str = payload.group(1).strip()
      if not xml_str.endswith("</function>"):
        xml_str += "\n</function>"
      action = parse_openhands_xml_action(xml_str)
      return thought, action
    elif match_type == "tool_call":
      thought = response_text[:match_start].strip()
      inner_match = xml_pattern.search(payload.group(1).strip())
      assert inner_match is not None
      xml_str = inner_match.group(1).strip()
      if not xml_str.endswith("</function>"):
        xml_str += "\n</function>"
      action = parse_openhands_xml_action(xml_str)
      return thought, action

  if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in response_text:
    thought = response_text.replace(
        "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", ""
    ).strip()
    action = swe_action_cls("submit", {})
    return thought, action

  thought = response_text.strip()
  action = swe_action_cls("", {})
  return thought, action


def parse_codeact_response_all(response_text: str) -> tuple[str, list[Any]]:
  """Parses every tool call of a CodeAct / OpenHands response, in order."""
  thought, first = parse_codeact_response(response_text)
  _, think_sep, after_think = response_text.partition("</think>")
  content = after_think if think_sep else response_text
  function_pattern = re.compile(
      r"(?s)<function\s*=\s*[^>]+>.*?(?:</function>|$)"
  )
  opener_pattern = re.compile(r"<tool_call>|<function\s*=")
  calls = []
  pos = 0
  previous = None
  while (opener := opener_pattern.search(content, pos)) is not None:
    if opener.group(0) == "<tool_call>":
      start = opener.end()
      ends = [
          i
          for i in (
              content.find("</tool_call>", start),
              content.find("<tool_call>", start),
          )
          if i != -1
      ]
      end = min(ends) if ends else len(content)
      function = function_pattern.search(content[start:end])
      if function:
        calls.append(function.group(0))
      previous = "tool_call"
      pos = end + (
          len("</tool_call>") if content.startswith("</tool_call>", end) else 0
      )
    else:
      function = function_pattern.match(content, opener.start())
      if function is None:
        pos = opener.end()
        continue
      if previous != "bare":
        calls.append(function.group(0))
      previous = "bare"
      pos = function.end()
  actions = []
  for xml_str in calls:
    xml_str = xml_str.strip()
    if not xml_str.endswith("</function>"):
      xml_str += "\n</function>"
    actions.append(parse_openhands_xml_action(xml_str))
  return thought, actions or [first]


@registry.register_env(DEEPSWE_ENV_NAME)
class DeepSWEEnv(swe_env.SWEEnv):
  """Registry adapter that lets RolloutWorker construct SWEEnv per request."""

  def __init__(
      self,
      entry: dict[str, Any] | None = None,
      prompt_id: str = "",
      group_index: int = 0,
      num_generations: int = 1,
      policy_version: int = 0,
      group_id: Any = None,
      pair_index: int | None = None,
      **kwargs: Any,
  ):
    entry = dict(entry or kwargs.pop("task", {}) or {})
    if prompt_id and "instance_id" not in entry:
      entry["instance_id"] = prompt_id
    if group_id is None:
      group_id = prompt_id or None
    if pair_index is None:
      pair_index = group_index
    scaffold = str(kwargs.get("scaffold", "r2egym"))
    assert scaffold in (
        "r2egym",
        "sweagent",
        "openhands",
    ), f"Invalid scaffold: {scaffold}"
    if (
        "use_agent_sandbox" in kwargs
        and kwargs["use_agent_sandbox"]
        and ("fleet" not in kwargs or kwargs["fleet"] is None)
    ):
      job_id = os.environ["ORCHESTRATOR_ID"]
      logging.info(
          "Attaching DeepSWE SandboxFleet in rollout worker (job_id=%s,"
          " num_generations=%d, warm_on_init=False, teardown_on_exit=False).",
          job_id,
          num_generations,
      )
      kwargs["fleet"] = sandbox_utils.init_global_fleet(
          tasks=[entry],
          max_concurrency=num_generations,
          num_generations=num_generations,
          scaffold=scaffold,
          job_id=job_id,
          warm_on_init=False,
          teardown_on_exit=False,
      )

    super_kwargs = dict(kwargs)
    super_kwargs["scaffold"] = (
        "r2egym"
        if scaffold in openhands_template.OPENHANDS_SCAFFOLDS
        else scaffold
    )
    super().__init__(
        entry=entry,
        group_id=group_id,
        pair_index=pair_index,
        **super_kwargs,
    )
    self.scaffold = scaffold
    self.workspace: Any = None
    self.task = {
        **entry,
        "prompt_id": prompt_id,
        "group_index": group_index,
        "num_generations": num_generations,
        "policy_version": policy_version,
    }

  def _init_agent_sandbox_env(self) -> None:
    sandbox_utils.patch_r2egym_for_agent_sandbox()
    from agent_sandbox_rl import Task  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import]  # pytype: disable=import-error
    from agent_sandbox_rl.adapters.r2egym import make_fleet_repo_env, r2egym_command_files  # pylint: disable=g-import-not-at-top,g-multiple-import  # pyrefly: ignore[missing-import]  # pytype: disable=import-error

    fleet: Any = self.fleet or sandbox_utils.get_global_fleet()
    logging.info("[SWEEnv] Acquiring SandboxHandle from SandboxFleet!")
    task_id = str(
        self.entry.get("instance_id", self.entry.get("docker_image", "default"))
    )
    task_img = self.entry.get("docker_image", "default")
    if isinstance(task_img, (list, np.ndarray)):
      task_img = task_img[0] if len(task_img) > 0 else "default"
    task_img_str = str(task_img)
    rewrite_fn = openhands_utils.get_image_rewrite_fn(
        getattr(fleet, "_image_rewrite_fn", None)
    )
    if rewrite_fn:
      task_img_str = rewrite_fn(task_img_str)
    task = Task(
        id=task_id,
        image=task_img_str,
        metadata={"ds": self.entry},
    )
    sandbox_utils.ensure_tasks_in_fleet_plan(
        fleet, [task], scaffold=self.scaffold
    )
    max_acquire_retries = (
        sandbox_utils.SandboxFailFastConfig.from_env().acquire_retries
    )
    for attempt in range(max_acquire_retries):
      try:
        self.handle = fleet.acquire(task)
        break
      except Exception as e:  # pylint: disable=broad-exception-caught
        if attempt < max_acquire_retries - 1:
          logging.warning(
              "[SWEEnv] fleet.acquire failed (attempt %d/%d): %s; retrying in"
              " %ds...",
              attempt + 1,
              max_acquire_retries,
              e,
              5 * (attempt + 1),
          )
          time.sleep(5 * (attempt + 1))
        else:
          raise
    is_openhands = self.scaffold in openhands_template.OPENHANDS_SCAFFOLDS
    if is_openhands:
      from agent_sandbox_rl.adapters.openhands import make_handle_workspace  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import]  # pytype: disable=import-error

      ws_kwargs: dict[str, Any] = {}
      if os.getenv("SANDBOX_SESSION_KEY"):
        ws_kwargs["api_key"] = os.getenv("SANDBOX_SESSION_KEY")
      if os.getenv("ROUTER_URL"):
        ws_kwargs["router_url"] = os.getenv("ROUTER_URL")
      if os.getenv("ROUTER_AUTH_TOKEN"):
        ws_kwargs["router_auth_token"] = os.getenv("ROUTER_AUTH_TOKEN")
      ws_kwargs["working_dir"] = os.getenv("OPENHANDS_WORKING_DIR", "/testbed")
      self.workspace = make_handle_workspace(self.handle, **ws_kwargs)

    cmd_files = None
    if not is_openhands:
      try:
        cmd_files = r2egym_command_files()
      except Exception:  # pylint: disable=broad-exception-caught
        cmd_files = None
    if is_openhands and self.handle is not None:
      self.handle._target_container = sandbox_utils.EVAL_CONTAINER_NAME  # pylint: disable=protected-access
      self.handle._defer_setup_env = True  # pylint: disable=protected-access
    try:
      self.env = make_fleet_repo_env(
          self.handle,
          command_files=cmd_files,
          step_timeout=self.step_timeout,
          reward_timeout=self.reward_timeout,
          verbose=self.verbose,
      )
    finally:
      if is_openhands and self.handle is not None:
        self.handle._target_container = sandbox_utils.RUNTIME_CONTAINER_NAME  # pylint: disable=protected-access
        self.handle._defer_setup_env = False  # pylint: disable=protected-access
    if is_openhands:
      sandbox_utils.set_runtime_container(
          self.env, sandbox_utils.EVAL_CONTAINER_NAME
      )
      openhands_utils.setup_openhands_workspace(self.workspace, self.entry)
    elif self.env is not None:
      openhands_utils.hide_r2e_tests_for_rollout(self.env, self.entry)

    if self.env is not None and hasattr(self.env, "compute_reward"):
      orig_compute_reward = self.env.compute_reward

      def _compute_reward_in_fresh_container(*args, **kwargs):
        rollout_target = self.workspace or self.env
        if self.workspace is None and self.env is not None:
          sandbox_utils.set_runtime_container(
              self.env, sandbox_utils.RUNTIME_CONTAINER_NAME
          )
        base_commit = openhands_utils.resolve_base_commit(self.entry)
        workspace_path = os.getenv("OPENHANDS_WORKING_DIR", "/testbed")
        patch = openhands_utils.extract_agent_patch(
            rollout_target,
            base_commit=base_commit,
            workspace_path=workspace_path,
            timeout=float(self.reward_timeout),
        )
        openhands_utils.cleanup_rollout_container_processes(rollout_target)
        return openhands_utils.evaluate_patch_in_fresh_container(
            self.env,
            patch,
            orig_compute_reward,
            *args,
            **kwargs,
        )

      self.env.compute_reward = _compute_reward_in_fresh_container

  def _initial_observation(self) -> Any:
    if not self.env and not self.workspace:
      if self.use_agent_sandbox:
        self._init_agent_sandbox_env()
      else:
        self._init_local_repo_env()
    elif self.env is not None:
      self.env.reset()

    env_any: Any = self.env
    self.final_reward_fn = (
        env_any.compute_reward if env_any is not None else None
    )
    self.total_steps = 0

    if self.workspace is not None:
      return str(
          self.entry.get("problem_statement")
          or self.entry.get("instruction")
          or ""
      )
    return env_any.get_task_instruction() if env_any is not None else ""

  def reset(self) -> tuple[Any, dict[str, Any]]:
    obs, info = super().reset()
    if self.scaffold in openhands_template.OPENHANDS_SCAFFOLDS:
      base_commit = openhands_utils.resolve_base_commit(self.entry)
      if base_commit:
        info["base_commit"] = base_commit
      if self.entry.get("repo_language"):
        info["repo_language"] = str(self.entry["repo_language"])
      info["workspace_path"] = os.getenv("OPENHANDS_WORKING_DIR", "/testbed")
    return obs, info

  def _step_openhands_calls(
      self, calls: list[Any]
  ) -> base_environment.EnvStepResult:
    """Runs the tool calls of one multi-call turn in order, like OpenHands."""
    budget = max(1, self.max_steps - self.step_count + 1)
    observations = []
    result = None
    for i, call in enumerate(calls[:budget]):
      if i > 0:
        self.step_count += 1
      result = self._step_impl(call)
      observations.append(result.observation)
      if result.done:
        break
    assert result is not None
    info = dict(result.info or {})
    info["num_tool_calls"] = len(observations)
    return base_environment.EnvStepResult(
        observation=observations if len(observations) > 1 else observations[0],
        reward=result.reward,
        done=result.done,
        info=info,
    )

  def _step_impl(self, action: Any) -> base_environment.EnvStepResult:
    if self.scaffold not in openhands_template.OPENHANDS_SCAFFOLDS:
      return super()._step_impl(action)

    if isinstance(action, (list, tuple)) and action:
      return self._step_openhands_calls(list(action))
    if isinstance(action, str):
      action_obj = openhands_utils.parse_openhands_action_str(action)
    else:
      action_obj = action

    if not action_obj.function_name:
      return base_environment.EnvStepResult(
          observation=openhands_template.OPENHANDS_FAKE_USER_RESPONSE,
          reward=0,
          done=False,
          info={
              "max_steps": self.max_steps,
              "is_fake_user_response": True,
          },
      )
    return openhands_utils.step_openhands(self, action_obj)

  def close(self) -> None:
    if self.workspace is not None:
      if hasattr(self.workspace, "close"):
        try:
          self.workspace.close()
        except Exception as e:  # pylint: disable=broad-exception-caught
          logging.warning("[SWEEnv] Workspace close note: %s", e)
      if hasattr(self.workspace, "cleanup"):
        try:
          self.workspace.cleanup()  # pytype: disable=attribute-error
        except Exception as e:  # pylint: disable=broad-exception-caught
          logging.warning("[SWEEnv] Workspace cleanup note: %s", e)
      self.workspace = None
    super().close()


@registry.register_agent(DEEPSWE_AGENT_NAME)
class DeepSWEAgent(swe_agent.SWEAgent):
  """Registry adapter for the DeepSWE XML-tool and OpenHands CodeAct agents."""

  name = DEEPSWE_AGENT_NAME

  def __init__(
      self,
      system_prompt: str | None = None,
      use_fn_calling: bool = False,
      format_model_response: bool = False,
      scaffold: str = "r2egym",
  ):
    assert scaffold in (
        "r2egym",
        "sweagent",
        "openhands",
    ), f"Invalid scaffold: {scaffold}"
    is_openhands = scaffold in openhands_template.OPENHANDS_SCAFFOLDS
    if is_openhands and system_prompt is None:
      system_prompt = openhands_template.OPENHANDS_SYSTEM_PROMPT
    super().__init__(
        system_prompt=system_prompt,
        use_fn_calling=use_fn_calling,
        format_model_response=format_model_response,
        scaffold="r2egym" if is_openhands else scaffold,
    )
    self.scaffold = scaffold
    self.multi_tool_calls = multi_tool_calls_enabled()
    if is_openhands:
      self.user_prompt_template = openhands_template.OPENHANDS_USER_PROMPT

  def update_from_env(
      self,
      observation: Any,
      reward: float,
      done: bool,
      info: dict[str, Any] | None = None,
      **kwargs,
  ) -> None:
    if self.scaffold not in openhands_template.OPENHANDS_SCAFFOLDS:
      super().update_from_env(observation, reward, done, info, **kwargs)
      return

    if isinstance(observation, list):
      observation = [str(o) for o in observation]
      base_agent.ConversationAgentBase.update_from_env(
          self, observation, reward, done, info
      )
      self.cur_step = agent_types.Step(observation=observation)
      return
    observation = str(observation)
    if info is None:
      info = {}
    if not self._trajectory.steps:
      if self.user_prompt_template == openhands_template.OPENHANDS_USER_PROMPT:
        observation = openhands_template.format_openhands_user_prompt(
            problem_statement=observation,
            workspace_path=str(info.get("workspace_path") or "/testbed"),
            repo_language=str(info.get("repo_language") or "python"),
            base_commit=str(info.get("base_commit") or ""),
        )
      else:
        observation = self.user_prompt_template.format(
            problem_statement=observation
        )
    elif not observation and not self._trajectory.steps[-1].action:
      observation = openhands_template.OPENHANDS_FAKE_USER_RESPONSE

    base_agent.ConversationAgentBase.update_from_env(
        self, observation, reward, done, info
    )
    self.cur_step = agent_types.Step(observation=observation)

  def _observation_to_messages(
      self, observation: Any, reward: float, done: bool, info: dict[str, Any]
  ) -> None:
    if self.scaffold not in openhands_template.OPENHANDS_SCAFFOLDS:
      super()._observation_to_messages(observation, reward, done, info)
      return
    if not self._trajectory.steps:
      self._messages.append({"role": "user", "content": str(observation)})
      return
    if isinstance(observation, list):
      for result in observation:
        self._messages.append({"role": "tool", "content": str(result)})
      return
    last_step = self._trajectory.steps[-1]
    if (info and info.get("is_fake_user_response")) or not last_step.action:
      self._messages.append({"role": "user", "content": str(observation)})
    else:
      self._messages.append({"role": "tool", "content": str(observation)})

  def _parse_model_response(self, response: str | Any) -> tuple[str, Any]:
    if self.use_fn_calling:
      return swe_agent.parse_oai_response(response)
    if self.scaffold in openhands_template.OPENHANDS_SCAFFOLDS:
      return parse_codeact_response(response)
    return swe_agent.parse_xml_response(response)

  def update_from_model(self, response: str, **kwargs) -> agent_types.Action:
    self._trajectory.steps.append(self.cur_step)
    thought, action = self._parse_model_response(response)
    action_str: Any = (
        action.to_xml_string() if getattr(action, "function_name", True) else ""
    )

    cur_step: Any = self._trajectory.steps[-1]
    cur_step.thought = thought
    cur_step.action = action_str
    cur_step.model_response = response

    if self.format_model_response:
      self._messages.append(
          {"role": "assistant", "content": f"{thought}\n\n{action_str}"}
      )
    else:
      self._messages.append({"role": "assistant", "content": response})
    self.step += 1
    result = agent_types.Action(action=cur_step.action)
    if (
        self.scaffold in openhands_template.OPENHANDS_SCAFFOLDS
        and self.multi_tool_calls
        and not self.use_fn_calling
    ):
      _, actions = parse_codeact_response_all(response)
      calls = [
          a.to_xml_string() for a in actions if getattr(a, "function_name", "")
      ]
      if calls and calls != [result.action]:
        if len(calls) > 1:
          logging.info("CodeActAgent: turn has %d tool calls.", len(calls))
        cur_step.action = "\n".join(calls)
        return agent_types.Action(action=calls if len(calls) > 1 else calls[0])
    return result


@registry.register_agent(CODEACT_AGENT_NAME)
class CodeActAgent(DeepSWEAgent):
  """Registry alias for the OpenHands CodeActAgent."""

  name = CODEACT_AGENT_NAME

  def __init__(
      self,
      system_prompt: str | None = None,
      use_fn_calling: bool = False,
      format_model_response: bool = False,
      scaffold: str = "openhands",
  ):
    super().__init__(
        system_prompt=system_prompt,
        use_fn_calling=use_fn_calling,
        format_model_response=format_model_response,
        scaffold=scaffold,
    )

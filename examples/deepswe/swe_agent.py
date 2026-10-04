"""DeepSWE Agent."""

import json
import re
from typing import Any
from typing import Optional, Union  # Added Union for pytype compatibility

from absl import logging

try:
  from examples.deepswe import template
except ImportError:
  import template  # pytype: disable=import-error

OPENHANDS_SCAFFOLDS = template.OPENHANDS_SCAFFOLDS
OPENHANDS_SYSTEM_PROMPT = template.OPENHANDS_SYSTEM_PROMPT
OPENHANDS_USER_PROMPT = template.OPENHANDS_USER_PROMPT
OPENHANDS_FAKE_USER_RESPONSE = template.OPENHANDS_FAKE_USER_RESPONSE
OPENHANDS_TOOLS = template.OPENHANDS_TOOLS
SWE_SYSTEM_PROMPT = template.SWE_SYSTEM_PROMPT
SWE_SYSTEM_PROMPT_FN_CALL = template.SWE_SYSTEM_PROMPT_FN_CALL
SWE_USER_PROMPT = template.SWE_USER_PROMPT
SWE_USER_PROMPT_FN_CALL = template.SWE_USER_PROMPT_FN_CALL
SWEAGENT_SYSTEM_PROMPT = template.SWEAGENT_SYSTEM_PROMPT
SWEAGENT_USER_PROMPT = template.SWEAGENT_USER_PROMPT
get_system_prompt = template.get_system_prompt
get_user_prompt_template = template.get_user_prompt_template
format_openhands_user_prompt = template.format_openhands_user_prompt



from tunix.rl.agentic.agents.agent_types import Action
from tunix.rl.agentic.agents.agent_types import Step
from tunix.rl.agentic.agents.agent_types import Trajectory
from tunix.rl.agentic.agents.base_agent import ConversationAgentBase


try:
  from r2egym.agenthub.action import Action as SWEAction  # pytype: disable=import-error
except ImportError:

  class SWEAction:  # type: ignore[no-redef]
    """Fallback Action representation when r2egym is not installed."""

    def __init__(
        self,
        function_name: str = "",
        parameters: Optional[dict[str, Any]] = None,
    ):
      self.function_name = function_name
      self.parameters = parameters if parameters is not None else {}

    @classmethod
    def from_string(cls, action_str: str) -> "SWEAction":
      if not action_str or not action_str.strip():
        return cls("", {})
      fn_match = re.search(r"<function\s*=\s*([^>]+)>", action_str)
      if not fn_match:
        return cls("", {})
      fn_name = fn_match.group(1).strip()
      params = {}
      for k, v in re.findall(
          r"<parameter\s*=\s*([^>]+)>(.*?)</parameter>",
          action_str,
          flags=re.DOTALL,
      ):
        params[k.strip()] = v.strip()
      return cls(fn_name, params)

    def to_xml_string(self) -> str:
      if not self.function_name:
        return ""
      parts = [f"<function={self.function_name}>"]
      for k, v in self.parameters.items():
        parts.append(f"<parameter={k}>{v}</parameter>")
      parts.append("</function>")
      return "\n".join(parts)

    def __str__(self) -> str:
      return self.to_xml_string()


TOKEN_WARNING_THRESHOLD = 28000


def parse_oai_response(response: Any):
  thought = response.choices[0].message.content
  if not thought:
    thought = ""
  try:
    function_name = response.choices[0].message.tool_calls[0].function.name
    parameters = json.loads(
        response.choices[0].message.tool_calls[0].function.arguments
    )
    action = SWEAction(function_name, parameters)
  except Exception:
    action = SWEAction(function_name="", parameters={})
  return thought, action


def parse_xml_response(response_text: str) -> tuple[str, Any]:
  """Extracts:

  - thought: everything before the first <function=...> block
  - action: the entire first <function=...></function> block
  Returns (thought, action).
  """
  # Regex to match (non-greedily) from `<function=` up to the first `</function>`
  pattern = re.compile(r"(?s)(<function=.*?</function>)")
  match = pattern.search(response_text)

  if match:
    action = match.group(1)  # The entire <function=...></function> block
    thought = response_text[: match.start()]  # Everything before the block
  else:
    # If no match, treat entire text as "thought"
    thought = response_text
    action = ""

  # Strip leading/trailing whitespace
  thought = thought.strip()
  action = action.strip()

  # convert action to Action object
  action = SWEAction.from_string(action)

  return thought, action


def parse_openhands_xml_action(action_str: str) -> SWEAction:
  """Parses an XML function call while preserving multiline code indentation.

  Matches Qwen3XMLToolParser behavior by stripping only a single leading and
  single trailing newline from parameter values rather than calling .strip(),
  which would strip leading indentation from str_replace_editor's old_str /
  new_str parameters.
  """
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

  return SWEAction(function_name, params)


def parse_codeact_response(response_text: str) -> tuple[str, Any]:
  """Parses a model response in CodeAct / OpenHands format.

  Supports:
  1. Qwen3/Qwen3.5 XML tool calls: <tool_call><function=...>...</function></tool_call>
  2. Bare XML function blocks: <function=...></function>
  3. JSON tool calls: <tool_call>{...}</tool_call> or ```json ... ```
  4. Markdown code blocks: ```(bash|sh|shell|python|py|ipython) ... ```
  5. Task completion indicators: COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT

  Returns:
    (thought, action): Tuple of reasoning string and SWEAction instance.
  """
  xml_pattern = re.compile(
      r"(?s)(<function\s*=\s*([^>]+)>.*?(?:</function>|$))"
  )
  tc_pattern = re.compile(
      r"(?s)<tool_call>\s*(.*?)\s*(?:</tool_call>|$)"
  )
  cb_pattern = re.compile(
      r"(?s)```(bash|sh|shell|python|py|ipython)\s*\n(.*?)(?:```|$)"
  )
  json_cb_pattern = re.compile(
      r"(?s)```json\s*\n(.*?)(?:```|$)"
  )

  def _collect_candidates(text_slice: str, offset: int):
    found = []
    tc_matches = list(tc_pattern.finditer(text_slice))
    for tc_match in tc_matches:
      found.append((0, offset + tc_match.start(), "tool_call", tc_match))

    for xml_match in xml_pattern.finditer(text_slice):
      # Only add bare xml candidate if it is not already inside a tc_match.
      if not any(
          tc_m.start() <= xml_match.start() <= tc_m.end() for tc_m in tc_matches
      ):
        found.append((0, offset + xml_match.start(), "xml", xml_match))

    json_match = json_cb_pattern.search(text_slice)
    if json_match:
      try:
        parsed_json = json.loads(json_match.group(1).strip())
        if isinstance(parsed_json, dict) and (
            "name" in parsed_json or "function" in parsed_json
        ):
          found.append((
              0,
              offset + json_match.start(),
              "json_block",
              (json_match, parsed_json),
          ))
      except Exception:
        pass

    cb_match = cb_pattern.search(text_slice)
    if cb_match:
      found.append((1, offset + cb_match.start(), "code_block", cb_match))
    return found

  def _parse_candidate_action(match_type: str, payload: Any) -> SWEAction:
    if match_type == "xml":
      xml_str = payload.group(1).strip()
      if not xml_str.endswith("</function>"):
        xml_str += "\n</function>"
      return parse_openhands_xml_action(xml_str)

    if match_type in ("tool_call", "json_block"):
      if match_type == "tool_call":
        raw_payload = payload.group(1).strip()
        xml_in_tc = xml_pattern.search(raw_payload)
        if xml_in_tc:
          xml_str = xml_in_tc.group(1).strip()
          if not xml_str.endswith("</function>"):
            xml_str += "\n</function>"
          return parse_openhands_xml_action(xml_str)
        try:
          data = json.loads(raw_payload)
        except Exception:
          data = {}
      else:
        _, data = payload

      if isinstance(data, list) and data:
        data = data[0]
      if isinstance(data, dict):
        if "function" in data and isinstance(data["function"], dict):
          fn_name = data["function"].get("name", "")
          args = data["function"].get("arguments", {})
        else:
          fn_name = data.get("name", "")
          args = data.get("arguments", data.get("parameters", {}))
        if isinstance(args, str):
          try:
            args = json.loads(args)
          except Exception:
            args = {"command": args}
        if not isinstance(args, dict):
          args = {"command": str(args)}
        normalized_args = {}
        for k, v in args.items():
          if isinstance(v, (dict, list)):
            normalized_args[str(k)] = json.dumps(v, ensure_ascii=False)
          else:
            normalized_args[str(k)] = str(v)
        return SWEAction(fn_name, normalized_args)
      return SWEAction(function_name="", parameters={})

    if match_type == "code_block":
      lang = payload.group(1).lower()
      code = payload.group(2).strip()
      if lang in ("bash", "sh", "shell"):
        if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in code:
          return SWEAction("submit", {})
        return SWEAction("execute_bash", {"command": code})
      return SWEAction("execute_ipython_cell", {"code": code})

    return SWEAction(function_name="", parameters={})

  # Prefer tool calls emitted after </think> so illustrative code fences or
  # snippets inside reasoning blocks do not shadow the actual tool invocation.
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
    # Prefer structured tool invocations (priority 0) over generic markdown
    # code fences (priority 1), then earliest position in the response.
    candidates.sort(key=lambda x: (x[0], x[1]))
    first_priority, first_match_start, first_type, first_payload = candidates[0]
    thought = response_text[:first_match_start].strip()
    action = _parse_candidate_action(first_type, first_payload)

    # If the model emitted a bookkeeping tool call (think or task_tracker)
    # followed by an actionable tool call in the same turn, execute the
    # actionable tool call rather than dropping it.
    if first_priority == 0 and action.function_name in ("think", "task_tracker"):
      for cand_prio, _, cand_type, cand_payload in candidates[1:]:
        if cand_prio != 0:
          break
        next_action = _parse_candidate_action(cand_type, cand_payload)
        if next_action.function_name and next_action.function_name not in (
            "think",
            "task_tracker",
        ):
          action = next_action
          break

    return thought, action

  # Fallback: check for completion trigger in plain text
  if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in response_text:
    thought = response_text.replace(
        "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", ""
    ).strip()
    action = SWEAction("submit", {})
    return thought, action

  thought = response_text.strip()
  action = SWEAction(function_name="", parameters={})
  return thought, action


_LOGGED_AGENT_CONFIGS: set[tuple[str, str, str, bool]] = set()


class SWEAgent(ConversationAgentBase):

  name = "swe_agent"

  def __init__(
      self,
      system_prompt: Optional[str] = None,
      use_fn_calling: bool = False,
      format_model_response: bool = False,
      scaffold: str = "r2egym",
  ):
    self.use_fn_calling = use_fn_calling
    self.format_model_response = format_model_response
    assert scaffold in [
        "r2egym",
        "sweagent",
        "openhands",
    ], (
        f"Invalid scaffold: {scaffold}, must be one of ['r2egym', 'sweagent',"
        " 'openhands']"
    )
    self.scaffold = scaffold
    if system_prompt is None:
      system_prompt = get_system_prompt(
          scaffold=scaffold, use_fn_calling=use_fn_calling
      )
    self.user_prompt_template = get_user_prompt_template(
        scaffold=scaffold, use_fn_calling=use_fn_calling
    )

    super().__init__(system_prompt)
    agent_key = (
        self.__class__.__name__,
        self.name,
        scaffold,
        use_fn_calling,
    )
    if agent_key not in _LOGGED_AGENT_CONFIGS:
      _LOGGED_AGENT_CONFIGS.add(agent_key)
      logging.info(
          "Initialized DeepSWE agent: class=%s, name=%s, scaffold=%s,"
          " use_fn_calling=%s",
          *agent_key,
      )

  def _format_initial_observation(
      self, observation: str, info: dict[str, Any]
  ) -> str:
    del info  # Unused in default implementation.
    return self.user_prompt_template.format(problem_statement=observation)

  def update_from_env(
      self,
      observation: Any,
      reward: float,
      done: bool,
      info: Optional[dict[str, Any]] = None,
      **kwargs,
  ) -> None:
    observation = str(observation)
    if info is None:
      info = {}
    # If it's the first step in environment, let's apply user prompt template
    if len(self._trajectory.steps) == 0:
      observation = self._format_initial_observation(observation, info)

    max_steps = info.get("max_steps", None)
    if max_steps:
      remaining_steps = max_steps - self.step - 1
      if remaining_steps > 0:
        observation += f"\nSteps Remaining: {remaining_steps}"
      else:
        observation += (
            "\nYou have reached the maximum number of steps. Please submit your"
            " answer NOW."
        )
    cur_tokens = info.get("cur_tokens", None)
    if cur_tokens is not None and cur_tokens >= TOKEN_WARNING_THRESHOLD:
      if self.scaffold in OPENHANDS_SCAFFOLDS:
        observation += (
            "\nYou are running out of tokens. Stop exploring now. Do not call"
            " file_editor, str_replace_editor, or execute_bash again. You must"
            " immediately submit using the finish tool. Output exactly this XML"
            " and nothing else:\n"
            "<tool_call>\n"
            "<function=finish>\n"
            "<parameter=message>\n"
            "Task completed.\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
        )
      else:
        observation += (
            "\nYou are running out of tokens. Stop exploring now. Do not call"
            " file_editor, str_replace_editor, search, execute_bash, or any"
            " view command again. You must immediately submit using the final"
            " tool. Output exactly this XML and nothing else:\n"
            "<function=finish>\n"
            "<parameter=command>submit</parameter>\n"
            "<parameter=result>FINAL_RESULT</parameter>\n"
            "</function>\n"
            "Do not include reasoning text. Do not include a result parameter."
            " Do not summarize the fix. If the submit tool is available instead"
            " of finish, output exactly this XML and nothing else:\n"
            "<function=submit>\n"
            "</function>\n"
        )

    super().update_from_env(observation, reward, done, info)
    self.cur_step = Step(observation=observation)

  def _observation_to_messages(
      self, observation: Any, reward: float, done: bool, info: dict[str, Any]
  ) -> None:

    self._messages.append({"role": "user", "content": str(observation)})

  def _parse_model_response(
      self, response: str | Any
  ) -> tuple[str, SWEAction]:
    if self.use_fn_calling:
      return parse_oai_response(response)
    if self.scaffold in OPENHANDS_SCAFFOLDS:
      return parse_codeact_response(response)
    return parse_xml_response(response)

  def update_from_model(self, response: str, **kwargs):
    """Updates the agent's internal state after an environment step.

    This function is called during environment interaction to incorporate the
    latest action's outcome into the agent's learning process.

    Args:
        response (str): The response from the model.

    Returns:
        Action: The action produced by the agent.
    """
    self._trajectory.steps.append(self.cur_step)
    thought, action = self._parse_model_response(response)
    action_str = (
        action.to_xml_string() if getattr(action, "function_name", True) else ""
    )

    # Update Trajectory
    cur_step = self._trajectory.steps[-1]
    cur_step.thought = thought
    cur_step.action = action_str
    cur_step.model_response = response

    # Update Chat Completions
    if self.format_model_response:
      self._messages.append(
          {"role": "assistant", "content": f"{thought}\n\n{action_str}"}
      )
    else:
      self._messages.append({"role": "assistant", "content": response})
    self.step += 1
    return Action(action=cur_step.action)


class CodeActAgent(SWEAgent):
  """CodeActAgent for OpenHands matching nv-OpenHands@0d766ad0.

  Uses the OpenHands 5-tool action space (`execute_bash`, `think`, `finish`,
  `task_tracker`, `str_replace_editor`), Qwen3/Qwen3.5 XML native tool-calling
  format, `swe_default.j2` user prompt, and `role="tool"` (`<tool_response>`)
  observations.
  """

  name = "codeact_agent"

  def __init__(
      self,
      system_prompt: Optional[str] = None,
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

  def _format_initial_observation(
      self, observation: str, info: dict[str, Any]
  ) -> str:
    if self.user_prompt_template == OPENHANDS_USER_PROMPT:
      return format_openhands_user_prompt(
          problem_statement=observation,
          workspace_path=str(info.get("workspace_path") or "/testbed"),
          repo_language=str(info.get("repo_language") or "python"),
          base_commit=str(info.get("base_commit") or ""),
      )
    return super()._format_initial_observation(observation, info)

  def update_from_env(
      self,
      observation: Any,
      reward: float,
      done: bool,
      info: Optional[dict[str, Any]] = None,
      **kwargs,
  ) -> None:
    if info is None:
      info = {}
    if (
        len(self._trajectory.steps) > 0
        and not observation
        and not self._trajectory.steps[-1].action
    ):
      observation = OPENHANDS_FAKE_USER_RESPONSE

    super().update_from_env(observation, reward, done, info, **kwargs)

  def _observation_to_messages(
      self, observation: Any, reward: float, done: bool, info: dict[str, Any]
  ) -> None:
    if len(self._trajectory.steps) == 0:
      self._messages.append({"role": "user", "content": str(observation)})
      return
    last_step = self._trajectory.steps[-1]
    if (info and info.get("is_fake_user_response")) or not last_step.action:
      self._messages.append({"role": "user", "content": str(observation)})
    else:
      self._messages.append({"role": "tool", "content": str(observation)})

  def _parse_model_response(
      self, response: str | Any
  ) -> tuple[str, SWEAction]:
    if self.use_fn_calling:
      return parse_oai_response(response)
    return parse_codeact_response(response)


__all__ = [
    "CodeActAgent",
    "SWEAgent",
    "parse_codeact_response",
    "parse_oai_response",
    "parse_openhands_xml_action",
    "parse_xml_response",
]


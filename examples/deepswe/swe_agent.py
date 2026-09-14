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

OPENHANDS_SYSTEM_PROMPT = template.OPENHANDS_SYSTEM_PROMPT
SWE_SYSTEM_PROMPT = template.SWE_SYSTEM_PROMPT
SWE_SYSTEM_PROMPT_FN_CALL = template.SWE_SYSTEM_PROMPT_FN_CALL
SWE_USER_PROMPT = template.SWE_USER_PROMPT
SWE_USER_PROMPT_FN_CALL = template.SWE_USER_PROMPT_FN_CALL
SWEAGENT_SYSTEM_PROMPT = template.SWEAGENT_SYSTEM_PROMPT
SWEAGENT_USER_PROMPT = template.SWEAGENT_USER_PROMPT
get_system_prompt = template.get_system_prompt
get_user_prompt_template = template.get_user_prompt_template



from tunix.rl.agentic.agents.agent_types import Action
from tunix.rl.agentic.agents.agent_types import Step
from tunix.rl.agentic.agents.agent_types import Trajectory
from tunix.rl.agentic.agents.base_agent import ConversationAgentBase


try:
  from r2egym.agenthub.action import Action as SWEAction  # pytype: disable=import-error
except ImportError:
  logging.error(
      "Failed to load SWEAction. Please ensure 'r2egym' is installed properly."
  )
  raise  # This halts execution and preserves the original traceback

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


def parse_codeact_response(response_text: str) -> tuple[str, Any]:
  """Parses a model response in CodeAct / OpenHands format.

  Supports:
  1. XML function blocks: <function=...></function>
  2. Tool calls: <tool_call>...</tool_call> or ```json ... ``` with tool call schema
  3. Markdown code blocks: ```(bash|sh|shell|python|py|ipython) ... ```
  4. Task completion indicators: COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT

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

  # Collect all potential matches with their start positions
  candidates = []

  xml_match = xml_pattern.search(response_text)
  if xml_match:
    candidates.append((xml_match.start(), "xml", xml_match))

  tc_match = tc_pattern.search(response_text)
  if tc_match:
    candidates.append((tc_match.start(), "tool_call", tc_match))

  cb_match = cb_pattern.search(response_text)
  if cb_match:
    candidates.append((cb_match.start(), "code_block", cb_match))

  json_match = json_cb_pattern.search(response_text)
  if json_match:
    try:
      parsed_json = json.loads(json_match.group(1).strip())
      if isinstance(parsed_json, dict) and (
          "name" in parsed_json or "function" in parsed_json
      ):
        candidates.append(
            (json_match.start(), "json_block", (json_match, parsed_json))
        )
    except Exception:
      pass

  if candidates:
    # Pick the match that appears earliest in the response text
    candidates.sort(key=lambda x: x[0])
    match_type = candidates[0][1]

    if match_type == "xml":
      m = candidates[0][2]
      thought = response_text[: m.start()].strip()
      action = SWEAction.from_string(m.group(1).strip())
      return thought, action

    elif match_type in ("tool_call", "json_block"):
      if match_type == "tool_call":
        m = candidates[0][2]
        thought = response_text[: m.start()].strip()
        raw_payload = m.group(1).strip()
        try:
          data = json.loads(raw_payload)
        except Exception:
          data = {}
      else:
        m, data = candidates[0][2]
        thought = response_text[: m.start()].strip()

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
        action = SWEAction(fn_name, {str(k): str(v) for k, v in args.items()})
        return thought, action

    elif match_type == "code_block":
      m = candidates[0][2]
      thought = response_text[: m.start()].strip()
      lang = m.group(1).lower()
      code = m.group(2).strip()
      if lang in ("bash", "sh", "shell"):
        if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in code:
          action = SWEAction("submit", {})
        else:
          action = SWEAction("execute_bash", {"command": code})
      else:  # python, py, ipython
        action = SWEAction("execute_ipython_cell", {"code": code})
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


class SWEAgent(ConversationAgentBase):

  def __init__(
      self,
      system_prompt: Optional[str] = None,
      use_fn_calling: bool = False,
      format_model_response: bool = False,
      scaffold: str = "openhands",
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
      observation = self.user_prompt_template.format(
          problem_statement=observation
      )

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
      if self.scaffold == "openhands":
        observation += (
            "\nYou are running out of tokens. Stop exploring now. Do not call"
            " file_editor, str_replace_editor, or execute_bash again. You must"
            " immediately submit using the submit tool. Output exactly this XML"
            " and nothing else:\n"
            "<function=submit>\n"
            "</function>\n"
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
    if self.use_fn_calling:
      thought, action = parse_oai_response(response)
    elif self.scaffold == "openhands":
      thought, action = parse_codeact_response(response)
    else:
      thought, action = parse_xml_response(response)
    action_str = action.to_xml_string() if action.function_name else ""

    # Update Trajectory
    cur_step = self._trajectory.steps[-1]
    cur_step.thought = thought
    cur_step.action = Action(action=action_str)
    cur_step.model_response = response

    # Update Chat Completions
    if self.format_model_response:
      self._messages.append(
          {"role": "assistant", "content": f"{thought}\n\n{action_str}"}
      )
    else:
      self._messages.append({"role": "assistant", "content": response})
    self.step += 1
    return cur_step.action


class CodeActAgent(SWEAgent):
  """CodeActAgent for OpenHands.

  Executes code (Bash and Python/IPython) directly as its primary action space,
  supporting markdown code blocks, JSON tool calls, and XML function calls.
  """

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


__all__ = [
    "CodeActAgent",
    "SWEAgent",
    "parse_codeact_response",
    "parse_oai_response",
    "parse_xml_response",
]


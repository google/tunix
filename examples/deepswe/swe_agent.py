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




class SWEAgent(ConversationAgentBase):

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
    ], f"Invalid scaffold: {scaffold}, must be one of ['r2egym', 'sweagent', 'openhands']"
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
    latest action's
    outcome into the agent's learning process.

    Args:
        response (str): The response from the model.

    Returns:
        None
    """
    self._trajectory.steps.append(self.cur_step)
    if self.use_fn_calling:
      thought, action = parse_oai_response(response)
    else:
      thought, action = parse_xml_response(response)
    action_str = action.to_xml_string()

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

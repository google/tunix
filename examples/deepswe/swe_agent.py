import json
import logging
import re
import traceback
try:
    from r2egym.agenthub.action import Action as SWEAction
except ImportError:
    SWEAction = None

from tunix.rl.agentic.agents.agent_types import Action, Step, Trajectory
from tunix.rl.agentic.agents.base_agent import ConversationAgentBase
from rllm.agents.system_prompts import SWE_SYSTEM_PROMPT, SWE_SYSTEM_PROMPT_FN_CALL, SWE_USER_PROMPT, SWE_USER_PROMPT_FN_CALL, SWEAGENT_SYSTEM_PROMPT, SWEAGENT_USER_PROMPT

TOKEN_WARNING_THRESHOLD = 28000


def parse_oai_response(response):
    logging.info("Parsing OpenAI function-calling response.")
    thought = response.choices[0].message.content
    if not thought:
        thought = ""
    try:
        function_name = response.choices[0].message.tool_calls[0].function.name
        parameters = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        action = SWEAction(function_name, parameters)
    except Exception:
        action = SWEAction(function_name="", parameters={})
    return thought, action


def parse_xml_response(response_text: str) -> tuple[str, SWEAction]:
    """
    Extracts:
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


logger = logging.getLogger(__name__)


class SWEAgent(ConversationAgentBase):
    def __init__(self,  system_prompt: str = None, use_fn_calling: bool = False, format_model_response: bool = False, scaffold: str = "r2egym"):
        self.use_fn_calling = use_fn_calling
        self.format_model_response = format_model_response
        assert scaffold in ["r2egym", "sweagent"], f"Invalid scaffold: {scaffold}, must be one of ['r2egym', 'sweagent']"
        if system_prompt is None:
            system_prompt = SWE_SYSTEM_PROMPT_FN_CALL if use_fn_calling else SWE_SYSTEM_PROMPT
            if scaffold == "sweagent":
                system_prompt = SWEAGENT_SYSTEM_PROMPT
        self.user_prompt_template = SWE_USER_PROMPT_FN_CALL if use_fn_calling else SWE_USER_PROMPT
        if scaffold == "sweagent":
            self.user_prompt_template = SWEAGENT_USER_PROMPT
        super().__init__(system_prompt)

    def update_from_env(self, observation, reward, done, info):
        observation = str(observation)
        # If it's the first step in environment, let's apply user prompt template
        if len(self._trajectory.steps) == 0:
            observation = self.user_prompt_template.format(problem_statement=observation)

        max_steps = info.get("max_steps", None)
        if max_steps:
            remaining_steps = max_steps - self.step - 1
            if remaining_steps > 0:
                observation += f"\nSteps Remaining: {remaining_steps}"
            else:
                observation += "\nYou have reached the maximum number of steps. Please submit your answer NOW."
        cur_tokens = info.get("cur_tokens", None)
        if cur_tokens is not None and cur_tokens >= TOKEN_WARNING_THRESHOLD:
            observation += "\nYou are running out of tokens. Please submit your answer NOW."

        self._messages.append({"role": "user", "content": observation})
        self.cur_step = Step(observation=observation)

    def update_from_model(self, response: str, **kwargs):
        """
        Updates the agent's internal state after an environment step.

        This function is called during environment interaction to incorporate the latest action's
        outcome into the agent's learning process.

        Args:
            response (str): The response from the model.

        Returns:
            None
        """
        self._trajectory.steps.append(self.cur_step)
        logging.info("update from model response()")
        logging.info("use_fn_calling: %s", self.use_fn_calling)
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
            self._messages.append({"role": "assistant", "content": f"{thought}\n\n{action_str}"})
        else:
            self._messages.append({"role": "assistant", "content": response})
        self.step += 1
        return Action(action=cur_step.action)


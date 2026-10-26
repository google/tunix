"""Failure-aware action guard for the DeepSWE agent.

A runtime policy layer that sits between agent.update_from_model() and env.step().
It blocks repeated failed actions, enforces failure-specific recovery transitions,
pre-checks high-risk actions, and requires post-edit verification.

Usage:
    guard = ActionGuard()
    # ... in the step loop, after agent.update_from_model():
    verdict = guard.evaluate(action_str)
    if verdict.blocked:
        agent.update_from_env(verdict.message, reward=0.0, done=False, info={})
    else:
        obs, rew, done, info = env.step(action_str)
        guard.record_outcome(action_str, str(obs))
        agent.update_from_env(obs, rew, done, info)
"""

import dataclasses
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class GuardConfig:
  """Configuration for the action guard."""

  max_consecutive_edit_failures: int = 3
  """Block all edits after this many consecutive edit failures."""

  require_view_after_not_found: bool = True
  """After a 'not found' error, require view/search before retrying edit."""

  require_view_after_non_unique: bool = True
  """After a 'non unique' error, require view/search before retrying edit."""

  require_view_before_finish: bool = False
  """Block finish/submit until all edited files have been viewed."""

  enabled: bool = True
  """Master switch. Set False to disable the guard entirely."""


@dataclasses.dataclass
class GuardVerdict:
  """Result of evaluating an action through the guard."""

  blocked: bool
  """True means do NOT call env.step(); inject *message* instead."""

  message: str = ""
  """Synthetic observation to inject when blocked."""

  reason: str = ""
  """Internal reason code for logging (e.g. 'repeated_failure:3')."""


@dataclasses.dataclass
class ActionRecord:
  """A record of a past action and its outcome."""

  action_str: str
  """The full action XML string."""

  observation: str
  """The observation returned by env.step() (truncated)."""

  failure_type: Optional[str]
  """Classified failure type, or None if success."""

  step_index: int
  """Which step this occurred at."""


# ---------------------------------------------------------------------------
# ActionGuard
# ---------------------------------------------------------------------------

class ActionGuard:
  """Failure-aware action guard.

  Sits between agent.update_from_model() and env.step(). Evaluates each
  action against a set of rules and either allows execution or blocks it
  with a synthetic observation message.
  """

  def __init__(self, config: Optional[GuardConfig] = None):
    self.config = config or GuardConfig()
    self._history: List[ActionRecord] = []
    self._last_failed_action: Optional[str] = None
    self._consecutive_edit_failures: int = 0
    self._last_failure_type: Optional[str] = None
    self._last_failed_path: Optional[str] = None
    self._files_edited_since_last_view: set = set()
    self._files_successfully_edited: set = set()
    self._step_index: int = 0

  # ---- lifecycle ----

  def reset(self) -> None:
    """Reset all guard state. Call at the start of each episode."""
    self._history.clear()
    self._last_failed_action = None
    self._consecutive_edit_failures = 0
    self._last_failure_type = None
    self._last_failed_path = None
    self._files_edited_since_last_view.clear()
    self._files_successfully_edited.clear()
    self._step_index = 0

  # ---- public API ----

  def evaluate(self, action_str: str) -> GuardVerdict:
    """Evaluate whether *action_str* should be executed or blocked.

    Args:
      action_str: The XML action string from agent.update_from_model().action.

    Returns:
      GuardVerdict with blocked=False to proceed, blocked=True to skip.
    """
    if not self.config.enabled:
      return GuardVerdict(blocked=False)

    if not action_str:
      return GuardVerdict(blocked=False)

    func_name, params = self._parse_action(action_str)

    # Rule 1: block repeated identical failures
    verdict = self._check_repeated_failure(action_str)
    if verdict:
      logger.info("Guard rule 1 fired: %s", verdict.reason)
      return verdict

    # Rule 2: enforce failure-specific transitions
    verdict = self._check_failure_transition(func_name, params)
    if verdict:
      logger.info("Guard rule 2 fired: %s", verdict.reason)
      return verdict

    # Rule 3: block edits after too many consecutive failures
    verdict = self._check_consecutive_edit_failures(func_name, params)
    if verdict:
      logger.info("Guard rule 3 fired: %s", verdict.reason)
      return verdict

    # Rule 4: pre-check finish/submit
    verdict = self._check_finish_preconditions(func_name, params)
    if verdict:
      logger.info("Guard rule 4 fired: %s", verdict.reason)
      return verdict

    return GuardVerdict(blocked=False)

  def record_outcome(self, action_str: str, observation: str) -> None:
    """Record the result of an executed action and update guard state.

    Must be called after every successful env.step().

    Args:
      action_str: The action that was executed.
      observation: The observation string returned by the environment.
    """
    func_name, params = self._parse_action(action_str)
    failure_type = self._classify_failure(observation)
    path = params.get("path", "")

    is_edit = self._is_edit_action(func_name, params)
    is_view = self._is_view_action(func_name, params)
    is_search = func_name == "search"
    is_bash_grep = (
        func_name == "execute_bash"
        and "grep" in params.get("cmd", params.get("command", ""))
    )
    is_recovery = is_view or is_search or is_bash_grep

    # Append to history
    self._history.append(
        ActionRecord(
            action_str=action_str,
            observation=observation[:500],
            failure_type=failure_type,
            step_index=self._step_index,
        )
    )

    if failure_type:
      # ---- failure ----
      self._last_failed_action = action_str
      self._last_failure_type = failure_type
      self._last_failed_path = path
      if is_edit:
        self._consecutive_edit_failures += 1
    else:
      # ---- success ----
      self._last_failed_action = None
      self._last_failure_type = None
      self._last_failed_path = None
      if is_edit:
        self._consecutive_edit_failures = 0
        self._files_successfully_edited.add(path)
        self._files_edited_since_last_view.add(path)

    # view clears "unverified" status for that file
    if is_view and path:
      self._files_edited_since_last_view.discard(path)

    # recovery actions (view/search/grep) clear transition constraints
    if is_recovery:
      self._last_failed_action = None
      self._last_failure_type = None
      self._last_failed_path = None
      # Also reset consecutive edit failure count so agent can try editing again
      self._consecutive_edit_failures = 0

    self._step_index += 1

  # ---- rules ----

  def _check_repeated_failure(self, action_str: str) -> Optional[GuardVerdict]:
    """Rule 1: block if this is the exact same action that just failed."""
    if self._last_failed_action is not None and action_str == self._last_failed_action:
      return GuardVerdict(
          blocked=True,
          reason="repeated_failure",
          message=(
              f"[ACTION GUARD] This exact action just failed. "
              f"Repeating it will produce the same result.\n"
              f"Please try a DIFFERENT approach:\n"
              f"- View the file around the relevant lines with a specific "
              f"view_range\n"
              f"- Use search or grep to find the correct string\n"
              f"- Include more context in old_str to make it unique\n"
              f"- Try a completely different editing strategy"
          ),
      )
    return None

  def _check_failure_transition(
      self, func_name: str, params: Dict[str, str]
  ) -> Optional[GuardVerdict]:
    """Rule 2: after certain failures, require recovery action first."""
    if self._last_failure_type is None:
      return None

    is_edit = self._is_edit_action(func_name, params)
    if not is_edit:
      return None

    path = params.get("path", "")
    same_path = path == self._last_failed_path if self._last_failed_path else True

    if not same_path:
      return None

    if (
        self._last_failure_type == "non_unique"
        and self.config.require_view_after_non_unique
    ):
      return GuardVerdict(
          blocked=True,
          reason="transition:non_unique_requires_view",
          message=(
              f"[ACTION GUARD] Your last str_replace failed because old_str "
              f"matched multiple locations in {self._last_failed_path}.\n"
              f"You MUST first view the file around the edit location to "
              f"gather more context, then include additional surrounding "
              f"lines in old_str to make it unique.\n"
              f"Suggested: file_editor view with a specific view_range on "
              f"{self._last_failed_path}."
          ),
      )

    if (
        self._last_failure_type == "not_found"
        and self.config.require_view_after_not_found
    ):
      return GuardVerdict(
          blocked=True,
          reason="transition:not_found_requires_view",
          message=(
              f"[ACTION GUARD] Your last str_replace failed because old_str "
              f"was not found in {self._last_failed_path}.\n"
              f"You MUST first view the file to see its current content. "
              f"The file may have changed from a previous edit, or the "
              f"old_str may have whitespace/indentation differences.\n"
              f"Suggested: file_editor view with a specific view_range on "
              f"{self._last_failed_path}."
          ),
      )

    if self._last_failure_type == "path_not_found":
      return GuardVerdict(
          blocked=True,
          reason="transition:path_not_found_requires_search",
          message=(
              f"[ACTION GUARD] The path '{self._last_failed_path}' does not "
              f"exist.\n"
              f"Verify the correct file path using:\n"
              f"- file_editor view on the parent directory\n"
              f"- search to find the correct file name\n"
              f"- execute_bash with find or ls"
          ),
      )

    return None

  def _check_consecutive_edit_failures(
      self, func_name: str, params: Dict[str, str]
  ) -> Optional[GuardVerdict]:
    """Rule 3: block edits after too many consecutive edit failures."""
    is_edit = self._is_edit_action(func_name, params)
    if not is_edit:
      return None

    if self._consecutive_edit_failures >= self.config.max_consecutive_edit_failures:
      return GuardVerdict(
          blocked=True,
          reason=f"consecutive_edit_failures:{self._consecutive_edit_failures}",
          message=(
              f"[ACTION GUARD] You have had "
              f"{self._consecutive_edit_failures} consecutive edit failures. "
              f"Stop trying to edit and take a step back.\n"
              f"Please:\n"
              f"1. View the file(s) you are trying to edit to see their "
              f"current state\n"
              f"2. Re-read the error messages from previous attempts\n"
              f"3. Consider using undo_edit if edits left the file in a bad "
              f"state\n"
              f"4. Try a completely different approach to the fix"
          ),
      )
    return None

  def _check_finish_preconditions(
      self, func_name: str, params: Dict[str, str]
  ) -> Optional[GuardVerdict]:
    """Rule 4: block finish/submit if edited files haven't been verified."""
    if func_name not in ("finish", "submit"):
      return None

    if not self.config.require_view_before_finish:
      return None

    unverified = self._files_edited_since_last_view
    if unverified:
      file_list = ", ".join(sorted(unverified))
      return GuardVerdict(
          blocked=True,
          reason="finish:unverified_edits",
          message=(
              f"[ACTION GUARD] You are trying to submit, but these edited "
              f"files have not been verified since your last edit:\n"
              f"  {file_list}\n"
              f"Please view or test them before submitting."
          ),
      )
    return None

  # ---- helpers ----

  @staticmethod
  def _parse_action(action_str: str) -> Tuple[str, Dict[str, str]]:
    """Extract function_name and parameters from an XML action string."""
    if not action_str:
      return "", {}

    fn_match = re.search(r"<function\s*=\s*([^>]+)>", action_str)
    func_name = fn_match.group(1).strip() if fn_match else ""

    pattern = r"<parameter\s*=\s*([^>]+)>(.*?)</parameter>"
    param_matches = re.findall(pattern, action_str, flags=re.DOTALL)
    params = {k.strip(): v.strip() for k, v in param_matches}

    return func_name, params

  @staticmethod
  def _is_edit_action(func_name: str, params: Dict[str, str]) -> bool:
    """Check if this action is a file-editing action."""
    if func_name in ("file_editor", "str_replace_editor"):
      return params.get("command") in ("str_replace", "insert", "create")
    return False

  @staticmethod
  def _is_view_action(func_name: str, params: Dict[str, str]) -> bool:
    """Check if this action is a file-viewing action."""
    if func_name in ("file_editor", "str_replace_editor"):
      return params.get("command") == "view"
    return False

  @staticmethod
  def _classify_failure(observation: str) -> Optional[str]:
    """Classify an observation into a failure type, or None for success.

    Uses re.search (not re.match) because observations are prefixed with
    'Execution output of [tool_name]:\\n'.
    """
    if not observation:
      return None

    if re.search(r"Multiple occurrences of .+ found in", observation):
      return "non_unique"
    if re.search(r"No occurrences of .+ found in .+ for replacement", observation):
      return "not_found"
    if "Your proposed edit has introduced new syntax error(s)" in observation:
      return "syntax_error"
    if re.search(r"File already exists at: .+\. Cannot overwrite", observation):
      return "file_exists"
    if re.search(r"The path '.+' does not exist", observation):
      return "path_not_found"

    # Check for generic ERROR prefix (from EditorError / EditorResult)
    # But skip if there's also normal output (e.g. "ERROR:" in bash output)
    lines = observation.strip().split("\n")
    for line in lines:
      stripped = line.strip()
      if stripped.startswith("ERROR:") and len(lines) <= 5:
        return "generic_error"

    return None

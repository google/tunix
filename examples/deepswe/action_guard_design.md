# Failure-Aware Action Guard — Design Document

## 1. Problem Statement

When solving SWE-bench problems, the DeepSWE agent frequently exhibits the following failure patterns:

| Problem | Symptom |
|---|---|
| Repeated ineffective actions | The same `str_replace` has already failed, yet the model retries it verbatim |
| Ignoring tool feedback | The tool returns `Multiple occurrences` / `No occurrences`, but the model keeps trying in place |
| No state transition after failure | After non-unique the agent should view more context first; after not-found it should re-read source — but it just keeps editing |
| Premature completion | The model "thinks it fixed the bug" but the patch was never confirmed, and it tries to finish |

**Goal**: Insert a **runtime policy** layer (action guard) between the agent and the environment that enforces failure recovery strategies. No prompt changes, no model changes — only **runtime behavior** changes.

---

## 2. Architecture Overview

```
Model Response
      ↓
agent.update_from_model(response)
      ↓
    action
      ↓
┌─────────────────────────────────────┐
│      ActionGuard.evaluate()         │
│                                     │
│  Rule 1: Repeated failure blocking  │
│  Rule 2: Failure state transitions  │
│  Rule 3: Consecutive edit cap       │
│  Rule 4: Finish pre-check           │
└──────────┬──────────────────────────┘
           │
     ┌─────┴─────┐
     │           │
  blocked     allowed
     │           │
     │      env.step(action)
     │           │
     │      guard.record_outcome(action, obs)
     │           │
     ↓           ↓
  synthetic    real obs
    obs           │
     │           │
     └─────┬─────┘
           ↓
  agent.update_from_env(obs)
```

**Core principle**: When the guard blocks an action, it does NOT call `env.step()`. Instead, it injects a synthetic observation (prefixed with `[ACTION GUARD]`) through the existing `update_from_env` channel. The model naturally sees this feedback in its next turn.

---

## 3. File Structure

| File | Type | Description |
|---|---|---|
| `examples/deepswe/action_guard.py` | New ~300 lines | Core guard module: ActionGuard, GuardConfig, GuardVerdict |
| `examples/deepswe/guarded_swe_env.py` | New ~50 lines | `SWEEnv` wrapper that applies guard checks before env execution |
| `examples/deepswe/debug_eval_deepswe.py` | Modified ~15 lines | Guard integration in the debug loop |
| `examples/deepswe/eval_deepswe.py` | Modified ~5 lines | Switches between `SWEEnv` and `GuardedSWEEnv` via `ENABLE_GUARD` |

**Untouched files**: base_agent.py, agent_types.py, trajectory_collect_engine.py, rollout_orchestrator.py — all framework-level files remain unchanged.

---

## 4. Core Data Structures

### 4.1 GuardConfig

```python
@dataclasses.dataclass
class GuardConfig:
    max_consecutive_edit_failures: int = 3  # Consecutive edit failure cap
    require_view_after_not_found: bool = True   # After not_found, must view first
    require_view_after_non_unique: bool = True  # After non_unique, must view first
    require_view_before_finish: bool = False    # Whether to require verifying edits before finish
    enabled: bool = True                        # Master switch
```

### 4.2 GuardVerdict

```python
@dataclasses.dataclass
class GuardVerdict:
    blocked: bool           # True = do NOT call env.step
    message: str = ""       # Synthetic observation to inject when blocked
    reason: str = ""        # Reason code for logging (e.g. "repeated_failure:3")
```

### 4.3 ActionRecord

```python
@dataclasses.dataclass
class ActionRecord:
    action_str: str              # Full action XML string
    observation: str             # Observation returned by env (truncated to 500 chars)
    failure_type: Optional[str]  # Classified failure type, None = success
    step_index: int              # Step number
```

---

## 5. ActionGuard Class

### 5.1 State Variables

```python
class ActionGuard:
    _history: List[ActionRecord]           # Complete history of actions and outcomes
    _last_failed_action: Optional[str]     # The action string that just failed (for consecutive repeat detection)
    _consecutive_edit_failures: int         # Running count of consecutive edit failures
    _last_failure_type: Optional[str]       # Type of the most recent failure
    _last_failed_path: Optional[str]        # File path involved in the most recent failure
    _files_edited_since_last_view: set      # Files edited but not yet verified
    _files_successfully_edited: set         # All files successfully edited this episode
    _step_index: int                        # Current step counter
```

### 5.2 Core Methods

```python
def evaluate(self, action_str: str) -> GuardVerdict:
    """Decide whether an action should be executed. Checks all 4 rules in order."""

def record_outcome(self, action_str: str, observation: str) -> None:
    """Record the outcome after execution and update guard state. Must be called after every env.step()."""

def reset(self) -> None:
    """Reset all state at the start of each episode."""
```

### 5.3 evaluate() Flow

```python
def evaluate(self, action_str):
    if not self.config.enabled:
        return GuardVerdict(blocked=False)

    func_name, params = self._parse_action(action_str)

    # Rule 1: Exact same action repeated after failure
    verdict = self._check_repeated_failure(action_str)
    if verdict: return verdict

    # Rule 2: Must recover before retrying after certain failures
    verdict = self._check_failure_transition(func_name, params)
    if verdict: return verdict

    # Rule 3: Too many consecutive edit failures
    verdict = self._check_consecutive_edit_failures(func_name, params)
    if verdict: return verdict

    # Rule 4: Pre-conditions for finish/submit
    verdict = self._check_finish_preconditions(func_name, params)
    if verdict: return verdict

    return GuardVerdict(blocked=False)
```

---

## 6. Rules in Detail

### Rule 1: Repeated Failure Blocking (Consecutive)

**Trigger condition**: The action string is **exactly identical** to the one that **just failed** in the immediately preceding step. If the same action appears right after a failure, it is blocked immediately — no threshold, no accumulated count.

**No fingerprint abstraction** — only blocks when the command is literally the same. **Consecutive only** — if a different action (successful or not) intervenes, the constraint resets.

```python
def _check_repeated_failure(self, action_str: str) -> Optional[GuardVerdict]:
    if self._last_failed_action is not None and action_str == self._last_failed_action:
        return GuardVerdict(
            blocked=True,
            reason="repeated_failure",
            message=(
                f"[ACTION GUARD] This exact action just failed. "
                f"Repeating it will produce the same result.\n"
                f"Please try a DIFFERENT approach:\n"
                f"- View the file around the relevant lines with a specific view_range\n"
                f"- Use search or grep to find the correct string\n"
                f"- Include more context in old_str to make it unique\n"
                f"- Try a completely different editing strategy"
            ),
        )
    return None
```

**Updated in record_outcome**:

```python
if failure_type:
    self._last_failed_action = action_str   # Remember this action as the most recent failure
else:
    self._last_failed_action = None          # Success clears the constraint

# Recovery actions (view/search/grep) also clear it:
if is_recovery:
    self._last_failed_action = None
```

---

### Rule 2: Failure State Transitions

**Core idea**: Each failure type maps to a required next step — no retrying without observing first.

| Last Failure | Current Action | Decision |
|---|---|---|
| `non_unique` | edit on same file | **block** — must view larger context first |
| `not_found` | edit on same file | **block** — must refresh source view first |
| `path_not_found` | edit on same path | **block** — must verify path exists first |
| any failure | view / search / grep | **allow** and clear transition constraint |

```python
def _check_failure_transition(self, func_name, params) -> Optional[GuardVerdict]:
    if self._last_failure_type is None:
        return None

    is_edit = (func_name in ("file_editor", "str_replace_editor") and
               params.get("command") in ("str_replace", "insert", "create"))
    path = params.get("path", "")
    same_path = (path == self._last_failed_path)

    if self._last_failure_type == "non_unique" and is_edit and same_path:
        return GuardVerdict(
            blocked=True,
            reason="transition:non_unique_requires_view",
            message=(
                f"[ACTION GUARD] Your last str_replace failed because old_str matched "
                f"multiple locations in {self._last_failed_path}.\n"
                f"You MUST first view the file around the edit location to gather more "
                f"context, then include additional surrounding lines in old_str.\n"
                f"Suggested: file_editor view with a specific view_range."
            ),
        )

    if self._last_failure_type == "not_found" and is_edit and same_path:
        return GuardVerdict(
            blocked=True,
            reason="transition:not_found_requires_view",
            message=(
                f"[ACTION GUARD] Your last str_replace failed because old_str was not "
                f"found in {self._last_failed_path}.\n"
                f"You MUST first view the file to see its current content. The file may "
                f"have changed, or old_str may have whitespace/indentation differences.\n"
                f"Suggested: file_editor view with a specific view_range."
            ),
        )

    if self._last_failure_type == "path_not_found" and is_edit and same_path:
        return GuardVerdict(
            blocked=True,
            reason="transition:path_not_found_requires_search",
            message=(
                f"[ACTION GUARD] The path '{self._last_failed_path}' does not exist.\n"
                f"Verify the correct file path using:\n"
                f"- file_editor view on the parent directory\n"
                f"- search to find the correct file name\n"
                f"- execute_bash with find or ls"
            ),
        )

    return None
```

**Transition clearing**: In `record_outcome`, when a view/search/grep action is executed:

```python
# view/search/grep clears transition constraints
if is_view or is_search or is_bash_grep:
    self._last_failure_type = None
    self._last_failed_path = None

# Successful actions also clear
if failure_type is None:
    self._last_failure_type = None
    self._last_failed_path = None
```

---

### Rule 3: Consecutive Edit Failure Cap

**Trigger condition**: `max_consecutive_edit_failures` consecutive edit failures (any edit, any file) — blocks all further edits.

```python
def _check_consecutive_edit_failures(self, func_name, params) -> Optional[GuardVerdict]:
    is_edit = (func_name in ("file_editor", "str_replace_editor") and
               params.get("command") in ("str_replace", "insert"))
    if is_edit and self._consecutive_edit_failures >= self.config.max_consecutive_edit_failures:
        return GuardVerdict(
            blocked=True,
            reason=f"consecutive_edit_failures:{self._consecutive_edit_failures}",
            message=(
                f"[ACTION GUARD] You have had {self._consecutive_edit_failures} consecutive "
                f"edit failures. Stop trying to edit and take a step back.\n"
                f"Please:\n"
                f"1. View the file(s) you are trying to edit to see their current state\n"
                f"2. Re-read the error messages from previous attempts\n"
                f"3. Consider using undo_edit if edits left the file in a bad state\n"
                f"4. Try a completely different approach"
            ),
        )
    return None
```

**Reset conditions**:
- Successful edit → reset to 0
- Successful view/search → reset to 0

---

### Rule 4: Finish Pre-check

**Trigger condition**: `require_view_before_finish=True` and there are files that were edited but not re-viewed.

```python
def _check_finish_preconditions(self, func_name, params) -> Optional[GuardVerdict]:
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
                f"[ACTION GUARD] You are trying to submit, but these edited files "
                f"have not been verified since your last edit:\n  {file_list}\n"
                f"Please view/test them before submitting."
            ),
        )
    return None
```

---

## 7. Failure Classification

`record_outcome` classifies failure types by regex-matching the observation string:

```python
def _classify_failure(self, observation: str) -> Optional[str]:
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
    if observation.strip().startswith("ERROR:"):
        return "generic_error"
    return None  # success
```

Uses `re.search` (not `re.match`) because observations are prefixed with `"Execution output of [file_editor]:\n"`.

---

## 8. record_outcome — Full Logic

```python
def record_outcome(self, action_str: str, observation: str) -> None:
    func_name, params = self._parse_action(action_str)
    failure_type = self._classify_failure(observation)
    path = params.get("path", "")

    is_edit = (func_name in ("file_editor", "str_replace_editor") and
               params.get("command") in ("str_replace", "insert", "create"))
    is_view = (func_name in ("file_editor", "str_replace_editor") and
               params.get("command") == "view")
    is_search = func_name == "search"
    is_bash_grep = (func_name == "execute_bash" and
                    "grep" in params.get("cmd", params.get("command", "")))

    # Append to history
    self._history.append(ActionRecord(
        action_str=action_str,
        observation=observation[:500],
        failure_type=failure_type,
        step_index=self._step_index,
    ))

    if failure_type:
        # Failure
        self._last_failed_action = action_str
        self._last_failure_type = failure_type
        self._last_failed_path = path
        if is_edit:
            self._consecutive_edit_failures += 1
    else:
        # Success
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

    # recovery actions (view/search/grep) clear all constraints
    is_recovery = is_view or is_search or is_bash_grep
    if is_recovery:
        self._last_failed_action = None
        self._last_failure_type = None
        self._last_failed_path = None
        self._consecutive_edit_failures = 0

    self._step_index += 1
```

---

## 9. Action Parsing

Extracts function_name and parameters from XML action strings via regex (needed by Rules 2–4):

```python
def _parse_action(self, action_str: str) -> Tuple[str, Dict[str, str]]:
    fn_match = re.search(r"<function\s*=\s*([^>]+)>", action_str)
    func_name = fn_match.group(1).strip() if fn_match else ""

    pattern = r"<parameter\s*=\s*([^>]+)>(.*?)</parameter>"
    param_matches = re.findall(pattern, action_str, flags=re.DOTALL)
    params = {k.strip(): v.strip() for k, v in param_matches}

    return func_name, params
```

---

## 10. Integration

### 10.1 debug_eval_deepswe.py (manual loop)

```python
from action_guard import ActionGuard, GuardConfig

guard = ActionGuard(GuardConfig())

# In the step loop:
action_result = agent.update_from_model(model_response)

verdict = guard.evaluate(action_result.action)
if verdict.blocked:
    logger.warning("GUARD BLOCKED: %s", verdict.reason)
    obs, reward, done, info = verdict.message, 0.0, False, {"guard_blocked": True}
else:
    obs, reward, done, info = env.step(action_result.action)
    guard.record_outcome(action_result.action, str(obs))

agent.update_from_env(observation=obs, reward=reward, done=done, info=info)
```

### 10.2 eval_deepswe.py (parallel eval)

```python
from guarded_swe_env import GuardedSWEEnv
from swe_env import SWEEnv

env_cls = GuardedSWEEnv if ENABLE_GUARD else SWEEnv
env = env_cls(entry=entry, max_steps=MAX_STEPS)
```

The rollout engine stays unchanged. Guarding lives in the SWE environment wrapper.

### 10.3 GuardedSWEEnv

```python
class GuardedSWEEnv(SWEEnv):
    def __init__(self, *args, guard_config=None, **kwargs):
        self.guard = ActionGuard(guard_config or GuardConfig())
        super().__init__(*args, **kwargs)

    def _initial_observation(self):
        self.guard.reset()
        return super()._initial_observation()

    def _step_impl(self, action):
        verdict = self.guard.evaluate(action)
        if verdict.blocked:
            return EnvStepResult(
                observation=verdict.message,
                reward=0.0,
                done=False,
                info={"guard_blocked": True, "guard_reason": verdict.reason},
            )
        return super()._step_impl(action)
```

---

## 11. Key Design Decisions

| Decision | Rationale |
|---|---|
| No framework file changes | The guard is DeepSWE-specific logic and should not pollute the generic framework |
| Inject via `update_from_env` channel | No new interfaces needed; the model naturally sees guard feedback next turn |
| Trajectory records faithfully | Blocked steps have guard messages as their observation, useful for post-hoc analysis |
| Guard steps consume turns | The model did waste a turn (already called `update_from_model`), and this is correct |
| `[ACTION GUARD]` prefix | Both humans and the model can distinguish guard feedback from real environment output |
| Rule 1 uses consecutive exact match | Only blocks when the command is literally identical to the one that **just** failed — no accumulated count, no fingerprint abstraction |
| `re.search` instead of `re.match` | Observations are prefixed with `"Execution output of [...]:\n"` |
| guard_config via engine_kwargs | Leverages `RolloutOrchestrator`'s existing `engine_cls` + `engine_kwargs` mechanism |

---

## 12. Implementation Order

1. **`action_guard.py`** — Core guard logic (no external dependencies, pure Python + re)
2. **`guarded_swe_env.py`** — SWE environment wrapper (depends on action_guard.py + swe_env.py)
3. **`debug_eval_deepswe.py`** — Integrate into debug loop (easiest to test)
4. **`eval_deepswe.py`** — Integrate into production eval

---

## 13. Verification Plan

1. **Unit tests**: Construct synthetic action XML strings + observations, verify all 4 rules trigger/pass correctly
2. **Debug eval**: Run `debug_eval_deepswe.py` on an instance known to cause repeated failures, observe `GUARD BLOCKED` in logs
3. **Conversation check**: Confirm guard messages appear in `agent.chat_completions` with correct format
4. **A/B eval**: Compare Pass@1 with and without guard (expected: equal or improved)

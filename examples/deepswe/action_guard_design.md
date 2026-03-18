# Failure-Aware Action Guard — 设计文档

## 1. 问题背景

DeepSWE agent 在解 SWE-bench 问题时，经常出现以下失败模式：

| 问题 | 表现 |
|---|---|
| 重复无效动作 | 同一个 `str_replace` 已经失败，模型又原样再来一次 |
| 没吸收 tool feedback | 工具返回 `Multiple occurrences` / `No occurrences`，模型继续原地试 |
| 失败后不转状态 | non-unique 应该先看更大上下文，not-found 应该先重新看源码，但模型继续 edit |
| 以为成功了 | 模型"觉得自己修好了"，patch 没确认写进去，就想 finish |

**目标**: 在 agent 和 env 之间加一层 **runtime policy** (action guard)，强制执行失败恢复策略。不改 prompt，不改模型，改 **运行时行为**。

---

## 2. 架构总览

```
Model Response
      ↓
agent.update_from_model(response)
      ↓
    action
      ↓
┌─────────────────────────────┐
│   ActionGuard.evaluate()    │
│                             │
│  Rule 1: 重复失败拦截        │
│  Rule 2: 失败状态转移        │
│  Rule 3: 连续 edit 失败上限  │
│  Rule 4: finish 前置检查     │
└──────────┬──────────────────┘
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

**核心原则**：Guard 被 block 时，不调用 `env.step()`，而是注入合成 observation（带 `[ACTION GUARD]` 前缀）通过已有的 `update_from_env` 通道回给 agent。模型下一轮自然能看到这条反馈。

---

## 3. 文件结构

| 文件 | 类型 | 说明 |
|---|---|---|
| `examples/deepswe/action_guard.py` | 新建 ~300 行 | 核心 guard 模块：ActionGuard, GuardConfig, GuardVerdict |
| `examples/deepswe/guarded_engine.py` | 新建 ~80 行 | GuardedTrajectoryCollectEngine，继承原 engine |
| `examples/deepswe/debug_eval_deepswe.py` | 修改 ~15 行 | 在 debug loop 中集成 guard |
| `examples/deepswe/eval_deepswe.py` | 修改 ~5 行 | 用 `engine_cls=GuardedTrajectoryCollectEngine` |

**不改的文件**：base_agent.py, agent_types.py, trajectory_collect_engine.py, rollout_orchestrator.py — 所有框架层不动。

---

## 4. 核心数据结构

### 4.1 GuardConfig

```python
@dataclasses.dataclass
class GuardConfig:
    max_identical_failures: int = 2        # 完全相同 action 允许失败几次
    max_consecutive_edit_failures: int = 3  # 连续 edit 失败上限
    require_view_after_not_found: bool = True   # not_found 后必须先 view
    require_view_after_non_unique: bool = True  # non_unique 后必须先 view
    require_view_before_finish: bool = False    # finish 前是否要求确认编辑
    enabled: bool = True                        # 总开关
```

### 4.2 GuardVerdict

```python
@dataclasses.dataclass
class GuardVerdict:
    blocked: bool           # True = 不调 env.step
    message: str = ""       # blocked 时注入的合成 observation
    reason: str = ""        # 日志用原因码（如 "repeated_failure:3"）
```

### 4.3 ActionRecord

```python
@dataclasses.dataclass
class ActionRecord:
    action_str: str              # 完整 action XML string
    observation: str             # env 返回的 observation（截断到 500 字符）
    failure_type: Optional[str]  # 分类后的失败类型，None = 成功
    step_index: int              # 第几步
```

---

## 5. ActionGuard 类

### 5.1 State Variables

```python
class ActionGuard:
    _history: List[ActionRecord]           # 完整历史记录
    _failure_counts: Dict[str, int]        # action_str 原文 → 失败次数 (exact match)
    _consecutive_edit_failures: int         # 连续 edit 失败计数
    _last_failure_type: Optional[str]       # 上一次失败类型
    _last_failed_path: Optional[str]        # 上一次失败涉及的文件路径
    _files_edited_since_last_view: set      # 编辑后未确认的文件集合
    _files_successfully_edited: set         # 本 episode 成功编辑过的文件集合
    _step_index: int                        # 当前步数
```

### 5.2 核心方法

```python
def evaluate(self, action_str: str) -> GuardVerdict:
    """判断 action 是否允许执行。依次检查 4 条规则。"""

def record_outcome(self, action_str: str, observation: str) -> None:
    """action 执行后记录结果，更新 guard 状态。必须在每次 env.step() 后调用。"""

def reset(self) -> None:
    """episode 开始时重置所有状态。"""
```

### 5.3 evaluate() 流程

```python
def evaluate(self, action_str):
    if not self.config.enabled:
        return GuardVerdict(blocked=False)

    func_name, params = self._parse_action(action_str)

    # Rule 1: 完全相同的 action 重复失败
    verdict = self._check_repeated_failure(action_str)
    if verdict: return verdict

    # Rule 2: 失败后必须先恢复再继续
    verdict = self._check_failure_transition(func_name, params)
    if verdict: return verdict

    # Rule 3: 连续 edit 失败太多次
    verdict = self._check_consecutive_edit_failures(func_name, params)
    if verdict: return verdict

    # Rule 4: finish/submit 前置条件检查
    verdict = self._check_finish_preconditions(func_name, params)
    if verdict: return verdict

    return GuardVerdict(blocked=False)
```

---

## 6. 四条规则详解

### Rule 1: 重复失败拦截

**触发条件**: action string **完全一样**（exact match），且已失败 >= `max_identical_failures` 次。

**不做 fingerprint 抽象**——只有 command 一模一样才拦。

```python
def _check_repeated_failure(self, action_str: str) -> Optional[GuardVerdict]:
    count = self._failure_counts.get(action_str, 0)
    if count >= self.config.max_identical_failures:
        return GuardVerdict(
            blocked=True,
            reason=f"repeated_failure:{count}",
            message=(
                f"[ACTION GUARD] This exact action has already failed {count} time(s). "
                f"Repeating it will produce the same result.\n"
                f"Please try a DIFFERENT approach:\n"
                f"- View the file around the relevant lines\n"
                f"- Use search/grep to find the correct string\n"
                f"- Include more context in old_str to make it unique"
            ),
        )
    return None
```

**record_outcome 中更新**：

```python
if failure_type:
    self._failure_counts[action_str] = self._failure_counts.get(action_str, 0) + 1
```

---

### Rule 2: 失败后状态转移

**核心思想**: 每种失败都映射到明确的下一步，不允许不看就重试。

| 上一次失败 | 当前 action | 判定 |
|---|---|---|
| `non_unique` | edit 同一文件 | **block** — 要求先 view 更大上下文 |
| `not_found` | edit 同一文件 | **block** — 要求先 view 刷新源码 |
| `path_not_found` | edit 同一路径 | **block** — 要求先确认路径存在 |
| 任意失败 | view / search / grep | **allow** 并清除 transition 约束 |

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

**Transition 清除**: 在 `record_outcome` 中，当执行 view/search/grep 时：

```python
# view/search/grep 清除 transition 约束
if is_view or is_search or is_bash_grep:
    self._last_failure_type = None
    self._last_failed_path = None

# 成功的 action 也清除
if failure_type is None:
    self._last_failure_type = None
    self._last_failed_path = None
```

---

### Rule 3: 连续 edit 失败上限

**触发条件**: 连续 `max_consecutive_edit_failures` 次 edit 失败（任何 edit，任何文件），再次 edit 时 block。

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

**重置条件**:
- edit 成功 → 重置为 0
- view/search 成功 → 重置为 0

---

### Rule 4: Finish 前置检查

**触发条件**: `require_view_before_finish=True` 且有编辑过但未重新查看的文件。

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

## 7. Failure 分类

`record_outcome` 中通过正则匹配 observation 来分类失败类型：

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
    return None  # 成功
```

用 `re.search`（不是 `re.match`）——observation 前面有 `"Execution output of [file_editor]:\n"` 前缀。

---

## 8. record_outcome 完整逻辑

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

    # 记录到历史
    self._history.append(ActionRecord(
        action_str=action_str,
        observation=observation[:500],
        failure_type=failure_type,
        step_index=self._step_index,
    ))

    if failure_type:
        # 失败
        self._failure_counts[action_str] = self._failure_counts.get(action_str, 0) + 1
        self._last_failure_type = failure_type
        self._last_failed_path = path
        if is_edit:
            self._consecutive_edit_failures += 1
    else:
        # 成功
        self._last_failure_type = None
        self._last_failed_path = None
        if is_edit:
            self._consecutive_edit_failures = 0
            self._files_successfully_edited.add(path)
            self._files_edited_since_last_view.add(path)

    # view 清除 "未确认" 状态
    if is_view and path:
        self._files_edited_since_last_view.discard(path)

    # view/search/grep 清除 transition 约束（即使上面已经因为成功清除了，这里覆盖也无妨）
    if is_view or is_search or is_bash_grep:
        self._last_failure_type = None
        self._last_failed_path = None

    self._step_index += 1
```

---

## 9. Action 解析

用正则从 XML action string 中提取 function_name 和 parameters（Rule 2-4 需要）：

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

## 10. 集成方式

### 10.1 debug_eval_deepswe.py（手动 loop）

```python
from action_guard import ActionGuard, GuardConfig

guard = ActionGuard(GuardConfig())

# 在 step loop 中:
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

### 10.2 eval_deepswe.py（并行 eval）

```python
from guarded_engine import GuardedTrajectoryCollectEngine
from action_guard import GuardConfig

orchestrator = RolloutOrchestrator(
    engine_cls=GuardedTrajectoryCollectEngine,
    engine_kwargs=dict(
        model_call=model_call,
        timeout=TIMEOUT,
        guard_config=GuardConfig(),
    ),
    max_concurrency=MAX_CONCURRENT,
    rollout_sync_lock=agentic_utils.RolloutSyncLock(),
)
```

`RolloutOrchestrator` 已有 `engine_cls` 参数，天然支持替换 engine 类。

### 10.3 GuardedTrajectoryCollectEngine

```python
class GuardedTrajectoryCollectEngine(TrajectoryCollectEngine):
    def __init__(self, *args, guard_config=None, **kwargs):
        # guard_config 从 kwargs 中取出，不传给 parent
        super().__init__(*args, **kwargs)
        self.guard = ActionGuard(guard_config or GuardConfig())

    async def _reset(self):
        await super()._reset()
        self.guard.reset()

    async def _one_step(self) -> bool:
        # model call
        resp = await asyncio.get_event_loop().run_in_executor(
            None, self.model_call, self.agent.chat_completions,
            self.env, **self.model_call_kwargs,
        )
        action = self.agent.update_from_model(resp).action
        if action is None:
            action = []

        # guard evaluate
        verdict = self.guard.evaluate(action)
        if verdict.blocked:
            obs, rew, done, info = verdict.message, 0.0, False, {"guard_blocked": True}
        else:
            obs, rew, done, info = await asyncio.get_event_loop().run_in_executor(
                None, self.env.step, action
            )
            self.guard.record_outcome(action, str(obs))

        self.agent.update_from_env(obs, rew, done, info)

        # tokenization (same as parent _one_step lines 303-334)
        ...

        # timeout check
        if time.time() - self._start_ts > self.timeout:
            self.agent.get_current_state().done = True
            return True
        return done
```

---

## 11. 关键设计决策

| 决策 | 理由 |
|---|---|
| 不改框架文件 | guard 是 deepswe 特有逻辑，不应污染通用框架 |
| 利用 `update_from_env` 通道注入 | 不需要新接口，model 下一轮自然看到 |
| Trajectory 如实记录 | blocked step 的 observation 是 guard message，可用于后续分析 |
| Guard step 消耗轮次 | model 确实浪费了一轮（已调 `update_from_model`），这是正确的 |
| `[ACTION GUARD]` 前缀 | 人类和模型都能区分 guard feedback 和真实环境输出 |
| Rule 1 用 exact match | 用户要求：只有 command 完全一样才拦截 |
| `re.search` 而非 `re.match` | observation 前有 `"Execution output of [...]:\n"` 前缀 |
| guard_config 通过 engine_kwargs 传入 | 利用 `RolloutOrchestrator` 已有的 `engine_cls` + `engine_kwargs` 机制 |

---

## 12. 实现顺序

1. **`action_guard.py`** — 核心 guard 逻辑（无外部依赖，纯 Python + re）
2. **`guarded_engine.py`** — engine 子类（依赖 action_guard.py + trajectory_collect_engine.py）
3. **`debug_eval_deepswe.py`** — 集成到 debug loop（最方便测试）
4. **`eval_deepswe.py`** — 集成到正式 eval

---

## 13. 验证方案

1. **Unit test**: 构造合成 action XML string + observation，测试 ActionGuard 的 4 条规则是否正确触发/放行
2. **Debug eval**: 用 `debug_eval_deepswe.py` 跑一个已知会反复失败的 instance，观察 log 中 `GUARD BLOCKED` 输出
3. **Conversation 检查**: 确认 guard message 出现在 `agent.chat_completions` 中，格式正确
4. **A/B eval**: 对比有/无 guard 的 Pass@1（预期持平或提升）

# Agentic RL Initial Design Proposal

Status: early design proposal

Scope: multi-turn / agentic RL frontend and backend split

## 0. Design Philosophy

For multi-turn and agentic RL, the system can be divided into two major parts:

| Layer | Responsibility |
|---|---|
| Frontend | Collect trajectories by running the model agent inside an environment |
| Backend | Train the policy model from collected trajectories |

The frontend is task-facing. It knows how to build an agent, create an
environment, execute tool calls, and record the interaction history. The
backend is trainer-facing. It consumes trajectories and performs RL updates.

The simplest mental model is:

```python
trajectories = []

for task in tasks:
  agent = ToolAgent()
  env = ToolEnv(task)
  trajectory = run_rollout(agent, env)
  trajectories.append(trajectory)

learner.train(trajectories)
```

For online RL, the same idea becomes a loop:

```text
Actor
  -> trajectory collector
  -> trajectory buffer
  -> learner
  -> new actor params
  -> actor
```

The frontend keeps interacting with environments and appending new trajectories.
The backend starts training once it has enough fresh data, instead of waiting
for all trajectories to be collected.

## 1. Offline vs Online RL

In post-training and RL fine-tuning, there are two common modes.

| Mode | Trajectory generation | Training start | Pros | Cons |
|---|---|---|---|---|
| Offline | Generate or collect a fixed trajectory dataset first | After collection finishes | Simple and reproducible | Learns only from old-policy data |
| Online | Continuously collect trajectories while training | As soon as enough new trajectories are available | Can bootstrap from the improving policy | Requires buffering, sync, and failure handling |

This design chooses the online loop as the target design, while keeping the
trajectory format reusable for offline replay.

## 2. Overall Architecture

The proposed package structure is:

```text
tunix/rl/multi_turn/
  agents/                 # Agent layer
  environments/           # Environment layer
  parser/tool_parser/     # Tool parsing layer
  tools/                  # Tool execution layer
  rewards/                # Reward system
  trajectory/             # Trajectory collection layer
  prompts/                # Prompt management layer
```

Design principle:

* agent code owns conversation and model-output parsing;
* environment code owns task state, tool execution, and reward feedback;
* parser code isolates model-specific tool-call syntax;
* tool code exposes executable capabilities through a common schema;
* trajectory collection coordinates the loop but avoids task-specific logic;
* learner code consumes trajectories and trains the model.

### 2.1 Design Requirements

The design should satisfy the following requirements.

| Requirement | Description | Design implication |
|---|---|---|
| Task modularity | New tasks should be added without rewriting the learner | Task-specific logic lives in `Agent` and `Environment` |
| Multi-turn support | A task may require repeated model/action/environment steps | Trajectories store ordered `Step` objects, not only final answers |
| Tool support | The model may emit structured tool calls | Parsing and tool execution are separate layers |
| Online training | The backend should train while the frontend continues collecting data | A buffer separates trajectory collection from optimization |
| Offline replay | Collected trajectories should also be usable as fixed data | Trajectory records must contain enough information for later training |
| Backend isolation | The learner should not directly execute tools or environment logic | Backend consumes trajectory data only |
| Debuggability | Failures should be inspectable after collection | Raw responses, parsed actions, observations, rewards, and metadata are stored |

### 2.2 Frontend and Backend Boundary

The frontend/backend boundary is the central design contract.

```text
frontend output:
  Trajectory(task, steps, reward, metadata)

backend input:
  Trajectory or batch[Trajectory]
```

The frontend is allowed to depend on task-specific objects such as tools,
repositories, sandboxes, parsers, and reward functions. The backend should not
depend on those objects. Instead, the backend should depend only on structured
trajectory data.

This boundary gives the system three useful properties:

* trajectories can be inspected and replayed;
* environment bugs can be debugged separately from trainer bugs;
* online training and offline training can share the same backend path.

### 2.3 Control Plane and Data Plane

The design separates control decisions from data movement.

| Plane | Examples | Owner |
|---|---|---|
| Control plane | When to start rollout, when to stop an episode, when to train, when to sync weights | Collector / learner loop |
| Data plane | Messages, model responses, actions, observations, rewards, trajectories | Agent / environment / buffer |

This distinction is important because multi-turn environments may be slow or
stateful. The control plane can schedule work and apply backpressure without
embedding task-specific payload logic.

## 3. Core Data Structures

### Step

One environment interaction step:

```python
@dataclass
class Step:
  chat_completions: list[dict[str, str]]
  thought: str
  action: Any
  observation: Any
  model_response: str
  reward: float
  done: bool
  mc_return: float
```

### Trajectory

One complete rollout:

```python
@dataclass
class Trajectory:
  task: Any
  steps: list[Step]
  reward: float
```

### ToolCall and ToolOutput

Common tool-call representation:

```python
@dataclass
class ToolCall:
  name: str
  arguments: dict[str, Any]


@dataclass
class ToolOutput:
  name: str
  output: str | list | dict
  error: str
  metadata: dict
```

These structures are the boundary between frontend collection and backend
training. The backend should be able to train from a serialized trajectory
without directly accessing the environment.

## 4. Module Design

### 4.1 Agent Layer

The agent converts environment observations into model messages and converts
model responses into executable actions.

```python
class BaseAgent(ABC):
  @property
  def chat_completions(self) -> list[dict[str, str]]:
    ...

  @property
  def trajectory(self) -> Trajectory:
    ...

  @abstractmethod
  def update_from_env(self, observation, reward, done, info):
    ...

  @abstractmethod
  def update_from_model(self, response: str) -> Action:
    ...

  @abstractmethod
  def reset(self):
    ...
```

`ToolAgent` adds tool parsing and tool prompt construction:

```python
class ToolAgent(BaseAgent):
  def __init__(self, system_prompt, parser_name, tool_map):
    self.tool_manager = ToolManager(tool_map)
    self.tool_parser = get_tool_parser(parser_name)
    self._messages = []
    self._trajectory = Trajectory()
```

Design choice: the agent should parse the model response but should not execute
tools directly. Tool execution belongs to the environment.

### 4.2 Environment Layer

The environment executes actions and returns observations, rewards, and
termination state.

```python
class BaseEnv(ABC):
  @abstractmethod
  def reset(self) -> tuple[dict, dict]:
    ...

  @abstractmethod
  def step(self, action) -> tuple[Any, float, bool, dict]:
    ...

  @staticmethod
  @abstractmethod
  def from_dict(env_args: dict) -> "BaseEnv":
    ...
```

`ToolEnvironment` is the default tool-execution environment:

```python
class ToolEnvironment(BaseEnv):
  def __init__(self, task, tool_map, reward_fn, max_steps=10):
    self.tool_manager = ToolManager(tool_map)
    self.reward_fn = reward_fn
    self.step_count = 0
```

Design choice: the environment owns task state, tool execution, max-step logic,
and reward feedback.

### 4.3 Tool Parser Layer

The parser converts model-specific text into structured tool calls.

```python
class ToolParser(ABC):
  @abstractmethod
  def parse(self, model_response: str) -> list[ToolCall]:
    ...

  @abstractmethod
  def get_tool_prompt(self, tools_schema: str) -> str:
    ...

  def parse_tool_outputs(self, model_response: str) -> dict:
    ...
```

Parser registration:

```python
_PARSER_REGISTRY = {
    "qwen": QwenToolParser,
}


def get_tool_parser(parser_name: str) -> type[ToolParser]:
  return _PARSER_REGISTRY[parser_name]
```

Design choice: parser differences should not leak into the learner.

### 4.4 Tool Execution Layer

Tools expose a schema and an execution method.

```python
class BaseTool(ABC):
  @property
  @abstractmethod
  def json(self) -> dict:
    ...

  def forward(self, **kwargs) -> ToolOutput:
    ...

  async def async_forward(self, **kwargs) -> ToolOutput:
    ...
```

`ToolManager` handles registration and dispatch:

```python
class ToolManager:
  def __init__(self, tool_map: dict[str, type[BaseTool]]):
    ...

  @property
  def json(self) -> list[dict]:
    ...

  def run(self, tool_name: str, **kwargs) -> ToolOutput:
    ...

  def execute_calls(
      self,
      calls: list[ToolCall],
      parallel: bool = True,
  ) -> dict[str, str]:
    ...
```

Design choice: tool execution can be parallelized when calls are independent,
but errors should be wrapped as `ToolOutput` when possible.

### 4.5 Reward System

Rewards can come from the environment or from registered reward functions.

```python
@dataclass
class RewardOutput:
  reward: float
  metadata: dict[str, Any]


_REGISTRY: dict[str, Callable] = {}


@register("reward_name")
def reward_function(task: dict, action: str) -> RewardOutput:
  ...
```

Design choice: reward metadata should be preserved for debugging, not only the
scalar reward.

### 4.6 Trajectory Collection Engine

The collection engine coordinates one complete rollout.

```python
class TrajectoryCollectEngine:
  def __init__(
      self,
      agent,
      env,
      model_call,
      final_reward_fn,
      max_steps=10,
      gamma=1.0,
      timeout=30.0,
  ):
    ...

  async def collect(self) -> Trajectory:
    ...
```

Design choice: the engine should be dependency-injected with `agent`, `env`,
and `model_call`, so it does not care whether inference is local, remote, or
batched.

## 5. Execution Flow

### 5.1 Startup

```text
ToolAgent
  -> ToolManager
  -> ToolParser
  -> system prompt with tool schema

ToolEnvironment
  -> task config
  -> tool map
  -> reward function

TrajectoryCollectEngine
  -> agent
  -> env
  -> model_call
```

### 5.2 Rollout Loop

```text
engine.collect()
  -> env.reset()
  -> agent.reset()
  -> agent.update_from_env(initial_observation)

while not done:
  -> agent.chat_completions
  -> model_call(messages)
  -> agent.update_from_model(response)
  -> env.step(action)
  -> agent.update_from_env(observation, reward, done, info)
  -> stop on done, max_steps, or timeout

finalize:
  -> append final reward if needed
  -> fill Monte Carlo returns
  -> cleanup environment resources
  -> return Trajectory
```

### 5.3 Tool Call Flow

```text
model response
  -> tool_parser.parse(response)
  -> ToolCall list
  -> env.step(action)
  -> tool_manager.execute_calls(calls)
  -> tool.forward(**arguments)
  -> ToolOutput
  -> next observation
```

### 5.4 Message Conversion

```text
task observation
  -> user message

tool output
  -> user/tool message with tool result

model response
  -> assistant message
```

Design choice: conversation history is owned by the agent, while environment
state and tool results are owned by the environment.

## 6. Backend Training Contract

The backend consumes trajectories and converts them into RL training examples.

The frontend should provide:

* task metadata;
* ordered steps;
* model input messages;
* raw model responses;
* parsed actions;
* observations;
* rewards;
* done status;
* optional tool metadata.

The backend is responsible for:

* tokenization;
* action or assistant-token masking;
* return and advantage computation;
* policy loss computation;
* optimizer update;
* parameter sync back to the actor.

This keeps environment execution out of the learner and keeps optimization
logic out of the environment.

## 7. Design Options and Decisions

### Option A: Offline Trajectory Dataset

Collect all trajectories first, then train from a fixed dataset.

Pros:

* easiest to debug;
* simple to reproduce;
* useful for validating the trajectory schema.

Cons:

* data is stale;
* the policy cannot improve the data distribution during training.

When to use:

* validating a new trajectory schema;
* debugging reward functions;
* reproducing a known training sample.

### Option B: Online Producer-Consumer Loop

Collect trajectories continuously while the backend trains.

Pros:

* supports policy improvement during data collection;
* hides environment latency;
* matches the target agentic RL use case.

Cons:

* requires buffering;
* requires parameter sync;
* requires careful failure handling.

When to use:

* target multi-turn RL training;
* tasks where the policy should improve its own future data;
* environments with enough latency to benefit from overlapping rollout and
  training.

### Option C: One Monolithic Learner

Put agent, environment, tool execution, and training logic into one learner.

Pros:

* fastest for a one-off prototype;
* fewer interfaces.

Cons:

* task logic leaks into trainer code;
* hard to reuse for other environments;
* hard to test components independently.

When to use:

* early throwaway experiments only;
* not recommended as the long-term Tunix agentic RL design.

### Decision Matrix

| Criterion | Offline dataset | Online producer-consumer | Monolithic learner |
|---|---|---|---|
| Supports policy improvement during collection | No | Yes | Yes |
| Easy to reproduce | High | Medium | Low |
| Supports multiple tasks cleanly | Medium | High | Low |
| Keeps learner task-agnostic | High | High | Low |
| Handles slow environments efficiently | Low | High | Medium |
| Implementation complexity | Low | Medium | Low initially, high later |

### Selected Design

Use Option B with the modular interface boundaries from Option A.

The intended evolution path is:

```text
local rollout prototype
  -> offline trajectory replay
  -> online producer-consumer loop
  -> scalable distributed rollout
```

The selected design is an online producer-consumer architecture. The frontend
collects trajectories from the latest available actor policy and writes them to
a buffer. The backend consumes buffered trajectories, performs RL updates, and
periodically publishes new parameters back to the actor.

This gives the system a clean separation of responsibilities:

| Responsibility | Selected owner |
|---|---|
| Conversation state | Agent |
| Action parsing | Agent plus parser |
| Tool execution | Environment through `ToolManager` |
| Reward feedback | Environment or reward registry |
| Trajectory lifecycle | `TrajectoryCollectEngine` |
| Batching and RL loss | Backend learner |
| Weight refresh | Online loop between backend and frontend |

### Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Frontend/backend boundary | Trajectory object | Gives one shared contract for online streaming and offline replay |
| Tool execution owner | Environment | Keeps the agent focused on model interaction and avoids executing model text directly |
| Parser ownership | Separate parser layer | Model-family formatting can evolve without changing the learner |
| Training input | Full trajectories, not only final answers | Multi-turn credit assignment and debugging require step history |
| Online buffer | FIFO first | Simple semantics for the initial design; replay or priority can be added later |
| Parameter update direction | Backend publishes to frontend | The trainer remains the source of truth for policy parameters |

### Design Invariants

These invariants should remain true even if implementation details change:

* the learner does not directly call task tools;
* the environment does not compute gradients or own optimizer state;
* the agent records enough conversation state to reconstruct model inputs;
* each trajectory records raw model output and parsed action;
* the backend can train from stored trajectories without re-running the
  environment;
* online collection has an explicit parameter-refresh boundary.

## 8. Open Questions

Design questions to resolve during implementation:

* Should tool errors terminate the episode or become observations?
* Should trajectory tokenization happen in the frontend or backend?
* How frequently should learner updates sync back to the actor?
* How should partial trajectories be handled after timeout?
* Should the buffer support replay, priority, or only FIFO?
* What metadata is required for deterministic offline replay?
* Which metrics are required before an online run is considered valid?

## 9. Summary

This design separates agentic RL into a frontend trajectory collector and a
backend learner. Agents manage conversation state, environments execute actions,
parsers handle model-specific tool-call formats, tools provide executable
capabilities, and trajectories become the contract between collection and
training.

The design focuses on a clean online RL loop while preserving a simple offline
trajectory replay path.

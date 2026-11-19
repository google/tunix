<!-- DO NOT REMOVE! Placeholder for TOC. -->

# Agentic RL

## Architecture

![Trajectory Collect Engine Overview](images/agentic_rollout_pipeline.png)

## Core Components

The framework consists of several key components:

*   **Agent**: Interacts with the environment by generating actions based on
    observations and conversation history.
*   **Environment**: Represents the task or problem to be solved, processes
    agent actions, and returns observations, rewards, and termination signals.
*   **Tool**: A reusable component that provides specific functionalities (e.g.,
    calculation, search) that an agent can invoke.
*   **Parser**: Translates between natural language model responses and
    structured data like tool calls, and formats conversation history into
    model-specific input formats.
*   **TrajectoryCollectEngine**: Manages the interaction loop for a single
    agent-environment pair to produce a complete trajectory.
*   **RolloutOrchestrator**: Manages multiple `TrajectoryCollectEngine`
    instances for parallel trajectory collection.
*   **AgenticRLLearner / AgenticGRPOLearner**: Coordinates asynchronous
    producer-consumer trajectory collection, group-wise reward/advantage
    computation, and policy updates (`tunix.AgenticGRPOLearner` and
    `tunix.AgenticGRPOConfig`).

### Agents

Agents inherit from `ConversationAgentBase`, which provides common functionality
for maintaining conversation history (`chat_completions`) and recording
interaction steps in a `Trajectory`.

*   **`ModelAgent`**: A simple agent for single-turn tasks where the model's
    response is treated as the final answer.
*   **`ToolAgent`**: A more complex agent that can parse model responses to
    detect and invoke tool calls. It uses a `ToolManager` to manage available
    tools and a `ToolParser` (e.g., `QwenToolParser`, `GeminiToolParser`) to
    understand model outputs and format tool schemas for the model prompt.

### Environments

Environments inherit from `BaseTaskEnv`, which handles episode lifecycle
management like `max_steps`.

*   **`TaskEnvironment`**: Designed for single-turn tasks. The environment
    terminates after the first agent action and computes a reward based on the
    final response using a provided `reward_fn`.
*   **`ToolEnvironment`**: Designed for multi-turn, tool-using tasks. It
    receives actions from the `ToolAgent`, and if they are tool calls, it uses
    its internal `ToolManager` to execute them via `execute_calls`. The results
    of tool execution are returned to the agent as a new observation in
    `{"tool_outputs": ...}` format. The episode terminates when the agent
    invokes a special `finish` function or `max_steps` is reached, at which
    point `reward_fn` is called on the final answer.

### Tool Integration

Tools inherit from `BaseTool` and must implement `get_json_schema()` to define
their interface (parameters, description) and either `apply()` (synchronous) or
`apply_async()` (asynchronous) to define their logic. The `ToolManager`
discovers, registers, and executes tools by name. It can execute multiple tool
calls in parallel for efficiency.

### Agent/Environment interaction

![Agent/Environment interaction](images/agentic_agent:env.png)

--------------------------------------------------------------------------------

## Key Features and Optimizations

### Multi-turn Tool Use

Tunix fully supports multi-turn interactions involving tool use. The typical
flow is:

1.  The `ToolAgent` sends the conversation history (including user query and
    prior tool results) to the LLM.

2.  The LLM responds with a tool call, e.g., `calculator(a=1, b=1)`.

3.  The `ToolAgent` uses its `ToolParser` to parse this into an `Action` object.

4.  The `ToolEnvironment` receives this action, uses its `ToolManager` to
    execute `calculator`, and receives the result "2".

5.  The `ToolEnvironment` returns an observation like `{"tool_outputs":
    {"call_id_123": "Tool returned result: 2"}}`, reward=1, and done=False.

6.  The `ToolAgent` adds the tool result to its history as a `role: tool`
    message.

7.  The loop continues until the agent calls `finish(answer=...)` or `max_steps`
    is reached.

### Asynchronous Rollouts

To accelerate trajectory collection, Tunix supports asynchronous rollouts via
the `RolloutOrchestrator`. It leverages Python's `asyncio` to manage multiple
concurrent agent-environment interactions using `TrajectoryCollectEngine`
instances, with parallelism controlled by `max_concurrency`. The
`run_producers_from_stream` method manages a pool of workers that draw
agent-environment pairs from a stream, run full episodes via `collect()`, and
queue the resulting trajectories. The `yield_batches` method allows a consumer
(like an RL learner) to receive trajectories as they are generated. This
parallel execution significantly speeds up data collection, especially when
interacting with external models or tools with high latency.

Furthermore, Tunix provides a `RolloutSyncLock` to manage concurrency between
rollouts and model weight synchronization in distributed training setups. This
lock ensures that rollouts (`acquire_rollout`) are temporarily paused when a
weight sync (`acquire_weight_sync`) is requested, preventing agents from
generating trajectories with stale parameters.

![Batch vs Async Rollout](images/batch_vs_async_rollout.png)

### Trajectory Batching and Grouping

Tunix supports batching of agentic trajectories through `GroupQueueManager`
(`tunix.rl.agentic.queue_manager.group_queue_manager.GroupQueueManager`). Used
within `RolloutOrchestrator`, it collects `TrajectoryItem` instances into
buckets using a configurable `key_fn` (defaulting in `RolloutOrchestrator` to
`item.prompt_id`) or a custom `group_fn(buckets, item)`. In
`RolloutOrchestrator.run_producers_from_stream`, each collected episode is
assigned a `prompt_id` via `group_key_fn(pair_index, env, trajectory)` (which
defaults to `pair_index` from `env.extra_kwargs["pair_index"]`, or can read
custom task metadata such as `env.extra_kwargs["group_id"]`).

Once a bucket reaches `num_generations` trajectories (as required by GRPO), it
is optionally validated through `filter_fn` (invoking `on_group_filtered` if an
entire group is rejected) and moved into ready groups for downstream consumption
by `yield_batches`.

### Training with `AgenticGRPOLearner`

`AgenticGRPOLearner` (`tunix.AgenticGRPOLearner`, configured via
`tunix.AgenticGRPOConfig`) connects `RolloutOrchestrator` and `RLEngine` in an
asynchronous producer-consumer loop: background workers collect multi-turn
trajectories with `agent_class` and `env_class`, group them by prompt, compute
environment and/or function rewards (`reward_mode="env"` by default), and feed
completed groups into the actor trainer.

```python
from tunix import AgenticGRPOConfig, AgenticGRPOLearner
from tunix.rl.agentic.parser.chat_template_parser import parser as chat_parser_lib

algo_config = AgenticGRPOConfig(
    num_generations=4,
    max_response_length=1024,
    max_concurrency=16,
    beta=0.04,
    epsilon=0.2,
)

chat_parser = chat_parser_lib.QwenChatTemplateParser(tokenizer)

learner = AgenticGRPOLearner(
    rl_engine=rl_engine,
    algo_config=algo_config,
    chat_parser=chat_parser,
    agent_class=MyToolAgent,
    agent_kwargs={"max_turns": 5},
    env_class=MyToolEnvironment,
    env_kwargs={"max_steps": 5},
)
learner.train(train_ds=train_dataset, eval_ds=eval_dataset)
```

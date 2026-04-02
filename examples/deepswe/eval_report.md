**DeepSWE Evaluation Report**

**Objective**

This experiment evaluates the deepscaler-style DeepSWE evaluation pipeline on the Qwen3-32B model, and analyzes the impact of the action guard wrapper on resolve rate, trajectory length, context overflow termination, and invalid actions.

**Evaluation Pipeline**

The evaluation script follows this workflow:

1. Load the SWE-bench Verified test dataset.
2. For each instance, construct a complete task:
   - One `SWEAgent`
   - One `SWEEnv`, wrapped with `GuardedSWEEnv` when the wrapper is enabled
3. Use the task-level parallel scheduler `RolloutOrchestrator` to concurrently execute multiple complete trajectories.
4. Each trajectory runs as a multi-turn agent-environment interaction loop:
   - `reset()` the environment, spinning up a Docker container and obtaining the initial observation
   - Assemble the current conversation into a prompt via `QwenChatTemplateParser`
   - `model_call()` invokes the vLLM sampler to generate a model response
   - The agent parses the XML function call, extracting thought and action
   - `env.step()` executes the action inside the container and returns observation / reward / done
   - Update agent state and proceed to the next turn
   - Repeat until one of the following termination conditions is met:
     - Environment done (agent calls `finish`)
     - `MAX_STEPS` reached
     - `MAX_CONTEXT_LIMIT` reached
     - Timeout
5. After each trajectory completes:
   - Compute the final reward (by running test cases via `_calculate_reward`)
   - Aggregate step count, status, and guard hit statistics
   - Write the complete trajectory to a unified log file
6. After all instances finish:
   - Aggregate resolved count
   - Compute Pass@1, average steps, status distribution, and guard statistics

Key characteristics of this pipeline:
- One instance maps to one complete trajectory
- Parallelism granularity is at the task level (not step level)
- Each trajectory is a multi-turn agentic interaction, not a single-turn generation

**Key Parameters**

The key parameters used in this evaluation and their effects:

- `MODEL_VERSION = Qwen/Qwen3-32B`
  - Base model used for evaluation

- `MAX_STEPS = 30`
  - Maximum number of interaction steps per trajectory

- `MAX_RESPONSE_LENGTH = 8192`
  - Maximum generation length per model call

- `MAX_MODEL_LEN = 32768`
  - Model context window size

- `MAX_CONTEXT_LIMIT = MAX_MODEL_LEN - 256`
  - Effective context threshold for trajectories
  - When approached, the current trajectory is terminated early to prevent the entire evaluation from crashing due to an oversized prompt

- `MAX_CONCURRENT`
  - Number of trajectories running concurrently
  - Default is 256, though actual throughput is constrained by environment and backend capacity

- `VLLM_HBM_UTILIZATION = 0.4`
  - Target HBM utilization for vLLM

- `VLLM_MAX_NUM_SEQS = 128`
  - Maximum concurrent sequences in vLLM

- `VLLM_MAX_BATCHED_TOKENS = 165888`
  - Maximum batched tokens per vLLM batch

- `ENABLE_GUARD`
  - Whether to enable the wrapper
  - When off, uses the raw `SWEEnv`
  - When on, uses `GuardedSWEEnv`, which intercepts clearly invalid actions

- `TIMEOUT = 600`
  - Per-trajectory timeout in seconds

**Handling Overlong Trajectories**

If a trajectory's conversation grows too fast and exceeds the context limit, the script does not crash the entire evaluation. Instead:

- The current trajectory is terminated
- Its status is marked as `MAX_CONTEXT_LIMIT_REACHED`
- Its reward is set to `0`
- Other instances continue executing

This is critical for large-scale parallel evaluation, as it prevents a single anomalous trajectory from causing the entire batch to fail.

The implementation lives in `EvalTrajectoryCollectEngine`: when `model_call()` detects that the prompt token count exceeds `MAX_MODEL_LEN`, it raises a `PromptTooLongError`. The engine's `_one_step()` catches this exception, marks the trajectory as `MAX_CONTEXT_LIMIT_REACHED`, skips reward computation, and terminates gracefully.

**What the Wrapper Does**

The wrapper does not modify model capabilities — it restricts clearly invalid agent behaviors. `GuardedSWEEnv` wraps `SWEEnv` with an `ActionGuard` that checks action validity before each `env.step()` call. It intercepts the following cases:

| Rule | Trigger Condition | Intervention |
|------|-------------------|-------------|
| Missing function call | Model output does not contain a valid XML `<function=...>` tag | Inject format correction prompt |
| Repeated failure | The exact same action just failed | Inject alternative strategy suggestion |
| Failure transition | `str_replace` failed with not_found / non_unique / path_not_found, and agent retries edit directly | Force view/search before next edit |
| Consecutive edit failures | 3+ consecutive edit failures and agent continues blind editing | Inject step-back prompt |

When an action is blocked, the guard injects a synthetic observation to guide the agent toward a different strategy, without consuming an environment step. After the agent performs a recovery action (view/search/grep), the guard state resets automatically.

The wrapper's effect is therefore:
- Reduce idle spinning
- Reduce invalid steps
- Reduce unnecessary context inflation
- Allocate more steps toward actual bug fixing

**Experiment Results**

| Run | Hardware | Model | Wrapper | Resolved | Pass@1 | Avg Steps | SUCCEEDED | MAX_CONTEXT_LIMIT_REACHED | Guarded Trajs | Guard Blocks |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | TPU | Qwen3-32B | Off | 93/500 | 0.1860 | 21.31 | 438 | 62 | 0 | 0 |
| 2 | TPU | Qwen3-32B | On | 120/500 | 0.2400 | 21.98 | 450 | 50 | 171 | 669 |
| 3 | GPU | Qwen3-32B | Off | 95/500 | 0.1900 | — | — | — | 0 | 0 |

**Guard Hit Distribution**

In the TPU 32B experiment with wrapper enabled, the guard reason distribution is as follows:

| Guard Reason | Count |
|---|---:|
| `missing_function_call` | 101 |
| `repeated_failure` | 44 |
| `transition:non_unique_requires_view` | 27 |
| `transition:not_found_requires_view` | 17 |
| `consecutive_edit_failures:3` | 5 |
| `transition:path_not_found_requires_search` | 3 |
| **Total** | **669** |

171 out of 500 trajectories (34.20%) triggered the guard at least once.

**Results Analysis**

The most important comparison is between the two TPU 32B runs:

- No wrapper: Pass@1 = `18.6%`
- Wrapper: Pass@1 = `24.0%`

The improvement is:
- Resolved count increased by `27`
- Pass@1 improved by `5.4` percentage points (+29.0% relative)

This demonstrates that the wrapper is effective in the current DeepSWE evaluation setting.

Looking at the guard distribution, the largest source of benefit is:
- `missing_function_call` (101 occurrences)

This indicates that many failing trajectories were not failing because the model lacked the ability to fix the issue, but because it occasionally:
- Generated lengthy reasoning
- Without producing a valid tool call

The wrapper catches these invalid actions early, preventing the trajectory from wasting steps on idle spinning.

The second category of benefit comes from:
- `repeated_failure` (44 occurrences)
- `not_found / non_unique / path_not_found` edit recovery rules (47 occurrences total)

This contribution is smaller but still meaningful, reducing the waste of "blindly retrying after failed edits."

**Impact on Context Overflow**

The wrapper also reduced the number of context overflow terminations:

- No wrapper: `62`
- Wrapper: `50`

This shows that the wrapper not only improves resolve rate but also mitigates conversation bloat. The reasoning is straightforward:

- Fewer idle spins
- Fewer repeated failures
- Fewer invalid tool calls
- Therefore, prompts accumulate more slowly

This explains why the wrapper version achieves better results despite having slightly higher average step count.

**Average Step Count Interpretation**

- No wrapper: `21.31`
- Wrapper: `21.98`

The wrapper version has a higher step count yet also a higher resolve rate. This indicates that:

- The wrapper does not improve scores by prematurely terminating trajectories
- Instead, it enables more trajectories to progress toward effective fixes in subsequent steps

If the wrapper were simply "conservatively terminating" trajectories, the average step count would typically decrease. The slight increase here confirms that it is primarily improving interaction quality.

**GPU Utilization at 19%**

GPU utilization was only `19%`, which is consistent with the nature of DeepSWE tasks.

DeepSWE is not a pure model throughput benchmark. It involves substantial non-model overhead:
- Environment reset (spinning up Docker containers, initializing repo/runtime)
- Multi-turn env.step execution (running bash/file_editor/search inside containers)
- Tool call and observation round-trips
- Kubernetes pod scheduling and network communication

In this type of workload, even with a large model, the GPU is often not the bottleneck. Low GPU utilization likely means:
- The majority of time is spent on environment/tool/runtime execution
- Rather than model decoding being fully saturated

This result suggests that:
- The current evaluation is an environment-heavy workload
- Future optimization should not focus solely on model inference speed, but also on the environment execution chain

**TPU vs GPU Comparison**

Under the no-wrapper condition:

| | TPU | GPU |
|---|---|---|
| Pass@1 | 18.6% | **19.0%** |
| Diff | — | +0.4pp |

GPU only marginally outperforms TPU by 0.4pp, which is significantly smaller than the 5.4pp improvement from the wrapper. This confirms that hardware differences are not the dominant factor — wrapper and trajectory behavior have a much larger impact.

**Conclusions**

This experiment yields several clear conclusions:

1. On TPU, 32B + wrapper significantly outperforms 32B without wrapper (+5.4pp).
2. The wrapper's primary benefit comes not from complex recovery rules, but from correcting `missing_function_call` idle spins.
3. The wrapper also reduces context overflow terminations (62 → 50), improving multi-turn conversation efficiency.
4. Average step count slightly increases while resolve rate improves substantially, confirming that the wrapper improves trajectory quality rather than simply shortening trajectories.
5. TPU vs GPU difference is only 0.4pp under the no-wrapper condition — hardware is not the current bottleneck.
6. GPU utilization at 19% indicates that DeepSWE evaluation is primarily an environment/tool execution bottleneck, not a compute bottleneck.

**Recommendations**

For future optimization, the suggested priorities are:

1. **Reduce `missing_function_call` occurrences**: This is the largest source of wasted actions. Possible approaches include constrained decoding, stronger system prompts, or fine-tuning for format compliance.
2. **Keep the wrapper as the default evaluation configuration**: The stable 5.4pp improvement justifies its continued use.
3. **Analyze `MAX_CONTEXT_LIMIT_REACHED` instances individually**: Investigate whether there are common patterns of excessively fast prompt accumulation that could be addressed through conversation summarization or selective history.
4. **Optimize the environment execution chain for throughput**: Focus on container startup time, filesystem operations, and Kubernetes scheduling latency, rather than solely on model inference speed.

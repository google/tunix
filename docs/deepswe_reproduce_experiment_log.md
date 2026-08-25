# DeepSWE Reproduce Experiment Log

Status: draft

Date documented: 2026-08-25

Scope: DeepSWE V7x-128 reproduction, mismatch debugging, and truncation
analysis on Tunix agentic RL.

## 1. Executive Summary

This document records the DeepSWE reproduction effort as an iterative debugging
process. The goal was to reproduce the original V7x-128 DeepSWE behavior on
Tunix, then identify why training was unstable.

The debugging process found two major classes of issues.

| Issue class | Symptom | Main finding |
|---|---|---|
| Sampler/trainer mismatch | `logps_diff_mean` was much higher than the FrozenLake reference | Runtime scheduling and reduction behavior affected logprob alignment |
| DeepSWE truncation | High clip ratio and many `MAX_CONTEXT_LIMIT_REACHED` trajectories | Token and step budget metadata was not reaching `SWEAgent`, so the existing warning logic never fired |

After aligning runtime settings with the FrozenLake-style setup, logp and
probability diffs improved to a reasonable range. However, training still
became unstable around step 80. The investigation then shifted to clipping and
context-limit failures.

The truncation issue was traced to missing `cur_tokens` and `max_steps` fields
in the `info` dict passed to `SWEAgent.update_from_env()`. Adding those fields
made the token-warning path work. A stronger final-answer warning further
reduced early truncation and produced a strong curve for the first roughly 70
steps. The remaining issue is the later collapse, which appears more related
to policy/logprob alignment or off-policy behavior than to the original
token-warning bug.

## 2. Setup

| Field | Value |
|---|---|
| Environment | GKE DeepSWE environment setup |
| Source branch | [`deepswe-quality`](https://github.com/google/tunix/tree/deepswe-quality) |
| Primary workload | DeepSWE V7x-128 |
| Main model family | Qwen |
| Debug model sizes | Qwen 4B first, then Qwen 32B |
| Main references | FrozenLake logprob alignment behavior and rLLM DeepSWE agent behavior |

## 3. Initial Reproduction

The first step was to reproduce the original V7x-128 result.

| Trial | Commit | W&B run |
|---|---|---|
| Trial 1 | `1efb7e7` | `yqrpcx9z` |
| Trial 2 | `25e488c` | `yyy5j97p` |
| Trial 3 | `ca93505` | `e7qrmsmw` |

Initial observation:

```text
DeepSWE logps_diff_mean >> FrozenLake reference logps_diff_mean
```

This made sampler/trainer mismatch the first debugging target. The initial
hypothesis was that the rollout engine and trainer were not scoring the exact
same generated tokens in an aligned way.

## 4. Iterative Debugging Log

The sections below show the debugging process attempt by attempt. Each attempt
records the hypothesis, change, evidence, conclusion, and the next decision.

### Attempt 1: Debug Mismatch on a Smaller 4B Model

| Field | Value |
|---|---|
| Goal | Reproduce and debug the mismatch more cheaply |
| Model | Qwen 4B |
| Code | [`4bc9a829bb4af21b427452ed7cc85655770fcef4`](https://github.com/google/tunix/commit/4bc9a829bb4af21b427452ed7cc85655770fcef4) |
| W&B | [v9uqioki](https://wandb.ai/linchai-google/tunix-deepswe/runs/v9uqioki?nw=nwuserhaoyugao) |

Hypothesis:

The large mismatch may come from runtime differences between DeepSWE and the
FrozenLake-style setup, rather than from the DeepSWE reward or environment
alone.

Change:

The run added debugging information and aligned the following rollout settings
with the FrozenLake-style setup:

```json
{
  "rollout_vllm_async_scheduling": false,
  "rollout_vllm_init_with_random_weights": true,
  "enable_prefix_caching": false
}
```

Evidence:

The 4B run looked reasonable after the runtime alignment.

Conclusion:

The mismatch was likely sensitive to rollout runtime behavior. The result was
good enough to validate the same direction on the 32B model.

Next decision:

Move from Qwen 4B back to Qwen 32B and check whether the mismatch remains under
the real model size.

### Attempt 2: Switch Back to 32B with 24k Output Length

| Field | Value |
|---|---|
| Goal | Verify whether the mismatch fix transfers to the target model size |
| Model | Qwen 32B |
| Output length | 24k |
| W&B | [hhmhx613](https://wandb.ai/linchai-google/tunix-deepswe/runs/hhmhx613?nw=nwuserhaoyugao) |

Hypothesis:

If runtime settings were the main reason for mismatch, then the 32B run should
show lower logp and probability diffs after applying the same alignment.

Change:

Use the same runtime alignment and switch from Qwen 4B to Qwen 32B.

Evidence:

Compared with the previous version, both logp diff and probability diff became
more reasonable.

Conclusion:

The runtime alignment helped. However, training still became unsuccessful
around step 80.

Next decision:

Investigate metrics after step 80 to identify the next bottleneck.

### Attempt 3: Diagnose Step-80 Instability

| Field | Value |
|---|---|
| Goal | Understand why training still fails after mismatch improves |
| Model | Qwen 32B |
| Approximate failure point | Around step 80 |

Hypothesis:

The remaining failure may be caused by bad or filtered trajectories, not only
by sampler/trainer mismatch.

Evidence:

The run showed:

* high clip ratio;
* `min_length` almost 0 for many steps;
* many trajectories filtered out;
* upward trend in instability starting around step 80.

Comparison:

FrozenLake did not show the same pattern. Its clip ratio stayed near 0.

Conclusion:

The next issue was DeepSWE-specific trajectory clipping. DeepSWE has long
environment observations, repeated file views, and repository/tool output, so
it can hit context limits in a way FrozenLake does not.

Next decision:

Increase the response length to test whether the clipping is simply a token
budget issue.

### Attempt 4: Increase Response Length to 32k

| Field | Value |
|---|---|
| Goal | Test whether a larger response budget removes clipping |
| Model | Qwen 32B |
| Output length | 32k |
| W&B | [q8pghwlq](https://wandb.ai/linchai-google/tunix-deepswe/runs/q8pghwlq?nw=nwuserhaoyugao) |

Hypothesis:

If clipping is caused only by an insufficient token budget, increasing the
output length to 32k should significantly reduce `MAX_CONTEXT_LIMIT_REACHED`.

Evidence:

The issue remained. Logs still contained many lines like:

```text
WARNING:absl:[step_idx=23, pair_index=6, group_id=63] trajectory clipped: MAX_CONTEXT_LIMIT_REACHED
```

Conclusion:

The problem was not just the maximum response length. The agent was failing to
stop and submit before reaching the context limit.

Next decision:

Compare Tunix DeepSWE behavior with rLLM and inspect how the agent receives
token-budget and step-budget information.

### Attempt 5: Compare Tunix Warning Path with rLLM

| Field | Value |
|---|---|
| Goal | Explain why the DeepSWE agent did not stop before context exhaustion |
| Reference | rLLM AgentExecutionEngine |

Hypothesis:

Tunix may already have the right warning logic, but the warning may not be
triggered because required rollout metadata is missing.

Evidence:

`SWEAgent.update_from_env()` already had token-warning logic:

```python
cur_tokens = info.get("cur_tokens", None)
if cur_tokens is not None and cur_tokens >= TOKEN_WARNING_THRESHOLD:
  observation += "\nYou are running out of tokens. Please submit your answer NOW."
```

The threshold matched rLLM:

```python
TOKEN_WARNING_THRESHOLD = 28000
```

However, Tunix was calling `agent.update_from_env(...)` without adding
`cur_tokens` or `max_steps` to `info`. Therefore:

* `cur_tokens` was always `None`;
* the token warning never fired;
* the step-budget warning also could not work;
* the model kept exploring until context exhaustion.

rLLM injects these fields:

```python
info["max_steps"] = self.max_steps
info["cur_tokens"] = response_token_len
```

Conclusion:

The first real root cause was a framework metadata bug: the collector knew the
rollout state, but it did not pass that state to the agent.

Next decision:

Patch `TrajectoryCollectEngine` to inject engine-managed rollout state into the
`info` dict.

### Attempt 6: Add Rollout-state Metadata to Tunix

| Field | Value |
|---|---|
| Goal | Make the existing DeepSWE token and step warning logic active |
| Commits | `03a73a1f`, `c2fba4cd` |
| Code | [`c2fba4cda6c5de626f394c6cb11eec85dd85c6f0`](https://github.com/google/tunix/commit/c2fba4cda6c5de626f394c6cb11eec85dd85c6f0) |
| W&B | [ysvldfqv](https://wandb.ai/linchai-google/tunix-deepswe/runs/ysvldfqv?nw=nwuserhaoyugao) |

Change:

Add engine-managed rollout metadata inside `TrajectoryCollectEngine`:

```python
info["max_steps"] = self.max_steps
info["cur_tokens"] = self._response_token_count
```

The helper was renamed to `_rollout_state_info()` so the implementation stayed
generic and not SWE-specific.

Evidence:

The code path was fixed, but the run was still unsuccessful.

Conclusion:

The metadata bug was necessary to fix, but it was not sufficient. The next
question was whether the warning actually appeared in real trajectories and how
the model behaved after seeing it.

Next decision:

Run smaller and full-size follow-up experiments to inspect trajectory logs.

### Attempt 7: Verify Warning Injection on 4B and 32B

| Field | Value |
|---|---|
| 4B code | [`8ff2db8cf6f83e1dc7997f85830cd40f40555aa6`](https://github.com/google/tunix/commit/8ff2db8cf6f83e1dc7997f85830cd40f40555aa6) |
| 4B W&B | [fptm8fto](https://wandb.ai/linchai-google/tunix-deepswe/runs/fptm8fto?nw=nwuserhaoyugao) |
| 32B W&B | [vbl6ivbk](https://wandb.ai/linchai-google/tunix-deepswe/runs/vbl6ivbk?nw=nwuserhaoyugao) |
| Result | Both unsuccessful, but logs confirmed warning injection |

Hypothesis:

If metadata injection works, trajectory logs should show the warning once
`cur_tokens` crosses the threshold.

Evidence:

Around `cur_tokens=28546`, the observation included:

```text
You are running out of tokens. Please submit your answer NOW.
```

The warning was active, but the model still did not submit. Instead, logs
showed:

* repeated `file_editor view` calls over consecutive line ranges;
* line ranges around 1300 to 2150;
* assistant outputs around 51 tokens per step;
* environment observations around 650 to 800 tokens per step;
* plain natural-language summaries after the warning instead of valid tool
  calls;
* empty parsed actions such as:

```text
<function=>
</function>
```

Conclusion:

The first warning was too weak. It told the model to submit, but it did not
explicitly forbid more file viewing or provide a concrete valid final-tool
format. The agent could continue consuming context through many small view
operations.

Next decision:

Make the warning prescriptive: forbid more exploration and require an immediate
final tool call with exact XML syntax.

### Attempt 8: Strengthen the Final-answer Warning

| Field | Value |
|---|---|
| Goal | Force the model to stop exploring near the token limit |
| Code | [`ae22c203ad6a4ae164c39ccf7d3e97f116d56fbf`](https://github.com/google/tunix/commit/ae22c203ad6a4ae164c39ccf7d3e97f116d56fbf) |
| W&B | [9t4clomy](https://wandb.ai/linchai-google/tunix-deepswe/runs/9t4clomy?nw=nwuserhaoyugao) |

Change:

Replace the weak warning:

```text
You are running out of tokens. Please submit your answer NOW.
```

with a stronger instruction:

```text
You are running out of tokens. Stop exploring now. Do not call file_editor,
str_replace_editor, search, execute_bash, or any view command again. You must
immediately submit a final answer using the final tool.
```

The new warning also provided exact XML examples for `finish` and `submit`.

Evidence:

The first roughly 70 steps showed a strong curve, and the truncation issue
appeared to be solved in the early phase.

Conclusion:

The stronger warning fixed the immediate behavior near the context boundary.
The model was more likely to stop exploring and submit instead of continuing
file views.

New problem:

The curve collapsed after roughly step 70. This looked less like pure
truncation and more like policy/logprob alignment or off-policy behavior.

Next decision:

Return to sampler/trainer mismatch and investigate whether batch variance or
reduction ordering can explain the later collapse.

### Attempt 9: Stabilize Inference and Training Mismatch

| Field | Value |
|---|---|
| Goal | Remove batch-variance-driven logprob mismatch |
| Target model | Qwen 32B |
| Reported mismatch improvement | Around `4e-2` to `7.6e-7` |

Hypothesis:

The remaining mismatch may come from two independent sources of batch variance:
one on the inference side and one on the training side. Fixing only one side is
not enough.

Inference-side change:

* Pin attention block sizes.
* Make the split of a sequence across scheduler steps deterministic.
* Result: bitwise-identical logprobs across concurrency 1 to 8.

Training-side change:

* Express reductions as fixed-block `lax.fori_loop`.
* This prevents XLA from reordering or fusing reductions across iterations.
* Result: mismatch improved from around `4e-2` to `7.6e-7` on full 32B.
* No padding was required for this improvement.

Conclusion:

Sampler/trainer alignment is not only a tokenizer or prompt-template issue. It
also depends on deterministic scheduling and deterministic reduction order.

Next decision:

Run the 32B DeepSWE recipe with both the strong warning fix and the mismatch
stabilization changes. The key validation is whether the post-70 collapse
disappears.

## 5. Iteration Summary

| Attempt | Question | Result | Decision |
|---|---|---|---|
| Initial reproduction | Can we reproduce V7x-128? | Reproduction exposed large logp mismatch | Start with sampler/trainer mismatch |
| 1 | Does FrozenLake-style runtime alignment help on 4B? | 4B looked fine | Try 32B |
| 2 | Does alignment transfer to 32B? | Logp/prob diff improved | Investigate remaining step-80 failure |
| 3 | Is step-80 failure caused by bad trajectories? | High clip ratio and near-zero min length | Investigate truncation |
| 4 | Does 32k response length fix clipping? | No, context-limit clipping remained | Compare with rLLM warning path |
| 5 | Is the warning path inactive? | `cur_tokens` and `max_steps` were missing | Inject rollout metadata |
| 6 | Does metadata injection solve training? | Warning path enabled but still unsuccessful | Inspect behavior after warning |
| 7 | Does model submit after warning? | No, it kept viewing files or produced invalid actions | Make warning stronger |
| 8 | Does strong warning solve truncation? | Early curve improved for ~70 steps | Investigate later collapse |
| 9 | Is later collapse tied to mismatch? | Deterministic inference/training reduced mismatch strongly | Validate combined fix |

## 6. Current Understanding

The reproduction effort uncovered a layered failure pattern:

```text
initial symptom:
  DeepSWE training unstable

first bottleneck:
  sampler/trainer mismatch too high

second bottleneck:
  trajectories clipped by context limit

root cause of clipping:
  token and step metadata missing from agent info

third bottleneck:
  weak warning did not force final submission

remaining bottleneck:
  post-70 collapse likely tied to policy/logprob alignment or off-policy effects
```

The token-warning fixes address the trajectory truncation component. The
remaining priority is to validate whether deterministic inference/training
alignment is sufficient to prevent the later collapse.

## 7. Metrics to Compare Across Attempts

| Metric | Why it matters |
|---|---|
| `logps_diff_mean` | Measures sampler/trainer scoring mismatch |
| Probability diff | Confirms whether logprob mismatch affects probability agreement |
| Clip ratio | Detects truncation or mask filtering |
| `MAX_CONTEXT_LIMIT_REACHED` count | Direct signal for context-budget failures |
| Average response length | Shows whether the agent is over-exploring |
| Average environment token length | Captures file-view and tool-output accumulation |
| Warning activation count | Confirms how often token-budget warning triggers |
| Final tool-call compliance | Confirms whether the model submits correctly after warning |
| Reward / resolved rate | Main task-quality signal |
| Collapse step | Separates early truncation fixes from later policy instability |

## 8. Recommended Next Steps

1. Re-run the 32B DeepSWE experiment with the strong final-answer warning and
   the inference/training mismatch stabilization changes together.
2. Compare the first 70 steps and post-70 behavior against run `9t4clomy`.
3. Track `logps_diff_mean`, probability diff, clip ratio, resolved reward,
   completion length, and `MAX_CONTEXT_LIMIT_REACHED` together.
4. Verify whether collapse happens only when policy lag increases, or even when
   rollout/training are tightly synchronized.
5. Add explicit metrics for token-warning activation count and final-tool
   compliance rate.
6. Keep Qwen 4B as the fast debugging path, but validate final fixes on Qwen
   32B because long-context numerical mismatch is model-size sensitive.

## 9. Figure Placement Suggestions

Suggested images to include when presenting:

| Section | Image |
|---|---|
| Attempt 1 or 2 | Logp/prob diff before and after FrozenLake-style runtime alignment |
| Attempt 3 or 4 | Clip ratio and `min_length` showing trajectory filtering |
| Attempt 7 | Trajectory log showing token warning injection |
| Attempt 8 | Strong early curve before the post-70 collapse |
| Attempt 9 | Mismatch reduction from `4e-2` to `7.6e-7` |

Recommended paths:

```text
docs/images/deepswe_reproduce/logprob_diff_before_after.png
docs/images/deepswe_reproduce/clip_ratio_32k.png
docs/images/deepswe_reproduce/token_warning_triggered.png
docs/images/deepswe_reproduce/post_70_collapse.png
docs/images/deepswe_reproduce/mismatch_stabilization.png
```

## 10. Caveats

* Some records include W&B ids but not full launch commands.
* The FrozenLake comparison is a mismatch reference, not a directly comparable
  task-quality benchmark.
* "Off-policy behavior" is an interpretation of the post-70 collapse and
  should be validated with policy-version, old-logp, and sampler/trainer
  metrics.
* The final mismatch stabilization attempt should be updated with exact commit
  ids once they are available.

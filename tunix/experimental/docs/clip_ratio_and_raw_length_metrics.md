<!-- disableFinding(LINE_OVER_80) -->

# Response-Length Clip Ratio and Raw Length Metrics in Experimental RL

This document describes the architecture and design of the response-length
truncation (`generation/completions/clip_ratio`) and raw response length
(`generation/completions/{mean,max,min}_raw_length`) metrics in Tunix's
experimental RL orchestrator.

[TOC]

---

## 1. Overview & Core Motivation

Tunix maintains two RL training stacks:

1.  **Agentic Stack (`tunix/rl/agentic/`)**: The monolithic trainer
    (`GRPOLearner` in
    [agentic_grpo_learner.py](https://github.com/google/tunix/blob/main/rl/agentic/agentic_grpo_learner.py)),
    where rollout collection and advantage/metric computation execute in a
    single loop.
2.  **Experimental Stack (`tunix/experimental/`)**: The distributed
    producer-consumer architecture (`StandardRLProgram` in
    [rl_program.py](https://github.com/google/tunix/blob/main/experimental/orchestrator/rl_program.py)
    and `TrajectoryCollectorEngine` in
    [collector.py](https://github.com/google/tunix/blob/main/experimental/rollout/collector.py)),
    where asynchronous rollout workers stream trajectories through queues to
    trainer workers.

When workloads migrate from Agentic GRPO to Experimental RL, engineering
dashboards rely on four critical completion telemetry signals:

*   `generation/completions/clip_ratio`: The fraction of rollouts cut off
    mid-generation by `max_response_length` rather than stopping naturally on an
    End-of-Sequence (EOS) token.
*   `generation/completions/{mean,max,min}_raw_length`: The total number of
    response tokens generated after the initial prompt, clamped to
    `max_response_length`.

Previously, the experimental orchestrator only reported
`rollout/completion_length_mean`, which sums the assistant-only loss mask
(`completion_mask`). In multi-turn agentic environments (e.g., tool use, code
execution in DeepSWE), environment and tool observations (`env_tokens`) consume
context budget alongside assistant turns. Counting only assistant loss-mask
tokens undercounts actual context consumption and hides budget exhaustion.
Following the rLLM and VERL `response_length` convention, `raw_length` spans the
entire post-prompt token stream (`conversation_tokens`), including both
assistant and environment/tool turns.

---

## 2. Producer-Consumer Architecture & Division of Responsibility

In a decoupled producer-consumer system, computing `clip_ratio` requires
splitting responsibility between the **Rollout Worker (Producer)** and the
**Trainer Orchestrator (Consumer)**:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ PRODUCER: Rollout Worker                                                     │
│                                                                              │
│  RolloutManager ([manager.py])                                               │
│    │  Reads recipe-configured RolloutConfig.eos_tokens                       │
│    ▼                                                                         │
│  TrajectoryCollectorEngine._annotate_response_budget ([collector.py])        │
│    │  1. Reads per-request max_response_length & configured eos_ids          │
│    │  2. Calls response_budget_facts(tokens, budget, eos_ids)                │
│    │  3. Stamps per-rollout facts onto Token-mode trajectory dict:           │
│    │       traj["raw_length"] = min(len(tokens), max_response_length)        │
│    │       traj["clipped"]    = (len >= budget) and (tokens[-1] not in eos)  │
└────┼─────────────────────────────────────────────────────────────────────────┘
     │
     │  TrajectoryItem streamed via TrajectoryQueueManager
     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ CONSUMER: Trainer Orchestrator                                               │
│                                                                              │
│  StandardRLProgram.run_async ([rl_program.py])                               │
│    │  Pulls B = full_batch_size prompt groups via scored_q.get_batch(1)      │
│    │  uncommitted_groups = [group_1, group_2, ..., group_B]                  │
│    ▼                                                                         │
│  _generation_metrics(uncommitted_groups)                                     │
│    │  Reduces per-rollout facts within each prompt group g, then aggregates  │
│    │  across all B prompt groups in the global training step                 │
│    ▼                                                                         │
│  MetricsLogger.log("generation/completions/{clip_ratio,mean_raw_length,...}")│
└──────────────────────────────────────────────────────────────────────────────┘
```

### Why the Producer Must Annotate Each Rollout

The consumer (`StandardRLProgram`) cannot derive `clipped` or `raw_length` from
`TrajectoryItem` alone because **only the producer knows the budget and stop set
that were actually enforced**:

1.  **Per-Request Budget Overrides**: While `StandardRLProgram` holds a default
    `generation_args.max_response_length`, `DistributedRLEngine` allows
    individual dataset items to override `max_response_length` inside
    `request.generation_kwargs`. Scoring rollouts against the program-level
    default at the consumer would compare overridden rollouts against a
    threshold that never applied to them.
2.  **Recipe-Configured Stop Sets**: Rollout workers are launched with custom
    stop tokens via `--eos_tokens` (stored in `RolloutConfig.eos_tokens`), which
    the trainer process does not necessarily hold.

### Why the Consumer Must Aggregate at the `get_batch(num_groups=1)` Boundary

The producer (`TrajectoryCollectorEngine`) only processes **one rollout at a
time**, whereas GRPO's `clip_ratio` and `mean_raw_length` are defined as
**per-prompt-group averages**. In `StandardRLProgram.run_async`,
`scored_q.get_batch(num_groups=1)` is the last boundary where rollouts are still
grouped by prompt (`uncommitted_groups`) before being flattened into
`all_step_items` and packed into hardware microbatches.

---

## 3. Scoring Formula & Recipe-Configured Stop Sets

### Per-Rollout Scoring: `response_budget_facts`

In [collector.py](https://github.com/google/tunix/blob/main/experimental/rollout/collector.py),
the per-rollout scoring formula is encapsulated in `response_budget_facts`:

```python
def response_budget_facts(
    response_tokens: Sequence[int] | np.ndarray,
    max_response_length: int,
    eos_ids: Collection[int],
) -> tuple[int, bool]:
  raw_length = len(response_tokens)
  stopped_on_eos = raw_length > 0 and int(response_tokens[-1]) in eos_ids
  clipped = raw_length >= max_response_length and not stopped_on_eos
  return min(raw_length, max_response_length), clipped
```

`TrajectoryCollectorEngine._annotate_response_budget` delegates to this helper.
*(Note: To isolate blast radius, `cl/981373441` scopes changes strictly to
`tunix/experimental/`. Unifying `GRPOLearner._process_results` in
`tunix/rl/agentic/agentic_grpo_learner.py` with this helper and plumbing
configured stop sets to the monolithic agentic stack is tracked as a follow-up
CL.)*

### Recipe-Configured Stop Sets (`RolloutConfig.eos_tokens`) vs Tokenizer Defaults

A rollout is truncated (`clipped = True`) if and only if it exhausted
`max_response_length` **without** emitting a configured stop token as its final
token.

Crucially, termination must be checked via **set membership (`in eos_ids`)**
against the sampler's configured stop set (`RolloutConfig.eos_tokens`), and
**the framework must not force or guess `tokenizer.eos_token_id`**:

*   **Why EOS Belongs at the Recipe Level**: A tokenizer's `eos_token_id` is a
    static vocabulary property from pretraining representing *end-of-document*
    (e.g., `<|endoftext|>` = `151643` in Qwen, `<eos>` = `1` in Gemma). In RL
    post-training (chat, multi-turn tool use, DeepSWE), turn termination is
    dictated by the **recipe and chat template** (`<|im_end|>` = `151645`,
    `<end_of_turn>` = `106`, or custom tool stop tokens)—not by the base
    tokenizer.
*   **No Framework-Level Tokenizer Fallback**: If the framework silently fell
    back to `tokenizer.eos_token_id` whenever `RolloutConfig.eos_tokens` was
    unset, forgetting `eos_tokens` on a Qwen chat run would silently evaluate
    `clip_ratio` against `<|endoftext|>` (`151643`), falsely marking every
    normal completion at budget $N$ as clipped without warning. Therefore,
    `TrajectoryCollectorEngine` relies **strictly** on `eos_ids` passed from
    `RolloutConfig.eos_tokens`. If `eos_ids` is unset or empty,
    `_annotate_response_budget` emits a one-shot warning
    (`logging.log_first_n(logging.WARNING, ...)`) and leaves the trajectory
    unannotated.

---

## 4. Why `TrajectoryStatus.MAX_CONTEXT_LIMIT_REACHED` Is Not Reused

`TrajectoryCollectEngine` sets
`traj["status"] = TrajectoryStatus.MAX_CONTEXT_LIMIT_REACHED` during rollout
collection. However, `clip_ratio` deliberately inspects the final token stream
via `response_budget_facts` rather than reading `traj["status"]`, for three
reasons:

| Dimension | `TrajectoryStatus.MAX_CONTEXT_LIMIT_REACHED` | `response_budget_facts` (`clip_ratio`) |
| :--- | :--- | :--- |
| **EOS Awareness** | **Blind to EOS.** [`_check_and_set_context_limit_reached`](https://github.com/google/tunix/blob/main/rl/agentic/trajectory/trajectory_collect_engine.py) only checks `_response_token_count >= max_response_length`. If a model naturally finishes on exact token $N = \text{max\_response\_length}$ with `<|im_end|>`, status is marked `MAX_CONTEXT_LIMIT_REACHED` (false positive). | **EOS-aware.** Explicitly checks `int(response_tokens[-1]) in eos_ids`. Finishing on EOS at exact budget $N$ is scored as `clipped = False`. |
| **Overwrite Vulnerability** | **Mutable single enum.** In `_one_step()`, if a step hits `max_response_length` at line 573 and also crosses the wall-clock episode timeout at line 673 (`step_timed_out`), `status` is overwritten with `TrajectoryStatus.TIMEOUT`, losing the truncation signal. | **Immutable token fact.** Computed directly from the final collected `conversation_tokens` array and the enforced budget. |
| **Agentic Parity** | **Ignored by Agentic GRPO.** `GRPOLearner._process_results` does not read `traj["status"]` when computing `generation/completions/clip_ratio`. | **Exact parity.** Uses the identical token-level check in both Agentic and Experimental stacks. |

---

## 5. Global Batch Aggregation & Per-Group Reduction Math

In `StandardRLProgram`, a single training step (one global batch) consumes
$B = \text{full\_batch\_size}$ prompt groups, where each prompt group
$g \in \{1, \dots, B\}$ contains $G_g$ annotated rollouts (nominally
$G = \text{group\_size}$).

To match `GRPOLearner._process_results` (which logs per-group metrics via
`buffer_metrics_async` with `np.mean`, `np.max`, and `np.min` reduction ops
across the global step), `_generation_metrics(uncommitted_groups)` performs a
**two-stage reduction**:

| Metric Name | Stage 1: Within Each Prompt Group $g \in \{1 \dots B\}$ | Stage 2: Across All $B$ Groups in Global Batch | Rationale |
| :--- | :--- | :--- | :--- |
| **`generation/completions/clip_ratio`** | Group clip ratio:<br>$$r_g = \frac{1}{G_g} \sum_{i=1}^{G_g} \mathbb{I}[\text{clipped}_{g,i}]$$ | Mean of group ratios:<br>$$\text{clip\_ratio} = \frac{1}{B} \sum_{g=1}^{B} r_g$$ | Ensures every prompt weighs equally ($1/B$) even if a group has fewer valid rollouts ($G_g < G$). Pooling rollouts globally ($\sum \text{clipped} / \sum G_g$) would down-weight smaller groups. |
| **`generation/completions/mean_raw_length`** | Group mean raw length:<br>$$\mu_g = \frac{1}{G_g} \sum_{i=1}^{G_g} \text{raw\_length}_{g,i}$$ | Mean of group means:<br>$$\text{mean\_raw\_length} = \frac{1}{B} \sum_{g=1}^{B} \mu_g$$ | Same per-prompt equal weighting as `clip_ratio`. |
| **`generation/completions/max_raw_length`** | Group max:<br>$$M_g = \max_{i} \text{raw\_length}_{g,i}$$ | Global max across batch:<br>$$\max_{g=1 \dots B} M_g = \max_{g,i} \text{raw\_length}_{g,i}$$ | Associative and grouping-invariant. |
| **`generation/completions/min_raw_length`** | Group min:<br>$$m_g = \min_{i} \text{raw\_length}_{g,i}$$ | Global min across batch:<br>$$\min_{g=1 \dots B} m_g = \min_{g,i} \text{raw\_length}_{g,i}$$ | Associative and grouping-invariant. |

### Edge Cases & Precondition Logging

*   **Zero-length responses (`len(tokens) == 0`)**: Annotated with
    `raw_length = 0, clipped = False` and **kept in the group denominator
    $G_g$**. A rollout that executed and produced zero tokens is still an
    attempted rollout; dropping it would artificially inflate `clip_ratio` for
    the remaining rollouts in that group.
*   **Unannotated trajectories & One-shot Warnings**: If a trajectory lacks
    annotations (e.g., `max_response_length` was `None` or non-positive,
    `RolloutConfig.eos_tokens` was unset/empty, or `conversation_tokens` was
    missing), it is excluded from $G_g$ rather than counted as `clipped = False`.
    To prevent silent metric disappearance on dashboards,
    `_annotate_response_budget` emits a one-shot
    `logging.log_first_n(logging.WARNING, ...)` naming the exact failed
    precondition.
*   **Half-annotated trajectories**: `_generation_metrics` guards both
    `"clipped" in traj` and `"raw_length" in traj` so a malformed producer
    payload skips cleanly instead of raising `KeyError` during training.

---

## 6. Summary of Code Changes

*   **[`tunix/experimental/rollout/manager.py`](https://github.com/google/tunix/blob/main/experimental/rollout/manager.py)**:
    Extracted `self.eos_ids` from `RolloutConfig.eos_tokens` and plumbed it into
    `TrajectoryCollectorEngine`.
*   **[`tunix/experimental/rollout/collector.py`](https://github.com/google/tunix/blob/main/experimental/rollout/collector.py)**:
    Added `response_budget_facts(response_tokens, max_response_length, eos_ids)`,
    `eos_ids` parameter, one-shot warning logs for missing preconditions, and
    `_annotate_response_budget`.
*   **[`tunix/experimental/orchestrator/rl_program.py`](https://github.com/google/tunix/blob/main/experimental/orchestrator/rl_program.py)**:
    Added `_generation_metrics(uncommitted_groups)` returning fully-qualified
    `generation/completions/*` metric names prior to `scored_q.commit()`, and
    logged them in `_collect_and_log_step_metrics`.
*   **Unit Tests**: Added comprehensive coverage in
    [`collector_test.py`](https://github.com/google/tunix/blob/main/experimental/rollout/collector_test.py),
    [`manager_test.py`](https://github.com/google/tunix/blob/main/experimental/rollout/manager_test.py),
    and
    [`rl_program_test.py`](https://github.com/google/tunix/blob/main/experimental/orchestrator/rl_program_test.py).
*   **Follow-up Work**: Unify `GRPOLearner._process_results` in
    [`agentic_grpo_learner.py`](https://github.com/google/tunix/blob/main/rl/agentic/agentic_grpo_learner.py)
    with `response_budget_facts` in a separate CL.

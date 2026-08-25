# DeepSWE Pre-training Evaluation Results

Status: draft

Date documented: 2026-08-25

Scope: DeepSWE evaluation before RL training

## Summary

This document records pre-training DeepSWE evaluation results for Qwen3 32B.
The main comparison is between a baseline 32B run without the action wrapper and
a 32B run with the wrapper enabled.

The wrapper run improved Pass@1 from `0.1860` to `0.2100`, corresponding to 12
additional resolved instances out of 500. It also reduced
`MAX_CONTEXT_LIMIT_REACHED` trajectories from 62 to 50. The wrapper introduced
guard intervention on 34.20% of trajectories, with `missing_function_call` as
the most common guard reason.

Two additional unlabeled runs are included for completeness. Their exact setup
needs confirmation before they should be used for final comparison.

## Evaluation Setup

| Field | Value |
|---|---|
| Task | DeepSWE |
| Phase | Before RL training |
| Model family | Qwen3 |
| Main model size | 32B |
| Evaluation size | 500 instances |
| Metric | Pass@1 / resolved count |
| Reward interpretation | Average reward equals Pass@1 in these runs |

Security note: an API bearer token was present in the raw notes. It is
intentionally omitted from this document. That token should be rotated or
revoked if it was a live credential.

## Main Results

| Run | Total | Resolved | Pass@1 | Avg reward | Avg steps | Succeeded | Max context | Guarded trajectories | Guard blocks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32B no wrapper | 500 | 93 | 0.1860 | 0.1860 | 21.31 | 438 | 62 | 0 / 500 (0.00%) | 0 |
| 32B wrapper | 500 | 105 | 0.2100 | 0.2100 | 21.98 | 450 | 50 | 171 / 500 (34.20%) | 669 |

## Main Comparison

| Metric | 32B no wrapper | 32B wrapper | Delta |
|---|---:|---:|---:|
| Resolved | 93 | 105 | +12 |
| Pass@1 | 0.1860 | 0.2100 | +0.0240 |
| Avg reward | 0.1860 | 0.2100 | +0.0240 |
| Avg steps | 21.31 | 21.98 | +0.67 |
| `SUCCEEDED` status | 438 | 450 | +12 |
| `MAX_CONTEXT_LIMIT_REACHED` status | 62 | 50 | -12 |
| Guarded trajectories | 0 | 171 | +171 |
| Guard blocks | 0 | 669 | +669 |

## Wrapper Guard Breakdown

Guard reasons for the 32B wrapper run:

| Guard reason | Count |
|---|---:|
| `missing_function_call` | 101 |
| `repeated_failure` | 44 |
| `transition:non_unique_requires_view` | 27 |
| `transition:not_found_requires_view` | 17 |
| `consecutive_edit_failures:3` | 5 |
| `transition:path_not_found_requires_search` | 3 |

The guard reason counts sum to more than the number of guarded trajectories
because a single trajectory can trigger multiple guard blocks.

## Additional Unlabeled Runs

The following two runs were present in the raw notes but did not include a clear
run label. They are recorded here without drawing conclusions from them.

| Run label | Total | Resolved | Pass@1 | Avg reward | Avg steps | Succeeded | Max context | Guarded trajectories | Guard blocks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Unlabeled run A | 500 | 4 | 0.0080 | 0.0080 | 14.20 | 475 | 25 | 0 / 500 (0.00%) | 0 |
| Unlabeled run B | 500 | 3 | 0.0060 | 0.0060 | 14.13 | 482 | 18 | 26 / 500 (5.20%) | 179 |

Guard reasons for unlabeled run B:

| Guard reason | Count |
|---|---:|
| `missing_function_call` | 25 |
| `consecutive_edit_failures:3` | 1 |

## Notes on OpenAPI / Server Path

The raw notes mention an OpenAPI/server-based Qwen3 32B path and approximately
23% GPU utilization. The associated bearer credential is not included here.

Before comparing the OpenAPI/server path against the main local 32B runs, the
following fields should be confirmed:

* whether unlabeled run A and unlabeled run B correspond to the OpenAPI/server
  path;
* whether the same prompt template, tool/action wrapper, decoding config, and
  max context settings were used;
* whether the same evaluation dataset split and 500-instance subset were used;
* whether the lower average step count reflects earlier termination, different
  max-turn settings, or different action parsing behavior.

## Preliminary Interpretation

The wrapper appears beneficial in the main 32B comparison:

* Pass@1 improves by 2.4 percentage points, from 18.60% to 21.00%.
* Resolved instances increase by 12 out of 500.
* Context-limit failures decrease by 12.
* Guard interventions are common, especially for missing function calls and
  repeated failures, suggesting the wrapper is actively correcting invalid or
  low-quality action patterns.

The unlabeled runs should not be mixed into the main conclusion until their
configuration is confirmed. Their Pass@1 values are much lower than the main
runs despite fewer context-limit statuses, which likely indicates a meaningful
setup difference rather than normal evaluation variance.

## Raw Result Blocks

### 32B No Wrapper

```text
Total instances:  500
Resolved:         93
Pass@1:           0.1860
Avg reward:       0.1860
Avg steps:        21.31
Status counts:    {'SUCCEEDED': 438, 'MAX_CONTEXT_LIMIT_REACHED': 62}
Guarded trajs:    0/500 (0.00%)
Guard blocks:     0
```

### 32B Wrapper

```text
Total instances:  500
Resolved:         105
Pass@1:           0.2100
Avg reward:       0.2100
Avg steps:        21.98
Status counts:    {'SUCCEEDED': 450, 'MAX_CONTEXT_LIMIT_REACHED': 50}
Guarded trajs:    171/500 (34.20%)
Guard blocks:     669
Guard reasons:    {'missing_function_call': 101, 'repeated_failure': 44, 'transition:not_found_requires_view': 17, 'transition:non_unique_requires_view': 27, 'consecutive_edit_failures:3': 5, 'transition:path_not_found_requires_search': 3}
```

### Unlabeled Run A

```text
Total instances:  500
Resolved:         4
Pass@1:           0.0080
Avg reward:       0.0080
Avg steps:        14.20
Status counts:    {'SUCCEEDED': 475, 'MAX_CONTEXT_LIMIT_REACHED': 25}
Guarded trajs:    0/500 (0.00%)
Guard blocks:     0
```

### Unlabeled Run B

```text
Total instances:  500
Resolved:         3
Pass@1:           0.0060
Avg reward:       0.0060
Avg steps:        14.13
Status counts:    {'SUCCEEDED': 482, 'MAX_CONTEXT_LIMIT_REACHED': 18}
Guarded trajs:    26/500 (5.20%)
Guard blocks:     179
Guard reasons:    {'missing_function_call': 25, 'consecutive_edit_failures:3': 1}
```

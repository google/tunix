# MLPerf offline evaluation recipes

These recipes evaluate the checkpoints of an MLPerf DeepSWE RL training run
(pass@4 on the validation split) and write the MLPerf compliance events into
that run's MLLOG, so the training log on its own is the submission log.

| Recipe | Model | Rollout topology |
| :--- | :--- | :--- |
| [`mlperf_397b_v7x_eval.sh`](mlperf_397b_v7x_eval.sh) | Qwen3.5-397B-A17B | `16x tpu7x:2x2x4` (256 chips), pod1 or pod2 |
| [`mlperf_397b_v5p_eval.sh`](mlperf_397b_v5p_eval.sh) | Qwen3.5-397B-A17B | `tpuv5p:2x2x4` replicas, `bodaborg-v5p-nap` |
| [`mlperf_35b_v5p_eval.sh`](mlperf_35b_v5p_eval.sh) | Qwen3.5-35B-A3B | `tpuv5:2x2x1` replicas |

All three source [`../mlperf_base.sh`](../mlperf_base.sh), which holds the
evaluation loop, and [`rcp_eval.py`](rcp_eval.py), a stdlib-only helper that
reads the manifest and MLLOG.

## Evaluate a training run

A training run with `RCP_LOGGING=true` writes two files next to each other:

*   `<METRIC_LOGGER_DIR>/seed_<seed>.out`: the MLLOG.
*   `<METRIC_LOGGER_DIR>/eval_checkpoints.jsonl`: one record per checkpoint,
    with its path, `samples_count` and `timestamp_ms`. The record is appended
    when `save_checkpoint` returns, so a run that dies mid-save leaves a final
    checkpoint dir with no record; `inspect` reports those as "Unrecorded".

Point the recipe at the manifest:

```bash
# Check what the loop will see (read-only).
python3 tunix/experimental/examples/recipes/evals/rcp_eval.py inspect \
  --manifest gs://atwigg-trellis-us-east1/maxtext/<run>/mllog/eval_checkpoints.jsonl

# Evaluate on v7x pod2 (us-east1). The cluster must be able to read the
# checkpoints and write the MLLOG. PREEMPTIBLE=true helps the JobSet get
# admitted in priority-dev; during a reservation window set
# K8S_NAMESPACE=priority-dev-scheduled instead.
CHECKPOINT_MANIFEST_FILE=gs://atwigg-trellis-us-east1/maxtext/<run>/mllog/eval_checkpoints.jsonl \
JOB_PREFIX=${USER}-ev POD=pod2 PREEMPTIBLE=true \
  bash tunix/experimental/examples/recipes/evals/mlperf_397b_v7x_eval.sh eval
```

Without `CHECKPOINT_MANIFEST_FILE`, `eval` runs one evaluation of
`MAXTEXT_CKPT` and does not touch any MLLOG.

### What the loop does

1.  `rcp_eval.py plan` reads the manifest and MLLOG and lists the checkpoints
    still to evaluate. It evaluates all checkpoints recorded in the manifest
    sequentially in step order. It stops with an error if:
    *   the manifest steps are not contiguous;
    *   the MLLOG already has `eval_accuracy` for a later step than one that
        is missing.

    Records older than the last `run_start` (left over from an earlier,
    restarted run) are ignored. Steps that already have `eval_accuracy` are
    skipped, so rerunning the same command resumes. If `run_stop` is already
    logged, there is nothing to do.
2.  For each checkpoint in step order, it launches the eval JobSet with
    `RCP_LOGGING=true`, `METRIC_LOGGER_DIR=<MLLOG>`, `CHECKPOINT_STEP`,
    `SAMPLES_COUNT`, `CHECKPOINT_TIMESTAMP_MS` and `IS_LAST_CHECKPOINT`, and
    waits for the main container to finish. The container appends
    `eval_start`, `eval_accuracy` and `eval_stop` to the MLLOG, then:
    *   `run_stop(status=success)` if pass@4 >= `TARGET_ACCURACY` (0.69);
    *   `run_stop(status=aborted)` if this is the run's final step
        (`step == max_steps`) and the target was missed.
3.  Whether a checkpoint was evaluated is read from the MLLOG, not from the
    container's exit code. A checkpoint without `eval_accuracy` is retried up
    to `MAX_EVAL_ATTEMPTS` times; after that the loop exits 1 instead of
    skipping it (a skipped checkpoint would make a later one look like the
    first to reach the target). Fix the cause and rerun the same command.
4.  It stops at the first `run_stop`, prints the final state (`inspect`) and
    runs the MLPerf compliance checker on the MLLOG.

If training stopped before `max_steps` (preempted or killed) and no evaluated
checkpoint reaches the target, the loop evaluates every checkpoint in the
manifest and leaves the MLLOG without `run_stop`; the compliance check then
fails with `Required EXACTLY_ONE occurrence of 'run_stop'`. Such a run is
incomplete and cannot be submitted.

### Timing

`run_stop` is backdated to the checkpoint's manifest `timestamp_ms`. The
trainer takes that timestamp right after the optimizer update and before the
checkpoint is written, so time to train (`run_stop - run_start`) excludes the
writing of the final checkpoint and all offline evaluation. The `eval_*`
events keep their wall-clock times.

### Requirements

*   The eval image (`TUNIX_IMAGE`) must include
    [google/tunix#2558](https://github.com/google/tunix/pull/2558): the eval
    container appends to the existing MLLOG instead of replacing it, and
    closes a training block left open by a preempted run before
    `eval_start`.
*   Run evaluation after training has stopped. Don't run two evaluators on
    the same run at once.
*   `check` and `rcp` need `mlperf_logging`
    (`pip install git+https://github.com/mlcommons/logging.git`); set
    `RCP_PYTHON` to a Python that has it.

### Options

| Variable | Default | Meaning |
| :--- | :--- | :--- |
| `CHECKPOINT_MANIFEST_FILE` | unset | Manifest to evaluate; enables the loop. |
| `MAX_EVAL_ATTEMPTS` | `3` | Attempts per checkpoint before the loop exits 1. |
| `EVAL_RETRY_DELAY_SECS` | `30` | Wait before retrying after a failed JobSet launch. |
| `RUN_COMPLIANCE_CHECK` | `true` | Run the compliance checker at the end. |
| `MLPERF_RULESET` | `6.1.0` | Ruleset for `check` and `rcp`. |
| `RCP_PYTHON` | `python3` | Python used for `rcp_eval.py`. |
| `EVAL_OUTPUT_DIR` | recipe default | Per-step results go to `<dir>/step_<N>`. |
| `DRY_RUN` / `--dry-run` | `false` | Render the JobSets for every planned step without applying. |
| `K8S_NAMESPACE` | recipe default (`priority-dev` on v7x) | Namespace for the eval JobSets, e.g. `priority-dev-scheduled` during a reservation. |
| `PREEMPTIBLE` | `false` | Mark the eval JobSets preemptible; recommended in `priority-dev`. |

## Validate a log

```bash
H=tunix/experimental/examples/recipes/evals/rcp_eval.py
M=gs://atwigg-trellis-us-east1/maxtext/<run>/mllog/eval_checkpoints.jsonl

python3 $H inspect --manifest $M  # per-step reward / pass@4, time to train
python3 $H check --manifest $M    # mlperf_logging.compliance_checker
python3 $H rcp --results_dir <dir with result_*.txt from several seeds>
```

`rcp --manifest $M` (or `--mllog`) copies the one log three times and runs the
RCP checker on it. That only shows the seed does not converge earlier than the
reference allows; it is not an RCP result for a submission.

# Distributed vs. agentic RL benchmark

Compare the distributed and ordinary agentic stacks on the same FrozenLake
training recipe with `google/gemma-4-E2B-it`. The benchmark generates matched
configurations for both stacks.

## Prerequisites

- Run from the repository root in Bash, using the Python environment that can
  run both training stacks: Tunix, JAX/libtpu, vLLM/tpu-inference, and Raiden.
  The distributed setup is described in [FrozenLake](../frozenlake_dist/README.md).
- Use one dedicated four-chip TPU host. By default, agentic training and rollout
  share all four chips. `--agentic-chip-layout split2x2` gives the actor and
  rollout separate two-chip meshes, matching the distributed chip allocation.
  Run the stacks sequentially on the same host.
- Download the complete Gemma checkpoint and tokenizer beforehand. `--model-dir`
  must contain the safetensors shards, `config.json`, and tokenizer files.
- Keep the code, dependencies, model files, hardware, and workload flags unchanged
  between runs being compared. Every run needs a **new output directory**.

The distributed runner requires a Raiden build whose
`RaidenController.register_work_unit` accepts `host_subgrid`. An older Raiden
checkout can report a completed transfer while leaving rollout weights
unchanged. Check the imported module path if a local checkout appears before
the installed package on `PYTHONPATH`; the benchmark rejects that older API.
For a one-time transfer check, run the distributed command with
`VERIFY_WEIGHTS=true` and compare `source checksums` in `trainer.log` with
`destination checksums` in `rollout.log`. Leave this unset for timed runs.

The commands below use the active training environment's `python3`. Replace the
model path and hardware label with your actual values. `--hardware` records a
label; it does not provision or select a TPU.

```bash
set -euo pipefail
BENCH_PYTHON=python3
BENCH_SCRIPT=tunix/experimental/examples/rl_efficiency_benchmark/frozenlake.py
BENCH_MODEL_DIR=/absolute/path/to/gemma-4-E2B-it
BENCH_HARDWARE=v5p-4
BENCH_RUN_ROOT="$PWD/artifacts/rl_efficiency/$(date -u +%Y%m%dT%H%M%SZ)"
BENCH_CACHE_ROOT="$PWD/artifacts/rl_efficiency_cache"

bench_common=(
  --model-dir "$BENCH_MODEL_DIR"
  --cache-root "$BENCH_CACHE_ROOT"
  --hardware "$BENCH_HARDWARE"
  --steps 150
  --warmup 5
  --batch 64
  --generations 8
  --micro-groups 1
  --agentic-chip-layout split2x2
  --match-training-microbatch
  --prompt 2048
  --response 2048
  --turns 8
  --concurrency 512
  --seed 42
  --timeout 172800
)
```

This measures the first 150 steps of the Gemma4 training recipe and excludes
the first 5 from the comparison. `--batch 64` is the full batch size in prompt
groups, not trajectories. Each global step samples 8 trajectories per group,
so both stacks process 512 trajectories before one optimizer update. With the
alignment flags above, agentic uses a two-chip actor and a separate two-chip
TP2 rollout. One prompt group per agentic microbatch means 8
trajectories. The distributed trainer and its log-prob path each use 8
trajectories per call, producing 64 calls per step on both sides. The report
verifies actual per-call shapes and call counts. Without
`--match-training-microbatch`, the distributed recipe retains its native
two-trajectory microbatch; that is a separate experiment.
On the measured two-chip distributed trainer, 16 trajectories per call
failed with an XLA out-of-memory error; eight per call completed.
Both stacks use BF16 compute, FP32 checkpoint loading, and exact token
continuity. The distributed launcher normally defaults to FP32 compute; this
benchmark overrides it to BF16 so both stacks train with the same precision.
The benchmark reserves 512 response tokens for the final FrozenLake observation
and chat-template boundary, so `--response` must exceed 512. Both paths use
the same reserve; training still pads to the requested `--response` length.
At full length, an epoch can take longer than a day on this hardware; the
48-hour per-run timeout accommodates it. Use a smaller `--steps` value for a
correctness check, then run enough steps for stable timing measurements.

## Preview the configuration

Without `--execute`, the command writes the manifest and generated configuration
and prints the launch command. It does not start training. Preview directories
cannot be reused as execution output directories or included in a report.

```bash
for stack in agentic dist; do
  "$BENCH_PYTHON" "$BENCH_SCRIPT" run \
    "${bench_common[@]}" \
    --mode "$stack" --timing-mode stages \
    --output "$BENCH_RUN_ROOT/plan-$stack"
done
```

## Compare the large modules

Use `stages` for collection, actor log-probs, training, and weight sync. This is
the default timing mode. Completion waits are included in module timings;
distributed training waits inside the trainer worker before returning to the
orchestrator. The distributed trace also records `trainer_worker_per_token_logps`,
`trainer_worker_fwd_bwd`, and `trainer_worker_update` spans inside their
corresponding `worker_rpc_*` spans. Each worker span includes its device
completion wait. Subtracting the worker span from the enclosing RPC span
estimates time outside the worker, including dispatch and response handling;
neither span is an isolated TPU kernel time.

```bash
for stack in agentic dist; do
  "$BENCH_PYTHON" "$BENCH_SCRIPT" run \
    "${bench_common[@]}" \
    --mode "$stack" --timing-mode stages \
    --output "$BENCH_RUN_ROOT/stages-$stack" --execute
done

"$BENCH_PYTHON" "$BENCH_SCRIPT" report \
  "$BENCH_RUN_ROOT/stages-agentic" \
  "$BENCH_RUN_ROOT/stages-dist" \
  > "$BENCH_RUN_ROOT/stages-comparison.json"
```

Read `pipeline_stages_seconds_per_step` in the comparison:

| Field | Timing boundary |
| --- | --- |
| `rollout_collection` | First trajectory starts through last trajectory completes, including reward and environment cleanup. Excludes dispatch before this interval and result delivery afterward. |
| `actor_log_probs` | Actor log-probability results are ready. |
| `training` | Forward/backward, gradient accumulation, and optimizer work complete; the relevant training state is ready. |
| `weight_sync` | Weight transfer and installation on the rollout side complete. |

These are mean seconds per full training step, including framework and RPC
overhead. They are not isolated kernel durations, and the four modules do not
account for every preprocessing or scheduling gap in a step. Nested API spans
must not be added to the module totals.

## Compare end-to-end pipeline time

Use a separate `pipeline` run to preserve intra-step asynchronous overlap.
Rollout weight completion still defines the full-step boundary. This mode does
not rank unfenced training API durations as completed module timings.

```bash
for stack in agentic dist; do
  "$BENCH_PYTHON" "$BENCH_SCRIPT" run \
    "${bench_common[@]}" \
    --mode "$stack" --timing-mode pipeline \
    --output "$BENCH_RUN_ROOT/pipeline-$stack" --execute
done

"$BENCH_PYTHON" "$BENCH_SCRIPT" report \
  "$BENCH_RUN_ROOT/pipeline-agentic" \
  "$BENCH_RUN_ROOT/pipeline-dist" \
  > "$BENCH_RUN_ROOT/pipeline-comparison.json"
```

Read `metrics.end_to_end_step_time_seconds`. Stage-mode fences alter overlap, so
its full-step time describes synchronized execution. Reports reject mixed
`stages` and `pipeline` runs.

## Repeat for a more reliable comparison

A single pair does not establish run-to-run variability. The following runs
three additional pairs in alternating order, retaining each stack's compilation
cache. Use `pipeline` instead of `stages` to repeat the end-to-end experiment.

```bash
BENCH_TIMING=stages
bench_runs=()
for repeat in 1 2 3; do
  bench_order=(agentic dist)
  if (( repeat % 2 == 0 )); then
    bench_order=(dist agentic)
  fi
  for stack in "${bench_order[@]}"; do
    bench_output="$BENCH_RUN_ROOT/repeat-$BENCH_TIMING-$repeat-$stack"
    "$BENCH_PYTHON" "$BENCH_SCRIPT" run \
      "${bench_common[@]}" \
      --mode "$stack" --timing-mode "$BENCH_TIMING" \
      --output "$bench_output" --execute
    bench_runs+=("$bench_output")
  done
done

"$BENCH_PYTHON" "$BENCH_SCRIPT" report "${bench_runs[@]}" \
  > "$BENCH_RUN_ROOT/repeated-$BENCH_TIMING-comparison.json"
```

The report takes the median of each stack's per-run summaries. For full-step
time, those summaries are run means; `step_time_per_run_seconds` retains each
run's mean to show variability. There is no confidence interval or significance
test yet. Check compilation logs and per-step times to confirm that warmup was
long enough; five steps is a default, not a stability guarantee.

## Other metrics and output files

`metrics` also includes mean/P90 trajectory latency, rollout start delay,
post-rollout tail time, generation/environment time per trajectory, and average
output tokens, prompt tokens, turns, and reward. Online sampling is independent
between stacks: compare work and reward alongside time. `SUCCEEDED` in
`trajectory_outcome_percent` means a normally completed episode, not necessarily
a solved task. `api_calls` is diagnostic host timing and call-count information.

For each compared metric, `agentic` and `dist` are the two aggregate values;
`dist_minus_agentic` is their difference and `dist_relative_to_agentic_percent`
is `100 * (dist / agentic - 1)`. Positive time differences mean distributed is
slower. Relative differences are null when the agentic baseline is zero.

Each executed run writes:

- `manifest.json`: configuration and provenance; `agentic.json` additionally
  contains the generated agentic overrides.
- `events*.jsonl`: timing events and completion evidence.
- `result.json`: launcher exit status and total launch duration.
- `summary.json`: per-run statistics, including warmup and measured step times.
- `launcher.log` and, for distributed runs, worker/orchestrator logs.

Reports reject failed/incomplete runs, missing completion evidence, invalid
timing boundaries, mismatched provenance/configuration/dataset, and mismatched
training/log-prob sequence lengths or per-step work volumes. Different native
micro-batch sizes are recorded but allowed. Old host-only timing results
must be rerun. Completion checks have CPU/JAX regression coverage; short TPU
smoke runs validate execution, but the full workload needs repeated runs for
performance conclusions.

To see all workload flags:

```bash
"$BENCH_PYTHON" "$BENCH_SCRIPT" run --help
```

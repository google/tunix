# Performance Instrumentation

Tunix now ships with a lightweight tracing layer that can break each training
step into component spans (data loading, shard preparation, TPU compute,
checkpointing, rollout orchestration, etc.). The instrumentation is available
for supervised fine tuning, the RL cluster, and the RL learner loops so you can
diagnose idle pockets or bottlenecks across the full pipeline.

## Enabling instrumentation

Pass a `PerformanceMetricsConfig` instance through the existing trainer
configuration objects. The same structure is shared by SFT and RL training:

```python
from tunix.perf import trace as perf_trace

perf_config = perf_trace.PerformanceMetricsConfig(
    enabled=True,
    sampling_rate=4,          # capture every 4th iteration
    components=(
        "train_data_load",
        "train_step_compute",
        "checkpoint_save",
    ),
    otel_scope="tunix",
)

training_config = peft_trainer.TrainingConfig(
    eval_every_n_steps=50,
    performance_metrics_config=perf_config,
    # ... other fields ...
)
```

When enabled the trainers automatically publish scalar metrics via
`MetricsLogger`. Metric names are prefixed with `perf/…` for SFT, `perf/eval/…`
for evaluation, `perf/rl/actor/…` for actor updates, and
`perf/rl/learner/…` for the learner orchestration loop. The values represent
seconds spent in the specific component averaged over the sampling window.

## Idle time visibility

Each session tracks three counters:

- `busy_time_sec`: time spent inside recorded spans.
- `blocked_time_sec`: time spent waiting on queues, throttlers, or mesh swaps.
- `idle_time_sec`: residual time (wall clock minus busy and blocked). This is
  a direct signal for unexpected gaps such as data starvation or host/device
  synchronization issues.

Component metrics mirror this breakdown, allowing reports such as
`perf/rl/learner/train_queue_wait/blocked_time_sec` and
`perf/train/checkpoint_save/busy_time_sec` in TensorBoard or whatever backend
is attached to `MetricsLogger`.

## OpenTelemetry compatibility

If `otel_scope` is set and the `opentelemetry` package is installed, the
collector will start native OpenTelemetry spans for every recorded component.
The spans inherit the same names as the Tunix metrics, making it simple to
forward traces to an external collector by configuring an OTel exporter in the
host application. When the library is absent, the spans are recorded only in
the local metrics/log pipeline with no additional dependencies.

## Sampling and filtering

Use `sampling_rate` to reduce overhead on tight loops and the optional
`components` tuple to restrict tracing to a subset of component names. When the
sampling rate is N, only iterations whose sequence id is divisible by N are
instrumented.

## JSONL export for post-processing

`PerformanceMetricsConfig.export_jsonl_path` allows you to dump raw span data
for offline analysis (for example, constructing full step timelines). Every
session writes a single JSON object containing the raw spans as well as the
aggregated counters described above.

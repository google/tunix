<!-- DO NOT REMOVE! Placeholder for TOC. -->

# RL Monitoring and Profiling with Google Cloud ML Diagnostics

Tunix integrates with the
[Google Cloud ML Diagnostics SDK (`google-cloud-mldiagnostics`)](https://github.com/AI-Hypercomputer/google-cloud-mldiagnostics?tab=readme-ov-file)
to provide real-time scalar metrics and targeted per-role XProf profiling for
Reinforcement Learning (RL) workloads (such as GRPO, GSPO-Token, and PPO) across
both **colocated single-slice** and **disaggregated multi-slice (Pathways)**
topologies.

## Overview

Unlike Supervised Fine-Tuning (SFT), where every training step runs a uniform
forward/backward pass on a single mesh, RL workloads coordinate distinct
training and inference roles within an `RLCluster`—often across disaggregated
trainer and sampler TPU slices:

-   **`actor`**: Executes policy gradient updates and actor evaluation on
    trainer devices.
-   **`rollout`**: Generates autoregressive completions using inference engines
    such as vLLM or SGLang-JAX on sampler devices.

Tunix provides two built-in integrations with Google Cloud ML Diagnostics:

1.  **RL Scalar Metrics (`MLDiagScalarBackend`)**: Streams training and
    evaluation scalars to Google Cloud Logging alongside TensorBoard.
2.  **Per-Role Profiling (`RLProfileConfig`)**: Captures targeted XProf traces
    independently for each RL role (`actor` and `rollout`) as separate
    `<role>_invocation_<n>` profile sessions (for example, `actor_invocation_20`
    and `rollout_invocation_20`).

For Google Cloud project setup, IAM permissions, and cluster prerequisites, see
the
[Google Cloud ML Diagnostics SDK documentation](https://github.com/AI-Hypercomputer/google-cloud-mldiagnostics?tab=readme-ov-file#prerequisites).

--------------------------------------------------------------------------------

## Enabling ML Diagnostics in Tunix RL Workloads

To enable ML Diagnostics in a training script that runs Tunix RL algorithms
(`RLCluster`, `GrpoLearner`, `PpoLearner`):

1.  **Install the SDK packages**:

    ```sh
    pip install google-cloud-mldiagnostics xprof
    ```

    *(Already included if your environment was installed with Tunix's
    `requirements/maxtext_requirements.txt`.)*

2.  **Initialize `machinelearning_run`** in your training script before creating
    `RLCluster`.

3.  **Pass `MLDiagScalarBackend` to `MetricsLoggerOptions`** (alongside
    `TensorboardBackend` if TensorBoard logging is also desired) and
    **`RLProfileConfig` to `RLTrainingConfig`** (with `profiler_options=None`).

> **Note:** If you are launching Tunix RL workloads through a training framework
> that already integrates ML Diagnostics (such as
> [MaxText](https://github.com/AI-Hypercomputer/maxtext)), the framework's
> entrypoint script wires `machinelearning_run`, `MLDiagScalarBackend`, and
> `RLProfileConfig` automatically when its ML Diagnostics configuration is
> enabled. See the
> [Google Cloud ML Diagnostics SDK documentation](https://github.com/AI-Hypercomputer/google-cloud-mldiagnostics?tab=readme-ov-file)
> for framework-specific launch flags.

```python
import functools
import google_cloud_mldiagnostics as mldiag
from tunix.common import configs
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo import grpo_learner
from tunix.rl.metrics import mldiag_backend
from tunix.sft import metrics_logger


def build_metrics_logging_options(
    tensorboard_dir: str,
    log_period: int = 10,
    enable_mldiagnostics: bool = True,
    enable_tensorboard: bool = True,
) -> metrics_logger.MetricsLoggerOptions:
  """Returns Tunix MetricsLoggerOptions configured for ML Diagnostics."""
  backends = []
  if enable_mldiagnostics:
    backends.append(mldiag_backend.MLDiagScalarBackend)
  if enable_tensorboard:
    backends.append(
        functools.partial(
            metrics_logger.TensorboardBackend,
            log_dir=tensorboard_dir,
            flush_every_n_steps=log_period,
        )
    )
  return metrics_logger.MetricsLoggerOptions(
      log_dir=tensorboard_dir,
      flush_every_n_steps=log_period,
      backend_kwargs={"custom_backend": backends},
  )


def build_rl_profiler_config(
    output_dir: str,
    enable_mldiagnostics: bool = True,
    start_invocation: int = 20,
    num_invocations: int = 1,
    timeout_secs: float = 60.0,
) -> configs.RLProfileConfig:
  """Configures per-role invocation profiling for Tunix RLCluster."""
  return configs.RLProfileConfig(
      start_invocation=start_invocation,
      num_invocations=num_invocations,
      output_dir=output_dir,
      managed_mldiagnostics=enable_mldiagnostics,
      timeout_secs=timeout_secs,
  )


# 1. Start the ML Diagnostics run before creating RLCluster.
mldiag.machinelearning_run(
    name="tunix-grpo-run",
    run_group="rl-experiments",
    configs={"algo": "grpo", "num_generations": 2},
    gcs_path="gs://<your_gcs_bucket>/mldiagnostics",
    on_demand_xprof=False,
)

# 2. Build metrics logging and per-role profiling configs.
tb_dir = "gs://<your_gcs_bucket>/tensorboard"
metrics_logging_options = build_metrics_logging_options(
    tensorboard_dir=tb_dir,
    log_period=10,
    enable_mldiagnostics=True,
)
rl_profiler_config = build_rl_profiler_config(
    output_dir=tb_dir,
    enable_mldiagnostics=True,
    start_invocation=20,  # Start after warmup / XLA compilation
    num_invocations=1,
)

# 3. Pass both into RLTrainingConfig (keep profiler_options=None).
training_config = rl_cluster_lib.RLTrainingConfig(
    actor_optimizer=optimizer,
    eval_every_n_steps=10,
    max_steps=100,
    mini_batch_size=4,
    metrics_logging_options=metrics_logging_options,
    profiler_options=None,
    rl_profiler_config=rl_profiler_config,
)

cluster_config = rl_cluster_lib.ClusterConfig(
    role_to_mesh={
        rl_cluster_lib.Role.ACTOR: actor_mesh,
        rl_cluster_lib.Role.REFERENCE: reference_mesh,
        rl_cluster_lib.Role.ROLLOUT: rollout_mesh,
    },
    training_config=training_config,
    rollout_config=rollout_config,
)

rl_cluster = rl_cluster_lib.RLCluster(
    actor=actor_model,
    reference=reference_model,
    tokenizer=tokenizer,
    cluster_config=cluster_config,
)
learner = grpo_learner.GrpoLearner(
    rl_cluster=rl_cluster,
    reward_fns=reward_fns,
    algo_config=grpo_config,
)
learner.train(train_dataset)
```

--------------------------------------------------------------------------------

## `RLProfileConfig` Options

`tunix.common.configs.RLProfileConfig` configures per-role trace capture across
`actor` and `rollout` roles:

| Attribute               | Type    | Default | Description                    |
| :---------------------- | :------ | :------ | :----------------------------- |
| `start_invocation`      | `int`   | `2`     | Zero-indexed invocation count  |
:                         :         :         : at which to start profiling    :
:                         :         :         : for each role. Set to a higher :
:                         :         :         : value such as `20` to skip     :
:                         :         :         : initial warmup and XLA         :
:                         :         :         : compilation.                   :
| `num_invocations`       | `int`   | `1`     | Number of invocations to       |
:                         :         :         : profile for each role once     :
:                         :         :         : `start_invocation` is reached. :
| `output_dir`            | `str`   | `""`    | Output directory for raw JAX   |
:                         :         :         : `.xplane.pb` traces when       :
:                         :         :         : `managed_mldiagnostics=False`. :
| `mldiagnostics_dir`     | `str`   | `""`    | Optional GCS output directory  |
:                         :         :         : when                           :
:                         :         :         : `managed_mldiagnostics=False`  :
:                         :         :         : (overrides `output_dir`).      :
| `managed_mldiagnostics` | `bool`  | `False` | If `True`, uploads traces to   |
:                         :         :         : the active `MLRun` for viewing :
:                         :         :         : in the ML Diagnostics UI.      :
| `timeout_secs`          | `float` | `60.0`  | Timeout in seconds to wait for |
:                         :         :         : trace finalization and upload. :

--------------------------------------------------------------------------------

## Metrics and Profiles Collected

| Telemetry Type                | Examples                                   |
| :---------------------------- | :----------------------------------------- |
| **Actor Training Metrics**    | `actor/train/loss`, `actor/train/kl`,      |
:                               : `actor/train/grad_norm`,                   :
:                               : `actor/train/learning_rate`                :
| **Reward Metrics**            | `rewards/train/mean`, `rewards/train/sum`, |
:                               : `rewards/train/min`, `rewards/train/max`,  :
:                               : `rewards/train/score/mean`,                :
:                               : `rewards/train/<reward_fn>`                :
| **Completion Length Metrics** | `completions/train/mean_length` (GRPO),    |
:                               : `generation/train/completions/mean_length` :
:                               : (PPO)                                      :
| **Performance Metrics**       | `perf/train/global_step_time`,             |
:                               : `perf/train/weight_sync_time`,             :
:                               : `perf/train/mean/rollout_time`,            :
:                               : `perf/train/mean/actor_train_time`         :
| **Per-Role XProf Traces**     | `actor_invocation_20`,                     |
:                               : `rollout_invocation_20`                    :

## Viewing Metrics and Profiles

Once your workload is running, view your `MLRun` in the Google Cloud Console
under **Cluster Director -> Diagnostics** or **GKE -> AI/ML -> Diagnostics**:

-   **Model Metrics**: Time-series charts for `actor/train/*`,
    `rewards/train/*`, `completions/train/*` (or
    `generation/train/completions/*`), and `perf/train/*`.
-   **Profiles**: Separate per-role XProf sessions (`actor_invocation_<n>` and
    `rollout_invocation_<n>`) with direct links to the managed XProf viewer.

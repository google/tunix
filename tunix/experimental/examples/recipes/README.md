# MLPerf 35B DeepSWE Recipe

`mlperf_35b_128_v5p.sh` runs the MLPerf DeepSWE recipe (Qwen3.5-35B-A3B MoE,
OpenHands CodeAct agent, 128 TPU v5p chips) on GKE through the shared launcher
(`tunix/dist_rl/launch/`). `mlperf_35b_128_v5p.yaml` holds the recipe; the
script adds the libtpu flags of the trainer, the Pathways workers and the
Pathways proxy, and passes its arguments to `launch.sh`. Pass the per-run
settings as `section.key=value` arguments:

```bash
B=gs://<bucket>
bash tunix/experimental/examples/recipes/mlperf_35b_128_v5p.sh start \
  cluster_config.k8s.tunix_image=<IMAGE> \
  actor_model_config.model_path=<MaxText checkpoint gs:// path> \
  data_config.dataset_path=<dataset gs:// path> \
  rl_training_config.metrics_logging_options.run_name="${USER}-mlperf-35b" \
  rl_training_config.metrics_logging_options.log_dir="${B}/logs/${USER}" \
  rl_training_config.checkpoint_root_directory="${B}/checkpoints/${USER}" \
  actor_model_config.kwargs.maxtext_output_dir="${B}/maxtext/${USER}" \
  trajectory_store_config.enable=true trajectory_store_config.backend=file \
  trajectory_store_config.dir="${B}/trajectories/${USER}/store" \
  cluster_config.pathways.gcs_scratch_location="${B}/pathways-staging/${USER}" \
  cluster_config.k8s.jax_cache.rollout_gcs_dir="${B}/jax_cache/v5p/qwen3.5-35b-a3b/rollout_2x2x1_ep4_tp1_8fb2c88823" \
  cluster_config.k8s.jax_cache.save=true
```

The JobSets are named `${USER}-orch`, `${USER}-train`, and `${USER}-roll-*`;
set `cluster_config.k8s.{orchestrator,trainer,rollout}_id` to run several
experiments at once. Pass `--dry-run` to print the resolved config and JobSets
without deploying, and `stop` with the same arguments to tear the run down.

## MLPerf 397B DeepSWE Recipe

`mlperf_397b_256_v7x.sh` runs the MLPerf 397B DeepSWE recipe
(Qwen3.5-397B-A17B MoE, 512 routed experts, OpenHands CodeAct agent, 256 TPU
v7x chips: 128-chip Pathways trainer + 32 single-host FP8 MoE `tpu7x:2x2x1`
rollout workers) on GKE through `tunix/dist_rl/launch/`:

```bash
B=gs://<bucket>
TPU_RESERVATION=<reservation-name> \
bash tunix/experimental/examples/recipes/mlperf_397b_256_v7x.sh start \
  cluster_config.k8s.tunix_image=<IMAGE> \
  actor_model_config.model_path=<MaxText checkpoint gs:// path> \
  data_config.dataset_path=<dataset gs:// path> \
  rl_training_config.metrics_logging_options.run_name="${USER}-mlperf-397b" \
  rl_training_config.metrics_logging_options.log_dir="${B}/logs/${USER}" \
  rl_training_config.checkpoint_root_directory="${B}/checkpoints/${USER}" \
  actor_model_config.kwargs.maxtext_output_dir="${B}/maxtext/${USER}" \
  trajectory_store_config.enable=true trajectory_store_config.backend=file \
  trajectory_store_config.dir="${B}/trajectories/${USER}/store" \
  cluster_config.k8s.gcs_scratch_location="${B}/pathways-staging/${USER}" \
  cluster_config.k8s.jax_cache.rollout_gcs_dir="${B}/jax_cache/v7x/qwen3.5-397b-a17b/rollout_2x2x1_ep8_fp8" \
  cluster_config.k8s.jax_cache.save=true
```

To scale rollout throughput to 512 rollout chips (`128` single-host replicas,
`GBS=512` or `GBS=1024`), pass `cluster_config.rollout_replicas=128`,
`batch_size=32` (or `64`), scaled `learning_rate` / `max_norm`,
`agentic_grpo_config.max_concurrency=2048` (or `4096`), and
`cluster_config.k8s.user_container_memory=96G` (or `160G`).

# MLPerf 35B DeepSWE Recipe

`mlperf_35b_128_v5p.sh` runs the MLPerf DeepSWE recipe (Qwen3.5-35B-A3B MoE,
OpenHands CodeAct agent, 128 TPU v5p chips) on GKE through the shared launcher
(`tunix/dist_rl/launch/`). `mlperf_35b_128_v5p.yaml` holds the recipe; the
script adds the Pathways worker and proxy flags, which have no config keys, and
passes its arguments to `launch.sh`. Pass the per-run settings as
`section.key=value` arguments:

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
  cluster_config.k8s.gcs_scratch_location="${B}/pathways-staging/${USER}" \
  cluster_config.k8s.jax_cache.rollout_gcs_dir="${B}/jax_cache/v5p/qwen3.5-35b-a3b/rollout_2x2x1_ep4_tp1_8fb2c88823" \
  cluster_config.k8s.jax_cache.save=true
```

The JobSets are named `${USER}-orch`, `${USER}-train`, and `${USER}-roll-*`;
set `cluster_config.k8s.{orchestrator,trainer,rollout}_id` to run several
experiments at once. Pass `--dry-run` to print the resolved config and JobSets
without deploying, and `stop` with the same arguments to tear the run down.

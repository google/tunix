# Multi-Worker MoE Routed-Experts Data-Plane Benchmark

Measures end-to-end Distributed RL data-plane throughput and CPU Orchestrator /
Rollout / Trainer memory footprint (`VmRSS` and `VmHWM`) for MoE router-replay
tensors (`routed_experts`) without model weight loading or XLA compilation.

## Architecture

- **Orchestrator**: Runs the unmodified stock
  `tunix.dist_rl.launch.nodes.orchestrator.main` (`StandardRLProgram` +
  `SequencePackedBatchAssembler` + gRPC Bulk TCP + LZ4 transport).
- **Rollout Workers**: Run `mock_rollout.rollout_main` (`RolloutWorker` backed
  by `MockBaseSamplerImpl`), slicing deterministic `int16` MoE routing tensors
  from a pre-allocated slab (default Kimi-K3 geometry: 61 layers, top-8, 384
  experts -> `uint10` = 610 bytes/token).
- **Trainer Worker**: Runs `mock_trainer.trainer_main` (`TrainerWorker` backed
  by `FakeTrainer`) with `actor_model_config.mesh` emulating multi-way FSDP
  packed-row shapes on CPU.

## Local Single-Host Run

```bash
bash tunix/dist_rl/benchmarks/data_plane/launcher.sh \
  batch_size=2 \
  rl_training_config.mini_batch_size=2 \
  rl_training_config.max_seq_tokens_per_device=4096 \
  rollout_config.total_generation_steps=2048
```

## Multi-Worker GKE TPU v5p Run

Build and push a container image from a commit containing this package, then
launch with `cluster_config.k8s.tunix_image=<IMAGE>`:

```bash
bash tunix/dist_rl/benchmarks/data_plane/k8s_launcher.sh start \
  cluster_config.k8s.tunix_image=<IMAGE> \
  cluster_config.run_id=dp_bench_s2 \
  cluster_config.rollout_replicas=4 \
  agentic_grpo_config.off_policy_steps=2
```

## Telemetry & A/B Comparison

Inspect pod logs for `[DataPlaneBench]` markers:

- `[DataPlaneBench][Orchestrator] VmRSS=... MiB VmHWM=... MiB batch_items=...`
  and `Final summary: VmRSS=... MiB VmHWM=... MiB`: emitted per reward batch
  and at Orchestrator exit (capturing trailing `train_stage` `VmHWM`).
- `[DataPlaneBench][Rollout <id>] Initialized / Draining VmRSS=... VmHWM=...`:
  emitted at startup and graceful drain on each Rollout pod.
- `[DataPlaneBench][Trainer <id>] Final summary: ...`: emitted on Trainer
  shutdown with `fwd_bwd_calls`, `total_tokens` (valid non-padded tokens),
  `total_token_slots` (total packed-row slots), `total_routed_GiB`,
  `last_routed_shape`, `VmRSS`, and `VmHWM`.

To run a controlled Eager vs. Orchestrator Relay A/B comparison, build one
container image on the baseline commit and one on the Relay commit, then run
identical `batch_size`, `rollout_replicas`, and `off_policy_steps` sweeps while
comparing Orchestrator step wall-time and `[DataPlaneBench]` `VmHWM` peaks.

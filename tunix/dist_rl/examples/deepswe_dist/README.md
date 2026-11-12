# Distributed DeepSWE GRPO Recipes

This directory provides the **Base DeepSWE** distributed GRPO recipe
(`Qwen/Qwen3-32B`, R2E-Gym XML agent) on the shared launcher
(`tunix/dist_rl/launch/`), alongside **MLPerf 35B DeepSWE**
(`Qwen/Qwen3.5-35B-A3B` MoE, OpenHands CodeAct agent) in
`tunix/experimental/examples/recipes/mlperf_35b_128_v5p.sh`.

| Recipe | Launcher | Model & Trainer | Rollout & Sandbox |
| :--- | :--- | :--- | :--- |
| **Base DeepSWE** | `tunix/dist_rl/examples/deepswe_dist/k8s_launcher_v2.sh` | `Qwen/Qwen3-32B` (Dense, Tunix `PeftTrainer`, `tpuv5:4x4x4` `8x8` mesh) | 16x `tpuv5:2x2x1` (`inprocess_vllm`, `TP=4`), `r2egym` single-container Agent Sandbox |
| **MLPerf 35B** | `tunix/experimental/examples/recipes/mlperf_35b_128_v5p.sh` | `Qwen/Qwen3.5-35B-A3B` (MoE, MaxText `FSDP=32, CP=2`, `tpuv5:4x4x4`) | 16x `tpuv5:2x2x1` (`vllm`, `EP=4` + router replay), `openhands` dual-container Agent Sandbox |

## Architecture

`launcher_v2.sh` (single TPU host) and `k8s_launcher_v2.sh` (GKE) run the shared
`tunix/dist_rl/launch/nodes/{orchestrator,trainer,rollout}` mains.
`deepswe.yaml` configures the 32B recipe (`Qwen/Qwen3-32B`, batch 8 x 8
generations, 201 steps, 50 turns, `4096+28672` context, `rloo`,
`sequence-mean-token-scale`, token IS `@2.0`, `epsilon=0.2/0.28`, `beta=0.0`,
`inprocess_vllm` + `raiden`). When `env_kwargs.use_agent_sandbox=true`, the
orchestrator enters `prompt_items_wrapper` to pre-warm upcoming batches'
`SandboxWarmPool` resources, and rollout workers attach to the same job-scoped
`SandboxFleet` via `ORCHESTRATOR_ID`.

## Local Single-Host Smoke Run

```bash
bash tunix/dist_rl/examples/deepswe_dist/launcher_v2.sh \
  model_config.model_name=Qwen3-1.7B model_config.model_id=Qwen/Qwen3-1.7B \
  tokenizer_config.tokenizer_path=Qwen/Qwen3-1.7B \
  rollout_config.max_prompt_length=1024 rollout_config.total_generation_steps=1024 \
  batch_size=1 rl_training_config.mini_batch_size=1 \
  rl_training_config.train_micro_batch_size=1 \
  rl_training_config.compute_logps_micro_batch_size=1 \
  agentic_grpo_config.num_generations=2 rl_training_config.max_steps=1 \
  agentic_grpo_config.max_turns=3 env_kwargs.max_steps=3 \
  weight_sync_config.mode=none vllm_config.init_with_random_weights=false
```

## Kubernetes / GKE Launch

### 1. Base DeepSWE (Qwen3-32B)

Trainer (`jobset.pathways.yaml`) and rollout (`jobset.tpu.yaml`) pods mount a
memory-backed `/dev/shm` volume so the 65.5 GB checkpoint downloads into RAM
(`/dev/shm/models`, `HF_HOME=/dev/shm/hf`). `k8s.sh` forwards
`kubernetes_config` (`NAMESPACE`, `NODE_SELECTOR_*`, `IMAGE_REWRITE_PREFIX`) and
`ORCHESTRATOR_ID` to orchestrator and rollout pods and cleans up
`agent-sandbox-rl-run` CRDs on `stop`:

```bash
bash tunix/dist_rl/examples/deepswe_dist/k8s_launcher_v2.sh start \
  cluster_config.k8s.tunix_image=<IMAGE> \
  env_kwargs.use_agent_sandbox=true \
  rl_training_config.checkpoint_root_directory=gs://<BUCKET>/checkpoints/$USER \
  rl_training_config.metrics_logging_options.log_dir=gs://<BUCKET>/logs/$USER

bash tunix/dist_rl/examples/deepswe_dist/k8s_launcher_v2.sh stop
```

### 2. MLPerf 35B DeepSWE (Qwen3.5-35B-A3B MoE)

```bash
bash tunix/experimental/examples/recipes/mlperf_35b_128_v5p.sh \
  --image=<IMAGE> --bucket=gs://<BUCKET> \
  --maxtext-ckpt=gs://<MAXTEXT_ORBAX_CKPT> \
  --dataset-path=gs://<R2EGYM_DATASET_PATH>

bash tunix/experimental/examples/recipes/mlperf_35b_128_v5p.sh stop
```

## Earlier Launchers

`launcher.sh` and `k8s_launcher.sh` run the legacy `run_deepswe_dist.py`
environment-variable flow (`USE_AGENT_SANDBOX=1 ./k8s_launcher.sh --command
start`).

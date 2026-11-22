# Distributed DeepSWE GRPO Recipe

This directory runs the DeepSWE R2E-Gym recipe (`examples/deepswe/`) on the
shared distributed RL launcher and node mains in `tunix/dist_rl/launch/`:
`launcher.sh` (one TPU host) and `k8s_launcher.sh` (GKE), configured by
`deepswe.yaml`.

The launchers run these node mains:

1. `tunix.dist_rl.launch.nodes.orchestrator` runs the CPU control plane and
   loads task records via `deepswe.create_dataset(**data_config)`.
2. `tunix.dist_rl.launch.nodes.rollout` imports `deepswe.py`
   (`registry_module`) and runs `DeepSWEEnv` (`SWEEnv`) and
   `DeepSWEAgent` (`SWEAgent`).
3. `tunix.dist_rl.launch.nodes.trainer` runs the distributed trainer worker.

`deepswe.yaml` (layered on top of `launch/common/base_dist_config.yaml`) sets
the standard 32B converging defaults: `Qwen/Qwen3-32B`, 8 prompt groups per
step, 8 generations, 201 steps, 50 turns, 4096 prompt / 28672 generation
tokens, stop token `151645` (`<|im_end|>`), `rloo` advantage estimation,
`sequence-mean-token-scale` loss aggregation, token-level importance sampling
(`sampler_is="token"`, threshold `2.0`), asymmetric GRPO clipping
(`epsilon=0.2`, `epsilon_high=0.28`), `beta=0.0`, an in-process vLLM rollout
(`rollout_engine="inprocess_vllm"`) with `weight_sync_config.mode="raiden"`,
and the `r2egym` Kubernetes environment backend.

## Local Single-Host Launch

Because `deepswe.yaml` defaults to `Qwen/Qwen3-32B` in `float32`, a single
4-chip TPU host (trainer chips `0,1`, rollout chips `2,3`) needs smaller smoke
overrides:

```bash
bash tunix/dist_rl/examples/deepswe_dist/launcher.sh \
  model_config.model_name=Qwen3-1.7B model_config.model_id=Qwen/Qwen3-1.7B \
  tokenizer_config.tokenizer_path=Qwen/Qwen3-1.7B \
  rollout_config.max_prompt_length=1024 \
  rollout_config.total_generation_steps=1024 \
  batch_size=1 rl_training_config.mini_batch_size=1 \
  rl_training_config.train_micro_batch_size=1 \
  rl_training_config.compute_logps_micro_batch_size=1 \
  agentic_grpo_config.num_generations=2 rl_training_config.max_steps=1 \
  agentic_grpo_config.max_turns=3 env_kwargs.max_steps=3 \
  weight_sync_config.mode=none vllm_config.init_with_random_weights=false
```

Enable `SandboxFleet` inside the rollout worker process and customize sandbox
pod placement via `section.key=value` overrides:

```bash
bash tunix/dist_rl/examples/deepswe_dist/launcher.sh \
  env_kwargs.use_agent_sandbox=true \
  kubernetes_config.namespace=rl-tunix-swebench \
  kubernetes_config.node_selector_key=cloud.google.com/gke-nodepool \
  kubernetes_config.node_selector_val=swe-sandbox-pool
```

Pass `--dry-run` to print the resolved node commands without starting workers.
`launch.sh` rejects the removed environment-variable launchers' settings (such
as `BATCH_SIZE`, `MAX_STEPS`, `WEIGHT_SYNC_MODE`, or `USE_AGENT_SANDBOX`): unset
them and pass `section.key=value` arguments instead. Keep
`env_kwargs.max_steps` equal to `agentic_grpo_config.max_turns`.

## Kubernetes / GKE Launch

Deploy the orchestrator, Pathways trainer (`tpuv5:4x4x4`, `8x8` mesh), and 16
rollout (`tpuv5:2x2x1`, `TP=4` vLLM engine each) JobSets on GKE:

```bash
bash tunix/dist_rl/examples/deepswe_dist/k8s_launcher.sh start \
  cluster_config.k8s.tunix_image=<IMAGE>
```

The trainer and rollout pods download the Qwen3-32B checkpoint (65.5 GB) to
`model_download_path`, which by default is in the container's ephemeral
storage. On node pools with less ephemeral storage than that (~47 GB on
standard GKE nodes), cache it in the pods' memory-backed `/dev/shm` and raise
the trainer container's memory limit to cover the cached weights:

```bash
bash tunix/dist_rl/examples/deepswe_dist/k8s_launcher.sh start \
  cluster_config.k8s.tunix_image=<IMAGE> \
  model_config.model_download_path=/dev/shm/models \
  cluster_config.roles.trainer.env.HF_HOME=/dev/shm/hf \
  cluster_config.roles.rollout.env.HF_HOME=/dev/shm/hf \
  cluster_config.roles.trainer.memory.limit=340G
```

Run with `SandboxFleet` enabled on GKE. The sandboxes run in the run's GKE
namespace (`cluster_config.k8s.namespace`, default `default`) unless
`kubernetes_config.namespace` names another one:

```bash
bash tunix/dist_rl/examples/deepswe_dist/k8s_launcher.sh start \
  cluster_config.k8s.tunix_image=<IMAGE> \
  env_kwargs.use_agent_sandbox=true
```

Deploy or remove the in-cluster OpenTelemetry collector, or tear down all
JobSets:

```bash
bash tunix/dist_rl/examples/deepswe_dist/k8s_launcher.sh otel
bash tunix/dist_rl/examples/deepswe_dist/k8s_launcher.sh stop-otel
bash tunix/dist_rl/examples/deepswe_dist/k8s_launcher.sh stop
```

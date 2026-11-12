# Distributed DeepSWE GRPO Recipe

This directory runs the DeepSWE R2E-Gym recipe (`examples/deepswe/`) with two
sets of launchers:

*   `launcher_v2.sh` (one TPU host) and `k8s_launcher_v2.sh` (GKE) run it on the
    shared distributed RL launcher and node mains in `tunix/dist_rl/launch/`,
    configured by `deepswe.yaml`. They are described below.
*   `launcher.sh` and `k8s_launcher.sh` are the earlier launchers. They run
    `run_deepswe_dist.py` with the `examples/common` node mains and are
    configured through environment variables; see
    [Earlier launchers](#earlier-launchers). They stay until the MLPerf DeepSWE
    recipe reaches parity on the v2 launchers.

The v2 launchers run these node mains:

1. `tunix.dist_rl.launch.nodes.orchestrator` runs the CPU control plane and
   loads task records via `deepswe.create_dataset(**data_config)`.
2. `tunix.dist_rl.launch.nodes.rollout` imports `deepswe.py`
   (`cluster_config.registry_module`) and runs `DeepSWEEnv` (`SWEEnv`) and
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
bash tunix/dist_rl/examples/deepswe_dist/launcher_v2.sh \
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
bash tunix/dist_rl/examples/deepswe_dist/launcher_v2.sh \
  env_kwargs.use_agent_sandbox=true \
  kubernetes_config.namespace=rl-tunix-swebench \
  kubernetes_config.node_selector_key=cloud.google.com/gke-nodepool \
  kubernetes_config.node_selector_val=swe-sandbox-pool
```

Pass `--dry-run` to print the resolved node commands without starting workers.
`launch.sh` rejects the earlier launchers' environment variables (such as
`BATCH_SIZE`, `MAX_STEPS`, `WEIGHT_SYNC_MODE`, or `USE_AGENT_SANDBOX`): unset
them and pass `section.key=value` arguments instead. Keep
`env_kwargs.max_steps` equal to `agentic_grpo_config.max_turns`.

## Kubernetes / GKE Launch

Deploy the orchestrator, Pathways trainer (`tpuv5:4x4x4`, `8x8` mesh), and 16
rollout (`tpuv5:2x2x1`, `TP=4` vLLM engine each) JobSets on GKE:

```bash
bash tunix/dist_rl/examples/deepswe_dist/k8s_launcher_v2.sh start \
  cluster_config.k8s.tunix_image=<IMAGE>
```

The trainer and rollout pods download the Qwen3-32B checkpoint (65.5 GB) to
`model_download_path`, which by default is in the container's ephemeral
storage. The launcher has no volume option: on node pools with less ephemeral
storage than that, add a volume to the JobSets that `--dry-run` prints and
point the `*model_config.model_download_path` keys at it.

Run with `SandboxFleet` enabled on GKE. The sandboxes run in the run's GKE
namespace (`cluster_config.k8s.namespace`, default `default`) unless
`kubernetes_config.namespace` names another one:

```bash
bash tunix/dist_rl/examples/deepswe_dist/k8s_launcher_v2.sh start \
  cluster_config.k8s.tunix_image=<IMAGE> \
  env_kwargs.use_agent_sandbox=true
```

Deploy or remove the in-cluster OpenTelemetry collector, or tear down all
JobSets:

```bash
bash tunix/dist_rl/examples/deepswe_dist/k8s_launcher_v2.sh otel
bash tunix/dist_rl/examples/deepswe_dist/k8s_launcher_v2.sh stop-otel
bash tunix/dist_rl/examples/deepswe_dist/k8s_launcher_v2.sh stop
```

## Earlier launchers

`launcher.sh` and `k8s_launcher.sh` run `run_deepswe_dist.py` (the CPU
orchestrator) with `../common/run_rollout_node.py` (configured with DeepSWE's
`SWEEnv` and `SWEAgent`) and `../common/run_trainer_node.py`. They read their
settings from environment variables. Set `USE_AGENT_SANDBOX=1` to construct
`SWEEnv` with `SandboxFleet` inside the rollout worker process.

Their defaults are the same converging Qwen3-32B recipe as `deepswe.yaml`
(`k8s_launcher.sh`: `tpuv5:4x4x4` trainer with an `(8, 8)` mesh and
`ROLLOUT_REPLICAS=16` single-host `tpuv5:2x2x1` rollouts).

```bash
cd tunix/dist_rl/examples/deepswe_dist
BETA=0.0 WEIGHT_SYNC_MODE=none MAX_STEPS=1 BATCH_SIZE=1 NUM_GENERATIONS=2 ./launcher.sh
```

For sandbox placement, set `SANDBOX_NAMESPACE`, `SANDBOX_NODE_SELECTOR_KEY`, and
`SANDBOX_NODE_SELECTOR_VAL` before launching. To deploy on GKE, or to stop all
JobSets:

```bash
cd tunix/dist_rl/examples/deepswe_dist
USE_AGENT_SANDBOX=1 ./k8s_launcher.sh --command start
./k8s_launcher.sh --command stop
```

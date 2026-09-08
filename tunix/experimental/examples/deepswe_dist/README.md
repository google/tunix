# Distributed DeepSWE GRPO Pipeline

This example ports the non-experimental `examples/deepswe` recipe to the
experimental distributed RL control plane. It reuses the recipe's
`examples/deepswe/deepswe_data.py`, `examples/deepswe/swe_agent.py`, and
`examples/deepswe/swe_env.py` directly; only the distributed registry and
request wiring, orchestration, and launchers live in this directory.

1. `run_deepswe_dist.py` runs the CPU orchestrator.
2. `../common/run_rollout_node.py` runs a rollout worker configured with the
   original recipe's `swe_env.SWEEnv` and `swe_agent.SWEAgent` implementations.
3. The trainer worker is reused from `../common/run_trainer_node.py` because it
   is already a generic PeftTrainer V2 worker.

The Python and launcher defaults match `examples/deepswe/train_deepswe_nb.py`:
Qwen3-32B, batch size 8, 8 generations, 4096 prompt tokens, 8192 response
tokens, 50 turns, RLOO advantages, asymmetric clipping (`0.2`/`0.28`),
`sequence-mean-token-scale` loss aggregation, a `1e-6` learning rate, AdamW
(`b1=0.9`, `b2=0.99`, `weight_decay=0.01`), global gradient clipping at `1.0`,
a three-hour episode timeout, overlong filtering, and rollout concurrency 200.
The reused `examples/deepswe/deepswe_data.py` loader defaults to
`R2E-Gym/R2E-Gym-V1`.

`BATCH_SIZE` is the number of prompt groups in one full/global step, while
`MINI_BATCH_SIZE` is the number of prompt groups in each optimizer update.
Therefore one full step performs `BATCH_SIZE / MINI_BATCH_SIZE` optimizer
updates and, when weight synchronization is enabled, synchronizes rollout
weights once. The trainer accumulates over
`MINI_BATCH_SIZE * NUM_GENERATIONS` trajectories per optimizer update; that
value must be divisible by `TRAIN_MICRO_BATCH_SIZE`. By default both batch sizes
are 8, preserving the original DeepSWE recipe's single update per full step.
For example, `BATCH_SIZE=8 MINI_BATCH_SIZE=2` performs four optimizer updates
before one weight synchronization.

Weight synchronization defaults to `none`, matching the other distributed
examples and keeping a one-step smoke test dependency-free. For a real
multi-step training run, set `WEIGHT_SYNC_MODE=raiden` so updated trainer
weights reach the rollout worker. `fallback` is intentionally rejected because
it acknowledges the synchronization protocol without transferring weights.
Like the non-experimental recipe, the environment defaults to R2E-Gym's
Kubernetes backend without Agent Sandbox (`USE_AGENT_SANDBOX=0` and
`ENV_BACKEND=kubernetes`).

For a one-step infrastructure smoke test, override the full-recipe defaults:

```bash
cd tunix/experimental/examples/deepswe_dist
BETA=0.0 MODEL_NAME=Qwen3-1.7B \
MODEL_ID=Qwen/Qwen3-1.7B MAX_STEPS=1 BATCH_SIZE=1 NUM_GENERATIONS=2 \
MAX_TURNS=3 MAX_PROMPT_LENGTH=1024 MAX_RESPONSE_LENGTH=1024 ./launcher.sh
```

To opt into the Kubernetes Agent Sandbox and its process-wide SandboxFleet,
explicitly set `USE_AGENT_SANDBOX=1`:

```bash
cd tunix/experimental/examples/deepswe_dist
USE_AGENT_SANDBOX=1 SANDBOX_MAX_CONCURRENCY=2 BETA=0.0 \
MAX_STEPS=1 BATCH_SIZE=1 NUM_GENERATIONS=2 ./launcher.sh
```

For sandbox placement, set `SANDBOX_NAMESPACE`, `SANDBOX_NODE_SELECTOR_KEY`, and
`SANDBOX_NODE_SELECTOR_VAL` before launching. The launcher forwards them to the
rollout worker as the `agent_sandbox_rl` variables consumed by `SWEEnv`. It
also forwards the dataset settings so the rollout worker initializes the global
`SandboxFleet` once from the same task set used by the orchestrator. Fleet
capacity is derived from `BATCH_SIZE` and `NUM_GENERATIONS`; set
`MAX_WARMPOOL_REPLICAS` to override the default of one warm replica per
generation. Agent Sandbox currently supports only `SCAFFOLD=r2egym`; the
standard R2E-Gym backend supports both `r2egym` and `sweagent` scaffolds.

The Kubernetes launcher uses the same settings:

```bash
TUNIX_IMAGE=<image> ./k8s_launcher.sh --command=start
```

To enable Agent Sandbox on GKE or stop all jobsets:

```bash
USE_AGENT_SANDBOX=1 TUNIX_IMAGE=<image> ./k8s_launcher.sh --command=start
./k8s_launcher.sh --command=stop
```

It uses the shared `../common/enter_kube_context.sh` setup from the distributed
examples. `HF_TOKEN` is forwarded to trainer and rollout workers. Pathways
deployments can override `PATHWAYS_SERVER_IMAGE`, `PATHWAYS_PROXY_IMAGE`,
`TRAINER_JOBSET_YAML`, and `ROLLOUT_JOBSET_YAML`.

`BETA` remains `0.0` because these launchers currently start only actor and
rollout workers. Enabling KL requires adding a reference worker.

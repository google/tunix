# Distributed DeepSWE GRPO Pipeline

This example ports the non-experimental `examples/deepswe` recipe to the
experimental distributed RL control plane. It reuses the recipe's dataset,
agent, and environment implementations directly:

1. `run_deepswe_dist.py` runs the CPU orchestrator.
2. `../common/run_rollout_node.py` runs a rollout worker configured with
   DeepSWE's `SWEEnv` and `SWEAgent`.
3. The trainer worker is reused from `../common/run_trainer_node.py` because it
   is already a generic PeftTrainer V2 worker.

The defaults match `examples/deepswe/train_deepswe_nb.py`: Qwen3-32B, batch
size 8, 8 generations, 4096 prompt tokens, 8192 response tokens, 50 turns,
RLOO advantages, asymmetric clipping (`0.2`/`0.28`),
`sequence-mean-token-scale` loss aggregation, FP32 parameter storage with BF16
compute, decoder rematerialization, flash attention (block size 1024), and
actor-side recomputation of start-of-step log probabilities. Weight sync uses
Raiden by default. Set `WEIGHT_SYNC_MODE=none` only for a one-step smoke test.

For a local four-chip infrastructure smoke test, split two chips each between
trainer and rollout and override the recipe model and batch sizes:

```bash
cd tunix/experimental/examples/deepswe_dist
MODEL_NAME=Qwen3-1.7B MODEL_ID=Qwen/Qwen3-1.7B \
BETA=0.0 WEIGHT_SYNC_MODE=none MAX_STEPS=1 \
BATCH_SIZE=1 MINI_BATCH_SIZE=1 NUM_GENERATIONS=2 \
MAX_PROMPT_LENGTH=1024 MAX_RESPONSE_LENGTH=1024 ./launcher.sh
```

```bash
cd tunix/experimental/examples/deepswe_dist
USE_AGENT_SANDBOX=1 BETA=0.0 WEIGHT_SYNC_MODE=none MAX_STEPS=1 BATCH_SIZE=1 NUM_GENERATIONS=2 ./launcher.sh
```

For sandbox placement, set `SANDBOX_NAMESPACE`, `SANDBOX_NODE_SELECTOR_KEY`, and
`SANDBOX_NODE_SELECTOR_VAL` before launching. The launcher forwards them to the
rollout worker as the `agent_sandbox_rl` variables consumed by `SWEEnv`.

To deploy on Kubernetes / GKE:

```bash
cd tunix/experimental/examples/deepswe_dist
./k8s_launcher.sh --command start
```

Or run with sandbox enabled on GKE:

```bash
cd tunix/experimental/examples/deepswe_dist
USE_AGENT_SANDBOX=1 ./k8s_launcher.sh --command start
```

To stop all jobsets:

```bash
./k8s_launcher.sh --command stop
```

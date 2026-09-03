# Distributed DeepSWE GRPO Pipeline

This example is the first DeepSWE-specific version of the experimental
distributed RL pipeline. It follows the same control-plane shape as the
distributed GSM8K example:

1. `run_deepswe_dist.py` runs the CPU orchestrator.
2. `../common/run_rollout_node.py` runs a rollout worker configured with
   DeepSWE's `SWEEnv` and `SWEAgent`.
3. The trainer worker is reused from `../common/run_trainer_node.py` because it
   is already a generic PeftTrainer V2 worker.

The first milestone is intentionally small: run one trainer+rollout pipeline
step with `BETA=0.0` and `WEIGHT_SYNC_MODE=none`. The default path uses the
real DeepSWE dataset (`R2E-Gym/R2E-Gym-V1`) and constructs `SWEEnv` with
`SandboxFleet` inside the rollout worker process.

```bash
cd tunix/experimental/examples/deepswe_dist
BETA=0.0 WEIGHT_SYNC_MODE=none MAX_STEPS=1 BATCH_SIZE=1 NUM_GENERATIONS=2 ./launcher.sh
```

To run without the agent sandbox, explicitly set `USE_AGENT_SANDBOX=0`.

```bash
cd tunix/experimental/examples/deepswe_dist
USE_AGENT_SANDBOX=0 BETA=0.0 WEIGHT_SYNC_MODE=none MAX_STEPS=1 BATCH_SIZE=1 NUM_GENERATIONS=2 ./launcher.sh
```

For sandbox placement, set `SANDBOX_NAMESPACE`, `SANDBOX_NODE_SELECTOR_KEY`, and
`SANDBOX_NODE_SELECTOR_VAL` before launching. The launcher forwards them to the
rollout worker as the `agent_sandbox_rl` variables consumed by `SWEEnv`. It
also forwards the dataset settings so the rollout worker initializes the global
`SandboxFleet` from the same task set used by the orchestrator.

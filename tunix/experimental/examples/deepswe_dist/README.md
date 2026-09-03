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
step with `BETA=0.0` and `WEIGHT_SYNC_MODE=none`. Reference KL, Raiden weight
sync, agent-sandbox prewarming, and production DeepSWE-scale settings can be
layered on after the basic pipeline is stable.

```bash
cd tunix/experimental/examples/deepswe_dist
BETA=0.0 WEIGHT_SYNC_MODE=none MAX_STEPS=1 BATCH_SIZE=1 NUM_GENERATIONS=2 ./launcher.sh
```
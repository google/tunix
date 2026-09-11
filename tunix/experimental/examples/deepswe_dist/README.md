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
regular DeepSWE `SWEEnv` backend. Set `USE_AGENT_SANDBOX=1` to construct
`SWEEnv` with `SandboxFleet` inside the rollout worker process.

```bash
cd tunix/experimental/examples/deepswe_dist
BETA=0.0 WEIGHT_SYNC_MODE=none MAX_STEPS=1 BATCH_SIZE=1 NUM_GENERATIONS=2 ./launcher.sh
```

```bash
cd tunix/experimental/examples/deepswe_dist
USE_AGENT_SANDBOX=1 BETA=0.0 WEIGHT_SYNC_MODE=none MAX_STEPS=1 BATCH_SIZE=1 NUM_GENERATIONS=2 ./launcher.sh
```

You can also launch components individually using `--command`:

```bash
# Terminal 1: CPU orchestrator
./launcher.sh --command orchestrator

# Terminal 2: Trainer worker
./launcher.sh --command trainer

# Terminal 3: Rollout worker
./launcher.sh --command rollout

# Stop running background workers
./launcher.sh --command stop
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

## Testing Orchestrator with Mock Workers (Zero TPUs)

To validate end-to-end orchestration, discovery, and batch assembly without
requiring physical TPU slices or Docker sandboxes, run `test_orchestrator`:

```bash
# On Kubernetes / GKE (all nodes on CPU):
./k8s_launcher.sh --command test_orchestrator

# Or locally on workstation:
./launcher.sh --command test_orchestrator
```

This launches:

- `run_deepswe_dist.py`: Orchestrator control-plane.
- `tunix/experimental/examples/common/run_mock_trainer_node.py`: Lightweight mock trainer on CPU.
- `tunix/experimental/examples/common/run_mock_rollout_node.py`: Lightweight mock rollout returning wire-compatible fixed completions or executing SWE episodes in sandboxes.


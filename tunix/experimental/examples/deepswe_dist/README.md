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

## Evaluate without a trainer

`eval_deepswe.py` uses the same SWE-Bench Verified task split, Qwen3-32B
chat template, one greedy trajectory per task, 30-turn limit, and Pass@1
calculation as `examples/deepswe/eval_deepswe.py`. It loads a Hugging Face
checkpoint directly into the rollout worker; it does not start a trainer or
perform weight synchronization. The defaults use eight TPU chips in a pure
tensor-parallel mesh. From the repository root, run:

```bash
python tunix/experimental/examples/deepswe_dist/eval_deepswe.py \
  --model_dir /path/to/Qwen3-32B \
  --output_dir /path/to/eval_results
```

For a short correctness run, add `--tasks_limit 2`. A full run omits that flag.
Results are written as one JSONL record per task and a neighboring
`*.summary.json` with Pass@1, average reward, average steps, statuses, and
guard counts. Failed tasks remain in the Pass@1 denominator and cause a
nonzero exit. `--enable_guard` enables the same optional action guard as the
agentic evaluation.

To use rollout workers on separate TPU hosts, start the script with
`--role worker --port 20001` on each host. Then start one CPU controller with
`--role controller --worker_addresses host1:20001 host2:20001` and the same
model and evaluation flags. Set `--mesh_fsdp` and `--mesh_tp` to the visible
chip layout on each worker.

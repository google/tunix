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

The training defaults use `Qwen/Qwen3-4B-Instruct-2507`. The orchestrator
loads the pinned `R2E-Gym/R2E-Gym-Subset` train split (revision
`2e8108ff942f24fcb5686badfaf7f9a8808566d5`) and joins its full task
records to the 1,012 Qwen3-4B learnable `docker_image` values in
`canon-zero-tim/clean_data/p46_q4_learnable/`, copied from
`origin/yuxzhang/canon-zero-tim`. It verifies the canonical
selector's SHA-256 and requires exactly one source row for each selected image.
The selector alone contains evaluation metadata, not the task fields needed by
`SWEEnv`.

For a local one-step smoke run, use a writable model and artifact location and
select a working sandbox backend:

```bash
MODEL_DIR=/path/to/Qwen3-4B-Instruct-2507 \
ARTIFACT_ROOT=/path/to/artifacts \
ENV_BACKEND=docker \
MAX_STEPS=1 ./tunix/experimental/examples/deepswe_dist/launcher.sh
```

`MODEL_DIR` may be empty; the launcher downloads the model from Hugging Face.
With the default `WEIGHT_SYNC_MODE=none`, the in-process rollout loads those
pretrained weights directly. Multi-step training needs a configured weight
sync mode so later rollouts use the updated policy.
For a different task selector, set `GOLD_WHITELIST` to a JSONL file containing
unique `docker_image` values. `DATASET_PATH` can point to a local Hugging Face
dataset; the same selector join is applied. A Kubernetes image must include the
`canon-zero-tim/clean_data/p46_q4_learnable/` directory because the
orchestrator resolves its default selector relative to the installed source.

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

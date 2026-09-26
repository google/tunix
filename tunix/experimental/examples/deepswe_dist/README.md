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

The Qwen3-4B canon training recipe uses 8 prompt groups with 16 generations
each (128 trajectories per optimizer update), 4096 prompt tokens, 16384 response
tokens, 50 turns, RLOO advantages, `sequence-mean-token-scale` loss,
`epsilon=0.2`, `epsilon_high=0.28`, and AdamW with learning rate `1e-6`,
`b2=0.99`, and weight decay `0.01`. On the local 2-trainer/2-rollout TPU split,
`TRAIN_MICRO_BATCH_SIZE=2` gives 64 gradient accumulation steps per update;
the original 64-trainer-chip recipe uses a microbatch of 8. For a 200-update
local run, set:

```bash
BATCH_SIZE=8 NUM_GENERATIONS=16 MINI_BATCH_SIZE=8 \
TRAIN_MICRO_BATCH_SIZE=2 MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=16384 \
MAX_TURNS=50 MAX_STEPS=200 WEIGHT_SYNC_MODE=raiden \
ENV_BACKEND=docker ./tunix/experimental/examples/deepswe_dist/launcher.sh
```

The clean-data evaluation used temperature 1, top-p 1, top-k 0, vLLM seed 42,
prefix caching off, and the `q4_r2egym_xml_v2` action adapter. The launcher
uses those settings by default. It also passes a 4800-second episode budget;
set `STEP_TIMEOUT_SECS=1800` and `REWARD_TIMEOUT_SECS=1800` for the evaluated
environment limits. The agent's R2E-Gym system prompt matches the evaluation
prompt. On Docker, `SWEEnv` installs a pinned `chardet` wheel offline so the
`file_editor` tool works when task containers cannot reach PyPI.

The local run uses two rollout TPU chips and Docker, while the clean-data
evaluation used a larger Kubernetes TPU mesh. Keep rollout concurrency within
the capacity of the local worker; it does not change the 8-by-16 update batch.

The canonical Zero-TIM arm also requires its serving/training alignment
overlay. This distributed example uses the native rollout log probabilities;
check `sampler-trainer` agreement metrics in the run logs when interpreting
training quality.

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

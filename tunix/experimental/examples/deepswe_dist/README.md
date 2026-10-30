# Distributed DeepSWE GRPO Pipeline

## Local rollout-only evaluation

`eval_launcher.py` starts a CPU controller and a TPU rollout worker, using
`RolloutWorker` and the same DeepSWE environment as training. It restores real
MaxText weights and runs generation, tools, and the task's final tests. No trainer
or weight synchronization is needed. The existing training commands follow below.

### Environment

Use an environment with Tunix's experimental dependencies, a compatible vLLM +
TPU inference installation, and MaxText's integrated
`maxtext.integration.vllm.maxtext_vllm_adapter`. The local smoke run was verified
with Python 3.12, JAX 0.11.0, Qwix 0.1.8, and MaxText commit `aae0386e8` plus the
Qwen3-1.7B HF model registry entry described below.

From the repository root, install the DeepSWE additions into that environment:

```bash
python -m pip install -e '.[experimental]'
bash tunix/experimental/examples/deepswe_dist/setup_local.sh
python -c 'import docker; print(docker.from_env().ping())'
```

The setup script pins R2E-Gym and applies the two compatibility fixes already used
by this repository's Dockerfile. It does not install the TPU inference stack or
the Docker daemon. The local `r2egym` scaffold needs a working Docker daemon and
space for task images. If your daemon uses a non-default socket, export
`DOCKER_HOST=unix:///path/to/docker.sock` before running. If access fails with
`Permission denied`, verify Docker group membership and refresh the login session
(or run `newgrp docker`).

### Convert the matching weights once

Set `MODEL_PATH` to a local `Qwen/Qwen3-1.7B` Hugging Face model directory containing
its safetensors, config, and tokenizer. Use a disk with room for the checkpoint and
Docker images. Changing `--model_id` alone does not change the restored weights.

```bash
export MODEL_PATH=/path/to/Qwen3-1.7B
export CONVERTED_PATH=/path/on/large/disk/qwen3-1.7b-orbax
MAXTEXT_CONFIG_DIR="$(python -c 'from maxtext.utils.globals import MAXTEXT_CONFIGS_DIR; print(MAXTEXT_CONFIGS_DIR)')"
JAX_PLATFORMS=cpu python -m maxtext.checkpoint_conversion.to_maxtext \
  "$MAXTEXT_CONFIG_DIR/base.yml" \
  model_name=qwen3-1.7b base_output_directory="$CONVERTED_PATH" \
  hardware=cpu skip_jax_distributed_system=True scan_layers=False \
  checkpoint_storage_use_ocdbt=True checkpoint_storage_use_zarr3=True \
  --hf_model_path "$MODEL_PATH" --lazy_load_tensors true
export CHECKPOINT_PATH="$CONVERTED_PATH/0/items"
```

MaxText must register `"qwen3-1.7b": "Qwen/Qwen3-1.7B"` in
`maxtext.utils.globals.HF_IDS`; some revisions have the architecture and converter
mapping but omit this entry. This upstream MaxText fix is a separate change.
Verify that the checkpoint matches the model: Qwen3-1.7B has hidden dimension
2048, MLP dimension 6144, and 28 layers. Do not reuse an unrelated `0/items` merely
because it shares a directory with the HF model. Shape-mismatch padding during
restore is not proof that the checkpoint is correct.

### Run on four TPU chips

```bash
bash tunix/experimental/examples/deepswe_dist/eval_local.sh
```

This runs one R2E-Gym task, one attempt, and at most two agent turns with thinking
disabled. It is an execution smoke test, not a model-quality evaluation. It uses
streaming for a limited remote dataset, so it does not materialize the full
training split. Set `DATASET_PATH` to use a dataset previously saved with
`datasets.save_to_disk`; `OUTPUT_DIR`, `PYTHON`, and `TPU_CHIPS` are also optional
environment overrides. Extra CLI arguments override the script's defaults:

```bash
bash tunix/experimental/examples/deepswe_dist/eval_local.sh \
  --tasks_limit 10 --max_steps 30 --num_rollouts_per_instance 4
```

For more concurrency, also adjust `--max_concurrent` and `--vllm_max_num_seqs`.
The underlying CLI supports separate worker/controller roles and SandboxFleet;
`python tunix/experimental/examples/deepswe_dist/eval_deepswe.py --help` lists all
options. Cluster-specific deployment configuration is maintained separately.

### Results and limits

Each run writes `config.json`, `attempts/eval_<task>_<attempt>.json`, and
`summary.json` under a unique output directory. The controller checks the worker's
model profile before dispatch. A successful smoke run exits 0, with
`complete=true`, `missing_attempts=0`, `error_attempts=0`, and `fatal_error=null`.
Also inspect `status_counts`: `SUCCEEDED` means an episode ended normally, not
that the issue was solved; `resolved` is determined by a positive final reward.
`MAX_STEPS_REACHED` indicates the configured turn limit.
Timeouts should be investigated before interpreting scores.

`avg_at_k` is the resolved-attempt fraction; `pass_at_k` reports the usual
combinatorial pass@k estimate. Missing and failed attempts remain in denominators.
`wall_seconds` measures dispatch through result return, including tool execution
and final scoring, but excludes worker startup. The seed controls the TPU engine
RNG; concurrent request ordering can affect samples, so it is not a per-attempt
reproducibility guarantee. This runner does not compare training performance.

### Tests

```bash
python tests/experimental/examples/deepswe_dist/eval_deepswe_test.py
JAX_PLATFORMS=cpu python -m pytest --import-mode=importlib \
  tests/experimental/rollout/collector_test.py \
  tests/rl/agentic/trajectory/trajectory_collect_engine_test.py \
  tests/generate/vllm_sampler_test.py -k 'not VllmSamplerTest'
```

## Distributed training

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

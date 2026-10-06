# DeepSWE MLPerf Distributed Recipes

This directory contains executable recipe scripts for running distributed DeepSWE RL on Google Cloud Platform (GCP) GKE TPU clusters with Tunix.

## Available Recipes

| Recipe | Model | Trainer Topology & Sharding | Rollout Topology & Sharding |
| :--- | :--- | :--- | :--- |
| [`mlperf_35b_128_v5p.sh`](mlperf_35b_128_v5p.sh) | Qwen3.5-35B-A3B | `1x tpuv5:4x4x4` (64 chips)<br>`FSDP=32, TP=1, EP=1, CP=2` | `16x tpuv5:2x2x1` (64 chips)<br>`DP=1, TP=1, EP=4` |
| [`mlperf_35b_256_v5p.sh`](mlperf_35b_256_v5p.sh) | Qwen3.5-35B-A3B | `1x tpuv5:4x4x4` (64 chips)<br>`FSDP=32, TP=1, EP=1, CP=2` | `32x tpuv5:2x2x1` (128 chips)<br>`DP=1, TP=1, EP=4` |
| [`mlperf_35b_128_v7x.sh`](mlperf_35b_128_v7x.sh) | Qwen3.5-35B-A3B | `1x tpu7x:4x4x4` (64 chips)<br>`FSDP=64, TP=1, EP=1, CP=2` | `16x tpu7x:2x2x1` (64 chips)<br>`DP=1, TP=1, EP=8` |
| [`mlperf_397b_512_v5p.sh`](mlperf_397b_512_v5p.sh) | Qwen3.5-397B-A17B | `1x tpuv5p:4x8x8` (256 chips)<br>`FSDP=16, TP=1, EP=2, CP=8` | `16x tpuv5p:2x2x4` (256 chips)<br>`DP=1, TP=1, EP=16` |
| [`mlperf_397b_256_v7x.sh`](mlperf_397b_256_v7x.sh) | Qwen3.5-397B-A17B | `1x tpu7x:4x4x8` (128 chips)<br>`FSDP=32, TP=1, EP=2, CP=4` | `32x tpu7x:2x2x2` (256 chips)<br>`DP=1, TP=1, EP=16` |
| [`mlperf_397b_512_v7x.sh`](mlperf_397b_512_v7x.sh) | Qwen3.5-397B-A17B | `1x tpu7x:4x4x8` (128 chips)<br>`FSDP=32, TP=1, EP=2, CP=4` | `32x tpu7x:2x2x4` (512 chips / 1024)<br>`DP=2, TP=1, EP=16` |
| [`mlperf_397b_1024_v7x.sh`](mlperf_397b_1024_v7x.sh) | Qwen3.5-397B-A17B | `1x tpu7x:4x4x8` (128 chips)<br>`FSDP=32, TP=1, EP=2, CP=4` | `64x tpu7x:2x2x4` (1024 chips)<br>`DP=2, TP=1, EP=16` |
| [`mlperf_35b_eval.sh`](mlperf_35b_eval.sh) | Qwen3.5-35B-A3B (offline eval, pass@4) | None (no trainer) | `16x tpuv5:2x2x1` (64 chips)<br>`DP=2, FSDP=2, TP=2` |

---

## Building the Docker Image

### 1. Build Command

From the root of the `tunix` repository:

```bash
IMAGE_TAG="gcr.io/cloud-tpu-multipod-dev/${USER}/trellis:latest"

docker build \
  --network=host \
  --build-arg INSTALL_MAXTEXT=true \
  --build-arg INSTALL_RAIDEN=true \
  --build-arg INSTALL_DEEPSWE_DEPS=true \
  -t "${IMAGE_TAG}" \
  -f Dockerfile .
```

**Build Arguments (`Dockerfile`)**:
- `INSTALL_MAXTEXT=true`: Installs MaxText, `maxtext-vllm-adapter`, and TPU diagnostics from `requirements/maxtext_requirements.txt`.
- `INSTALL_RAIDEN=true`: Installs the Raiden (`tpu_sync_jax`) wheel for direct DCN weight synchronization.
- `INSTALL_DEEPSWE_DEPS=true`: **(Required for DeepSWE)** Installs the agentic evaluation and Kubernetes sandbox client dependencies (`swebench`, `openhands-sdk`, `k8s-agent-sandbox`, `agent-sandbox-rl`, `r2e-gym`, and `kubernetes`).
- `INSTALL_K8S_TOOLS=true`: *(Optional, omitted above)* Installs interactive CLI debugging tools (`gcloud`, `kubectl`, `k9s`, `vim`, `lsof`, `procps`) inside the container; not required at runtime.

### 2. Push Image to Google Container Registry (GCR)

```bash
gcloud auth configure-docker
docker push "${IMAGE_TAG}"
```

---

## How to Launch a Run

### 1. Prerequisites

- **GKE Cluster Access**: Ensure you have credentials and context for the cluster:
  ```bash
  gcloud container clusters get-credentials bodaborg-v5p-nap \
    --region europe-west4 \
    --project cloud-tpu-shared-capacity
  ```
- **Weights & Biases (WandB)**: Obtain your API key from [wandb.ai/authorize](https://wandb.ai/authorize).
- **Storage Bucket**: Ensure your GCS bucket (e.g., `gs://<user>-storage-europe-west4/`) is accessible by the cluster's service account (`xpk-sa`).

### 2. Launching the Run

You can override configuration variables via environment variables at launch time:

```bash
# Set your environment variables
export WANDB_API_KEY="your_wandb_api_key_here"
export MAXTEXT_OUTPUT_DIR="gs://<your-bucket>/trellis/maxtext"
export TUNIX_IMAGE="gcr.io/cloud-tpu-multipod-dev/${USER}/trellis:latest" 

# Launch the run (change recipe script as needed)
SEED=42 bash tunix/experimental/examples/recipes/mlperf_35b_128_v5p.sh start
```

#### FP8 MoE

Every recipe can run its routed experts in FP8 (see `fp8_moe.sh`):

```bash
# FP8 rollout experts (W8A8), bf16 trainer
ROLLOUT_FP8=true bash tunix/experimental/examples/recipes/mlperf_35b_128_v5p.sh start
# Experimental: also round the trainer's experts to the rollout's FP8 grid in the forward pass
ROLLOUT_FP8=true TRAINER_FP8=true bash tunix/experimental/examples/recipes/mlperf_35b_128_v5p.sh start
```

Both need an image whose MaxText has the `rollout_fp8_moe` and `fp8_moe_fake_quant` flags.

#### Dropless MoE (`ragged_buffer_factor=-1`)

The trainer's default `ragged_buffer_factor=2.0` caps each expert's receive buffer, so a skewed routing batch can
drop tokens silently. [`mlperf_gbs1024_dropless.sh`](mlperf_gbs1024_dropless.sh) runs the gbs1024 submission
setup with worst-case buffers instead. It sets the following (each can be overridden from the environment) and then
calls `mlperf_gbs1024_submission.sh`:

| Variable | Value | Why |
|---|---|---|
| `RAGGED_BUFFER_FACTOR` | `-1.0` | worst-case (dropless) ring-of-experts buffers |
| `NUM_MOE_TOKEN_CHUNKS` | `4` | smaller per-chunk dispatch buffers |
| `CONTEXT_REMAT_POLICY` | `offload` | attention output saved to host |
| `TRAINER_EXTRA_LIBTPU_INIT_ARGS` | `--xla_tpu_max_hbm_size_mib=84992` | about 14.5 GiB of HBM is resident when `fwd_bwd` loads; without the cap XLA schedules against all 94.74 GiB and the program does not fit |
| `TUNIX_IMAGE` | `gcr.io/cloud-tpu-multipod-dev/atwigg/trellis-experimental:1005` | retries vLLM engine start (see below) |

Run it from the repo root with `MAXTEXT_EXTRA_FLAGS` unset (it replaces the recipe's MaxText flag list, and the
script refuses to start when it is set; use `MAXTEXT_USER_EXTRA_FLAGS` to append overrides):

```bash
unset MAXTEXT_EXTRA_FLAGS
export K8S_NAMESPACE=<namespace> JOB_PREFIX=<short-prefix> SEED=<seed>
export WANDB_API_KEY=<your key>          # optional; W&B is skipped without it

DRY_RUN=true bash tunix/experimental/examples/recipes/mlperf_gbs1024_dropless.sh   # render only
bash tunix/experimental/examples/recipes/mlperf_gbs1024_dropless.sh                # launch
```

Check the trainer at start-up and at the first step:

```bash
# Resolved MaxText config: expect -1.0, 4, OFFLOAD and True
kubectl logs -n ${K8S_NAMESPACE} -l jobset.sigs.k8s.io/jobset-name=${JOB_PREFIX}-train,jobset.sigs.k8s.io/replicatedjob-name=proc -c main \
  | grep -E "Config param (ragged_buffer_factor|num_moe_token_chunks|context|use_gdn_kernel):"
# fwd_bwd program size from the runtime compile (about 72 GiB with the cap, about 82 GiB without)
kubectl logs -n ${K8S_NAMESPACE} -l jobset.sigs.k8s.io/jobset-name=${JOB_PREFIX}-train,jobset.sigs.k8s.io/replicatedjob-name=proc --all-containers \
  | grep -E "XLA::TPU program HBM usage: [0-9.]+G"
# Steps, or a runtime HBM failure (RuntimeProgramAllocationFailure means the cap is too high)
kubectl logs -f -n ${K8S_NAMESPACE} -l jobset.sigs.k8s.io/jobset-name=${JOB_PREFIX}-orch \
  | grep -E "Train step [0-9]+ - |RESOURCE_EXHAUSTED"
```

Tear it down with the same script, which stops the orchestrator, the trainer and all 128 rollout replicas:

```bash
bash tunix/experimental/examples/recipes/mlperf_gbs1024_dropless.sh stop
```

Use an image whose `vllm_sampler_v2` retries engine start (`_START_MAX_ATTEMPTS`). Rollout engine cores
occasionally segfault right after `Enabled custom fusions`; with the retry they recover, and without it the first one
stops the run under fail-fast. On bodaborg-tpu7x-gsc-elm (2026-10-06, 10 steps) this recipe cost roughly 6-9% of
trainer time per microbatch against the production flags, and pathways workers used about 315 GiB of host memory.

### 3. Monitoring and Managing the Run

```bash
# Check JobSets status in the trellis namespace
kubectl get jobsets -n trellis

# Check Pods
kubectl get pods -n trellis -l kueue.x-k8s.io/local-queue-name=multislice-queue

# Follow Orchestrator logs
kubectl logs -f -n trellis -l jobset.sigs.k8s.io/jobset-name=${USER}-orch

# Follow Trainer logs
kubectl logs -f -n trellis -l jobset.sigs.k8s.io/jobset-name=${USER}-train -c main

# Tear down the run
bash tunix/experimental/examples/recipes/mlperf_35b_128_v5p.sh stop
```

---

## Raiden wheel and pathways images

Latest tested raiden wheel:
```
export RAIDEN_WHL=https://storage.googleapis.com/tunix-ci-artifacts/raiden/tpu_sync_jax-0.0.1.dev20260926082434-cp312-cp312-manylinux_2_31_x86_64.whl`
```
pathways images are defined in tunix/experimental/examples/recipes/mlperf_pathways_config.sh

### When to Rebuild: Pathways Images vs. Python Wheels

| Change Type | Rebuild Pathways Image? | Rebuild Python Wheel? | Notes |
| :--- | :---: | :---: | :--- |
| **C++ Code (`.cc`, `.h`, protos)** in `tpu_sync/` | **YES** | **YES** | Pathways workers run C++ inside `cloud_pathways_server`; McJAX rollout workers run C++ from `.so` files inside the Python wheel. Both must be updated. |
| **Pure Python** in `tunix/` or `tpu_sync/` (e.g. `broadcast_engine.py`, `raiden_controller.py`) | **NO** | **YES** | Pathways workers do not run Python. Only the controller/runner containers need the updated wheel. |
| **Pathways Infrastructure** (`cloud/tpu/multipod/pathways/`) | **YES** | **NO** | Only affects the Pathways server/proxy binaries. |

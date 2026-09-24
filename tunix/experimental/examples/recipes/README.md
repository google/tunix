# DeepSWE MLPerf Distributed Recipes

This directory contains executable recipe scripts for running distributed DeepSWE RL on Google Cloud Platform (GCP) GKE TPU clusters with Tunix.

## Available Recipes

Currently available DeepSWE MLPerf recipes:

| Recipe Script | Description | Hardware Setup |
| :--- | :--- | :--- |
| [`mlperf_35b_256_v5p.sh`](mlperf_35b_256_v5p.sh) | Qwen3.5-35B-A3B distributed GRPO recipe with 256 trajectories per step on Trellis GKE TPU v5p cluster (`europe-west4`). | **Trainer**: 1x TPU v5p-64 (`tpuv5:4x4x4`)<br>**Rollouts**: 16x TPU v5p-8 (`tpuv5:2x2x1`)<br>**Sandboxes**: GKE CPU pool (`sandbox-cpu-pool`) |
| [`mlperf_35b_256_v7x.sh`](mlperf_35b_256_v7x.sh) | Qwen3.5-35B-A3B distributed GRPO recipe with 256 trajectories per step on Trellis GKE TPU v7x cluster (`us-central1`). | **Trainer**: 1x TPU v7x-64 (`tpu7x:4x4x4`)<br>**Rollouts**: 16x TPU v7x-4 (`tpu7x:2x2x1`)<br>**Sandboxes**: GKE CPU pool (`sandbox-np`) |
| [`mlperf_397b_256_v5p.sh`](mlperf_397b_256_v5p.sh) | Qwen3.5-397B-A17B distributed GRPO recipe on Trellis GKE TPU v5p cluster (`europe-west4`). | **Trainer**: 1x TPU v5p-256 (`tpuv5p:4x8x8`)<br>**Rollouts**: 4x TPU v5p-16 (`tpuv5p:2x2x4`)<br>**Sandboxes**: GKE CPU pool (`sandbox-cpu-pool`) |
| [`mlperf_397b_256_v7x.sh`](mlperf_397b_256_v7x.sh) | Qwen3.5-397B-A17B distributed GRPO recipe on Trellis GKE TPU v7x cluster (`us-central1`). | **Trainer**: 1x TPU v7x-128 (`tpu7x:4x4x8`)<br>**Rollouts**: 16x TPU v7x-8 (`tpu7x:2x2x2`)<br>**Sandboxes**: GKE CPU pool (`sandbox-np`) |

---

## Building the Docker Image

The cluster runs Tunix components inside a custom container image that includes DeepSWE agentic dependencies (OpenHands, SWE-bench, R2E-Gym), MaxText, and Raiden weight synchronization.

### 1. Build Command

From the root of the `tunix` repository:

```bash
# Define your image tag
IMAGE_TAG="gcr.io/cloud-tpu-multipod-dev/${USER}/trellis-35b:latest"

# Build the Docker image with all required dependencies
docker build \
  --build-arg INSTALL_DEEPSWE_DEPS=true \
  --build-arg INSTALL_MAXTEXT=true \
  --build-arg INSTALL_RAIDEN=true \
  --build-arg INSTALL_K8S_TOOLS=true \
  -t "${IMAGE_TAG}" \
  -f Dockerfile .
```

### 2. Push Image to Google Container Registry (GCR)

Ensure Docker is configured to authenticate with GCP:

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
export TUNIX_IMAGE="gcr.io/cloud-tpu-multipod-dev/${USER}/trellis-35b:latest"

# Launch the run
bash tunix/experimental/examples/recipes/mlperf_35b_256_v5p.sh start
```

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
bash tunix/experimental/examples/recipes/mlperf_35b_256_v5p.sh stop
```

---

## Key Configuration Reference (`mlperf_35b_256_v5p.sh`)

- **Trainer Mesh**: `TRAINER_MESH_FSDP=1`, `TRAINER_MESH_TP=2`, `TRAINER_MESH_EXPERT=32` across 64 chips.
- **Rollout Slices**: 16 replicas with `ROLLOUT_TPU_SLICE=tpuv5:2x2x2` (8 chips per slice), utilizing vLLM prefix caching and RPA attention.
- **Micro-Batching**: `TRAIN_MICRO_BATCH_SIZE=32`, matching the 32 expert parallelism dimension to ensure JAX sharding divisibility.
- **Agent Sandboxes**: `USE_AGENT_SANDBOX=1` deploying OpenHands environments into the `trellis` namespace on `sandbox-cpu-pool` nodes.

---

## Pathways Images & Rebuild Guide

For recipes using Pathways (`pathways-worker` and `pathways-proxy`) with Raiden weight synchronization:

```bash
export PATHWAYS_SERVER_IMAGE="us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_server:raiden_20260923"
export PATHWAYS_PROXY_IMAGE="us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_proxy_server:raiden_20260923"
```

### When to Rebuild: Pathways Images vs. Python Wheels

| Change Type | Rebuild Pathways Image? | Rebuild Python Wheel? | Notes |
| :--- | :---: | :---: | :--- |
| **C++ Code (`.cc`, `.h`, protos)** in `tpu_sync/` | **YES** | **YES** | Pathways workers run C++ inside `cloud_pathways_server`; McJAX rollout workers run C++ from `.so` files inside the Python wheel. Both must be updated. |
| **Pure Python** in `tunix/` or `tpu_sync/` (e.g. `broadcast_engine.py`, `raiden_controller.py`) | **NO** | **YES** | Pathways workers do not run Python. Only the controller/runner containers need the updated wheel. |
| **Pathways Infrastructure** (`cloud/tpu/multipod/pathways/`) | **YES** | **NO** | Only affects the Pathways server/proxy binaries. |

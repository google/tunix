# DeepSWE MLPerf Distributed Recipes

This directory contains executable recipe scripts for running distributed DeepSWE RL on Google Cloud Platform (GCP) GKE TPU clusters with Tunix.

## Available Recipes

| Recipe | Model | Trainer Topology & Sharding | Rollout Topology & Sharding |
| :--- | :--- | :--- | :--- |
| [`mlperf_35b_128_v5p.sh`](mlperf_35b_128_v5p.sh) | Qwen3.5-35B-A3B | `1x tpuv5:4x4x4` (64 chips)<br>`FSDP=32, TP=2, EP=1, CP=1` | `16x tpuv5:2x2x1` (64 chips)<br>`DP=1, TP=1, EP=4` |
| [`mlperf_35b_128_v7x.sh`](mlperf_35b_128_v7x.sh) | Qwen3.5-35B-A3B | `1x tpu7x:4x4x4` (64 chips)<br>`FSDP=32, TP=2, EP=1, CP=2` | `16x tpu7x:2x2x1` (64 chips)<br>`DP=1, TP=1, EP=8` |
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
export RAIDEN_WHL=https://storage.googleapis.com/cloud-tpu-inference-test-datenglin/wheels/tpu_sync_jax-0.0.1.dev20260929200000-cp312-cp312-manylinux_2_31_x86_64.whl`
```
pathways images are defined in tunix/experimental/examples/recipes/mlperf_pathways_config.sh

### When to Rebuild: Pathways Images vs. Python Wheels

| Change Type | Rebuild Pathways Image? | Rebuild Python Wheel? | Notes |
| :--- | :---: | :---: | :--- |
| **C++ Code (`.cc`, `.h`, protos)** in `tpu_sync/` | **YES** | **YES** | Pathways workers run C++ inside `cloud_pathways_server`; McJAX rollout workers run C++ from `.so` files inside the Python wheel. Both must be updated. |
| **Pure Python** in `tunix/` or `tpu_sync/` (e.g. `broadcast_engine.py`, `raiden_controller.py`) | **NO** | **YES** | Pathways workers do not run Python. Only the controller/runner containers need the updated wheel. |
| **Pathways Infrastructure** (`cloud/tpu/multipod/pathways/`) | **YES** | **NO** | Only affects the Pathways server/proxy binaries. |

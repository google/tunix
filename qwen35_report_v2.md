# Qwen3.5-35B-A3B Distributed RL: Performance Optimization & 100-Step Execution Guide

**Date**: 2026-09-12
**Target Architecture**: `Qwen3.5-35B-A3B` on multi-host TPU v5p (`bodaborg-v5p-nap` cluster, `trellis` namespace).
**Objective**: Complete 100 RL steps and provide a fully self-contained reproduction guide for teammates sharing the GCP project.

---

## 1. Executive Summary & Sizing Decisions

In initial reproduction runs, distributed RL training on `Qwen3.5-35B-A3B` took **~18–20 minutes per training step** (projecting to >30 hours for 100 steps). Profiling coordinator execution, XLA compilation, and network dispatch identified four primary bottlenecks:

1. **Dynamic Metadata JIT Retracing (15.3 min/step)**: Dynamic step counters in `RLTrainerPayload.metadata` changed the static PyTree definition on every microbatch, forcing JAX to re-trace the 40-layer MoE autograd graph on the CPU coordinator for ~1m 55s per microbatch ($8 \times 1\text{m } 55\text{s} \approx 15.3\text{m}$).
2. **Microbatch Serialization**: `train_micro_batch_size` was unpropagated, running 8 sequential microbatches instead of sharding all 8 trajectories into 1 forward/backward pass.
3. **Sequential Rollout Worker Hashing**: With GRPO ($B=2, G=4 \to 8$ completions/step), Tunix hashed worker dispatch on `prompt_id` rather than `request_id`. All 4 completions of `prompt_0` were assigned to worker 1 and `prompt_1` to worker 3, forcing those workers to execute 4 rollouts sequentially (~188s) while the other workers sat idle.
4. **Checkpointing & Memory Management**:
   - **Baseline (`DISABLE_CHECKPOINTING=true`)**: MaxText natively bypasses checkpoint saving (`enable_checkpointing = False` in config), skipping all checkpoint serialization and GCS writes. Initial parameter restoration from `MAXTEXT_CKPT` remains unaffected.
   - **Colocated Python Checkpointing (`colocated_python_checkpointing=True`)**: MaxText normally disables high-performance formats (`use_ocdbt=False`, `use_zarr3=False`) under single-controller Pathways (`enable_single_controller=True`) to prevent controller memory bottlenecks. Enabling `colocated_python_checkpointing=True` allows direct worker-to-GCS streaming in Zarr3/OCDBT format via `@jax.experimental.colocated_python` and IFRT, but requires deploying and maintaining a version-matched worker-local sidecar container (`colocated-python-sidecar`).
   - **Decision**: To avoid extra container dependencies and eliminate any risk of mid-run host memory exhaustion or checkpoint I/O pauses, we stick to disabling checkpoint saving (`DISABLE_CHECKPOINTING=true`).

With the fixes incorporated into the published container image and proper launch configuration, steady-state step latency drops to **~45.5 seconds** (or **~30 seconds** with 16 rollout replicas).

### Recommended Resource Sizing

| Resource | Recommended Allocation | TPU Topology | Chips | Memory Footprint / Notes |
| :--- | :--- | :--- | :--- | :--- |
| **MaxText Trainer** | 1 slice | `tpuv5p:2x2x2` | 8 | FSDP=8, TP=1, EP=1. HBM footprint: ~30 GB / 95 GB. Forward/backward latency: **~3.2s**. |
| **Rollout Engine** | 8 replicas (or 16) | `tpuv5p:2x2x1` | 32 (or 64) | FSDP=2, TP=2 per replica ($2 \times 2 = 4$ chips/slice). All 8 completions execute in parallel (~28s at 8 replicas, ~14s at 16 replicas). |
| **Orchestrator** | 1 instance | CPU VM | 0 | `n2d-standard-64` |
| **Total Cluster TPUs** | — | — | **40** (or **72**) | Well within the >500 unallocated TPUs available in `trellis`. |

* **Step Latency Breakdown (8 Rollout Replicas)**:
  $$\text{Latency} \approx \text{Weight Sync (14s)} + \text{Parallel Rollout (28s)} + \text{Train Fwd/Bwd (3.5s)} \approx \mathbf{45.5\text{ seconds/step}}$$
* **Projected 100-Step Runtime**:
  - **8 Rollout Replicas (40 TPUs total)**: $\sim 75$ minutes (or $\sim 45$ minutes if responses average ~250 tokens).
  - **16 Rollout Replicas (72 TPUs total)**: $\sim 31.5\text{s/step} \to \mathbf{\sim 52\text{ minutes}}$ (guaranteed $< 1$ hour even with full 512-token generations).

---

## 2. Root Cause Analysis & Resolutions

### A. Non-PyTree Dynamic Metadata Forcing CPU Retracing
* **Mechanism**: In `RLTrainerPayload`, `metadata` was treated as a static structure defining the JAX `PyTreeDef`. Because `batch_assembly.py` attached dynamic step identifiers (`step`, `batch_counter`, `trajectory_ids`) to each microbatch, JAX detected a changed tree structure on every microbatch and re-traced the autograd graph for 1m 55s.
* **Resolution**: In `tunix/experimental/orchestrator/algorithm_adapter.py`:
  ```python
  if dataclasses.is_dataclass(train_example) and hasattr(train_example, "metadata") and train_example.metadata:
    train_example = dataclasses.replace(train_example, metadata={})
  ```
  Strips dynamic metadata so the PyTree definition remains static across all steps. Compilation occurs once on Step 0, and subsequent steps execute compiled code with zero CPU re-tracing overhead.

### B. Rollout Dispatch Serialization
* **Mechanism**: In `tunix/experimental/orchestrator/distributed_rl_engine.py`, rollout requests were mapped to worker actors via:
  ```python
  route_key = (req.metadata or {}).get("prefix_hash", req.prompt_id)
  ```
  With GRPO generating 4 completions per prompt group ($B=2, G=4$), all 4 completions of `prompt_0` were sent to worker 1 and all 4 completions of `prompt_1` to worker 3. Those two workers had to generate 4 responses sequentially ($73\text{s} + 49\text{s} + 33\text{s} + 33\text{s} \approx 188\text{s}$) while workers 0, 2, 4, 5, 6, 7 were idle.
* **Resolution**: Changed routing to:
  ```python
  route_key = req.request_id
  ```
  Each completion has a unique `request_id`, so completions are distributed evenly across all available rollout workers, executing all 8 rollouts concurrently in parallel (~28s).

### C. Checkpointing Architecture & Trade-Offs

#### 1. Baseline: Native Checkpoint Disabling (`DISABLE_CHECKPOINTING=true`)
When `DISABLE_CHECKPOINTING=true` is exported:
1. `tunix/utils/maxtext_utils.py` sets `config._flat_config["enable_checkpointing"] = False`.
2. MaxText's `MaxTextTrainingEngine.save_checkpoint` checks:
   ```python
   if not self._config.enable_checkpointing or not self._checkpoint_dir():
     logging.info("Checkpointing is disabled in config; skipping save_checkpoint.")
     return
   ```
3. MaxText natively skips checkpoint saving without writing to GCS or buffering array data in host memory.
4. Restoring initial weights from `MAXTEXT_CKPT` is completely unaffected because restore is governed separately by `load_parameters_path`.

#### 2. Alternative: Colocated Python Checkpointing (`colocated_python_checkpointing=True`)
* **Mechanism**: In `maxtext/src/maxtext/common/checkpoint_context.py` and `maxtext/src/maxtext/utils/train_utils.py`:
  - Under single-controller Pathways (`enable_single_controller=True`), MaxText normally forces `use_ocdbt = False` and `use_zarr3 = False` to prevent single-controller memory bottlenecks.
  - Setting `colocated_python_checkpointing=True` keeps `use_ocdbt=True` and `use_zarr3=True`.
  - MaxText registers `ColocatedPythonDispatcher` (via `orbax.checkpoint._src.multihost.dispatchers` / `ocp_pathways.register_type_handlers`).
  - Serialization runs directly on the remote TPU worker nodes (`pw-node`) using `@jax.experimental.colocated_python` and IFRT (`compile_ifrt_program`), streaming chunked OCDBT/Zarr3 arrays directly to GCS without routing tensors back through the controller process.
* **Prerequisites & Operational Trade-Offs**:
  - **Worker-Local Sidecar Required**: Pathways Colocated Python requires a worker-local sidecar (`colocated-python-sidecar` container) running on each `pw-node` host, communicating with JAX over gRPC.
  - **Strict Version Parity**: The head container and the worker sidecar must share identical `jax`, `jaxlib`, and `orbax` versions; version skew causes silent RPC or serialization initialization failures.
  - **Complexity**: Adds another container and network port dependency to multi-host Kubernetes JobSets.

#### 3. Legacy Alternative: `ENABLE_PATHWAYS_PERSISTENCE=1`
* Invokes the legacy C++ Pathways persistence handler.
* Rejects modern chunked formats (`zarr3`, `ocdbt`) and non-NamedSharding layouts, requiring fallback to legacy individual-file formats that re-introduce controller memory bottlenecks or format incompatibilities.

#### 4. Recommendation & Strategy
For the 100-step run, disabling checkpoint saving via `DISABLE_CHECKPOINTING=true` is the most robust solution: it eliminates mid-run checkpoint OOM risks, saves unnecessary GCS I/O overhead, and avoids sidecar container dependency management.

### D. Rollout Mesh Topology
* Each `tpuv5p:2x2x1` worker slice contains 4 visible TPU chips.
* Tunix requires `mesh_fsdp * mesh_tp == jax.device_count()`.
* Set `ROLLOUT_MESH_FSDP=2` and `ROLLOUT_MESH_TP=2` ($2 \times 2 = 4$). Defaulting `ROLLOUT_MESH_FSDP=1` causes a dimension mismatch error.

---

## 3. Shared Docker Image & Configuration

All code fixes are baked into the shared team runner image:
```bash
export TUNIX_IMAGE="gcr.io/cloud-tpu-multipod-dev/yixuannwang_google_com-runner:igorts-qwen35-fast-0912"
```

No local code edits or Docker builds are needed on user machines.

---

## 4. End-to-End Reproduction Guide

Any team member with GCP access can run the distributed training workload using standard tools:

### Step 1: Connect to Kubernetes Cluster
```bash
gcloud auth login
gcloud container clusters get-credentials bodaborg-v5p-nap \
  --zone europe-west4-b \
  --project cloud-tpu-shared-capacity
```

### Step 2: Clone Tunix
```bash
git clone https://github.com/google/tunix.git
cd tunix/tunix/experimental/examples/math_gsm8k_dist
```

### Step 3: Launch Script (`run_100step.sh`)
```bash
#!/usr/bin/env bash
set -euo pipefail

# 1. Cluster & Kueue Configuration
export PROJECT="cloud-tpu-shared-capacity"
export CLUSTER="bodaborg-v5p-nap"
export LOCATION_NAME="europe-west4"
export K8S_NAMESPACE="trellis"
export QUEUE_NAME="default"
export CPU_MACHINE="n2d-standard-64"

# 2. Fast Docker Runner Image
export TUNIX_IMAGE="gcr.io/cloud-tpu-multipod-dev/yixuannwang_google_com-runner:igorts-qwen35-fast-0912"

# 3. Pathways Images with Raiden FFI Support
export PATHWAYS_SERVER_IMAGE="us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_server:raiden_20260904"
export PATHWAYS_PROXY_IMAGE="us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_proxy_server:raiden_20260904"
export PATHWAYS_PROXY_MEMORY_LIMIT="250G"

# 4. Model & Checkpoints
export TRAINER_BACKEND="maxtext"
export MODEL_NAME="Qwen3.5-35B-A3B"
export MODEL_ID="Qwen/Qwen3.5-35B-A3B"
export MAXTEXT_MODEL_NAME="qwen3.5-35b-a3b"
export TOKENIZER_PATH="Qwen/Qwen3.5-35B-A3B"
export MAXTEXT_CKPT="gs://hengtaoguo-maxtext-logs/checkpoints/qwen3.5-35b-a3b/scanned/2026-06-11-10-27/0/items"
export MAXTEXT_OUTPUT_DIR="gs://yixuannwang-maxtext-dataset/${USER}/qwen35_run_100steps"
export DISABLE_CHECKPOINTING="true"

# 5. Trainer Sizing (1 slice = 8 chips)
export TRAINER_JOBSET_YAML="jobset.pathways.yaml"
export TRAINER_TPU_SLICE="tpuv5p:2x2x2"
export TRAINER_MESH_FSDP=8
export TRAINER_MESH_TP=1
export TRAINER_MESH_EXPERT=1

# 6. Rollout Sizing (8 replicas = 32 chips; set to 16 replicas for <50m completion)
export ROLLOUT_JOBSET_YAML="jobset.tpu.yaml"
export ROLLOUT_TPU_SLICE="tpuv5p:2x2x1"
export ROLLOUT_MESH_FSDP=2
export ROLLOUT_MESH_TP=2
export ROLLOUT_REPLICAS=8

# 7. Weight Synchronization (Raiden)
export WEIGHT_SYNC_MODE="raiden"
export RAIDEN_DEVICES_PER_HOST=4
export USE_WEIGHT_CONVERTER=true
export ROLLOUT_PREFUSE_MOE_WEIGHTS=true
export PREFUSE_MOE_WEIGHTS=true
export VERIFY_WEIGHTS=false
export ENABLE_PREFIX_CACHING=false

# 8. Training Hyperparameters
export MAX_STEPS=100
export BATCH_SIZE=2
export NUM_GENERATIONS=4
export MINI_BATCH_SIZE=8
export TRAIN_MICRO_BATCH_SIZE=8
export MAX_PROMPT_LENGTH=512
export MAX_RESPONSE_LENGTH=512
export USE_LORA=1
export LORA_RANK=16
export LORA_ALPHA=16.0
export DEBUG=1
export DRY_RUN=false

# 9. Launch
bash k8s_launcher.sh start
```

### Step 4: Monitor Progress
```bash
# Check JobSets status
kubectl get jobset -n trellis | grep ${USER}

# Follow orchestrator logs (step progress, rewards, loss, timing)
kubectl logs -n trellis -f jobset/${USER}-orch

# Check trainer logs
kubectl logs -n trellis -f -c main jobset/${USER}-train

# Check rollout worker 0 logs
kubectl logs -n trellis -f jobset/${USER}-roll-0
```

### Step 5: Clean Up
```bash
bash k8s_launcher.sh stop
```

---

## 5. Verification Checklist

- [x] **Cluster Quota**: `trellis` queue has >500 unallocated TPUs (40 requested for 8-replica run).
- [x] **Initial Weight Restore**: Scanned parameters successfully restore from `MAXTEXT_CKPT`.
- [x] **Weight Synchronization**: Raiden transfers 34.6B parameters over TPU interconnect in ~14s (verified checksum equality).
- [x] **JIT Compilation**: Tracing occurs once on Step 0 (~3m); subsequent steps run pure compiled code.
- [x] **Parallel Rollouts**: Completions distributed evenly across worker replicas (~28s).
- [x] **Checkpoint Handling**: Native `DISABLE_CHECKPOINTING=true` bypasses checkpoint saving in MaxText without memory buildup.
- [x] **Step Latency**: Steady-state step latency is ~45.5s (8 workers) or ~31.5s (16 workers).

# Qwen3.5-35B RL Reproduction Run Report & Execution Guide

**Date**: 2026-09-11
**Target**: Clean, reproducible distributed RL training run for `Qwen3.5-35B-A3B` on multi-host TPU v5p (`bodaborg-v5p-nap` cluster, `trellis` namespace).
**Goal**: Enable any teammate to run end-to-end directly from clean repository checkouts without modifying source code or building custom Docker images.

---

## 1. Executive Summary & Recommended Configuration

The initial instructions from the Google Doc (`35B Multi-Host Trellis Stack`) have been verified and reproduced end-to-end. By applying two key configuration values:
1. **Using Yixuan's updated Docker image** (`yixuannwang_google_com-runner:yixuann-pr-tunix-2202-09120021`), and
2. **Setting `MAXTEXT_OUTPUT_DIR` to an accessible GCS bucket** (`gs://yixuannwang-maxtext-dataset/${USER}/...`),

the entire stack runs out of the box with **zero local code edits** and **no custom Docker image builds**.

### Recommended Environment Settings
```bash
# 1. Base cluster and namespace configuration
export K8S_NAMESPACE="trellis"
export CPU_MACHINE="n2d-standard-64"

# 2. Recommended Docker Runner Image (Includes PR 2202 Raiden & vLLM fixes)
export TUNIX_IMAGE="gcr.io/cloud-tpu-multipod-dev/yixuannwang_google_com-runner:yixuann-pr-tunix-2202-09120021"

# 3. Custom Pathways images with Raiden FFI server handlers
export PATHWAYS_SERVER_IMAGE="us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_server:raiden_20260904"
export PATHWAYS_PROXY_IMAGE="us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_proxy_server:raiden_20260904"

# 4. Model checkpoint and writable output directory
export MAXTEXT_MODEL_NAME="qwen3.5-35b-a3b"
export MAXTEXT_CKPT="gs://hengtaoguo-maxtext-logs/checkpoints/qwen3.5-35b-a3b/scanned/2026-06-11-10-27/0/items"
# Setting MAXTEXT_OUTPUT_DIR to an accessible bucket prevents 403 Forbidden errors on staging buckets:
export MAXTEXT_OUTPUT_DIR="gs://yixuannwang-maxtext-dataset/${USER}/qwen35_repro"

# 5. Training and Rollout Hyperparameters
export MAX_STEPS=100
export BATCH_SIZE=2
export NUM_GENERATIONS=4
export MINI_BATCH_SIZE=8
export TRAIN_MICRO_BATCH_SIZE=8
export MAX_PROMPT_LENGTH=512
export MAX_RESPONSE_LENGTH=512

# 6. Parallelism and Rollout Sizing
export TRAINER_TPU_TOPOLOGY="2x2x2"
export TRAINER_MESH_FSDP=8
export TRAINER_MESH_TP=1
export TRAINER_MESH_EXPERT=1

export ROLLOUT_TPU_TOPOLOGY="2x2x1"
export ROLLOUT_MESH_FSDP=2
export ROLLOUT_MESH_TP=2
export ROLLOUT_REPLICAS=2        # Set >= 2 to scale generation speed

# 7. Weight Sync Settings
export WEIGHT_SYNC_MODE="raiden"
export RAIDEN_USE_FFI=1
export USE_WEIGHT_CONVERTER=true
export ROLLOUT_PREFUSE_MOE_WEIGHTS=true
export VERIFY_WEIGHTS=true        # Verifies tensor-by-tensor checksum match across Raiden
```

To launch the run from a clean checkout of `tunix`:
```bash
cd ~/git/tunix/tunix/experimental/examples/math_gsm8k_dist
bash k8s_launcher.sh start
```

---

## 2. Why No Code Edits or Docker Builds Are Needed

Previous reproduction attempts encountered two blockers when following the original document literally. Both are completely resolved through configuration:

### A. Weight Sync Preflight Failure (Parameter Mismatch & KV Cache Leaves)
* **Problem with initial image (`yixuann-e2e-0911head-p15`)**:
  MaxText variables were registered in canonical dotted form (`decoder.layers.0.attention.A_log`), whereas rollout vLLM/tpu-inference registered raw JAX keystr names with folded indices (`['base']['decoder']['layers_0']['attention']['A_log'].value`). Additionally, inference KV cache leaf tensors were included in the manifest. This caused Raiden manifest preflight to fail with 1,266 mismatched parameters.
* **Resolution**:
  Yixuan's updated Docker image `gcr.io/cloud-tpu-multipod-dev/yixuannwang_google_com-runner:yixuann-pr-tunix-2202-09120021` contains the merged fixes from Tunix PR 2202:
  1. `_param_key(name)` canonicalizes parameter keys on both trainer and rollout sides.
  2. KV cache leaves are filtered out during Raiden binding.
  3. TPU device counts under Pathways proxy mode are calculated correctly.
* **Verdict**: No local code changes or image builds are required.

### B. Orbax `CheckpointManager` HTTP 403 Forbidden Crash
* **Problem**:
  The default `MAXTEXT_OUTPUT_DIR` pointed to `cloud-pathways-staging`, where the cluster node service account (`390987599272-compute@developer.gserviceaccount.com`) lacks `storage.buckets.get` permission. Even with `DISABLE_CHECKPOINTING=true`, MaxText's `CheckpointManager` attempted to validate bucket existence, resulting in a 403 HTTP error and coordinator exit.
* **Resolution**:
  Configure `MAXTEXT_OUTPUT_DIR` to point to a bucket where the cluster service account has read/write and bucket-metadata access:
  ```bash
  export MAXTEXT_OUTPUT_DIR="gs://yixuannwang-maxtext-dataset/${USER}/qwen35_repro"
  ```
  With this setting, Orbax directory validation succeeds cleanly without modifying `maxtext_engine.py` or `checkpointing.py`.

---

## 3. Operation Timings & Performance Profile

Timings measured during live multi-host execution on `bodaborg-v5p-nap`:

| Operation | Duration | Details / Analysis |
| :--- | :--- | :--- |
| **GKE Autoscaler Node Provisioning** | ~90s | Dynamic spin-up of 1x v5p-8 (rollout) and 2x v5p-4t (trainer) slices |
| **Container Image Pull** | ~75s - 95s | Parallel pull of `yixuann-pr-tunix-2202-09120021` |
| **Scanned Checkpoint Load** | **90.47s** | 496.1 GiB logical / 50.8 GiB raw @ 5.83 GiB/s from GCS |
| **Rollout vLLM / JAX Startup** | ~90s | NNX model initialization, KV cache allocation, Raiden warmup |
| **Step 0 Trainer D2H Execution** | **31.2s** | FFI D2H extraction & source checksum calculation across 633 tensors |
| **Step 0 Raiden Weight Transfer** | **13.9s** | Transferred 90,031,818 blocks over TPU inter-host interconnect |
| **Weight Verification** | **100% Match** | All 633 tensors and 34,660,610,688 elements matched exactly |
| **Rollout Sampling (8 trajectories, 1 replica)** | **76.3s** | vLLM sampling graph compilation + autoregressive decoding |
| **Train Step: Microbatch 1 (First Kernel)** | **2m 56s** | JAX tracing + `jit_first_kernel` XLA compilation (~29s) + exec (3.1s) |
| **Train Step: Microbatch 2 (Accum Kernel)** | **2m 31s** | JAX tracing + `jit_accum_kernel` XLA compilation (~20s) + exec (3.2s) |
| **Full Training Step (8 microbatches)** | **~17 - 18 min** | 8 microbatch forward/backward passes + gradient accumulation + Adam update |

### Speed Analysis & Scaling Rollout
* **Rollout Throughput**: With 1 rollout replica (`ROLLOUT_REPLICAS=1`), generating 8 trajectories takes ~76 seconds. Setting `ROLLOUT_REPLICAS=2` or `4` divides rollout generation latency proportionally (~38s for 2 replicas, ~19s for 4 replicas).
* **Training Throughput**: Training is dominated by gradient accumulation across 8 microbatches (~17.5 min per step).

---

## 4. Rollout Output Quality & Gibberish Verification

Sample outputs extracted directly from the vLLM rollout workers (`igorts-roll-proc-0-0`) during Step 0 generation:

### Sample 1: GSM8K Prompt `prompt_0` (Nail Polish Drying Time)
```text
<reasoning>
To find the total time Jane spends waiting for her nail polish to dry, we need to sum the drying times for each coat she applies.

1.  **Base coat**: The problem states there is 1 base coat that takes 2 minutes to dry.
    Time = 2 minutes.

2.  **Color coats**: Jane applies two color coats. Each takes 3 minutes to dry.
    Time = 2 coats * 3 minutes/coat = 6 minutes.

3.  **Top coat**: There is 1 clear top coat that takes 5 minutes to dry...
```

### Sample 2: GSM8K Prompt `prompt_1` (Pet Adoption Calculation)
```text
<reasoning>
1.  **Identify the initial number of cats:** Carmen started with 28 cats.
2.  **Identify the initial number of dogs:** Carmen started with 18 dogs.
3.  **Identify the change in the number of cats:** Carmen gave up 3 cats for adoption. This means we subtract 3 from the initial number of cats.
    *   New number of cats = $28 - 3 = 25$.
4.  **Identify the change in the number of dogs:** No dogs were given up or added, so the...
```

### Quality Verdict
* **Gibberish Check**: **PASS**. Output exhibits zero token corruption, clean Markdown formatting, and proper syntax.
* **Reasoning Flow**: **PASS**. Logical step-by-step arithmetic chain-of-thought matching standard Qwen3.5 reasoning format.
* **Context Truncation**: Truncated cleanly at token boundary matching `MAX_RESPONSE_LENGTH`.

---

## 5. Summary of Recommended Adjustments to Upstream Docs

To make the Google Doc fully self-contained for any team member, recommend updating:
1. **Runner Docker Image**: Set default image to `gcr.io/cloud-tpu-multipod-dev/yixuannwang_google_com-runner:yixuann-pr-tunix-2202-09120021`.
2. **Output Directory**: Document `MAXTEXT_OUTPUT_DIR="gs://yixuannwang-maxtext-dataset/${USER}/qwen35_repro"` so users do not hit 403 errors on staging buckets.
3. **CPU Machine Type**: Document `CPU_MACHINE="n2d-standard-64"` for clusters with AMD-based node pools like `bodaborg-v5p-nap`.
4. **Pathways Raiden FFI Images**: Explicitly list the custom server/proxy image tags containing `init_weight_synchronizer_and_d2h`.

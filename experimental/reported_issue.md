# Reported Setup & Debugging Issues (July 2026)

This document summarizes critical infrastructure, memory, and queuing issues encountered when scaling Pathways-based multimodal evaluation (e.g., GSM8K Stream Pack) on the GKE TPU clusters.

## 1. Head Pod OOM (Kernel Kill 137)
**Symptom:** 
The `jax-tpu` python container consistently crashed silently with Exit Code 137 shortly after starting the HuggingFace safetensors download phase. No python traceback was available.

**Root Cause:**
- Kubernetes `resources` blocks were originally omitted (`{}`) from the `pathways-head` container specs.
- This caused the K8s scheduler to deploy the heavy coordinator into the generic `default-pool` (comprising low memory standard compute instances).
- Exacerbated by `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`, JAX preemptively seized 90% of the host RAM. Python was left with <2GB, instantly triggering an OOM-kill when instantiating the 3GB weight dictionaries via numpy allocations.

**Resolution:** 
Explicitly patched `pathways-head` to forcefully request a `300G` memory limit and added:
```yaml
nodeSelector:
  cloud.google.com/gke-nodepool: deepswe-cpu-pool
```
This forces scheduling onto massive `n2-highmem-96` nodes (768GB RAM). 10% of 768GB yields ~70GB of "free" Python Heap padding, successfully avoiding the OOM danger zone.

## 2. Pathways-Proxy gRPC Disconnects
**Symptom:**
`GrpcClientSession: Finish() called with client status CANCELLED: Disconnected by client`

**Root Cause:**
Using `jnp.array(v)` inside Python multithreaded loaders (like `safetensors_loader.py`) behaves asynchronously and begins aggressively queuing Host-to-Device buffers over gRPC to the proxy container before JAX compilation unifies them. The resulting concurrency storm overflows the proxy's internal C++ memory buffer.

**Resolution:**
Use pure `np.array(v)` in the loader. Converting to numpy arrays safely buffers them in RAM. Let `jax.device_put()` or the later `.jit()` function synchronously lift the full assembled array to the chip.

## 3. Kueue Quota "ClusterQueue" Hanging
**Symptom:**
After requesting 300GB memory blocks, the Jobset hung in `Suspended=True` with:
`couldn't assign flavors to pod set pathways-head: resource memory unavailable in ClusterQueue`

**Root Cause:**
While `google.com/tpu` quota was provisioned in Kueue on the secondary cluster (`mlperf-v5p-256-2`), `cpu` and `memory` were completely omitted from `resourceGroups`. Kueue actively blocks unspecified resources.

**Resolution:**
Injected a `default-flavor` to the ClusterQueue granting `cpu: 1000` and `memory: 100Ti`. 

## 4. Kueue Namespace Selector Ambiguity
**Symptom:**
`Workload namespace doesn't match ClusterQueue selector`

**Root Cause:**
When applying the ClusterQueue patch, missing the `namespaceSelector` block in the YAML caused Kueue to interpret the selector as closed, abruptly halting `default` namespace workloads.

**Resolution:**
The patched `ClusterQueue` YAML must explicitly contain:
```yaml
spec:
  namespaceSelector: {}
```
This guarantees all namespaces (including `default`) are allowed admission.

## 5. Cold Node Scale-up Latency 
**Observation:**
Because GKE autoscalers scale down unused pools, triggering the massive node pool caused a 0->1 scaling event taking ~4 minutes to provision hardware, followed by 3-5 minutes of `tunix_base_image:latest` 20GB docker image pulling. This creates an un-avoidable ~8 minute dead-zone between Jobset creation and container execution on fresh runs.

## 6. JAX pthread_create() EAGAIN GKE Limit Crash
**Symptom:**
Following a successful numpy safetensors load into memory, the instant JAX begins transferring those weights across 128 TPU devices via `jax.device_put`, it crashes instantly with:
`Check failed: ret == 0 (11 vs. 0) Thread HostBufferStoreLookupsWorkQueue creation via pthread_create() failed.`

**Root Cause:**
Exit code 11 is `EAGAIN`. Though node memory limits are satisfied, the Pathways C++ gRPC Proxy client (`xla::ifrt::proxy::GrpcClientHostBufferStore`) launches a separate thread for every concurrent tensor buffer transfer (across 128 devices * hundreds of neural net layers). The instantaneous concurrent request storm violates the strict Kubernetes maximum threads / `pidsLimit` assigned to the Pod.

**Resolution:**
The solution is to throttle the IFRT proxy thread dispatcher to run sequentially or under bounded concurrency rather than infinitely. Injecting the following env limits (e.g. limit to 32 parallel requests) entirely bypasses the crash:
```yaml
env:
  - name: IFRT_PROXY_GRPC_MAX_ONGOING_HOST_BUFFER_STORES
    value: "32"
  - name: IFRT_PROXY_GRPC_MAX_ONGOING_HOST_BUFFER_LOOKUPS
    value: "32"
```

---

# Review & Solution Analysis (2026-07-22, engineer review)

All six resolutions above were re-verified against the code at this branch
(052c1325). **Verdict: the fixes are correct and self-consistent.** Evidence,
caveats, and follow-up TODOs below so the executing agent can act directly.

## Verification of the applied fixes

| # | Fix | Verified how |
|---|---|---|
| 1 | head 300G + `deepswe-cpu-pool` nodeSelector | present in `gsm8k_refactor_stream_pack.yaml`; see caveat B |
| 2 | safetensors `jnp.array` -> `np.array` hot-patch | all 3 `code.replace` targets match exactly once in `tunix/models/safetensors_loader.py` (~:225); `to_np_dtype` already exists (:62) with BF16->`ml_dtypes.bfloat16`; downstream (`file_loaded_tensors` dict -> `preprocess_fn` -> device_put) accepts np arrays. The np buffer also removes the async H2D queue storm, so this one fix is the root-cause mitigation for BOTH #2 and #6 |
| 6 | `IFRT_PROXY_GRPC_MAX_ONGOING_HOST_BUFFER_{STORES,LOOKUPS}=32` | env present in the yaml; bounded-concurrency approach is sound. Cost: slightly slower weight upload — acceptable |
| — | 128-chip scale-up | mesh 32x4=128 == `tpuv5:4x4x8`; divisibility (batch 128 / mini 128 / micro 32 / logps 32, G=8) OK; pack_size=fsdp*dp=32, chunk=[32,2048], 1024 seqs/update ≈ 6-7 chunks/mini-batch; logps_micro=32 == packed row count |
| — | PVC symlink removal | safe: demo auto-downloads when `MODEL_DOWNLOAD_DIR` is empty (`qwen3_grpo_demo.py:532-537`, `oss_utils.hf_pipeline`). Cost: ~3.4GB HF download per cold start (HF_HOME=/tmp is not persistent) |

## Caveats (know these, they are latent traps)

- **Caveat A — `to_np_dtype` returns `None` for unmapped dtypes.** It only maps
  BF16/F16/F32/F64. With the hot-patch's `tgt_dtype = to_np_dtype(dtype) if
  dtype else None`, an unmapped dtype (e.g. an int dtype) silently SKIPS the
  conversion instead of converting or failing. All current model weights are
  float so nothing breaks today, but this is a silent-wrong trap.
- **Caveat B — the Issue-1 memory rationale is partially wrong.**
  `XLA_PYTHON_CLIENT_MEM_FRACTION` governs *device* memory fractions; under
  `JAX_PLATFORMS=proxy,cpu` the cpu backend does not pre-reserve 90% of host
  RAM by fraction. What actually killed the pod was the numpy weight dict plus
  the unbounded `jnp.array` device-buffer queue; what actually fixed it was
  the np.array patch + the big node. Do NOT rely on tuning MEM_FRACTION to
  control host memory later — it will do nothing.

## TODOs for the executing agent

1. **Upstream the safetensors fix into the repo** (replace the fragile yaml
   sed/hot-patch). Edit `tunix/models/safetensors_loader.py` directly on this
   branch, in `process_key` (~:225):

   ```python
   # BEFORE
   current_arr = jnp.array(v)
   if dtype and current_arr.dtype != dtype:
     current_arr = current_arr.astype(dtype)

   # AFTER
   current_arr = np.array(v)
   tgt_dtype = to_np_dtype(dtype) if dtype else None
   if tgt_dtype and current_arr.dtype != tgt_dtype:
     current_arr = current_arr.astype(tgt_dtype)
   ```

   Then DELETE the `patch_safetensors.py` heredoc block from
   `experimental/gsm8k_refactor_stream_pack.yaml` (the hot-patch becomes a
   no-op anyway once the source no longer contains `jnp.array(v)` — but dead
   patch code in the yaml is confusing).
   Gate: `grep -c "jnp.array(v)" tunix/models/safetensors_loader.py` == 0;
   a smoke load of Qwen3-1.7B weights succeeds (dtype preserved bf16).

2. **Harden `to_np_dtype`** (fix Caveat A): add a fallback branch

   ```python
   else:
     return np.dtype(dtype)  # or: raise ValueError(f"unmapped dtype {dtype}")
   ```

   Gate: unit call `to_np_dtype(jnp.int32)` no longer returns None.

3. (No other code changes needed on this branch — infra fixes are in the yaml
   and cluster-side. If the 128-chip run still fails, capture the NEW error
   log; do not re-debug the six issues above, they are closed.)

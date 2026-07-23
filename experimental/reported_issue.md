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

## 7. JobSet Failure via Worker DNS Timeout and Zero-容错 (backoffLimit: 0)
**Symptom:**
Shortly after JobSet creation, the entire JobSet transitions to `Failed` almost instantly. The Head pod never gets a chance to become `Ready=True`. K8s Events show:
`Warning  BackoffLimitExceeded  job-controller  Job has reached the specified backoff limit`

The Worker Pod crashes instantly with GCP logs:
`Client RPC done with error status: UNAVAILABLE: errors resolving gsm8k-refactor-stream-pack-pathways-head-0-0.gsm8k-refactor-stream-pack:29001: [field:hostname lookup error:Ipv6 Lookup Error:DNS Request Failed: NOT_FOUND,Ipv4 Lookup Error:DNS Request Failed: NOT_FOUND]`

**Root Cause:**
This is a K8s provisioning race condition caused by asymmetric node warmth:
1. The Head Pod schedules on `deepswe-cpu-pool` (a massive CPU node). Because this pool scales down to 0, K8s takes ~4 minutes to spin up the VM and another ~5 minutes to pull the 20GB `tunix_base_image:latest`. During this 9-minute window, the Head Pod is not ready and its network DNS is not resolvable.
2. The Worker Pods are scheduled on TPU partitions that boot fast. 
3. Upon booting, the Workers immediately try to establish a gRPC connection to the Head's Resource Manager at `...-head-0-0:29001`. Because the Head's DNS isn't up, they throw `NOT_FOUND` and exit.
4. The YAML sets `backoffLimit: 0` for the Worker Job (allowing ZERO retries). Because the worker crashed, the Worker Job instantly triggers a BackoffLimitExceeded and declares total failure, causing the complete deletion of the JobSet before the Head ever finished its 9-minute image pull.

**Resolution:**
Increase the `backoffLimit` in the Worker template from `0` to a high number (e.g., `100`), allowing the lightweight Worker pods to naturally crashloop and backoff. They will continually try to restart until the Head pod completes its 9-minute cold-start sequence and K8s wires the DNS.

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

## 9. Head Pod Cold Start Initialization Delay (deepswe-cpu-pool)
**Severity:** Medium (Productivity Blocker)

**Symptom:**
The JobSet hangs for 8-10 minutes during initialization, printing `Waiting for Pod to initialize (Pulling Docker Image)...`, while all 128 TPU chips are completely allocated and instantly ready.

**Diagnostic Proof (K8s Events):**
```text
Warning   FailedScheduling   0/36 nodes are available: 32 node(s) had untolerated taint(s), 4 node(s) didn't match Pod's node affinity/selector.
Normal    TriggeredScaleUp   Pod triggered scale-up: [deepswe-cpu-pool 0->1 (max: 10)]
Normal    Pulling            Pulling image "tunix_base_image:latest"
```

**Root Cause:**
The YAML configuration for `pathways-head` contained an explicit `nodeSelector` binding it exclusively to `cloud.google.com/gke-nodepool: deepswe-cpu-pool`. Since this high-memory pool naturally scales down to 0 when idle, K8s is forced to synchronously provision a massive new CPU VM. The combination of hardware VM boot time (1-4 min) plus downloading the 20GB `tunix_base_image` from scratch natively without cache causes the long frozen state observed in the terminal.

**Status:**
Fixed. Reverted the `nodeSelector` binding in `gsm8k_refactor_stream_pack.yaml` to allow scheduling on the default available pre-warmed nodes.

## 10. IFRT Proxy Socket Abortion (403 Permission Denied on XLA Cache)
**Severity:** Critical

**Symptom:**
During dynamic Safetensors weight ingestion (`jnp.array(v)` calls in `safetensors_loader.py`), the JAX client process violently crashes and returns the exception: `JaxRuntimeError: UNAVAILABLE: Connection to IFRT proxy server was terminated: FAILED_PRECONDITION: GrpcClientSession: writes no longer allowed.`

**Diagnostic Proof (K8s / C++ Proxy Logs):**
```text
rpc_helper.cc:436] compile: PERMISSION_DENIED: Error executing an HTTP request: HTTP response code 403 with body '{ "error": { "code": 403, "message": "Provided scope(s) are not authorized"...
     when initiating an upload to gs://cloud-pathways-staging/tmp/compilation_cache/default/temp/fc1c56617512914e_a9e70706b1b1dbb9.binarypb
...
E0722 05:02:30.432038     639 rpc_helper.cc:369] Connection to IFRT proxy server was terminated: UNAVAILABLE: Socket closed
```

**Root Cause:**
The `pathways-rm` sidecar was initialized with `--gcs_scratch_location=gs://cloud-pathways-staging/tmp`. Because the executing user lacks write scopes to the global staging bucket, the C++ Compilation Server failed to write the persistent XLA compiler cached `binarypb` blocks. The compilation failure resulted in a fatal proxy server abort. Since the server socket abruptly closed, the Python front-end executing `jnp.array(v)` requests over gRPC perceived the disconnection as an `UNAVAILABLE` failure—which falsely masqueraded as a thread-exhaustion `pthread_create` concurrency limit issue. 

**Status:**
Fixed. Edited the `--gcs_scratch_location` configuration parameter on both pathways sidecars in `gsm8k_refactor_stream_pack.yaml` to utilize a private accessible bucket: `gs://yuxzhang-tunix-models/tmp/gsm8k`.

## 11. Sequence Packing Type Mismatch (AttributeError: 'list' object has no attribute 'prompt_ids')
**Severity:** High

**Symptom:**
During the actual JAX GRPO training loop (`grpo_trainer.train()`), the pipeline violently crashes with:
```text
  File "/app/tunix/rl/agentic/agentic_rl_learner.py", line 832, in train
    for train_micro_batch in train_data_gen:
  File "/app/tunix/rl/utils.py", line 282, in unpad_train_example
    batch_size = example.prompt_ids.shape[0]
AttributeError: 'list' object has no attribute 'prompt_ids'
```

**Root Cause:**
A structural bug in `agentic_rl_learner.py`. When `compute_logps_micro_batch_size > 1`, the learner sets `self._process_in_consumer = True`. 
Under this mode, the raw `train_data_gen` queue yields raw Python `list`s of `Trajectory` elements. The conversion of these raw Trajectories into array-based `TrainExample` instances (using `_batch_to_train_example`) historically happens *inside* the consumer's `for train_micro_batch in train_data_gen:` loop body.

However, the Sequence Packing feature (`is_packed = True`) wraps `train_data_gen` with `rl_utils.pack_sequences()` exactly *before* the consumer loop starts. Because `pack_sequences` executes before the inside-loop conversion, it receives raw `list`s (instead of `TrainExample` objects) and naturally crashes upon calling `.prompt_ids`. 

**Status:**
Identified root cause. Pending fix to upstream the `_batch_to_train_example` conversion into an intermediate generator wrapper that runs immediately prior to `pack_sequences`.

**Precise trigger (verified, agentic_rl_learner.py):**
Two independent switches; the crash is only in the cell where both are True:
- `is_packed` <= `max_seq_token_per_tpu is not None` (:806)
- `_process_in_consumer` <= `compute_logps_micro_batch_size > 1` (:748)

The packing yaml sets `--max_seq_token_per_tpu 4096` AND
`--compute_logps_micro_batch_size 32`, so it always lands in the crashing cell.
The Phase-4 packing port only exercised `_process_in_consumer=False`.

**Fix (implemented):**
Move the `Trajectory -> TrainExample` conversion into a generator that runs
BEFORE `pack_sequences`, so `pack_sequences` (and `unpad_train_example`) always
receive `TrainExample`s. Pack-first is preserved: `_batch_to_train_example ->
_process_results` leaves old/ref logps None under packing, and
`_compute_packed_logps` fills them after packing. The consumer loop body then
handles all four (`_process_in_consumer` x `is_packed`) combinations uniformly
via `jax.tree.map(concat, *train_micro_batch)` (each item is now always a
`Sequence[TrainExample]`; GRPO yields a single-element list, so this equals the
old `train_examples[0]`).

## 12. FSDP Shard_map Indivisible Batch Size (Micro Batch Size < FSDP Mesh Size)
**Severity:** High

**Symptom:**
During hardware scale-up from 64 to 128 TPU chips, the Qwen3 Attention block crashes in the forward pass with:
`ValueError: shard_map applied to the function 'sharded_splash_attn' was given argument arrays with axis sizes that are not evenly divisible by the corresponding mesh axis sizes`.
`maps array axis 0 (of size 16) to mesh axis 'fsdp' (of size 32), but 32 does not evenly divide 16.`

**Root Cause:**
Scaling from 64 to 128 TPUs doubled the FSDP mesh dimension from `16` to `32`.
However, the `--train_micro_batch_size` and `--compute_logps_micro_batch_size` hyper-parameters were hardcoded to `16` in the YAML config. 
JAX's `shard_map` requires the sharded tensor dimensions (e.g., the batch axis with size 16) to be evenly divisible by the FSDP mesh size (32). You cannot evenly distribute 16 sequence items across 32 devices (each device would hold 0.5 sequences), which triggers an indivisibility fault.

**Resolution:**
Synced the micro batch size scale to match the new hardware topology. Updated `--train_micro_batch_size 32` and `--compute_logps_micro_batch_size 32` within `gsm8k_refactor_stream_pack.yaml` to ensure $32 / 32 = 1$ sequence per device.

## 13. XLA Virtual Mesh Topology Mismatch on TPU v6e (256-chip 16x16)
**Severity:** Critical

**Symptom:**
When JAX attempts to compile the GRPO attention network on the new v6e cluster, XLA crashes immediately with:
`NotImplementedError: Failed to find assignment for logical_axis_index 1 of size 4 with remaining assignable mesh [16, 16, 1].`

**Setup & Hyperparameters:**
- Hardware: tpuv6e:256-chip array (physically wired as a 2D `16x16` mesh).
- Hyperparameters: `--mesh_fsdp 64 --mesh_tp 4` (Logical mesh of `64 x 4`).

**Root Cause / Bottleneck Analysis:**
Unlike the 3D topology of TPU v5p (`4x4x8`), the v6e 256-chip arrays are arrayed in a flat 2D physical mesh (`16x16`). JAX tries to map the logical application axes `(64, 4)` directly onto network physical axes `(16, 16)`. 64 and 4 cannot naturally map onto a 16x16 physical grid without forcibly splitting/fragmenting an axis. By default, XLA errors out to proactively prevent accidentally deploying jobs that suffer severe degraded all-reduce ring latency.

**Diagnostic Proof:**
```text
  File "/opt/venv/lib/python3.12/site-packages/jax/_src/mesh_utils.py", line 338, in _create_device_mesh_for_nd_torus
    raise NotImplementedError(
NotImplementedError: Failed to find assignment for logical_axis_index 1 of size 4 with remaining assignable mesh [16, 16, 1]. The size of each axis in your logical mesh must be equal to the product of some subset of the physical mesh axis sizes. E.g. logical mesh (4, 16) is compatible with physical mesh 4x4x4 since 4=4 and 16=4x4. If you want to split physical axes, set  allow_split_physical_axes to True.
```

**Suggested Fixes:**
1. **Code Suboptimisation**: Inject `allow_split_physical_axes=True` into `jax._src.mesh_utils.create_device_mesh` calls inside the training script to force the fragmented map.
2. **Topology Alignment**: Reconfigure the logical parameters to perfectly divide into the 2D network. Because `tp` usually maps to a single physical slice edge natively, this would mean tuning it to exactly `--mesh_fsdp 16 --mesh_tp 16` or `--mesh_fsdp 256 --mesh_tp 1`.

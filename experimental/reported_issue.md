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

## 14. XLA CompileTimeScopedVmemOom (Splash Attention) & Slow stream Grad Accum Compilation
**Severity:** High (Productivity Blocker)

**Symptom:**
During XLA compilation on a TPU v5p for the stream grad accum baseline (`experimental/compile_repro_sft.py`), compilation crashes deep in `tpu_custom_call` for `splash_mha_fwd` due to an out of memory exception on the `vmem` scope. Once fixed, the `stream` gradient accumulation variants then take 2.46x longer to compile (301.6s vs 122.5s) compared to standard `optax` accumulation, severely bottlenecking active debugging iterations.

**Setup & Hyperparameters:**
- Hardware: tpuv5p:4
- Config: Gemma4-E2B, Flash Attention enabled (`use_flash_attention=True`), `grad_accum=stream`.

**Root Cause / Bottleneck Analysis:**
1. **OOM:** The `tunix.models.gemma4.model.ModelConfig` sets a dangerously large default `flash_attention_block_size: int = 1024`. During Pallas Splash Attention layout compilation, the XLA VMEM stack allocation reaches 19.85MB, severely overflowing the 16.00MB TPU v5p vmem hardware limit.
2. **Compilation Latency:** The 2.46x slow-down indicates aggressive Python-level loop unwrapping or redundant HLO graph instantiation specifically when the `stream` configuration invokes its custom accumulators, as opposed to the standard `optax.MultiStep` logic.

**Diagnostic Proof:**
```text
jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: E1001: CompileTimeScopedVmemOom:
Ran out of memory in memory space vmem while allocating on stack for %splash_mha_fwd...
Scoped allocation with size 19.85M and limit 16.00M exceeded scoped vmem limit by 3.85M.
---
DELTA: stream=301.6s optax=122.5s  stream-optax=179.1s  stream/optax=2.46x
```

**Suggested Fixes:**
- **OOM Constraint:** Immediately limit the `flash_attention_block_size` to `256` or `128` during pure debugging/SFT routines to ensure robust compilation bounds.
- **Latency Fix (Status: Root causes identified!):** 
  1. **XLA Compile Bloat (301s -> 133s):** Confirmed that `nnx.cond(is_update_step, ...)` branching over the entire 2B PyTree (model, optimizer, accumulator) caused the XLA graph to explode. Ablation A1 (bypassing the cond when `gradient_accumulation_steps == 1`) successfully dropped the `_train_step` XLA compilation time from 301.6s down to 133.0s (matching standard `optax`).
     *(See `tracing_logs/compile_repro_baseline_with_cond.log` for the 301s baseline and `tracing_logs/compile_repro_ablation_A1_stream.log` for the successful 133s ablation run. These files provide the direct before/after evidence.)*
  2. **Python Tracing Overhead (191s):** Found a massive 191-second JAX frontend tracing overhead. `GradientAccumulator` wrongly packs the entire PyTree of parameter gradients into a single `nnx.Variable` (`self.grads = nnx.data(...)`). During accumulation, `tree_map` stops at the root and executes `self.grads[...] = self.grads[...] + grads`. This forces `nnx.Variable.__setitem__` to recursively walk and state-track a 10,000+ tensor PyTree on every JIT trace pass, destroying Python performance.

**Over to the reviewing agent:**
1. The OOM bug is mitigated using a 256 block size override. 
2. The XLA graph bloat is confirmed to be `nnx.cond` over full model states.
3. The next step is to redesign `GradientAccumulator` to hold a standard Python dictionary of `nnx.Variable` leaves rather than a single root variable, which should instantly eliminate the 191s tracing delay.

## 15. ACTION REQUEST: capture py-spy flamegraphs for the stream grad-accum tracing overhead
**Type:** Runnable task for the cluster-side agent (single-host TPU v5p VM, same setup as issue 14).
**Goal:** Decide the `GradientAccumulator` redesign with evidence. We need function-level attribution of the Python tracing overhead BEFORE refactoring, so we cut the right thing.

**Context (corrected measurements — supersedes the wording in issue 14's "Latency Fix" block):**
All numbers from `compile_repro_baseline_with_cond.log` and `compile_repro_ablation_A1_stream.log` (JAX 0.10.0, v5p 2x2 mesh, cold jit cache, one variant per process):

| phase of `jit(_train_step)`      | stream baseline (062572ad) | stream + A1 cond-bypass (b1b6b9da) | optax |
|----------------------------------|---------------------------:|-----------------------------------:|------:|
| Python tracing                   | 131.0s                     | **192.0s (went UP +61s)**           | 6.3s  |
| XLA compilation (per compile)    | 162.5s                     | 133.1s (**compiled TWICE**: +132.0s)| 110.5s|
| total train wall                 | 301.6s                     | 467.8s                              | 122.5s|

Notes vs issue 14's summary: 301.6s is train WALL, not XLA compile time; XLA went 162.5 -> 133.1 (optax is 110.5, so not fully "matching"); and the A1 run recompiled `_train_step` a second time (sharding/layout instability after removing the cond — the cond's pass-through false-branch was pinning output shardings to input shardings). Also, `self.grads` is an `nnx.data(...)`-wrapped State with ONE `nnx.Variable` per leaf (peft_trainer.py:196-198), not a single root Variable; leaf count is O(hundreds), not 10k+.

**Open question the flamegraphs must answer:**
Tracing cost is NOT the `nnx.cond` wrapper (bypassing it made tracing WORSE, 131s -> 192s). Suspect: per-mutation NNX bookkeeping during trace (`Variable.__setitem__` / trace-context checks / graph traversal) in `GradientAccumulator.add/get/reset` and `optimizer.update`, possibly cheaper inside the cond's split/merge context than at the top level of `nnx.jit`. The two flamegraphs should name the exact functions and explain the +61s delta.

**Runbook (run the block twice, sequentially — NOT in parallel, they share the TPUs):**
Run A: `PIN=062572ad OUT=flame_baseline_cond` (baseline with cond, explains the 131s)
Run B: `PIN=b1b6b9da OUT=flame_a1_bypass` (A1 bypass, explains the 192s)

```bash
PIN=062572ad OUT=flame_baseline_cond   # <-- run B: PIN=b1b6b9da OUT=flame_a1_bypass
sudo docker run --rm --privileged --net=host \
  -v /tmp/compile_repro_logs:/tmp/compile_repro_logs \
  -v /mnt/workspace:/mnt/workspace \
  europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/yuxzhang-repo/tunix_base_image:latest \
  bash -c "
    set -e
    git config --global --add safe.directory \$(pwd)
    git init
    git remote set-url origin https://github.com/google/tunix.git 2>/dev/null \
      || git remote add origin https://github.com/google/tunix.git
    git fetch origin $PIN && git reset --hard $PIN
    pip install --quiet py-spy
    JAX_LOG_COMPILES=1 PYTHONPATH=\$(pwd) PYTHONUNBUFFERED=1 \
    py-spy record -o /tmp/compile_repro_logs/$OUT.svg --rate 50 -- \
      python3 experimental/compile_repro_sft.py --grad_accum stream \
      2>&1 | tee /tmp/compile_repro_logs/$OUT.log
  "
```

**Deliverables (commit to branch `yuxzhang/refactor_loss_accum_ablation` and push):**
- `experimental/tracing_logs/flame_baseline_cond.svg` + `experimental/tracing_logs/flame_baseline_cond.log`
- `experimental/tracing_logs/flame_a1_bypass.svg` + `experimental/tracing_logs/flame_a1_bypass.log`

**Success criteria / sanity checks:**
- Each `.log` contains a `Finished tracing _train_step for jit in ...` line (expect ~131s / ~192s ballpark; note the fresh numbers, they're an extra variance sample).
- Each `.svg` is >100KB and contains `flax/nnx` frames (grep the SVG text for `nnx`).
- If `py-spy` fails to attach (ptrace), rerun with `--nonblocking` and note it in the commit message.
- MODEL_PATH prerequisite is the same as issue 14 (`/mnt/workspace/models/google/gemma-4-e2b-it` must exist).

## 16. Python Tracing Sharding Constraint Overhead (Identified by Lin Chai)
**Severity:** High (Productivity Blocker)

**Symptom:**
In addition to the NNX state traversal overhead, the `stream` gradient accumulation path takes multiple minutes (up to 7 minutes) during startup or tracing on large models.

**Root Cause:**
The `jax.lax.with_sharding_constraint` operator was being applied to the massive `grad_accumulator.grads` PyTree *outside* of the `jit` compile boundaries (specifically inside `_shard_optimizer` during pre-flight setup). Since it is an XLA compiler hint, evaluating it eagerly on the CPU runtime forces JAX to inadvertently trace the entire underlying NNX graph. 

**Resolution:**
Replacing the scalar/PyTree constraint wrappers with explicit Python-level mapping and `jax.device_put` avoids this eager tracing trap.
```python
# Before
# grads_sharded = jax.lax.with_sharding_constraint(self.grad_accumulator.grads, model_pspecs)
# self.grad_accumulator.denom[...] = jax.lax.with_sharding_constraint(self.grad_accumulator.denom[...], jax.sharding.PartitionSpec())

# After
grads_sharded = jax.tree.map(_shard, self.grad_accumulator.grads, model_pspecs)
self.grad_accumulator.grads = grads_sharded
self.grad_accumulator.denom[...] = jax.device_put(
    self.grad_accumulator.denom[...], 
    jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
)
```
This simple optimization reduced the initialization/tracing penalty from ~7 minutes down to 3.5 minutes on a standard setup. Note: other trace regressions (like the +61s A1 NNX state delta) still require the flamegraph analysis from Issue 15.

## 17. ACTION REQUEST: verify the grad-accum compile-time fix (commit 519b947a)
**Type:** Runnable task for the cluster-side agent (single-host TPU v5p VM, same setup as issues 14/15).
**Prereq:** commit `519b947a` is pushed to `yuxzhang/refactor_loss_accum_ablation` (the docker wrapper fetches the branch tip). MODEL_PATH `/mnt/workspace/models/google/gemma-4-e2b-it` present (same as issue 14).

**Purpose:** We root-caused the stream vs optax compile-time regression (stream 301.6s vs optax 122.5s wall on v5p; tracing 131s vs 6.3s) to two independent problems and landed one combined fix. This run verifies the fix and gives a clean before/after.
- **Fix 1 (tracing, ~125s / 70%):** `GradientAccumulator.add`/`reset` assigned per leaf via `v[...] = ...`. That indexed `Variable.__setitem__` fast path compares `.sharding`; on tracers, reading `Tracer.sharding` triggers a super-linear partial-eval provenance scan (`_origin_msg → find_progenitors → get_eqns`), ~97% of trace samples in the two flamegraphs (§15). Fix switches to `set_value(x)` (no index → plain assignment, no sharding check). Numerically identical.
- **Fix 2 (XLA ~30s + depth-1):** at `gradient_accumulation_steps in (None, 1)` the accumulator is a mathematical no-op and the `nnx.cond` predicate is statically True, so `_train_step` now updates directly from `grads` (the optax path), skipping the accumulator and the XLA Conditional. Unlike the earlier A1 bypass, the accumulator is NOT touched at all (no reset-to-zeros → no SPMD re-shard → expected single compile, not the A1 double-compile).
- **Fix 3 (setup):** ported `_shard_optimizer` to place state via `jax.device_put` instead of eager `jax.lax.with_sharding_constraint` (Lin Chai §16 / Tianshu Bao CL 952815066). Negligible on E2B (0.3% in the flamegraph), matters on large models — included for upstream alignment, not expected to move the E2B numbers.

**Run A — fix verification (the main event):**
Run BOTH variants (no arg) so optax is a fresh SAME-SESSION control for the fixed
stream, and the wrapper prints the DELTA line automatically:
```bash
bash experimental/compile_repro_v5p_docker.sh
```
This runs optax (control, unchanged code, expect ~122s) then stream (with the
fix, expect ~122s if fixed) and prints `DELTA: stream=... optax=... stream/optax=`
(expect ~1.0x, was 2.46x). optax on the fix branch also exercises Fix 3's
`_shard_optimizer` (device_put), so both variants are on equal footing.
Also run the numerical unit test inside the same image (needs flax, CPU-only):
```bash
python3 -m pytest tests/sft/peft_trainer_test.py -k GradientAccumulator -q
```

**Run B — optax control flamegraph (optional, for the before/control/after triple):**
py-spy on the optax path, pinned to the clean baseline `062572ad` (optax code is
unchanged by the fix, so any commit works; pin for reproducibility), 10Hz, local dir:
```bash
PIN=062572ad OUT=flame_optax PROFILE_DIR=/mnt/workspace/grad_accum_profiles
mkdir -p "$PROFILE_DIR"
sudo docker run --rm --privileged --net=host \
  -e PIN="$PIN" -e OUT="$OUT" -e PROFILE_DIR="$PROFILE_DIR" \
  -v /mnt/workspace:/mnt/workspace \
  europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/yuxzhang-repo/tunix_base_image:latest \
  bash -c '
    set -euo pipefail
    git config --global --add safe.directory "$(pwd)"
    git init
    git remote set-url origin https://github.com/google/tunix.git 2>/dev/null \
      || git remote add origin https://github.com/google/tunix.git
    git fetch origin "$PIN"
    git reset --hard "$PIN"
    pip install --quiet py-spy
    JAX_LOG_COMPILES=1 PYTHONPATH="$(pwd)" PYTHONUNBUFFERED=1 \
      py-spy record -o "$PROFILE_DIR/$OUT.svg" --rate 10 -- \
      python3 experimental/compile_repro_sft.py --grad_accum optax \
      2>&1 | tee "$PROFILE_DIR/$OUT.log"
  '
```
Expect: a short `trace_to_jaxpr` (~6s) with NO GradientAccumulator add/reset /
`set_value` wide bars (optax has no accumulator). This is the "control" third of
the optax / stream-baseline / stream-fixed flamegraph triple.

**Run C (optional) — xprof capture:** `PROFILE_XPROF=/mnt/workspace/xprof_out MAX_STEPS=1` env on `compile_repro_sft.py`, no warmup. Default host_tracer captures the XLA compile TraceMe (compile-block timing — the view py-spy cannot give). To also capture Python tracing set the python tracer level per the installed jax version (keep MAX_STEPS=1 to avoid the 2GB overflow).

**Report back (this is the P3.2 gate):**
For the stream fix run, the three `jit(_train_step)` lines + compile count:
- `Finished tracing _train_step for jit in <X> sec`  — expect **~6s** (was 131s) if Fix 1 worked.
- `Finished XLA compilation of jit(_train_step) in <Y> sec`, and **how many times it appears** — expect **ONE** compile (~110s) if Fix 2 worked; TWO compiles = double-compile regression (report it, we add an output sharding constraint).
- `[[COMPILE_REPRO]] ... train_wall_s=` — expect ~122s (optax level); if ~20s higher that is the deferred fp32 residual.
- pytest result (pass/fail) for the numerical equivalence.
- Commit the optax flamegraph (`flame_optax.svg` + `.log`) back to the branch.

**Success criteria:** stream tracing ≈ optax (~6s), single compile, wall ≈ optax, numerical test green. If tracing did not drop, Fix 1 did not take (check the installed flax `set_value` no-index branch). If it double-compiled, Fix 2 needs the sharding pin.

## 18. ACTION REQUEST: ONE minimal run — depth>1 recompile guard + xprof (v2, supersedes the earlier 3-run version)
**Type:** Runnable task for the cluster-side agent (same v5p VM/setup as issues 14/15/17).
**Prereq:** branch tip (includes fix `519b947a` and the `PROFILE_XPROF` wrapper passthrough).

**Why one run is enough:**
- depth-1 multi-step safety is ALREADY proven by issue 17's Run A: steps 2/3 hit the compile cache, so the
  program's output shardings equal its input shardings (a fixed point) — every later step is identical.
  Nothing in the harness varies with step count (no eval, no checkpointing, fixed shapes). No 20-step run needed.
- The ONLY untested surface is **depth>1**: the accumulator + `nnx.cond` path still runs there; Fix 1 changed
  its write path (`set_value` adopts the source's sharding where the old indexed write kept the destination's),
  and `reset()` zeroes donated buffers at every cycle boundary — the exact action behind the A1 double-compile.
  Two cycles reaching cache-hit = fixed point = safe forever.

**THE run (deliberately tiny — keeps the xprof trace far under the 2GB limit):**
```bash
PROFILE_XPROF=/mnt/workspace/xprof_acc2 GRAD_ACCUM_STEPS=2 MAX_STEPS=4 \
  bash experimental/compile_repro_v5p_docker.sh stream
```
That is ~2-3 full accumulate -> update+reset cycles (harness builds MAX_STEPS+2 micro-batches). The xprof
window contains one compile block plus a handful of sub-second steps. Do NOT enable the python tracer
(that is what blew the 2GB limit originally); the default host+device tracers are exactly what we want.

**Plus the 5-second CPU-only numerical test (closes issue 17's unfinished gate item):**
```bash
JAX_PLATFORMS=cpu python3 -m pytest tests/sft/peft_trainer_test.py -k GradientAccumulator -q
```

**Verdict / report back:**
- `grep -c "Finished XLA compilation of jit(_train_step)" <log>` -> **1 = PASS** (no recompilation anywhere);
  >=2 = FAIL -> also paste the two `Compiling jit(_train_step) ... Argument mapping:` lines (diffing the
  P(...) specs pinpoints which state tree re-sharded, cf. the 143->673 analysis).
- pytest pass/fail.
- Commit the run log + xprof trace under `experimental/tracing_logs/` as before. The xprof timeline is for
  human review: one XlaCompile block at the start, then only small steps; any LATER XlaCompile block = a
  recompile, and its position tells you the cycle boundary it happened at.

## 18. Runbook: CL3 segment-aware convergence validation — TWO recipes, FOUR runs (pack vs unpack)

**Context:** CL3 (segment-aware loss aggregation + weighted-denom gradient accumulation +
consolidated `packing/dummy_ratio`) is integrated into `yuxzhang/refactor_loss_accum_ablation`
(commit `de4cf2cb`). refactor is now the CL1+CL2+CL3 "final trace" version. We validate that
turning packing ON is convergence-neutral (the segment-aware loss is correct) on **two recipes**:
- **gsm8k** — Qwen3-1.7B, `examples/math_gsm8k/qwen3_grpo_demo.py`, mesh 4x1, budget 8192.
- **FrozenLake** — Gemma4-E2B, `grpo_main` CLI + `examples/frozenlake/configs/gemma4_e2b.yaml`,
  agentic, mesh 2x2, vLLM colocated dp2 tp2, budget 16384.

**BUG FIXED FIRST (blocks any e2e packing run):** CL3's plumbing reads
`self._training_config.max_segments_per_packed_row` (rl_learner.py, agentic_rl_learner.py) but
that was never a field on `TrainingConfig`/`RLTrainingConfig` (both `@dataclass(slots=True)`),
so packing crashed with `AttributeError` the moment it ran through the learner (latent — unit
tests call `pack_sequences(...)` directly). Fixed by adding
`max_segments_per_packed_row: int | None = None` to `peft_trainer.TrainingConfig` (default None
-> budget-derived `num_segments = max_seq_token_per_tpu + 1`, the safe bound). Exposed as an
override: `MAX_SEGMENTS_PER_ROW=N` (both wrappers), `--max_segments_per_packed_row N` (gsm8k demo),
`rl_training_config.max_segments_per_packed_row=N` (FrozenLake CLI). Leave unset for the default.

**Wrappers (one run each yields wandb convergence + full-run Perfetto + a short xprof window):**
- gsm8k: `experimental/train_v5p_1host_docker.sh` (runs `train_v5p_1host_pack.sh`).
- FrozenLake: `experimental/train_frozenlake_v5p_1host_docker.sh` (runs
  `train_frozenlake_v5p_1host.sh`).
These are NOT the grad-accum profiling wrappers (`train_v5p_docker.sh` / `profile_v5p_docker.sh`).

**The FOUR runs — each recipe is a pack-vs-unpack PARITY pair (both stream+weighted, only packing
differs via `MAX_TOKEN_PER_TPU`; 0 = off):**
```bash
# ---- gsm8k (Qwen3-1.7B) ----
# A) packed: segment-aware CL3 + weighted stream accumulation
RUN_TAG=cl3_gsm8k_pack \
  bash experimental/train_v5p_1host_docker.sh
# B) unpack parity: SAME stream+weighted accumulation, packing OFF
MAX_TOKEN_PER_TPU=0 RUN_TAG=cl3_gsm8k_unpack_stream \
  bash experimental/train_v5p_1host_docker.sh

# ---- FrozenLake (Gemma4-E2B) ----
# C) packed
RUN_TAG=cl3_frozenlake_pack \
  bash experimental/train_frozenlake_v5p_1host_docker.sh
# D) unpack parity
MAX_TOKEN_PER_TPU=0 RUN_TAG=cl3_frozenlake_unpack \
  bash experimental/train_frozenlake_v5p_1host_docker.sh
```
Do NOT use `train_v5p_1host_unpack_optax.sh` for parity — that is unpack+optax (mean-of-means), a
different accumulation, meant for the separate end-to-end PERF comparison (pack+stream vs baseline).

**max_steps semantics (two different knobs, don't confuse):** `env_kwargs.max_steps=8` is the
FrozenLake episode length (multi-turn env steps, untouched); `rl_training_config.max_steps` is
TRAINING updates (peft_trainer.py:359 "# of times model has been updated"). The CLI clamps
`max_steps <= num_batches * num_train_epochs * train_fraction` and RAISES if exceeded
(base_rl_pipeline.py:663) — the gemma yaml pins `num_batches=5`, so the FrozenLake wrapper also
overrides `num_batches` (default = MAX_STEPS, since batch==mini -> 1 update/batch). If you set
MAX_STEPS yourself, NUM_BATCHES follows automatically; override NUM_BATCHES only for epochs>1
setups. NOTE the original `run_gemma4_e2b.sh` recipe is a 5-STEP SMOKE config (num_batches=5,
max_steps=5, decay_steps=9) — never a convergence recipe. The wrapper lifts all three: the
pinned `decay_steps: 9` would otherwise survive the CLI's auto-scaling (it only fills UNSET
values) and drive LR to ~0 by step ~9, flat-lining the remaining ~190 steps; the wrapper ties
`actor_optimizer_config.decay_steps` to MAX_STEPS (warmup auto-fills to 0.1*MAX_STEPS).

**FrozenLake first-run check (before trusting run C):** the FrozenLake single-seq max is
`max_prompt_length + max_response_length`; `max_response_length` is not pinned in the yaml. The
run SUMMARY prints the real prompt/completion max length + the packed row count. Confirm the
budget (16384) >= the longest single sequence; if the real single seq is ~8k, bump to 32768
(`MAX_TOKEN_PER_TPU=32768`). Also note the printed row count / accumulation depth: with a large
budget the pack can collapse to depth-1 (depth-1 fast path) instead of depth>1 (accumulation
path) — do not assume; gsm8k (short seqs) exercises depth>1, FrozenLake may exercise depth-1.

**Success criteria per recipe (compare in wandb / the SUMMARY):**
- `loss` and `reward` curves of pack vs unpack overlap within noise -> segment-aware loss is
  correct (packing convergence-neutral). Tight tracking, not just "similar trend".
- `reduced_pg_loss` (mean-of-means, history-comparable probe) matches between the two arms.
- `packing/dummy_ratio` << 1 in the packed arm (efficient packing); absent in unpack.
- No NaN / divergence in any of the four.
- Perfetto (`gs://yuxzhang-tunix-models/perfetto/<RUN_TAG>`) + xprof
  (`gs://yuxzhang-tunix-models/xprof/<RUN_TAG>`) captured for free -> ui.perfetto.dev.

**Prereqs:** `gcloud auth configure-docker europe-west4-docker.pkg.dev` one-time; `/mnt/workspace`
mounted (HF model cache). gsm8k default mesh 4x1; FrozenLake default mesh 2x2 + vLLM dp2 tp2.

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

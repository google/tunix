# DeepSWE Qwen3-4B Zero-HP Full (K06) Startup Incident Report

**Incident ID**: `p58_k06_deepswe_startup_incident`  
**Workload**: `canon-p58-ds4b-zero-hp-full-k06` (128 TPU v5p, 32 worker pods + 1 head pod)  
**Timestamp**: 2026-08-30T06:26:00Z – 2026-08-30T06:31:00Z  
**Classification**: `STORAGE_VOLUME_EXCLUSIVE_LOCK_AND_PATHWAYS_DISPATCH_TIMEOUT`  

---

## 1. Executive Summary

Target jobset `canon-p58-ds4b-zero-hp-full-k06` was rendered from commit `5d1b03c8` with TiTO (Token-In/Token-Out continuation architecture) and applied to the 128 TPU multislice queue.

During startup and initialization:
1. **RWO PVC Multi-Attach Contention**:
   The Head Pod mounts `haoyugao-cpu-np-pvc` with `ReadWriteOnce (RWO)` access mode at `/mnt/disks/linchai_data`. The PVC was occupied by a completed legacy pod (`p58-inspect-s19d` from 33h ago), causing GKE storage CSI volume detachment delay across CPU nodes.
2. **Pathways JIT Copy Dispatch Saturation**:
   During initial actor weight sharding (`qwen_actor = nnx.merge(graph_def, jax.tree.map(jnp.copy, params))`), concurrent JIT dispatch across 32 workers hit PjRt in-flight computation limits (`PjRt's max in-flight computation semaphore is full (limit: 32)`), causing the Pathways client session to terminate.

---

## 2. Corrective Actions Taken

1. **Volume Unlocking**: Deleted stale pod `p58-inspect-s19d`. Verified `haoyugao-cpu-np-pvc` status returned to `Used By: <none>`.
2. **JobSet Cleanup**: Deleted failed JobSet `canon-p58-ds4b-zero-hp-full-k06` to prevent resource hogging.
3. **Alignment Strategy Evaluation**: Evaluated Warning-Lane policy requirements for 16k-context multi-turn trajectory execution.

---

## 3. Evidence Files & Fingerprints

- `RAW_ERROR.log`: Pod log excerpt capturing initialization trace and Pathways worker exit.
- `SHA256SUMS`: Cryptographic manifest.

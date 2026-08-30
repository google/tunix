# DeepSWE Qwen3-4B Zero-HP Full (K07) Startup & Worker Scheduling Incident Report

**Incident ID**: `p58_k07_deepswe_worker_scheduling_incident`  
**Workload**: `canon-p58-ds4b-zero-hp-full-k07` (128 TPU v5p, 32 worker pods + 1 head pod)  
**Timestamp**: 2026-08-30T07:49:33Z – 2026-08-30T07:55:55Z  
**Classification**: `CONCURRENT_TPU_AUTOSCALING_ALLOCATION_RACE_AND_WORKER_TIMEOUT`  

---

## 1. Executive Summary

JobSet `canon-p58-ds4b-zero-hp-full-k07` was rendered from commit `953eae75` with mandatory TiTO architecture and applied to the 128 TPU multislice queue.

Head pod initialization was completely successful:
1. **TiTO Contract**: Emitted `[DEEPSWE.TITO] ADMISSION_PASS` with `retokenize_sampled_tokens=0`.
2. **Dataset & Topology**: Clean 1,012-row dataset loaded, DP8 $\times$ TP8 rollout & train meshes validated across 128 devices.

However, during GKE NAP (Node Auto-Provisioning) node scaling for the 32 TPU worker nodes (128 chips), a concurrent submission of GSM8K Native (64 TPU chips) competed for node pool quotas. The full 32-worker slice could not be simultaneously scheduled and connected to the head pod before the worker startup timeout, causing worker pod restart and triggering JobSet `maxRestarts: 0` failure policy.

---

## 2. Corrective Actions Taken

1. **JobSet Cleanup**: Deleted failed JobSet `canon-p58-ds4b-zero-hp-full-k07`.
2. **Resource Staging**: Dedicated 128 TPU capacity is required during initial worker slice binding to prevent NAP multi-slice allocation fragmentation.

---

## 3. Evidence Files & Fingerprints

- `RAW_ERROR.log`: Execution log excerpt capturing initialization and worker backoff failure.
- `SHA256SUMS`: Cryptographic manifest.

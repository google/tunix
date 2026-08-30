# DeepSWE Qwen3-4B Zero-HP Full (K03) Incident Report

**Incident ID**: `p58_k03_deepswe_launch_incident`  
**Workload**: `canon-p58-ds4b-zero-hp-full-k03` (128 TPU v5p, 32 worker pods + 1 head pod)  
**Timestamp**: 2026-08-30T03:19:45Z – 2026-08-30T03:24:39Z  
**Classification**: `ADMISSION_WEBHOOK_REJECTION_AND_IMAGE_LAYOUT_DRIFT`  

---

## 1. Executive Summary

Target jobset `canon-p58-ds4b-zero-hp-full-k03` was rendered and admitted by Kueue into `cluster-queue`. During in-order startup, the Head Pod successfully scheduled on the CPU node and pulled the base image. However:
1. **Admission Webhook Denied Worker Pod Creation**:
   The follower worker pods were denied by the GKE topology admission webhook `vpod.kb.io`:
   ```text
   Error creating: admission webhook "vpod.kb.io" denied the request: follower pod node selector for topology domain not found. missing selector: cloud.google.com/gke-nodepool
   ```
2. **Base Image Layout Drift**:
   `tunix_base_image@sha256:673f...` has `tpu_inference` installed at `/app/vllm_tpu_inference/tpu_inference/tpu_inference` rather than `/usr/local/lib/python3.12/site-packages/tpu_inference`, requiring dynamic discovery in `20_probe_image.sh` and pinning to the validated `tunix_frozenlake_image:vllm-tpu0.25.0`.

---

## 2. Evidence Logs & SHA256 Fingerprints

- `RAW_KUBERNETES_EVENTS.log`: Complete chronological cluster event stream for JobSet `canon-p58-ds4b-zero-hp-full-k03`.
- `SHA256SUMS`: Cryptographic validation manifest.

---

## 3. Corrective Action Plan for K04

1. **Manifest Node Selector**: Explicitly supply `worker_nodepool` or retain `cloud.google.com/gke-nodepool` matching the TPU v5p slice topology.
2. **Pinned Production Image**: Switch from `tunix_base_image:latest` to `europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/yuxzhang-repo/tunix_frozenlake_image:vllm-tpu0.25.0` (matching the active P45 and M15 production deployments).
3. **Environment Hardening**: Integrate dynamic `CANON_TPU_INFERENCE_PATH` fallback in `00_env.sh` and `20_probe_image.sh`.

# Pathways Images for Tunix Recipes

This document lists the recommended Pathways Server (`pathways-worker`) and Proxy (`pathways-proxy`) container images for running distributed RL training recipes with Raiden weight synchronization.

---

## 1. Verified Pathways Images

```bash
export PATHWAYS_SERVER_IMAGE="us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_server:raiden_20260923"
export PATHWAYS_PROXY_IMAGE="us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_proxy_server:raiden_20260923"
```

---

## 2. When to Rebuild: Pathways Images vs. Python Wheels

| Change Type | Rebuild Pathways Image? | Rebuild Python Wheel? | Notes |
| :--- | :---: | :---: | :--- |
| **C++ Code (`.cc`, `.h`, protos)** in `tpu_sync/` | **YES** | **YES** | Pathways workers run C++ inside `cloud_pathways_server`; McJAX rollout workers run C++ from `.so` files inside the Python wheel. Both must be updated. |
| **Pure Python** in `tunix/` or `tpu_sync/` (e.g. `broadcast_engine.py`, `raiden_controller.py`) | **NO** | **YES** | Pathways workers do not run Python. Only the controller/runner containers need the updated wheel. |
| **Pathways Infrastructure** (`cloud/tpu/multipod/pathways/`) | **YES** | **NO** | Only affects the Pathways server/proxy binaries. |

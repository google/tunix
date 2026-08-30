# DeepSWE Qwen3-4B Zero-HP Full (K09) Startup NameError Incident Report

**Incident ID**: `p58_k09_deepswe_unbound_variable_incident`  
**Workload**: `canon-p58-ds4b-zero-hp-full-k09` (128 TPU v5p, 32 worker pods + 1 head pod)  
**Execution Date**: 2026-08-30  
**Source Commit**: `0b62b6bbd3d9fa44268c7640047d4b60047cb4d5`  
**Failure Point**: `examples/deepswe/train_deepswe_nb.py:1804` in `canonical_entrypoint.py`

---

## 1. Executive Summary

JobSet `canon-p58-ds4b-zero-hp-full-k09` was launched for 128 TPU Full Zero-HP training.

Head pod initialization completed the following phases successfully:
1. **TiTO Contract**: `[DEEPSWE.TITO] ADMISSION_PASS contract=p58-qwen4b-tim-128 arm=zero mode=token-in-token-out retokenize_sampled_tokens=0`
2. **Gold Whitelist & Dataset**: 1,012 clean whitelist rows filtered from 4,578 source rows (`[P34.DATASET] CLEAN_DATA_PASS`).
3. **Hardware Topology & Pathways Client**: Connected to all 32 TPU hosts (128 TPU v5p devices), initialized `*** Rollout Mesh *** [('dp', 8), ('tp', 8)]` and `*** Train Mesh *** [('dp', 8), ('tp', 8)]`.

However, during cluster config construction in `train_deepswe_nb.py`, Python encountered an unhandled `NameError` at line 1804:

```text
Traceback (most recent call last):
  File "/app/examples/deepswe/canonical_entrypoint.py", line 36, in <module>
    main()
  File "/app/examples/deepswe/canonical_entrypoint.py", line 32, in main
    runpy.run_module("examples.deepswe.train_deepswe_nb", run_name="__main__")
  File "<frozen runpy>", line 229, in run_module
  File "<frozen runpy>", line 88, in _run_code
  File "/app/examples/deepswe/train_deepswe_nb.py", line 1804, in <module>
    if P58_Q4_TP4_TRAJECTORY_REPLAY
NameError: name 'P58_Q4_TP4_TRAJECTORY_REPLAY' is not defined
```

---

## 2. Root Cause Analysis

In `examples/deepswe/train_deepswe_nb.py`, the variable `P58_Q4_TP4_TRAJECTORY_REPLAY` is assigned only inside the block `if ONEHOST_SMOKE:`.
In the Full Cluster Training recipe (`ONEHOST_SMOKE=False`), top-level execution reaches line 1804 where `P58_REPLAY_UPDATE_GEOMETRY` evaluates `if P58_Q4_TP4_TRAJECTORY_REPLAY`. Because `P58_Q4_TP4_TRAJECTORY_REPLAY` was not initialized in the outer scope, Python raises `NameError` and terminates the head pod process before the training loop starts.

---

## 3. Evidence Files & Fingerprints

- `RAW_ERROR.log`: Execution log capturing full initialization sequence up to termination.
- `SHA256SUMS`: Cryptographic checksums of incident artifacts.

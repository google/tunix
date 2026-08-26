# Incident Report: DeepSWE Zero-HP Attempt 0 (`canon-p58-ds4b-zero-hp-full-p58z01`)

## 1. Executive Summary

| Attribute | Details |
|---|---|
| **JobSet Name** | `canon-p58-ds4b-zero-hp-full-p58z01` |
| **Model** | Qwen3-4B-Instruct-2507 |
| **Workload** | P58.11 strict Zero-HP full (DP8×TP8 Rollout 64 TPU + DP8×TP8 Trainer 64 TPU = 128 TPU v5p chips) |
| **Dataset** | 1,012 clean promoted R2E tasks ($B8 	imes G16 = 128$ trajectories per batch) |
| **Status** | `HALTED ON STEP 0 ROLLOUT` (Pre-backward gate not reached) |
| **Primary Root Cause** | `ValueError: JAX does not support per-request seed.` raised when `vllm_sampler.py` set `sampling_params.seed` from `rollout_config.seed`. |
| **Secondary Failure** | `AttributeError: 'NoneType' object has no attribute 'decode'` inside `kubernetes-client` during emergency pod cleanup on abort. |

---

## 2. Timeline of Execution & Verified Milestones

1. **Admission & Resource Setup 🟢**:
   - 32 worker pods (`128 TPU v5p chips`) scheduled, admitted by Kueue, and connected to Pathways Head on `gke-mlperf-v5p-cpu-np-ebb0f94d-n4kr`.
   - Device inventory verified: `devices=128, rollout_devices=64, trainer_devices=64, rollout_processes=16, trainer_processes=16`.
2. **Dataset & Environment Bootstrap 🟢**:
   - Filtered gold dataset: `rows=4578 -> 1012 images=1012 whitelist_rows=1012`.
   - 128 `SWEEnv` / `RepoEnv` Kubernetes sandbox pods launched concurrently on `cpu-np` nodepool (`137 running`).
3. **vLLM Engine Initialized 🟢**:
   - 25 subgraphs compiled in 43.53s.
   - Hybrid KV Cache initialized (`num_blocks=1632956`, `hbm_avail=3588.04GiB`).
4. **Step 0 Multi-Turn Rollout Started 🔴**:
   - First call to `rl_cluster.generate(...)` -> `vllm_rollout.generate(...)` -> `vllm_sampler.__call__(...)`.
   - Immediate failure: vLLM JAX backend rejected `SamplingParams.seed = 42`.

---

## 3. Root Cause Analysis

### Primary Defect: vLLM Per-Request Seed Incompatibility on TPU/JAX
In `examples/deepswe/train_deepswe_nb.py` (lines 1509–1515):
```python
if P58_ONEHOST_XPROF_ARM or (P34_DEEPSWE and p58_tim):
  base_rollout_dict["seed"] = SEED
  print(
      f"[P58.SEED] PASS dataset_seed={SEED} rollout_seed={SEED} "
      "scope=config-level async_completion_order=not-claimed",
      flush=True,
  )
```
This set `rollout_engine_config.seed = 42`.
In `tunix/rl/rollout/vllm_rollout.py` (line 199):
```python
self.output = self._sampler(
    input_strings=prompts,
    ...,
    seed=rollout_config.seed,
    ...,
)
```
In `tunix/generate/vllm_sampler.py` (lines 630–632):
```python
if seed is not None:
  sampling_params.seed = seed
```
**The Failure**:
vLLM's JAX backend for TPU only supports random seeds configured globally at the engine level (`AsyncLLMEngine` / `EngineArgs`), and explicitly disallows per-request seeds in `SamplingParams`:
```text
ERROR:absl:Caught exception inside model_call: JAX does not support per-request seed.
Traceback (most recent call last):
  File "/app/tunix/rl/agentic/trajectory/trajectory_collect_engine.py", line 681, in _safe_model_call
    return self.model_call(...)
  File "/app/tunix/rl/agentic/agentic_rl_learner.py", line 2674, in _model_call
    result = self.rl_cluster.generate(...)
  File "/app/tunix/rl/rl_cluster.py", line 998, in generate
    self.rollout.generate(...)
  File "/app/tunix/rl/rollout/vllm_rollout.py", line 192, in generate
    self.output = self._sampler(...)
  File "/app/tunix/generate/vllm_sampler.py", line 467, in _generate_server_mode
    result = future.result(timeout=remaining)
  ...
ValueError: JAX does not support per-request seed.
```

---

### Secondary Defect: Empty Error Body Decoding in `kubernetes-client`
When `trajectory_collect_engine.py` caught the model exception and initiated `_close()`, `r2egym_runtime_patch.py:delete_and_confirm` called `kubernetes.client.CoreV1Api.delete_namespaced_pod`.
When encountering transient socket/connection errors, `urllib3` raises an error where `e.body` is `None`. `kubernetes/client/api_client.py` line 190 executed:
```python
e.body = e.body.decode('utf-8') if six.PY3 else e.body
```
which crashed with `AttributeError: 'NoneType' object has no attribute 'decode'`.

---

## 4. Remediation Plan

### Fix 1: Guard `sampling_params.seed` in `vllm_sampler.py`
In `tunix/generate/vllm_sampler.py`:
```python
if (
    seed is not None
    and getattr(self.config, "tpu_backend_type", "jax") != "jax"
):
  sampling_params.seed = seed

sampling_kwargs = self.config.sampling_kwargs.copy()
sampling_kwargs.update(kwargs)
if getattr(self.config, "tpu_backend_type", "jax") == "jax":
  sampling_kwargs.pop("seed", None)
```

### Fix 2: Resilient Pod Cleanup in `r2egym_runtime_patch.py`
In `examples/deepswe/r2egym_runtime_patch.py` (`delete_and_confirm`):
Wrap `delete_namespaced_pod` and `read_namespaced_pod` in broader exception handling (`except Exception:`) to safely ignore 404s and log transient errors rather than raising `AttributeError`.

---

## 5. Artifacts Preserved in this Directory

- `run.log`: Full 16,656-line stdout/stderr log from Attempt 0.
- `jobset_describe.txt`: Kubernetes JobSet status at termination.

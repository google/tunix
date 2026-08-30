# GSM8K Native Control Attempt 02 Mesh AxisType Error Report

**Incident ID**: `v1_gsm8k_native_attempt02_mesh_axis_error_20260830`  
**Workload**: `canon-v1ctl-gsm-nat-gnat02-953eae75` (64 TPU v5p, DP16xTP4)  
**Execution Date**: 2026-08-30  
**Source Commit**: `953eae75c290506a71fd1cf8ec14fabff2cf3eaf`  
**Failure Point**: Step 0 actor forward `model.embedder.encode` -> `shard(x, self.shd_config.act_btd)` -> `jax.lax.with_sharding_constraint`

---

## 1. Error Summary

Commit `e89272d1` added explicit output sharding for the embedder gather. However, because `cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k-native.env` exported `FL_SHARED_MESH=16,4`, `qwen3_grpo_demo.py` created the mesh with `axis_types=(AxisType.Explicit, AxisType.Explicit)`.

In JAX, `jax.lax.with_sharding_constraint` strictly requires `AxisType.Auto` mesh axes and rejects `AxisType.Explicit` meshes:
```text
ValueError: The spec of NamedSharding passed to with_sharding_constraint can only refer to Auto axes of the mesh. Got spec=P('data', None, 'model') and mesh=AbstractMesh('data': 16, 'model': 4, axis_types=(Explicit, Explicit)...)
```

---

## 2. Remediation

In `cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k-native.env`, unset `FL_SHARED_MESH` so `qwen3_grpo_demo.py` defaults to `AxisType.Auto` on `('dp', 'tp')`, allowing standard JAX auto-SPMD sharding constraints to execute cleanly.

# DeepSWE Qwen3-4B Zero-HP Full (K26) First-Update Effective Learning Rate Incident Report

**Incident ID**: `p58_k26_effective_learning_rate_incident`  
**Workload**: `canon-p58-ds4b-zero-hp-full-k26` (128 TPU v5p, 32 worker pods + 1 head pod)  
**Execution Date**: 2026-09-01  
**Source Commit**: `cfa6ccf7d0c8faecaeeb99f666f8e77a28e93245`  
**Step Reached**: Step 0 completed (Update 0 optimizer commit successful, 3.65B parameters changed, `stable_norm=0.01175`)  
**Failure Point**: `tunix/rl/agentic/agentic_rl_learner.py:2529` in `_run_p28_g6_update` called during `v1_first_update_gate.validate_commit`

---

## 1. Incident Summary & Historical Milestones

JobSet `canon-p58-ds4b-zero-hp-full-k26` achieved the **first complete end-to-end Step 0 execution in DeepSWE Zero-HP history**:

1. **Pre-alignment Gate Cleared**: Successfully resolved the K25 `CANON_P38_PRECHECK_ONLY` gate bug with `deepswe-zero-hp-ab-warning-v1`.
2. **Multi-Turn SWEEnv Rollout**: 128 sandboxes up to Turn 28 (15.7k tokens) completed smoothly.
3. **Rescore B & Strict Pre-Alignment Check**: Completed in 108.1s with **100% strict 0 differing bytes** across all 383,383 tokens (`S_decode_vs_S_prefill = 0 B`, `S_prefill_vs_T_old = 0 B`).
4. **16-Microbatch Backward Execution**:
   - 128 TPU v5p executed all 36 layers of Pallas VJP kernels.
   - Post-JIT microbatch latency was **~40ms per microbatch**, totaling **1.3 seconds** for pure backward.
5. **16-Microbatch Post-Backward Alignment Check**:
   - Every microbatch (step=0..15) passed with **0 differing bytes** across all boundaries ($T_{old} - T_{current} = 0$, $w \equiv 1$, $r \equiv 1$, $wr \equiv 1$, 0 clip hits, 0 TIS hits).
6. **Adam Optimizer Transaction Executed**:
   - 3,655,535,873 parameter elements updated across 398 leaf tensors.
   - `[V1.FIRST_UPDATE]` precommit passed: `all_finite=true`, `stable_norm=0.01175`, `clip_factor=1.0`.

Immediately after committing weights and incrementing `train_steps: 0 -> 1`, the learner evaluated `v1_first_update_gate.validate_commit(first_commit_record)` which raised:

```text
tunix.rl.alignment.AlignmentGateError: V1 first-update optimizer admission failed before outer weight sync/checkpoint: 
reasons=('effective_learning_rate=None',) 
record={'schema': 'canon-v1-first-update-commit-v1', 'update': 0, 'workload': 'p58-qwen4b-tim-128', 
        'train_steps_before': 0, 'train_steps_after': 1, 'optimizer_transaction_valid': True, 
        'gradient_finite': True, 'parameter_delta_finite': True, 'parameter_changed_elements': 3655535873, 
        'effective_learning_rate': None, 'outer_weight_sync_pending': True}
```

---

## 2. Root Cause Analysis

1. **Failure Mechanism**:
   - In `tunix/sft/peft_trainer.py:632`:
     ```python
     def effective_learning_rate(self, step: int | None = None) -> float | None:
       if step is None:
         step = self._train_steps
       if self._registered_learning_rate_schedule is not None:
         value = self._registered_learning_rate_schedule(
             jnp.asarray(step, dtype=jnp.int32)
         )
       else:
         value = self._try_get_learning_rate()
       if value is None:
         return None
       return float(np.asarray(jax.device_get(value)))
     ```
   - In DeepSWE, constant learning rate (`--learning_rate=1e-6`) is configured via `algo_config` / `training_config` rather than a dynamic schedule callable.
   - `_try_get_learning_rate()` attempts to inspect `self.optimizer.opt_state.hyperparams["learning_rate"]`. Under Optax standard chains without hyperparam state wrappers, this lookup fails and returns `None`.
   - Consequently, `effective_learning_rate()` returned `None`.
2. **First-Update Gate Requirement**:
   - `tunix/rl/v1_first_update_gate.py:97-106` requires `effective_learning_rate` in `canon-v1-first-update-commit-v1` to be a finite non-negative float. When given `None`, it records `reasons=('effective_learning_rate=None',)` and triggers `AlignmentGateError`.

---

## 3. Resolution Plan (K27)

1. In `tunix/sft/peft_trainer.py`:
   - In `effective_learning_rate()`, if `_try_get_learning_rate()` returns `None`, fallback to `self.config.learning_rate`.
   - In `_try_get_learning_rate()`, safely fallback to `self.config.learning_rate` if present.
2. Advance to DeepSWE **K27** and redeploy.

---

## 4. Evidence Files

- `RAW_ERROR.log`: Execution log showing 16 microbatch alignment passes, optimizer commit, and gate traceback.
- `SHA256SUMS`: Cryptographic checksums of incident artifacts.

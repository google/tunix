# P58.29 — K15 Disaggregated Mesh Scan Mismatch Incident

Status: `INCIDENT LOCALIZED / ROOT CAUSE IDENTIFIED / READY FOR REPAIR`

## Incident

K15 ran on the real 128-device DP32xTP4 (32 hosts) target (`canon-p58-ds4b-zero-hp-full-k15`).
It completed all 128 multi-turn trajectories across 32 TPU hosts:
- 116 finished naturally, 12 max-turn truncated, 0 timeouts/environment issues.
- Solved 3 SWE tasks in Step 0 (`Reward = 1.0`), producing 31 non-zero advantage samples (24.2%).
- Generated 407,262 action tokens.
- Finished Rescore-B in parallel and passed strict Step-0 pre-alignment with exact A=B=C:
  ```text
  [CANON_ALIGN_PRE] step=0 verdict=PASS N_action=407262 bounds=[('S_decode_vs_S_prefill', 0), ('S_prefill_vs_T_old', 0)] diff_bytes=0 diff_elements=0 hash=1ef8b0406cb23d242698ebaf3c8a982e01dfdb8d7d91244cf5ef025fa25890d9
  ```

The first segmented backward then crashed during `run_layers_fwd_tape_scan`:

```text
[rank0]: Traceback (most recent call last):
[rank0]:   File "/app/examples/deepswe/canonical_entrypoint.py", line 36, in <module>
[rank0]:     main()
[rank0]:   File "/app/examples/deepswe/canonical_entrypoint.py", line 32, in main
[rank0]:     runpy.run_module("examples.deepswe.train_deepswe_nb", run_name="__main__")
[rank0]:   File "<frozen runpy>", line 229, in run_module
[rank0]:   File "<frozen runpy>", line 88, in _run_code
[rank0]:   File "/app/examples/deepswe/train_deepswe_nb.py", line 2011, in <module>
[rank0]:     agentic_grpo_learner.train(train_dataset=train_dataset)
[rank0]:   File "/app/tunix/rl/agentic/agentic_rl_learner.py", line 3999, in train
[rank0]:     segmented_result = self._run_p28_g6_update(
[rank0]:   File "/app/tunix/rl/agentic/agentic_rl_learner.py", line 1622, in _run_p28_g6_update
[rank0]:     result = adapter.segmented_dp_grpo_value_and_grad(
[rank0]:   File "/app/tunix/rl/canonical_qwen3_adapter.py", line 8100, in run_train_step
[rank0]:     stacked_cache_ins, stacked_hidden_ins, hidden = (
[rank0]:         segmented.run_layers_fwd_tape_scan(
[rank0]:             engine_leaves, caches, hidden, attention_metadata
[rank0]:         )
[rank0]:     )
[rank0]:   File "/app/tunix/rl/canonical_qwen3_adapter.py", line 3687, in run_layers_fwd_tape_scan
[rank0]:     stacked_hidden_ins, new_caches, hidden_out = self._p71_fwd_scan_fn(
[rank0]:         stacked_leaves, stacked_cache_ins, hidden, metadata
[rank0]:     )
[rank0]:   File "/app/tunix/models/qwen3/qwen3_p22xh.py", line 144, in __call__
[rank0]: ValueError: Received incompatible devices for jitted computation. Got argument stacked_leaves[0] of zt_tr_fwd_scan with shape bfloat16[36,2560] and with device ids [2, 3, 18, 19, ...] on platform TPU and shard_map inside jit with device ids [0, 4, 8, 12, 1, 5, ...] on platform TPU at qwen3_p22xh.py:144:11 (P22XHRmsNorm.__call__)
```

The immutable incident directory is `canon-zero-tim/evidence/p58_k15_disaggregated_mesh_scan_incident/`.

## Root Cause

In a 128-TPU disaggregated setup (DP32xTP4, 32 worker nodes), the 128 chips are split into two 64-chip meshes:
- `_source_engine_mesh` (Rollout/Serving mesh: devices `[0, 4, 8, 12, ...]`)
- `_engine_mesh` (Trainer execution mesh: devices `[2, 3, 18, 19, ...]`)

During Rollout, `linear_module._CANON_MESH` is initialized to `_source_engine_mesh`.
In `SegmentedEngine.__init__`, per-layer callables are wrapped with `bind_execution_mesh` (which enters `_canonical_fixed_ar_execution_mesh(self._source_engine_mesh, self._engine_mesh, ...)`).

However, P71/P50 layer scan callables (`run_layers_fwd_tape_scan`, `run_layers_scan`, `run_layers_tape_scan`, `run_layers_rev_scan`) in `SegmentedEngine` invoke `self._p71_fwd_scan_fn`, `self._layer_scan_fn`, etc., **without** wrapping with `_canonical_fixed_ar_execution_mesh`.

Consequently, during JIT tracing of `_p71_fwd_scan_fn`, `P22XHRmsNorm.__call__` reads `linear_module._CANON_MESH` (which points to the serving mesh `[0, 4, 8, ...]`), while `stacked_leaves` is sharded on the trainer mesh `[2, 3, 18, ...]`, triggering JAX's incompatible device assertion upon entering reverse reduce.

## Planned Repair

1. In `SegmentedEngine`:
   - Store `self._disaggregated = disaggregated`.
   - Provide a unified `_bind_execution_mesh(fn, stage)` method.
   - Wrap `_p71_fwd_scan_fn`, `_layer_scan_fn`, `_layer_tape_scan_fn`, `_layer_rev_scan_fn` with `_bind_execution_mesh`.
2. Commit and push fix.
3. Relaunch as Attempt K16.

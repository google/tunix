# DeepSWE K22 P59 Data Axis Resolution Incident Report

## 1. Incident Overview
- **Run Identifier**: `canon-p58-ds4b-zero-hp-full-k22`
- **Hardware Topology**: 128 TPU v5p (32 Hosts), Disaggregated 64 TPU Rollout (`dp=8, tp=8`) + 64 TPU Trainer (`dp=8, tp=8`)
- **Step Reached**: Step 0 Rollout (128 trajectories, 4 solved, solve rate 3.125%), Pre-alignment Gate (393,135 tokens, 0B difference, 100% PASS), VJP Backward Pullback (Layers 35 -> 0 PASS)
- **Termination Reason**: `tunix.rl.canonical_qwen3_adapter.FunctionalMappingError: P59 report and grouped trainer data axes differ` at `reverse_reduce_group` line 9044.

## 2. Root Cause Analysis
1. In DeepSWE (P34), the Trainer Mesh is shaped `('dp', 8), ('tp', 8)` where the data-parallel replicated axis name is `'dp'`.
2. In `tunix/rl/canonical_qwen3_adapter.py` lines 8525-8529:
   ```python
   trainer_dp_axis = self._dp_axis # Default initialized to 'data'
   if not p34:
     _, trainer_dp_axis = _p59_replicated_data_mesh(
         trainer_state, "P32 grouped trainer state"
     )
   ```
3. Because of the `if not p34:` guard, `trainer_dp_axis` remained `'data'` when `p34=True`.
4. During `_p59_rank_parallel_report_adjoint`, the reducer dynamically resolved `reducer_dp_axis = 'dp'` directly from `trainer_state`'s actual mesh.
5. When executing the safety check `if reducer_dp_axis != trainer_dp_axis:`, `'dp' != 'data'` failed closed with a string mismatch exception.

## 3. Resolution
- Removed the `if not p34:` guard so `trainer_dp_axis` is unconditionally derived via `_p59_replicated_data_mesh(trainer_state, ...)`.
- Verified that both sides resolve to `'dp'` on DP8xTP8 meshes and `'data'` on DP8xTP8/data-model meshes.

# Incident Report: DeepSWE K25 AlignmentGateError on Unset CANON_P38_PRECHECK_ONLY

- **Date**: 2026-09-01T03:57:39Z
- **JobSet**: `canon-p58-ds4b-zero-hp-full-k25`
- **Head Pod**: `canon-p58-ds4b-zero-hp-full-k25-pathways-head-0-0-ms5vb`
- **Workload**: DeepSWE Qwen3-4B Zero-HP Full Production Training (128 TPU v5p)

## 1. Summary
During Step 0 training of DeepSWE K25, rollout (128 trajectories) and rescoring completed successfully. However, before entering backward propagation, `alignment.check_pre_backward` failed with `AlignmentGateError`.

## 2. Root Cause
In `tunix/rl/alignment.py:L552`, the `p58_zero_ab_warning` policy gate checked:
```python
os.environ.get("CANON_P38_PRECHECK_ONLY", "") == "0"
```
In full production runs, `CANON_P38_PRECHECK_ONLY` is unset (defaulting to `0` in bash scripts via `${CANON_P38_PRECHECK_ONLY:-0}`). In Python, `.get(..., "")` returned `""`, which failed the `"" == "0"` equality check, causing `p58_zero_ab_warning` to evaluate to `False` and rejecting the valid Zero-HP warning policy.

## 3. Resolution
Modify `tunix/rl/alignment.py:L552` to:
```python
os.environ.get("CANON_P38_PRECHECK_ONLY", "0") in ("", "0")
```
This admits both unset and explicitly zero values for `CANON_P38_PRECHECK_ONLY`.

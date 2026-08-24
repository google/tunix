# Plan

1. Freeze one common GSM8K workload: Qwen3-1.7B, DP4×TP1, seed 42, eight
   prompts, eight generations, response cap 256, concurrency one, three real
   optimizer commits, resident optimizer. Both arms require the existing
   `CANON_P60_DETERMINISTIC_AB=1` hash-carrier contract.
2. Native installs no inference overlay and selects only the existing
   `CANON_GSM8K_VANILLA=1` stock yardstick.
3. Zero-HP installs `qwen1p7b_tp1`, sources the V1 DP4 profile, enables P59
   rank-parallel backward, and keeps strict alignment.
4. Both capture the second update with `phase=update`, host tracer 1, Python
   tracer 0, and `TRACE_COMPUTE`; both emit exact work hashes before update.
5. Each arm passes an arm-aware device census. Native requires exactly 16
   monolithic `jit__train_step` modules on every plane; Zero-HP requires the
   P59 layer/head/norm/embed/adjoint families. Both require decode absent and
   one complete semantic update. The pair passes only if source, image, model,
   topology, window, and the profiled work receipt are identical.
6. Use unprofiled `[PERF]`/wall measurements for speed decisions. XProf is for
   causal/shape attribution, not an A/B stopwatch.
7. Treat the deterministic seed as a test, not an assumption. If independently
   sampled Native and Zero-HP completions differ, accept each arm's backward
   capture but classify the pair `INCONCLUSIVE_INPUT_MISMATCH`. A causal timing
   comparison then requires a separately reviewed frozen-train-batch replay;
   do not silently compare the two different updates.

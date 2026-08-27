# DeepSWE 128 TPU Zero-HP Attempt 7 (p58z07) Incident Report

## 1. Executive Summary

| Attribute | Value |
|---|---|
| **Workload** | DeepSWE Qwen3-4B Zero-HP Full (128 TPU v5p, DP8xTP8 Rollout + DP8xTP8 Trainer) |
| **JobSet** | `canon-p58-ds4b-zero-hp-full-p58z07` |
| **Source Commit** | `ef46b0b3a5d8754160f0cce323ec3861b04dccdc` |
| **Image** | `us-central1-docker.pkg.dev/cloud-tpu-v2-images-dev/yux-large-dev/tunix@sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a` |
| **Raw Log** | `canon-zero-tim/debug_logs/p58_p58z07_deepswe_s_decode_vs_prefill_gate.raw.log` |
| **Verdict** | `PRE_BACKWARD_ALIGNMENT_GATE_RED` |

---

## 2. Verified Milestones & Achievements

1. **NNX Metadata Fix (P58.16) 100% Validated**:
   - `_canonical_nnx_state_treedef()` eliminated the previous `FunctionalMappingError` caused by `_is_loaded=True` in NNX State treedef.
   - Live runner and weight-free trainer mesh reconstruction passed contract with 398 leaves.
2. **Full Step-0 Rollout Completed**:
   - All 128 trajectories (379,496 Action Tokens) sampled across 128 TPU chips and 128 R2E docker sandboxes.
3. **Prefill vs Token Old Alignment Exact**:
   - `bounds=[('S_decode_vs_S_prefill', 71797), ('S_prefill_vs_T_old', 0)]`
   - `S_prefill_vs_T_old = 0` (bitwise exact across 379,496 action tokens).

---

## 3. Incident & Error Analysis

During `_process_results` / `check_pre_backward` before backward VJP execution:
- The alignment gate detected 71,797 token logprob mismatches between decode-time logprobs (`S_decode`) and prefill logprobs (`S_prefill`).
- Under `--arm zero --high-performance`, this triggered a fail-closed `AlignmentGateError`:
  ```text
  [CANON_ALIGN_PRE_EVIDENCE] path=/mnt/disks/linchai_data/deepswe_zero_tim/canon-p58-ds4b-zero-hp-full-p58z07/pre_alignment.jsonl sha256=57ded48b3f46e973e456030b2abf15a7fc73b2550d8c2d078943761dd7c0804b
  [CANON_ALIGN_PRE] step=0 verdict=FAIL N_action=379496 bounds=[('S_decode_vs_S_prefill', 71797), ('S_prefill_vs_T_old', 0)]
  [rank0]: Traceback (most recent call last):
  [rank0]:   File "/app/examples/deepswe/canonical_entrypoint.py", line 36, in <module>
  [rank0]:     main()
  [rank0]:   File "/app/examples/deepswe/canonical_entrypoint.py", line 32, in main
  [rank0]:     runpy.run_module("examples.deepswe.train_deepswe_nb", run_name="__main__")
  [rank0]:   File "<frozen runpy>", line 229, in run_module
  [rank0]:   File "<frozen runpy>", line 88, in _run_code
  [rank0]:   File "/app/examples/deepswe/train_deepswe_nb.py", line 1812, in <module>
  [rank0]:     agentic_grpo_learner.train(train_dataset=train_dataset)
  [rank0]:   File "/app/tunix/rl/agentic/agentic_rl_learner.py", line 3489, in train
  [rank0]:     train_examples = self._batch_to_train_example(
  [rank0]:                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  [rank0]:   File "/app/tunix/rl/agentic/agentic_rl_learner.py", line 2866, in _batch_to_train_example
  [rank0]:     return self._process_results(
  [rank0]:            ^^^^^^^^^^^^^^^^^^^^^^
  [rank0]:   File "/app/tunix/rl/agentic/agentic_grpo_learner.py", line 1938, in _process_results
  [rank0]:     precheck_record = alignment.check_pre_backward(
  [rank0]:                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  [rank0]:   File "/app/tunix/rl/alignment.py", line 1562, in check_pre_backward
  [rank0]:     raise AlignmentGateError(
  [rank0]: tunix.rl.alignment.AlignmentGateError: pre-backward alignment gate RED: ['S_decode_vs_S_prefill']; report=/mnt/disks/linchai_data/deepswe_zero_tim/canon-p58-ds4b-zero-hp-full-p58z07/pre_alignment.jsonl
  ```

---

## 4. Next Steps

1. Native mode or relaxed warning-only alignment policy on `S_decode_vs_S_prefill` allows full 1,000 updates training progression.
2. In parallel, analyze the numerical divergence cause between Qwen3-4B decode attention and prefill attention for Zero-HP mode.

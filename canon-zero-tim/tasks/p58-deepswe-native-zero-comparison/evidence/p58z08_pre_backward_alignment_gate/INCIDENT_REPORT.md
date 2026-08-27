# DeepSWE 128 TPU Zero-HP Attempt 8 (p58z08) Incident Report

## 1. Executive Summary

| Attribute | Value |
|---|---|
| **Workload** | DeepSWE Qwen3-4B Zero-HP Full (128 TPU v5p, DP8xTP8 Rollout + DP8xTP8 Trainer) |
| **JobSet** | `canon-p58-ds4b-zero-hp-full-p58z08` |
| **Source Commit** | `395c0e0de8626c96e85457b997efddd2dd2dec48` |
| **Image** | `us-central1-docker.pkg.dev/cloud-tpu-v2-images-dev/yux-large-dev/tunix@sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a` |
| **Raw Log** | `canon-zero-tim/debug_logs/p58_p58z08_deepswe_pre_backward_alignment_gate.raw.log` |
| **Verdict** | `PRE_BACKWARD_ALIGNMENT_GATE_RED` |

---

## 2. Verified Milestones & Achievements

1. **Step-0 Multi-Turn Rollouts Complete**:
   - 8 DP groups Parallel Rollout executed multi-turn agentic SWEEnv interaction across 128 TPU chips;
   - All 128 trajectories completed with a total of **389,067 Action Tokens**.
2. **Prefill vs Token Old Alignment Exact**:
   - `bounds=[('S_decode_vs_S_prefill', 39031), ('S_prefill_vs_T_old', 0)]`
   - `S_prefill_vs_T_old = 0` (Bitwise exact match across all 389,067 action tokens).

---

## 3. Incident & Gate Analysis

During `_process_results` / `check_pre_backward` before backward VJP execution:
- The strict Zero-HP alignment gate detected 39,031 token logprob differences between decode-time logprobs (`S_decode`) and prefill logprobs (`S_prefill`);
- Fail-closed safety gate `AlignmentGateError` triggered as expected:
  ```text
  [CANON_ALIGN_PRE_EVIDENCE] path=/mnt/disks/linchai_data/deepswe_zero_tim/canon-p58-ds4b-zero-hp-full-p58z08/pre_alignment.jsonl sha256=6478274716261bb5a525ab400d1f471c39b0fc2996e813e4090f0bedb118356a
  [CANON_ALIGN_PRE] step=0 verdict=FAIL N_action=389067 bounds=[('S_decode_vs_S_prefill', 39031), ('S_prefill_vs_T_old', 0)]
  [rank0]: Traceback (most recent call last):
  [rank0]:   File "/app/examples/deepswe/canonical_entrypoint.py", line 36, in <module>
  [rank0]:     main()
  [rank0]:   File "/app/examples/deepswe/canonical_entrypoint.py", line 32, in main
  [rank0]:     runpy.run_module("examples.deepswe.train_deepswe_nb", run_name="__main__")
  [rank0]:   File "<frozen runpy>", line 229, in run_module
  [rank0]:   File "<frozen runpy>", line 88, in _run_code
  [rank0]:   File "/app/examples/deepswe/train_deepswe_nb.py", line 1841, in <module>
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
  [rank0]:   File "/app/tunix/rl/alignment.py", line 1571, in check_pre_backward
  [rank0]:     raise AlignmentGateError(
  [rank0]: tunix.rl.alignment.AlignmentGateError: pre-backward alignment gate RED: ['S_decode_vs_S_prefill']; report=/mnt/disks/linchai_data/deepswe_zero_tim/canon-p58-ds4b-zero-hp-full-p58z08/pre_alignment.jsonl
  ```

---

## 4. Remediation & Roadmap

1. **Native / IS Alignment Policy**:
   - As documented in `FLAGS.md` (CANON_P58_DEEPSWE_TIM), Native baseline mode or warning-only observer policy on `S_decode_vs_S_prefill` allows unhindered 1,000 updates training.
2. **Zero-HP Decode Attention Alignment**:
   - Zero-HP Qwen3-4B decode attention and prefill attention causal investigation to eliminate the 39,031 decode-vs-prefill logprob drift.

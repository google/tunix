# P44.7 — r02 topology/runtime repair

- Status: passed

## Finding

- Confirmed: `p44r02` admitted 256 unique `(id, coords)` devices with physical extents `(4, 8, 8)` and then grouped all of them under `process_index=0`.
- Confirmed: Tunix already parses Pathways device repr `logical_task` as the per-host identity in `tunix/utils/topology.py`; the DeepSWE splitter duplicates older `process_index`-only logic.
- Confirmed: Current agentic rollout passes one conversation as a flat prompt collection, and current Agentic GRPO passes the configured prompt microbatch directly to trajectory-batched logprob calls.
- Hypothesis: Reusing the existing Pathways host-key semantics will preserve host-complete physical splits, while the two small reviewed main fixes will prevent the next rollout and trainer-path faults after topology admission.

## Execution

1. Compare `yuxzhang/deepswe-quality-fix@023978b976dd6d94e7a42948c3f3a68e34d73744` and main commits `38a6fbfc`/`7a15620d` without merging them.
2. Add target-derived Pathways device doubles and fail-closed 64/256 host inventory controls.
3. Implement one shared host-key placement path and emit deterministic inventory evidence.
4. Port single-conversation prompt batching and prompt-to-trajectory logprob execution sizing with a P44 diagnostic marker.
5. Run P44, adjacent P43/P39/P34, exact-image, compile, and diff gates.
6. Correct Section 45, the operator runbook, and handoff only from actual evidence.

## Exit gate

- Command: `bash canon-zero-tim/tests/p44_deepswe_qwen4b_parity/run_cpu.sh`
- Pass: Both Pathways-style topologies report 4 devices per host, 8/8 role hosts on 64 and 32/32 role hosts on 256; crossed/missing/wrong-cardinality identities fail; prompt generation receives a batch of one conversation; configured logprob prompts `4` execute as `16` trajectories.
- Fail: Keep P44.7 active and do not render or rerun a target JobSet.

## Result

PASS locally on the exact dependency image.

- The role splitter now derives a Pathways host key from device repr
  `logical_task`, while retaining `process_index` for direct-attached devices.
- Both 64-device and 256-device target-derived doubles admit exactly four
  devices per host and exact 8/8 or 32/32 host-complete role halves.
- Missing, crossed, mixed-source, and wrong-cardinality host inventories fail
  closed.
- One conversation is passed to generation as one prompt batch, and configured
  prompt microbatch `4` becomes logprob execution microbatch `16` for four
  generations.
- `P44_DEEPSWE_QWEN4B_PARITY_CPU_PASS` reports 32/32 cases.
- The pinned image
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
  passes the two affected learner unit tests and emits
  `P44_EXACT_IMAGE_CPU_PASS overlay=qwen4b`.
- Adjacent P43, P39, and P34 gates remain green, including the P43 and P34
  exact-image terminal markers.

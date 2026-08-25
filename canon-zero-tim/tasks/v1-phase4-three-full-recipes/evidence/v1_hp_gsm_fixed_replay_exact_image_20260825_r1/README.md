# GSM fixed-replay DP16xTP4 pinned-image receipt

- Timestamp: `2026-08-25T07:53:03Z`
- Image: `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
- Execution: 64 forced CPU devices, topology `DP16xTP4`, bounded projection,
  one frozen replay capsule, zero optimizer commits.
- Terminal: `V1_GSM_FIXED_REPLAY_PASS topology=DP16xTP4 groups=16
  rank_ownership=16/16 fp64=ordinary,p59 negatives=denominator,dp_sum
  optimizer_commits=0`.
- Claim ceiling: this verifies the registered DP ownership, grouping, scaling,
  fixed reducer, and FP64 behavior for the bounded projection. It does not run
  a full Qwen model, a TPU target, a rollout, or an optimizer transaction.
- The complete stdout is preserved in the execution transcript, not as a raw
  filesystem artifact. `receipt.json` is the durable structured result.

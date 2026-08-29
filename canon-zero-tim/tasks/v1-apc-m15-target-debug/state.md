# State

- Status: active; Phase D3e canonical first-action classifier accounting is
  implemented and has passed host plus pinned exact-image gates. No numerical
  APC repair has been made.
- Delivery base: `b74c4ba38f293606000398c29818cea0c8ca5c8b`. The D3e analysis source is
  the full published commit containing this state, not that base commit. A
  downstream executor must be supplied the exact full D3e commit SHA and use
  a clean `local/*` worktree at that SHA.
- Attempt-17 runtime source:
  `16c224aa80eb6b3a544be19f693c0542ab4b0dcb`.
- Verified D3d evidence:
  `evidence/v1_apc_m15_attempt17_d36_offline_binding_20260829/`; all three
  manifest members verify and the manifest SHA256 is
  `c3dd6ab4e8ee191e1012b011a6e8ff8d845e528aa85f59936c06315b10cbbb31`.
- Numerical fact: APC-off rounds 0/1/2 are sealed exact. APC-on Round 0 is
  sealed with A-B=207 bytes / 95 elements, B-C=0, and 119,150 actions. APC-on
  Round 1 failed Stage-10 assembly with exit 2; Round 2 is absent. Root
  `COLLECTED`/`COMPLETE` is absent, so this remains analysis-grade partial
  evidence rather than target PASS.
- D3d identity result: source row 217 / completion position 0 / source position
  1225 uniquely binds to A request `79-b8334848`; proof prefix 1300 exceeds the
  required elimination horizon 1227 and eliminates seven alternatives.
- Candidate tensor interval: the unique first-action anchor is Layer 0
  `k_post_rope -> rpa_output`. Its observer fingerprint geometry is
  `[2048,1,15,8]` and final geometry is `[2048,8]`; request/call/token and
  cache-page receipts are present in the sealed return.
- Remaining evidence boundary: all seven joinable red points have global
  signatures Layer-0 `rpa_output` and `final_norm`; 88/95 red points remain
  unobserved under continue-decode. D3e keeps those facts explicit while using
  completion-position-zero as the declared decision scope.
- D3e implementation: classifier decision-scope/global-signature separation,
  reviewer debt fields with `numerical_repair_authorized=false`, a read-only
  bucket wrapper, focused negatives, and updated phase/handoff/runbook.
- Host validation: task-local discovery 161/161 PASS; classifier 23/23 PASS;
  reviewer/wrapper and committed-evidence audit 5/5 PASS; P38 persistence
  `PERSISTENCE_TEST_PASS`; flag audit 395/395 PASS; Python/Bash syntax, D3e
  scope audit, secret scan, and `git diff --check` PASS.
- Pinned exact-image validation: PASS on immutable image
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`;
  official aggregate exited 0 with `apc_m15_carrier=68`, `m15_d3e=1`,
  `m15_durability=1`, `m15_round_provenance=1`, and `manifests=3`. Raw log:
  `/tmp/m15-d3e-exact-image-b74c4ba3-20260829.log`, SHA256
  `59efa6ddc6e0399050cbbbbc5b463fc6b94486d96834f1e8b50f4fd9d3b22d97`.
- External gates: read-only D3e GCS execution NOT RUN; fresh target NOT RUN.
  The exact-image run used no TPU, Kubernetes, or GCS. A future target launch
  is not currently admitted.
- Numerical scope: RoPE, attention/RPA, KV values, LM head, A/B/C, loss,
  backward, optimizer, B full reset, production profiles, and production
  APC-off defaults are unchanged.
- Next action after publication: request separate GCS-read approval for a clean
  bucket-capable executor to run the checked-in D3e wrapper. Do not launch TPU.
- Claim ceiling:
  `ATTEMPT17_PARTIAL_ROUNDS_RECOVERED /
  REQUEST_IDENTITY_UNIQUE_FIRST_ACTION /
  D3E_CANONICAL_ACTION_SCOPE_IMPLEMENTED /
  HOST_PASS /
  EXACT_IMAGE_PASS /
  D3E_GCS_RECLASSIFICATION_NOT_RUN /
  TARGET_NOT_RERUN /
  APC_NUMERICAL_FIX_NOT_IMPLEMENTED /
  PHASE_E_CLOSED`.
- Updated: 2026-08-29.

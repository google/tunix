# State

- Status: active; Phase D3d offline source-row/request binding is implemented
  locally. No numerical APC repair has been made.
- Worktree: `local/m15-apc-attempt17-review-0829` at published evidence base
  `6e4e7f587941ee7e0c83753bc321a995912c8021`; Phase D3d changes are
  uncommitted.
- Attempt-17 runtime source:
  `16c224aa80eb6b3a544be19f693c0542ab4b0dcb`.
- Attempt-17 evidence:
  `evidence/v1_apc_m15_attempt17_d36_operator_return_20260829/`; the
  self-excluding manifest contains 84 verified members, with manifest SHA256
  `edbd3c8809daff85cee71cc712579990646d57fca1f7432e717fc2f5a8fab5bd`.
- Numerical fact: APC-off rounds 0/1/2 are sealed and exact. APC-on Round 0 is
  sealed with A-B=207 differing bytes / 95 elements, B-C=0, and
  `M15_INTERNAL_FIRST_RED_CANDIDATE_SET`. APC-on Round 1 failed during
  assembly with exit code 2; Round 2 is absent.
- Evidence ceiling: root `COLLECTED`/`COMPLETE`, terminal JobSet conditions,
  raw-log receipts, and the original render binding were not returned. The
  package is analysis-grade partial-round evidence, not a signed target PASS.
- First-red state: source row 217 / completion position 0 has both an
  exact-through candidate and a Layer-0 `rpa_output` candidate. No unique
  last-exact/first-red interval is currently admitted.
- Phase D3d implementation: fail-closed future token-prefix binding, safe
  immutable-bundle review, deterministic d36 render reconstruction, and one
  read-only GCS/CPU wrapper for a bucket-capable agent.
- Local validation: task-local discovery PASS; focused classifier/reviewer
  tests PASS; Python/Bash syntax and `git diff --check` PASS. Pinned
  exact-image has not been run for this change.
- Numerical changes: none. RoPE, attention/RPA, KV values, LM head, loss,
  backward, optimizer, B full reset, production profiles, and production
  APC-off defaults are unchanged.
- Next action: review and, only with explicit user approval, commit/push the
  Phase D3d analysis-only change so a clean bucket-capable agent can run
  `run_m15_attempt17_d36_offline_binding.sh`. GCS read access is a second,
  separate approval; the wrapper performs no GCS writes, Kubernetes actions,
  or TPU launch.
- Claim ceiling:
  `ATTEMPT17_PARTIAL_ROUNDS_RECOVERED /
  REQUEST_AWARE_CLASSIFIER_CLUSTER_PASS /
  FIRST_RED_CANDIDATE_SET_CAPTURED /
  OFFLINE_REQUEST_BINDING_LOCAL_PASS /
  OFFLINE_GCS_RECLASSIFICATION_NOT_RUN /
  FIRST_RED_NOT_YET_LOCALIZED /
  APC_NUMERICAL_FIX_NOT_IMPLEMENTED /
  EXACT_IMAGE_NOT_RUN /
  PHASE_E_CLOSED`.
- Updated: 2026-08-29.

# State

- Status: active; D3e returned `FIRST_RED_LOCALIZED` for the canonical
  completion-position-zero action. The Phase E0 Layer-0 live-KV discriminator
  and its additive launch-readiness follow-up are published. No APC numerical
  repair has been made or authorized.
- Published E0 implementation:
  `1c7391da5336033abd0727e610f7bad4c5c4e2be`. Published follow-up base:
  `12207e3281db13461350fe7ef68dbaadfe713a58`. The latter added an unsafe
  mutable-image fallback; the published additive follow-up replaces it with an
  immutable already-local image gate, aligns the wrapper run-id contract with
  the renderer, preserves failed scratch, and checkpoints the classifier
  runtime route. A downstream executor must use a clean `local/*` worktree at
  the exact full published SHA containing this follow-up, not either older SHA.
- Attempt-17 runtime source:
  `16c224aa80eb6b3a544be19f693c0542ab4b0dcb`; D3e analysis source:
  `d83707e3cdbf13f912c489d6ad3568b9e84e16ad`.
- Verified D3e evidence:
  `evidence/v1_apc_m15_attempt17_d3e_canonical_action_20260829/`; all three
  manifest members verify and the manifest SHA256 is
  `cdf4130bcab5ffeeb38d19fe40dfca9e15898f6a8a7208d21fcbeb9a2e957858`.
- Numerical fact: APC-off rounds 0/1/2 are sealed exact. APC-on Round 0 is
  sealed with A-B=207 bytes / 95 elements, B-C=0, and 119,150 actions. APC-on
  Round 1 failed Stage-10 assembly with exit 2; Round 2 and root
  `COLLECTED`/`COMPLETE` are absent. This is analysis-grade partial evidence,
  not a complete target PASS.
- Localized boundary: source row 217 / completion position 0 / source position
  1225, A call 83, Layer 0 `k_post_rope -> rpa_output`, observer shape
  `[2048,1,15,8]`, final shape `[2048,8]`. The prefix length is 1226 tokens
  over 77 logical cache pages at block size 16.
- Request identity: eight A requests share the exact red prefix. Future-prefix
  proof through length 1300 uniquely binds request `79-b8334848` beyond the
  elimination horizon 1227 and explicitly conflicts with all seven aliases.
- Remaining evidence boundary: all seven joinable red points have global
  signatures at Layer-0 `rpa_output` and `final_norm`; 88/95 red points remain
  unobserved under continue-decode. E0 therefore asks only whether the bound
  request's stored Layer-0 live-KV fingerprint already differs before RPA.
- E0 implementation: append-only default-absent Patch 35; exact-prefix Layer-0
  KV observer; request-aware replay-ledger binding; one-round matched renderer;
  prepare-only wrapper; and compact read-only GCS-return wrapper. The observer
  captures all eight aliases, masks 77 valid pages within a 96-page static
  bound, and keeps B on the independent full-reset path.
- Host validation: PASS. Task-local discovery 173/173; KV classifier 7/7;
  target carrier 19/19; resolved environment 11/11; E0 admission/runtime 9/9;
  V1 CPU 91/91; P3 contract 12/12; P38 persistence
  `PERSISTENCE_TEST_PASS`; flag audit 398/398; Patch 35 applied to the
  registered overlay, compiled, and matched manifest SHA256
  `b8f7e9577003ebb6ffdd3a2b12694261a0eaf0cb65b6d116953976734d849588`.
  The real host-Python route and mocked forced-Docker route PASS; missing and
  wrong local image identities fail before `docker run`. Real Docker was not
  executed. Python/Bash syntax and `git diff --check` PASS. The optional broad
  P33 host aggregate is
  INCONCLUSIVE on this host because `datasets` and `metrax` are absent; the
  official pinned-image gate remains the dependency-complete judge.
- D3e pinned exact-image validation remains PASS on immutable image
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`.
  E0 changes have **not** run the official pinned exact-image aggregate.
- External gates: E0 pinned exact-image NOT RUN; fresh E0 DP8xTP8 pair NOT RUN;
  E0 compact GCS return NOT RUN. No TPU, Kubernetes, or GCS action occurred in
  the E0 implementation turn.
- Numerical scope: RoPE, attention/RPA arithmetic, KV values, LM head, A/B/C,
  loss, backward, optimizer, B full reset, production profiles, and production
  APC-off defaults are unchanged.
- Next action after publication: on a clean exact-SHA worktree, run the
  prepare-only wrapper with a fresh 1-16 character run label. The wrapper emits
  a self-hashed classifier-runtime receipt and cannot pull or network a Docker
  fallback. Then request separate approval for the official pinned exact-image
  aggregate. Only a later, separate approval may launch the rendered matched
  DP8xTP8 pair.
- Claim ceiling:
  `ATTEMPT17_PARTIAL_ROUNDS_RECOVERED /
  FIRST_RED_LOCALIZED_K_POST_ROPE_TO_RPA_OUTPUT /
  E0_LAYER0_LIVE_KV_DISCRIMINATOR_IMPLEMENTED_PUBLISHED /
  E0_LAUNCH_READINESS_FOLLOWUP_PUBLISHED /
  E0_HOST_PASS / REAL_DOCKER_NOT_RUN /
  E0_EXACT_IMAGE_NOT_RUN /
  E0_TARGET_NOT_RUN /
  APC_NUMERICAL_FIX_NOT_IMPLEMENTED /
  NUMERICAL_REPAIR_NOT_AUTHORIZED`.
- Updated: 2026-08-29.

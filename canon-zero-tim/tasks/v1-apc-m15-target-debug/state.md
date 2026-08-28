# State

- Status: active; Phase D3c request-aware candidate classification and
  pre-classification input durability are LOCAL PASS. No numerical APC repair
  has been made.
- Working base: pulled cleanly to
  `fbc4fa03cdb35ac519d183b03ecd25ede485a5e3` on
  `origin/yuxzhang/canon-zero-tim` before the Phase D3c implementation. The
  delivered source identity is always the actual branch `HEAD`.
- Attempt-16 evidence:
  `evidence/v1_apc_m15_attempt16_d35_20260828/`; all eight manifest members
  verify.
- Target fact: APC-on Round 0 reached 92.5% hits and A-B=1,711 bytes / 786
  elements while B-C=0. The numerical APC defect remains real and unfixed.
- Control fact: three APC-off numerical rounds were exact. Returned evidence
  proves complete seal/upload/ACK for rounds 0 and 1 only; round 2 is shown as
  requested, not terminally complete.
- Failure fact: Round-0 assembly passed (70 shards / 2,187 pairs). The
  classifier then conflated distinct requests sharing one token-prefix hash
  and failed on an alias conflict. The tuple's first value was diagnostic
  round 0, not position 0.
- Implemented locally: request-aware observation identity, exact numeric
  candidate grouping, fail-closed B and same-request duplicates, mixed
  candidate-set verdicts without a fake interval, unique red-point coverage,
  complete candidate packaging, and a self-hashed classifier-input GCS
  checkpoint before analysis.
- Local gates: task discovery PASS; classifier/packager 18/18;
  durability/checkpoint 11/11; P38 persistence PASS; Python/Bash syntax and
  `git diff --check` PASS.
- Numerical changes: none. RoPE, attention/RPA, KV values, LM head, loss,
  backward, optimizer, B full reset, and production APC-off defaults are
  unchanged.
- Attempt-16 recovery ceiling: its incident subset is integrity-complete for
  the eight listed files but lacks the assembled classifier input. Unless the
  original pod-local round still exists, it cannot be reclassified offline.
  Future runs checkpoint those inputs before classification.
- Next experiment action: pinned exact-image. A fresh matched target pair is a
  separate user approval boundary after the published source passes that gate.
- Claim ceiling:
  `REQUEST_AWARE_CLASSIFIER_LOCAL_PASS /
  PRECLASSIFY_INPUT_DURABILITY_LOCAL_PASS / NUMERICAL_PATH_UNCHANGED /
  ATTEMPT16_TARGET_RED_PRESERVED / FIRST_RED_NOT_YET_LOCALIZED /
  APC_NUMERICAL_FIX_NOT_IMPLEMENTED / EXACT_IMAGE_NOT_RUN /
  TARGET_NOT_RERUN / PHASE_E_CLOSED`.
- Updated: 2026-08-28.

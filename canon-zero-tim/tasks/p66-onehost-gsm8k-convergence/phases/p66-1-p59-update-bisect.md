# P66.1/P66.2: P59 same-input update-level bisection

## Question

On the current source and an identical one-update GSM8K input/pre-state, does P59 rank-parallel backward remain inside the already frozen P61 numerical envelope for both the full gradient and the resulting real AdamW parameter delta?

## Pre-registration

- Control: `numerical-control`, P59 off.
- Candidate: `numerical-candidate`, P59 on.
- Geometry: DP4xTP1, one update, 17 alignment checks per arm.
- Mandatory identity checks: seven update hashes equal and all model-before leaves byte-identical.
- Gradient and parameter-delta thresholds: use the P61 Tier-1 baseline and caps without modification.
- A strict alignment failure overrides all numerical metrics.
- Input/pre-state mismatch yields `INCONCLUSIVE_INPUT_MISMATCH`, not KEEP or REJECT.
- Missing/tampered/non-finite evidence yields an evidence/carrier failure.
- Full-tree capture makes this run ineligible for performance claims.

## P66.1 source gate

1. Restore the comparator and its unit tests from the exact implementation used to classify P61n2.
2. Make the wrapper always write a top-level manifest for valid KEEP, REJECT, and INCONCLUSIVE outcomes; preserve the comparator exit status.
3. Add a negative test proving a classified REJECT is packaged and cannot be printed as GREEN.
4. Run comparator tests, P59 focused tests, shell syntax, and `git diff --check`.

## P66.2 TPU gate

1. Verify the one-host v5p lane is idle and the expected source/image/assets are present.
2. Use three fresh immutable labels: control, candidate, comparison bundle.
3. Run control to completion, then candidate; never overlap them.
4. Run the comparator and record its exact verdict and metrics.
5. Hash all raw evidence and update `log.md`/`state.md` with verified file paths and SHAs.

## Decision table

| Observation | Decision |
|---|---|
| Gradient and AdamW delta pass | `P59_UPDATE_PROXY_KEEP`; proceed to the next trainer dependency closure |
| Gradient passes, AdamW delta fails | `P59_UPDATE_REJECT`; keep P59 out of convergence recipes despite gradient-only correctness |
| Gradient fails | `P59_GRADIENT_REJECT`; stop and repair P59 backward |
| Alignment failure | `ZERO_TIM_REJECT`; revert/repair at this gate |
| Input or pre-state mismatch | `INCONCLUSIVE_INPUT_MISMATCH`; repair carrier and rerun fresh labels |
| Evidence missing/tampered/non-finite | `INCONCLUSIVE_EVIDENCE`; repair evidence chain |

## Rollback

All P66 source changes are diagnostic-only and default unreachable. Removing the P66 task directory plus the restored missing comparator/tests returns runtime behavior to the source base. No production profile is changed in this phase.

## Result

- P66.1 source gate: PASS (P61 6/6, P59 37/37, pinned-image manifest 36/36 and container tests 35/35).
- P66.2 control and candidate: each 17/17 strict PASS, 0 FAIL, one real optimizer commit.
- Same-input and full-prestate gates: PASS.
- Gradient envelope: PASS.
- AdamW update envelope: FAIL on rel-L2 and one-minus-cos.
- Frozen verdict: `P59_UPDATE_REJECT` / comparator `NUMERICAL_REJECT`.
- Claim ceiling: gradient correctness is verified under the accepted proxy; serial trajectory equality is disproven; non-convergence is not established by this one-step experiment.
- Evidence: `/mnt/disks/tunix-data/logp_probe_1host/p61_dp4_numerical_ab_p66ab1_20260825t1813z/`, result SHA256 `28f576472dd3c625376d9f93dfa1627e87f31e38631ea1418473a57d2c469454`.

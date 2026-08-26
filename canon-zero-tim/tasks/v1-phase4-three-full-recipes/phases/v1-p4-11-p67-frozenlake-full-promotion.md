# V1.P4.11 — Promote P67 scoping into the two FrozenLake full recipes

Status: implementation complete; host and pinned-image admission PASS;
publication and full targets pending.

## Decision and scope

Wave 5 proved on real P45 DP8xTP8 serving geometry that both the checked-VMA-off
control and `CANON_P67_P66_VMA_P59_ONLY=1` restore strict A-B/B-C `0/0` while
the scoped arm retains the checked-VMA P59 backward implementation. The user
accepted that evidence and explicitly waived another M15 scope precheck. The
next target wave is therefore exactly two uninterrupted full trains: P45 and
M15/main, both 300 updates.

This phase promotes the already default-off P67 implementation only into the
exact FrozenLake V1 high-performance full profile. It does not change the
GSM8K profile, serving mathematics, fixed TP reduction order, base JobSet,
autoscaling, exclusive-topology placement, checkpoint cadence, evaluation
schedule, or strict Zero-TIM gates. It does not launch, commit, or push.

## Implementation contract

1. The exact P45-readiness and M15/main, DP8xTP8, strict zero-arm, 300-update
   full profile resolves `CANON_P67_P66_VMA_P59_ONLY=1` together with P59
   checked VMA. No other production or diagnostic profile inherits it.
2. A two-manifest render-only carrier creates fresh immutable P45 and M15
   full JobSets from one exact clean 40-character source SHA. It preserves the
   reviewed P57 YAML/topology and prints, but never executes, exactly two
   unpiped `kubectl apply` commands.
3. Renderer and postflight require P67 for both FrozenLake recipes. GSM8K is a
   negative control and must not resolve or render P67 on.
4. The existing first-update gate remains fail-closed before and after AdamW:
   finite, nonzero, stable norm at most `1e6`, coherent parameter delta, then
   normal strict alignment on every later step. APC remains off for both.

## Admission gates before publication

- profile and `00_env.sh` positive/negative controls for the exact two full
  contexts and rejection of wrong profile/stage/arm/shape;
- Phase4 renderer/classifier CPU suite, P57 suite, P59/P66 suites, APC suite,
  flag registry audit, shell syntax, and `git diff --check`;
- immutable pinned-image full gate, because P67 changes program identity in
  the production profile even though the implementation was already image
  tested in the diagnostic arm.

The user explicitly waived another M15 scope precheck and any separate short
canary. Host/image green makes the source launch-ready, not target-certified.

## Full-run target verdict

P45 and M15 may be launched together after publication from the same exact
source SHA with fresh run IDs. Each run is judged independently and must:

- complete all 300 optimizer commits and all rollout-only evaluations at
  policy steps `0,50,100,150,200,250,300`;
- record zero `CANON_ALIGN`/`CANON_ALIGN_PRE` FAIL and no warn-only mode;
- pass the first-update gradient and AdamW receipts before weight sync;
- retain P59 checked-VMA and P67 P59-only scope receipts/resolved environment;
- preserve update, XProf, Perfetto, JAX-cache, evaluation, and final checkpoint
  artifacts required by the existing full classifier.

A red in one run does not cancel or reinterpret the other. Any strict mismatch
is fatal for that recipe. A backward/non-finite/optimizer red is not repaired
by clipping or by weakening the forward contract.

## Claim ceiling and rollback

Wave 5 verifies P45 serving-forward restoration only. M15 serving-forward,
both workloads' real P59 backward, optimizer transactions, performance, and
convergence remain unverified until these full runs finish. Revert this phase
by disabling/removing P67 from the FrozenLake full profile and its two-recipe
renderer/postflight admission; the default-off implementation and all Wave 5
evidence remain intact. Never delete failed run directories.

## Result log

- 2026-08-26: phase opened after the user accepted Wave 5 P45 evidence and
  explicitly chose direct P45 plus M15 full trains without another M15 scope
  precheck. Verified by Wave 5 that P45 serving-scope recorded A-B/B-C `0/0`,
  48,594 action tokens, depth 2,472, controlled exit 42, and zero backward or
  optimizer commits. M15 full target is unverified because it has not run with
  P67 production admission.
- 2026-08-26: implemented production admission without changing the base
  JobSet or P57 renderer. The exact FrozenLake V1 full profile now resolves
  P67; `00_env.sh` admits only P45-readiness/M15-main DP8xTP8 strict zero full
  tuples or the historical serving-scope diagnostic. The legacy three-recipe
  renderer keeps GSM8K P67-off and requires P67 for its two FrozenLake outputs.
  A new render-only two-recipe wrapper emits exactly P45 and M15 manifests and
  two unpiped apply commands. Full postflight requires P67 for FrozenLake and
  rejects it for GSM8K.
- 2026-08-26: verified by Phase4 89/89, P57 146/146, P59 37/37, P66 16/16,
  APC 31/31, flag audit 385/385, shell/Python syntax, and `git diff --check`.
  The two-recipe tests prove the raw and resolved P67 contract, wrong-profile
  and wrong-mesh rejection, APC-off/full/eval/checkpoint identities, and
  unchanged worker `4x4x4` selector plus `exclusive-topology` annotation.
- 2026-08-26: immutable image
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
  passed the P67-enabled installed DP2xTP4/TP8 fixed-head/projection/attention,
  report-adjoint, staged-spec and reducer ladder and the complete
  `V1_HP_EXACT_IMAGE_PASS ... p59_checked_vma_real_shim=4 ... manifests=3`
  regression. Verified by execution transcript; no durable raw image log was
  saved, so this is admission-grade and not target evidence. No TPU JobSet,
  full optimizer transaction, commit, or push occurred.

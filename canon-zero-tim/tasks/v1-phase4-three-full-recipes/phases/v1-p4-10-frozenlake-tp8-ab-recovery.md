# V1.P4.10 — FrozenLake TP8 A/B recovery bisection

Status: dual-arm implementation complete; host, focused P59 image admission,
and full image regression PASS; target not run.

## Problem

Attempt 9 reproduced a pre-backward forward-only regression on the published
P66 stack. P45 recorded A−B/B−C `1755/0` differing bytes across 46,879 action
tokens; M15/main recorded `93/0` across 124,308. Both were APC-off and stopped
before backward or AdamW. The exact same TP8 recipes were strict `0/0` in
earlier attempts, so the current target is recovery of the Zero-TIM forward
contract, not gradient or optimizer analysis.

Source inspection found one unclosed bisection boundary: the process-wide
checked-VMA implementation alias remains visible to installed serving shims
outside the P59 backward context. The prior repair scoped the linear
projection `pmean`, but four forward mutations remained process-wide: Pallas
operand `pcast`, Pallas output manual-axis types, RPA output manual-axis types,
and the fixed-AR embedding `pmean`. This is a hypothesis, not yet a target
verdict.

## Pre-registered paired arms

`CANON_V1_FL_TP8_AB_ARM=p66-off` changes exactly one selector relative to
Attempt 9: `CANON_P59_CHECKED_VMA=0` and its internal P66 alias `=0`. It keeps
TP8, fixed-AR gather, continue-decode 8, gathered logprobs, step fusion,
fixed LM head, APC-off, seed 42, and the production P45/M15 token geometry.

Both carriers use the full 32-prompt/256-trajectory producer unit and one
frozen pre-backward round. P45 requires max logical KV prefix at least 1686;
M15 requires at least 3936. They accept only finite A−B evidence with B−C
exact, then exit code 42 with `backward=0 optimizer_commits=0`.

`CANON_V1_FL_TP8_AB_ARM=serving-scope` is the matched candidate arm. It keeps
`CANON_P59_CHECKED_VMA=1` and the P66 implementation alias enabled, but the
new default-off `CANON_P67_P66_VMA_P59_ONLY=1` admits those four mutations
only inside the exact outer manual `data/model` P59 pullback. Ordinary serving
decode/prefill therefore retains the historical graph. This arm does not
claim a production fix: it exits before backward, and target evidence must
also prove that B−C remains exact.

## Gate and interpretation

- Both arms `ZERO_TIM_RECOVERED`: checked-VMA serving leakage is the target
  cause family and `serving-scope` is the preferred candidate because it
  retains the repaired P59 backward path.
- `p66-off` recovers while `serving-scope` remains red: scoping is incomplete;
  inspect program fingerprints and bisect the four registered leak sites.
- Both arms remain `A_B_RED_REPRODUCED`: checked-VMA is exonerated and the next
  phase must bisect fixed-AR gather or continue-decode without changing this
  source/data contract.
- Any asymmetric outcome not covered above is inconclusive, not permission to
  train.
- Any B−C drift, non-finite value, insufficient depth, missing controlled exit,
  backward marker, or optimizer activity is fatal/inconclusive.

Launch the matched P45 pair together when 128 chips are intentionally
allocated. If only 64 chips are available, run `p66-off` first, then render a
fresh `serving-scope` ID. M15 is a later replication after the P45 verdict;
never infer one workload's verdict from the other.

## Rollback

Revert the profile, renderer/classifier, runner admission, learner sampler
admission, flag registration, and this phase together. No production profile
or default behavior is changed. All failed target directories remain.

## Result log

- 2026-08-26: verified by Attempt-9 durable fixtures that the classifier
  returns `A_B_RED_REPRODUCED` for P45 `1755/0` and M15 `93/0`; verified by
  synthetic recovery that `0/0` returns `ZERO_TIM_RECOVERED`, while B−C drift
  is fatal.
- 2026-08-26: the user approved a paired candidate arm. Source audit added the
  previously omitted fixed-AR embedding `pmean` to Pallas operand/out-shape
  and RPA out-shape scoping. Host gates pass Phase4 82/82, P57 146/146, P59
  37/37, P66 16/16, APC 31/31, and flags 385/385. The immutable image
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
  verifies the rebuilt shim manifest `37/37`; with both checked-VMA and
  P59-only scoping enabled, the installed DP2×TP4/TP8 fixed-head, projection,
  attention, report-adjoint, staged-spec and reducer gate exits zero. The full
  exact-image regression also exits zero with
  `V1_HP_EXACT_IMAGE_PASS ... p59_checked_vma_real_shim=4 ... manifests=3`.
  Its execution transcript was observed directly but was not durably saved as
  a raw-log artifact. Not verified on TPU because no JobSet was launched. No
  optimizer transaction, commit, push, or Kubernetes mutation was performed.

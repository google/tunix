# V1.P4.17 — P45 finite decode/prefill warning lane

- Status: host and exact-image admission pass; target pending

## Motivation

The user prioritizes obtaining the optimized P45 FrozenLake full-training
curve. A finite serving decode versus canonical full-prefill residual may be
recorded without killing the run, but this must not weaken the independent
prefill-versus-trainer, finiteness, backward-health, replica, or optimizer
contracts.

## Registered scope

- `frozenlake-dp8-tp8`, v1-hp profile, Zero arm;
- P45 identity is empty workload candidate and empty data split;
- exactly 300 committed updates, no evaluation, no checkpoint;
- `CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY=1` is rendered, not hand-edited;
- production M15 retains the same narrow warning shape but remains non-TiTO.

Native, IS, eval, P64, APC/debug, resident, partial P45 identity, wrong
profile/topology/horizon, and no-commit carriers are rejected.

## Gate shape

Only finite `S_decode_vs_S_prefill` and direct `w`/`wr`/clip/TIS consequences
are warnings. `S_prefill_vs_T_old`, `T_old_vs_T_current`, `r`, nonfinite
values, gradients, backward-health, replica equality, and optimizer
transactions remain fatal. Postflight must accept and count
`PASS_WITH_ALIGNMENT_WARNINGS`, require zero real FAIL, and classify the run
as `convergence-only / alignment-degraded`.

## Exit gates

1. policy positive plus P45 partial/foreign identity negatives;
2. finite A-B warning, B-C and nonfinite negatives;
3. renderer and real `00_env.sh` resolution for P45/M15, with neighbors off;
4. full classifier warning acceptance and any-FAIL negative;
5. P57/V1 host suites, flag audit, exact-image gate;
6. separately approved full target returns 300 commits and complete warning
   dose without any fatal boundary.

## Result

Verified by host P57 183/183, V1 Phase4 92/92, flag audit 409/409,
`git diff --check`, and shell syntax. The complete immutable-image gate against
`sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
exited zero with terminal
`V1_HP_EXACT_IMAGE_PASS ... frozenlake_ab_warning=2 ... manifests=3`.
The first image attempt correctly failed only a stale HANDOFF routing check;
after the top section named both P74 GSM8K and P75 FrozenLake routes, the full
gate was rerun rather than inferred. The successful execution transcript was
observed directly but not durably redirected, so it has no raw-log SHA and is
an admission receipt rather than a signed artifact.

Not verified because no P45 DP8xTP8 300-update target was launched. The claim
ceiling remains `convergence-only / alignment-degraded`.

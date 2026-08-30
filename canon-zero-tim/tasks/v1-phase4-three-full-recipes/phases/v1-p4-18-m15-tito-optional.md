# V1.P4.18 — optional exact M15 TiTO, default off

- Status: host and pinned-image admission pass; publication/target pending

## Motivation

The matched M15 one-host r7/r8 pair proved that the exact-token carrier keeps
all 17 later-turn prompt streams and all three strict A/B/C rounds identical
to the legacy path. That supports retaining exact TiTO as a selectable full
experiment, but it does not transfer DP1xTP4 evidence to DP8xTP8 or justify
changing the default training curve.

## Registered scope

- the P67 two-full renderer defaults to selector absent for both recipes;
- `--m15-tito-exact` sets `CANON_M15_TOKEN_CONTINUITY=exact` only for the
  exact M15/main Zero v1-hp DP8xTP8, 300-update, no-eval/no-checkpoint full
  identity;
- P45 always requires the selector absent;
- empty, `0`, `verify`, partial identities, APC-on, and neighboring workloads
  fail closed.

## Gates

1. default render proves both manifests selector-absent and records
   `m15_tito_exact=false`;
2. explicit render changes M15 alone, resolves through the real profile/env,
   and records `m15_tito_exact=true`;
3. classifier requires zero receipts in off mode, or only exact/equal receipts
   plus exactly one env receipt in exact mode;
4. P45 leakage, missing receipt, unequal receipt, wrong mode, and wrong
   identity are negative controls;
5. full P57/V1/flag/exact-image admission passes before publication;
6. the first explicit DP8xTP8 exact run remains a target certification, not a
   claim inherited from one host.

## Current launch decision

The authorized restart is P45 full only. Render with no optional sixth
argument, require `m15_tito=off`, verify P45 selector absence, and apply only
the fresh P45 YAML. Do not launch M15 in this action.

## Result

Verified after rebasing onto fetched operator tip `2f61f8fc` by P57 184/184,
V1 Phase4 93/93, flag audit 409/409, focused renderer and classifier 35/35,
shell syntax, and `git diff --check`. The complete pinned
image gate against
`sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
exited zero with terminal `V1_HP_EXACT_IMAGE_PASS ... m15_tito_option=exact
m15_tito_default=off ... manifests=3`. The console transcript was observed
directly but was not redirected to a durable raw log.

The first post-rebase image run stopped on two stale source-shape assertions:
they still required the old exact-only one-host variable and equality spelling,
while upstream now deliberately shares the one-host carrier between `verify`
and `exact`. The runtime contracts were unchanged by this follow-up; the
focused test passed 3/3 after checking the new semantic spelling, and the
complete pinned-image gate was rerun from the beginning to the terminal above.

Not verified because no exact remote readback, fresh publication-SHA render,
P45 DP8xTP8 restart, or M15 DP8xTP8 exact target has occurred under this phase.

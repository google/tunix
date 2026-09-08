# T7 — P45 and M15 exact-token treatment

- Status: complete

## Finding

- Confirmed: P45 and M15 are both multi-turn FrozenLake recipes, but the
  published exact-token selector admits M15 only.
- Confirmed: M15 exact token transport passed the bounded DP1xTP4 r8 carrier;
  P45 has no equivalent target evidence, and neither result certifies DP8xTP8.
- Confirmed: at source `6842edae`, the base P57 renderer already injects the
  seven full-system optimization keys. The P67 wrapper injects them again and
  fails with `refusing to overwrite rendered env` before a launchable manifest
  exists.
- Hypothesis: the same integer-ledger reconstruction is correct for P45
  because both FrozenLake recipes store the initial prompt, sampled assistant
  IDs, and nonterminal environment IDs in the shared trajectory structure.

## Treatment contract

The renderer exposes one closed enum:

| Value | P45 | M15 |
|---|---|---|
| `legacy` | selector absent | selector absent |
| `p45-exact` | generic selector `exact` | selector absent |
| `m15-exact` | selector absent | generic selector `exact` |
| `both-exact` | generic selector `exact` | generic selector `exact` |

The default is `legacy`. The historical `--m15-tito-exact` CLI spelling may
remain as a temporary alias for `m15-exact`, but combining it with the enum is
fatal. The runtime selector is `CANON_P57_TOKEN_CONTINUITY=exact`; the existing
`CANON_M15_TOKEN_CONTINUITY` remains available only for already-published M15
debug/one-host/full evidence, and both environment keys present is fatal.

## Execution

1. Remove P67's duplicate performance-bundle write; validate the exact bundle
   emitted by the base P57 renderer and preserve autoscaling, exclusive
   topology, selectors, and Pathways anti-affinity unchanged.
2. Add the generic full-only selector and exact P45/M15 workload identity
   checks at renderer, profile, `00_env.sh`, Python reader, and receipt layers.
3. Generalize shared prompt reconstruction without changing the first-turn
   path, chat parser, sampling, rewards, loss, backward, or optimizer.
4. Extend the full classifier so selected exact recipes require complete
   workload-labelled equal receipts plus one per-trajectory summary. The full
   classifier requires exactly `256 trajectories * 300 updates` summaries and
   contiguous later-turn receipts within each trajectory; legacy recipes
   require all continuity selectors, receipts, and summaries absent.
5. Run focused host suites, flag audit, exact-image installer/runtime probes,
   and intent/topology diff checks.

## Exit gate

- Pass: all four rendered selector modes resolve through real `00_env.sh`;
  P45 and M15 exact positives reconstruct the integer ledger and emit exact
  equal receipts; default/neighbor/malformed/mixed/profile/topology/poison and
  missing-receipt controls fail; pinned-image gate passes; no topology or
  non-token recipe field changes.
- Fail: stop at the first red gate. Do not weaken token equality, alignment,
  classifier completeness, or workload identity. Preserve the failing output.
- Target boundary: host and pinned-image green means construction only. P45
  and M15 DP8xTP8 remain `TARGET NOT RUN` until separately approved full jobs
  return complete receipts and their ordinary training verdicts.

## Result

Host admission passed. P57 is 191/191, V1 is 101/101, the flag registry is
411/411, shell/Python syntax and `git diff --check` pass. The structural
control proves `both-exact` changes each complete rendered JobSet only by one
`CANON_P57_TOKEN_CONTINUITY=exact` entry; recipe, scheduling, topology, model,
data, optimization, eval/checkpoint, and warning policy remain identical.
Malformed, mixed, wrong-profile, wrong-topology, unselected, cross-workload,
and missing-receipt controls fail. The complete immutable-image gate passed
from start to terminal against local image
`sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`;
its terminal is `V1_HP_EXACT_IMAGE_PASS ... frozenlake_tito_impl=2
frozenlake_tito_selector=closed frozenlake_tito_summary=1
frozenlake_tito_debug=1 frozenlake_tito_capsule_integrity=1
frozenlake_tito_default=legacy ... manifests=3`. The raw local admission log is
`/tmp/p57_tito_pair_pinned_20260902_r2.log`, SHA256
`9bc28afb41ac0a7049eb66a2c65aa47abb912bdcf42ed4620603c961700446a3`;
it has not been copied to durable GCS. No TPU or Kubernetes target ran, so
both DP8xTP8 treatments remain `TARGET NOT RUN`.

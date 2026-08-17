# P38.2t — Target-aware tail join and terminal split

Status: active; implementation, local gates, and user publication approval are
complete. The next gate is the zero-TPU GCS-side v3 reduction from the clean
published branch. No source GCS object, TPU executable, or training path is
changed by this phase CL.

## Entering evidence

The immutable P38s18r2 Round-0 v2 compact bundle is mechanically intact:

- 371 bundle SHA entries verify;
- the standalone auditor reproduces
  `INCONCLUSIVE_REDUCTION_JOIN` with 32 red points, 64/64 seam keys, and
  63/64 tail keys; and
- the sole conflict is arm A at token-prefix SHA
  `e7427e602f65095a7ce9930f03fc118831ff3a5345ebca45841b2c56b563a7b0`.

That conflict is not two observations of the same scored event. Records 510
and 723 score target token 54852 and have byte-identical numerical payloads;
record 539 shares the source-token-history prefix but scores target token
13598. The mismatch capsule requires target token 54852. A terminal-tail row
is therefore identified by:

```text
(diagnostic round, source-token-history prefix SHA, arm, target token ID)
```

The seam observer remains identified by the original three fields because it
observes the source-token hidden state before a next-token target is chosen.

## Objective

Repair the offline join contract without weakening its ambiguity discipline,
then use the already sealed Round 0 to answer two questions:

1. do all 32 red actions join both seam and terminal-tail observations after
   the scored target is included in tail identity; and
2. is the first measured A/B difference before the terminal tail, in the raw
   target logit, or in the raw log normalizer?

This phase does not claim that logsumexp reduction is the root cause. A
normalizer can differ because non-target vocabulary logits already differ.

## Deliverable A — target-aware reducer contract

The reducer must:

1. derive one required target token from the capsule for every required
   `(round, prefix SHA, arm)` tail key;
2. retain every same-prefix source candidate under `candidates/`;
3. exclude a wrong-target candidate from alias/conflict resolution and record
   it under `tail_target_mismatch_candidates`;
4. admit an alias only among same-target candidates whose complete registered
   provenance and numerical payload bytes agree;
5. remain fail-closed if the capsule target is absent or if same-target
   candidates conflict; and
6. keep the old v2 bundle independently auditable with byte-identical audit
   output.

The official classifier's target-aware path keys tail joins by the complete
four-tuple and therefore admits two capsule red points with the same source
prefix but different next-token targets. Legacy manifests retain the old
three-field interpretation. The auditor permits only the explicitly tested
old-classifier-to-new-classifier SHA pair for that legacy bundle; all new
target-aware bundles require exact classifier-source SHA identity.

The immutable remote output uses a new contract and destination:

```text
scripts/p38s18r2_round0_target_join_contract.json
.../derived/p38s18r2-round0-seam-tail-target-aware-v3
```

The v1/v2 source and derived objects are never overwritten.

## Local gates

Required positive and negative controls:

- unique seam/tail joins pass;
- equivalent same-target aliases pass and remain enumerated;
- same-prefix, different-target tail rows are retained but do not conflict;
- a missing capsule target fails closed;
- a same-target numerical conflict fails closed;
- missing A/B seam or tail arms fail closed;
- source, selected record, capsule, manifest, classifier, and alias-selection
  tampering are rejected;
- fake-GCS wrapper upload is completion-marker-last and immutable; and
- the new auditor reproduces the committed legacy v2 audit byte-for-byte.

Run the complete P38 Python and shell gates. A missing host-only dependency is
`TARGET NOT RUN`, never a pass.

## Deliverable B — one-host construction discriminator

Run the existing bounded real-v5p construction probe at production tail shape
`[256,151936]`. It sends identical logits through two distinct outer JIT
programs which call the same canonical log-softmax function object. Require:

```text
backend = tpu
device_count = 4
differing_elements = 0
negative_control_differing_elements = 1
```

This can refute a same-input canonical-reducer construction failure on one
host. It cannot prove that production A and B enter the tail with equal full
logit rows, and it cannot reproduce a 64-chip Pathways envelope.

## Deliverable C — immutable Round-0 reclassification

After the implementation is reviewed, committed, and published, a
GCS-authorized agent runs the checked-in target-aware contract wrapper once.
Acceptance requires:

```text
red_points = 32
matched_seam_keys = 64
matched_tail_keys = 64
payload_conflict_keys = []
joined_red_points = 32
tail_target_identity_required = true
standalone bundle audit = PASS
scientific verdict = INCONCLUSIVE_PARTIAL_RUN
```

The partial-run verdict remains mandatory because the source run completed
only Round 0 of three.

## Decision table

| Result | Decision |
|---|---|
| Hidden/final fingerprints first red | Continue the ordered hidden seam walk |
| Hidden/final fingerprints equal; raw target logit first red | Split pre-lm-head hidden bytes from lm-head target-dot accumulation |
| Hidden/final fingerprints equal; raw normalizer first red | Split full-vocabulary logits from max/exp-sum/log normalizer stages |
| Multiple first-difference signatures | Preserve every signature; do not force one root cause |
| Equivalent aliases only | Admit the deterministic target-aware subset with provenance |
| Any missing target or same-target conflict | `INCONCLUSIVE_REDUCTION_JOIN` |
| Same-input one-host canonical log-softmax red | Repair the canonical reducer before another production run |
| Same-input one-host canonical log-softmax exact | Reducer-alone hypothesis is insufficient; observe its production input |

## Next diagnostic if Round 0 confirms a terminal split

Add one bounded, observer-neutral terminal discriminator. For every captured
candidate it must record enough information to distinguish:

```text
pre-lm-head hidden -> full logits row -> target-logit dot
                   -> row max -> exp-sum -> log normalizer
```

The production endpoint must be bitwise identical with the observer disabled
and enabled, and an injected one-bit change must make the discriminator red.
Do not name the reducer as the repair target until identical full logits are
shown to enter different normalizer results.

## Claim ceiling

This phase may establish a target-aware, analysis-grade first-measured-tail
classification for one immutable Round 0 and may reject a same-input
one-host reducer construction bug. Fingerprint equality is not full hidden
tensor byte equality. The phase cannot prove a 64-chip reducer cause, select a
numerical repair, manufacture the missing rounds, or close zero-TIM.

## Rollback

The code is offline analysis tooling and a default-unused contract. Revert the
target-aware reducer/auditor/wrapper changes or leave the v3 destination
unused. No model, training, evaluation, optimizer, prefix-cache, or serving
runtime default is changed.

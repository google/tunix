# canon-zero-tim review rubric

Load this reference when reviewing the branch or changing a gate status.

## Original six-CL baseline

The original additive package ended at `3f037d8d`:

| Commit | Role | Durable review note |
|---|---|---|
| `7101b4a5` | Engine patches against a pinned upstream | Patch identity was checked; two patches are diagnostic only. |
| `53c0448b` | Shim vendoring, relative resolution, and installer | Large mixed CL: mechanical volume and new installer behavior share one review unit. |
| `7748dbeb` | CPU differentiable-attention gates | CPU mathematical evidence only; includes a negative control. |
| `53198034` | T1 topology probes | Originally not run on target TPU; compilation/static checks do not repay this debt. |
| `370d00c3` | Cluster scripts, profiles, and JobSet | Originally not applied on GKE; Pathways branch and inferred DNS required target validation. |
| `3f037d8d` | Anchors, evidence map, runbook, and footguns | Prose was not machine-checked against code; artifact hashing does not validate prose. |

Later commits may have repaid or changed these facts. Inspect current `EVIDENCE.md`, run logs,
and commit history before repeating an original `NOT RUN` statement.

## Mechanical facts versus conclusions

These are useful mechanical facts:

- all paths are additive and contained under `canon-zero-tim/`;
- a stack has zero deletions;
- manifests match;
- scripts parse or compile;
- commit messages disclose limitations.

None proves:

- the probes execute on TPU;
- Pathways initializes before JAX and uses the requested backend;
- a JobSet is accepted or starts correctly;
- the fixed-order reduction works at a new width or across slices;
- a fresh install reproduces signed A/B/C numbers;
- training converges or a production topology is admitted.

## CL-quality questions

Ask these in order:

1. Is the semantic concern singular and reviewable?
2. Is a mechanical move isolated from a behavioral change?
3. Are dependency edges explicit and minimal?
4. Does the message state problem, reason, verification, drawbacks, and context?
5. Does verification exercise the newly introduced behavior rather than only syntax?
6. Are unrun target paths labeled `TARGET NOT RUN`?
7. Can a reviewer identify a narrow rollback?

A disclosed drawback is evidence of honesty, not evidence that the drawback is harmless.

## Evidence promotion rules

Promote a row only when all are present:

1. exact command or frozen runner;
2. exact revision and pinned runtime/image;
3. intended hardware/backend/topology;
4. expected measurement count equals actual count;
5. negative controls reject where required;
6. raw artifact path and SHA-256;
7. postflight/classifier passes;
8. no undeclared override;
9. rollback is stated.

An infrastructure failure is `INCONCLUSIVE`, not a numerical failure. A missing output line is
also `INCONCLUSIVE` or red, never green.

## Published-history policy

For a public clean branch, do not rewrite the six commits merely to split the large C2. The
review benefit is usually smaller than the coordination cost and invalidated references. Add a
follow-up CL when behavior must change; add target evidence when the debt is validation. Rewrite
only after explicit user approval and confirmation that no consumer relies on the published refs.

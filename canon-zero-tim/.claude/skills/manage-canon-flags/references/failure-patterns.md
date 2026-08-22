# Flag failure patterns

Every pattern below has occurred in this repository. Use the source and runtime
marker named by the current workload; paths can evolve.

| Symptom | Actual failure | Correct response | Required regression |
|---|---|---|---|
| Profile unsets a key, but Python still sees the renderer value | `unset` ran in child `00_env.sh`; parent retained stale `CANON_LOGPROB_M=256` | make `env.sh` an authoritative managed snapshot | seed stale parent value, run real `00_env.sh`, reload, assert absence |
| A disabled implementation still runs | code branches on presence, so `FLAG=0` is present | use `unset`; separately validate boolean flags | missing/empty/zero truth table |
| Parser crashes on `int("")` | Docker/Kubernetes exported an empty key | treat empty explicitly or omit the env entry | empty-key negative control |
| Launcher says an XLA flag is present but TPU behavior is unchanged | flag reached JAX client, not Pathways compiler | deliver through proxy/server env | rendered manifest assertion plus target runtime evidence |
| A workload-specific exception leaks to another recipe | code checked a truthy generic workload flag | compare exact workload/profile/arm identity | intended workload positive plus neighbor negative |
| Parameter is set but implementation is stock | parameter and activation switch are distinct | verify the activating switch and runtime receipt | switch-off/parameter-on negative |
| Flag is on but patched path never executed | missing/stale chain member or wrong endpoint | verify manifest/import and require endpoint marker | postflight rejects zero or foreign markers |
| Renderer test passes, learner rejects first batch | manifest was never resolved through `00_env.sh` and learner contract | test renderer -> real env -> learner policy | workload-specific end-to-end contract |
| Short-stage policy test passes, full training is rejected after rollout | workload was added to a shared admission count but the stage enumeration still allowed only debug updates | give the workload its own signed stage/horizon predicate; do not broaden the debug branch | renderer-derived full-stage positive plus missing-admission/wrong-horizon/opposite-arm negatives |
| Flag name suggests a path is off, but training still uses it | flag gates a standalone diagnostic, not the training callable | trace readers and inspect stage markers | runtime stage assertion, not name-based inference |
| Shared API accepts only canonical flag, Native fails after rollout | processed `S_prefill` contract was hard-coded to canonical processing | add an independently signed observer-only Native route; keep canonical flag off | arm truth table, opposite-arm negative, exact-image value probe |
| Native reaches alignment but a finite serving/trainer boundary blocks | warning scope was narrowed to one seam even though the treatment spans two untreated programs | enumerate the exact signed Native serving boundaries; keep trainer repeat and Zero strict | real precheck positive, nonfinite negative, classifier dose and trainer-repeat controls |
| Native is “fixed” by enabling canonical processing | the control treatment is contaminated | stop; do not trade experiment identity for progress | contract rejects mixed Native/Zero tuple |
| Grep finds no marker in a live log | control characters made grep treat it as binary | use `grep -a`; confirm negative claims independently | postflight marker count |

## Debug order

1. Find the first failure, not the final derivative exception.
2. Print names and resolved non-secret values only; never print tokens or full
   secret-bearing environments.
3. Compare renderer env, profile result, persisted `env.sh`, reloaded parent,
   and exact reader process.
4. Inspect source branch conditions for presence, exact string, and default.
5. Require execution provenance. Absence of a marker is not evidence of the
   stock or canonical path.
6. Preserve the failed log and classify missing prerequisites as
   `INCONCLUSIVE`.

For the longer historical set, read `canon-zero-tim/KNOWN_FOOTGUNS.md`, notably
entries on empty Docker env keys, global/per-rank units, workload exceptions,
Pathways flag delivery, renderer-vs-learner gaps, and misleading flag names.

# P58.4N — Native 128-chip three-update canary

Status: active.

## Purpose

Prove that the untreated native DeepSWE-derived Qwen3-4B-Instruct path can
perform real rollout and training on the frozen P58 recipe before any zero arm
is launched. This is a native integration/training gate, not a numerical
treatment comparison and not zero-TIM evidence.

## Immutable contract

- source: exact post-push readback of `yuxzhang/canon-zero-tim`;
- image: registry digest, never a mutable tag;
- model/data: Qwen3-4B-Instruct-2507 and the frozen 1,012-task clean list;
- topology: one 128-chip `4x4x8` slice split into rollout DP8 x TP8 and trainer
  DP8 x TP8;
- recipe: B8 x G16, response 16,384, 50 turns, RLOO, fixed-context
  `sequence-mean-token-scale`, TPU-resident optimizer, optional interventions
  off, prefix cache off;
- stage: `three-update`, with exactly three optimizer commits;
- arm: `native` only. Rendering or applying `zero` is outside this phase.

## Exit gate

The native classifier must report `PASS` from complete, digest-verified
artifacts. It must prove exactly three commits, finite nonzero A-B treatment
dose, exact B-C, finite training values, device-resident optimizer state,
complete 128-row trajectory batches, journal continuity, sandbox cleanup, and
checkpoint/transaction integrity.

An all-filtered batch may add a trajectory batch without adding an optimizer
commit. It must have a zero-commit receipt and unchanged state. Interrupted
infrastructure, missing evidence, exact native A-B (`NO_TREATMENT`), or any
B-C drift is `INCONCLUSIVE`/failure under the classifier, never PASS.

## After the gate

Preserve and package the immutable run before proposing another launch. A
native canary PASS does not authorize a 1,000-update run and does not reactivate
zero; either decision requires a new user instruction.

## Attempt history

### p58c01 — bootstrap INCONCLUSIVE

Attempt-0 reached only `00_env.sh` and exited before a Pathways device probe,
model initialization, rollout, trajectory, backward, or optimizer action. The
profile correctly kept native stock DP reduction unadmitted, but the inherited
P34 shell check incorrectly demanded admission `1`; the same stock check
required three FrozenLake-only zeros that the DeepSWE profile left unset.

The failed run root is immutable and must not be resumed or reused. Its raw log
is `../evidence/p58c01/run.log`, SHA-256
`f551712696c9c36dbf4f1f2fb713a4c975ff49f2184cf62e887341679341d0bc`.
The next attempt was `p58c02` from the published admission fix SHA. This phase
remains active; p58c01 closed no target gate.

### p58c02 — direct-entrypoint bootstrap INCONCLUSIVE

Pathways initialized exactly once, but the DeepSWE program did not import.
The JobSet executes `/app/examples/deepswe/canonical_entrypoint.py` as a file;
Python consequently exposed `/app/examples/deepswe`, not `/app`, on
`sys.path`. The wrapper's package-qualified
`examples.deepswe.train_deepswe_nb` target was undiscoverable. No model,
rollout, trajectory, forward, backward, optimizer, or checkpoint action ran.

The immutable raw log is `../evidence/p58c02/run.log`, SHA-256
`8983ab0a61355a32c9992e09f33f3e42d3bf673463cf0ca500e54b749fba56de`.
The local fix derives the repository root from the wrapper's own file path,
adds it before package import, and makes the native stock preflight execute
the same direct-file path as the JobSet. A direct `/tmp` launch and the full
pinned-image gate pass. The fix was published as
`82d82f72a7220d945737d95f6266b5b7e2cfe706` and its first readback matched
with ahead/behind `0/0`. The next attempt is fresh run-id `p58c03` from the
final post-checkpoint operator tip.

### p58c03 — resolved-environment reload bootstrap INCONCLUSIVE

Attempt-0 passed `00_env.sh`, exact source sync, the pinned R2E install and
adapter check, stock-engine preflight, Pathways initialization, the direct
entrypoint, TPU device discovery, and bounded R2E runtime patching. It then
stopped before model initialization when the DeepSWE Python contract found
`CANON_LOGPROB_M=256` in the native process environment.

The native profile had correctly unset that presence-sensitive zero-TIM
switch inside `00_env.sh`. The bug was the process boundary: `00_env.sh` is a
child, while its generated `env.sh` contained exports only. Sourcing that file
in the parent layered resolved values over the raw renderer environment and
could not remove its stale `CANON_LOGPROB_M`. The contract rejection was
correct and must not be relaxed. The later one-W&B-run fatal is derivative of
the earlier Python exit, not a second root cause.

The immutable logs are `../evidence/p58c03/run.log`, SHA-256
`15aa9968200c55a02ef47c72c5e209277397835e1752a4dbd9699fce3b2c42b4`, and
`../evidence/p58c03/head_container.log`, SHA-256
`d5e8b5b1941aa5632fa6267cfdac445727c175bf8d2bbcc79c1ece7cf7aba1e2`.
No model, rollout, trajectory, forward, backward, optimizer transaction, or
checkpoint ran.

The local fix makes the generated `env.sh` clear its managed non-secret
namespaces before exporting the exact resolved set. The p58c03 regression
seeds the parent with the renderer's stale value, executes real `00_env.sh`,
reloads its snapshot, asserts the native-only absences, and calls the Python
contract. Focused P58/P34 tests, the P57 81-test adjacent suite, and the full
pinned-image gate pass. After explicit publication approval, the next and
only admissible run-id is fresh native `p58c04`; p58c03 is not resumable.

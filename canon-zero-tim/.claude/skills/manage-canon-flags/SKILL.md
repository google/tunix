---
name: manage-canon-flags
description: Audit, add, change, retire, or diagnose CANON_* runtime flags, especially layered renderer/profile/env behavior, presence-sensitive switches, process delivery, and paired Native/Zero isolation. Use when a flag change or flag explosion affects program identity; do not use for unrelated training debugging.
---

# Manage canon flags

Treat a flag as part of the executable contract, not as a loose configuration
string. The desired result is a traceable flag surface whose resolved values,
consumer process, runtime execution, lifecycle, and treatment ownership are all
proved.

This skill supplements `manage-canon-zero-tim-branch`; it does not replace that
skill's branch, evidence, authorization, or launch rules.

## Start with the actual chain

Before editing, read `canon-zero-tim/FLAGS.md`, the owning task's `state.md`,
`plan.md`, phase file, handoff, and runbook. Trace every affected key through:

```text
renderer/raw environment
  -> shared profile(s)
  -> workload profile
  -> cluster/steps/00_env.sh
  -> authoritative env.sh reload
  -> entrypoint/installer
  -> exact reader process
  -> runtime marker/postflight
```

Use `rg` to find every writer, unset, default, parser, reader, and marker. A
value visible in the launcher does not prove it reached the process that uses
it. One-host docker runners have their own chain — launch env prefix ->
runner script -> `docker -e` array -> in-container inner script -> **sourced
profile re-exports** -> reader — and a profile `export` OVERRIDES the
`docker -e` value; an env-toggled arm designed without tracing this chain can
silently run both arms identically.

Read [references/flag-model.md](references/flag-model.md) when classifying or
designing flags. Read
[references/failure-patterns.md](references/failure-patterns.md) when debugging
a failed or suspicious run. Read
[references/p58-native-zero.md](references/p58-native-zero.md) only for the P58
DeepSWE Native/Zero comparison.

## Required decisions

For each touched flag, record:

- semantic tier: numerical, observational/performance, diagnostic, or
  workload/infrastructure;
- parser kind: presence-sensitive, boolean, enum, integer/float, path, or list;
- default and whether absent, empty, and `0` are distinct;
- exact writer and reader processes, including Pathways proxy/server delivery;
- owning workload and neighboring workloads that must reject it;
- treatment ownership for paired experiments;
- lifecycle and objective sunset condition.

If these cannot be determined from source and current evidence, stop and label
the flag unresolved. Do not guess from its name.

## Change rules

When adding a flag:

1. Prefer one workload-level mode or treatment selector that derives subordinate
   values over many independently combinable booleans.
2. Make experimental behavior default-off and guard it by exact workload
   identity, admission, profile, and arm where relevant.
3. Register the name, semantics, default, lifecycle, and sunset in `FLAGS.md`.
   Add it once to the appendix and update the declared count.
4. Add a positive control plus a neighboring-workload or opposite-arm negative
   control.

When changing a flag:

- preserve absence when source branches on key presence; `FLAG=0` is not a
  substitute for `unset`;
- update the renderer/profile truth table, real `00_env.sh` reload contract,
  Python/runtime contract, and postflight together;
- never relax a hard gate or enable a canonical numerical flag merely to make a
  shared interface accept another arm;
- keep observer-only plumbing on an independent, signed flag when reusing the
  numerical treatment flag would contaminate the experiment.

When retiring or welding a flag:

- confirm its registered sunset condition and callers first;
- preserve historical evidence and veto records;
- treat welding a numerical flag as a program change requiring the same
  certification ladder as enabling it;
- do not delete or silently repurpose disabled flags without the user's
  explicit approval for that change.

## Verification ladder

Run the smallest sufficient ladder, escalating for numerical or process-boundary
changes:

1. registry audit and `git diff --check`;
2. profile truth table, including absent-vs-zero assertions;
3. renderer output through real `00_env.sh`, then reload `env.sh` in the parent
   environment and run the Python contract;
4. opposite arm and neighboring workload negative controls;
5. exact-image installer/import/value probes;
6. target runtime marker and arm-aware postflight;
7. for flags that select between lowered programs (reducer modes, kernel
   paths): a runtime path-fingerprint attestation per arm — xplane
   collective/module census (e.g. ppermute vs all-gather vs all-reduce
   counts) or jit module names — proving each arm executed its own path.

Use the deterministic registry helper before publication:

```bash
python3 canon-zero-tim/.claude/skills/manage-canon-flags/scripts/audit_flag_registry.py \
  --repo . --changed-base origin/yuxzhang/canon-zero-tim
```

A rendered value, installed file, imported module, or zero exit is construction
evidence only. Require the expected runtime marker in the correct process;
missing output is a failure to observe, not a pass.

## Handoff

State the resolved tuple, absence-sensitive keys, exact reader process,
positive/negative tests, terminal markers, and what remains target-unverified.
Update `FLAGS.md` and the owning phase/runbook/handoff in the same concern.
Never claim Native/Zero isolation from environment text alone.

## System-optimization flag family (P68-P71)

The 2026-08 reverse-path optimization wave left three classes of switch.
Classify a new optimization flag into one of them before designing it.

**Flagless (no switch at all).** The jitted gradient assembly and tree
programs, the per-group dispatch-prep hoist, and the reducer program
cache changed dispatch packaging only: same operations, same order, same
dtypes, one program instead of many. Bitwise identity is structural, so
a switch would only add a way to be wrong. Prefer this class whenever
the change cannot alter a value — and prove it (norm anchor plus module
fingerprints), never assert it.

**Selector with a proven geometry (`CANON_DP_COLLECTIVE_REDUCE`,
`CANON_P71_SCAN`).** These change how a computation is expressed, so
their verdict is geometry-scoped and the ledger must say where. Record
the topology each value was proven on, not just "green": psum is
bitwise-verified at DP4xTP1 and unverified against the FP64 oracle at
DP16; `fwd` is bitwise at DP4xTP1 and DP16xTP1 and untested at TP>1;
`bwd` fails closed on a non-unit model axis by construction. A value
that cannot run on a topology must refuse, not degrade.

**Receipt selector (`CANON_DP_COMPARE_MODE`,
`CANON_DP_DISTINCT_SCHEDULE`, `CANON_DP_FINITE_FETCH`).** These leave
every gradient bit alone and change only how often, and how thoroughly,
a receipt is taken. Each needs its own R1 conservation argument, its own
kill-test in the bench registry, and its own retirement path — which is
why they are three enums and not one mode. Never fold receipt selectors
into a performance selector: a future retirement of one must not rename
the values of another.

Delivery differs by lane and is the usual place these break. One-host
docker carriers need an explicit `-e` per flag in the runner *and* the
launch prefix; a workload profile sourced inside the container overrides
both. Target JobSets take neither: the pod environment is assembled by
the renderer's env dict plus the profile's `export` lines, so a flag
that only exists in a docker runner silently does not reach the target.
Add the passthrough in the same change as the flag, and verify delivery
by a runtime fingerprint (module or collective census), never by
reading the launch command.

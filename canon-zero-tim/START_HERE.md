# Start here

You have this package and no context. This page gets you to a useful action in five minutes.

---

## What this is

Rollout engines and training code disagree about logprobs — the same weights and tokens give
different numbers, and that difference is what corrupts the importance ratio in RL. This
package contains the canonical interventions and the gates that test whether a
named workload is **bit-identical**. Some target workloads remain red or
unrun; never promote a local package pass into a production numerical claim.

It is packaging of work already done, not a new result. What is signed and what is not is in
`EVIDENCE.md`; read that before repeating any claim from here.

## Active handoff router (2026-08-10)

The current work is deliberately split into two evidence ledgers. Do not mix
their manifests, artifacts, or promotion claims:

| Workstream | Read first | Target allocation |
|---|---|---|
| P38 GSM8K/FrozenLake alignment | `tasks/p38-pathways-decode-prefill-carrier/HANDOFF.md` | 64-chip Pathways |
| P39 Qwen3-32B DeepSWE | `tasks/p39-deepswe-production/HANDOFF.md` | one 4x8x8 (256-device) Pathways slice |

The DeepSWE workload behavior is referenced from
`yuxzhang/deepswe-quality-fix@023978b976dd6d94e7a42948c3f3a68e34d73744`,
but every target JobSet must fetch an exact published commit from
`yuxzhang/canon-zero-tim`.

---

## Where things stand

| Gate | Status |
|---|---|
| Package matches the signed release sources, byte for byte | **verified** — 25/25 files |
| Engine changes reproduce their patched files exactly | **verified** — 6/6, from a pinned image |
| Chain installs and loads outside its original directory | **verified** — in-image, off `/mnt/disks` |
| CPU mathematical gates | **pass** — plus a negative control that rejects 4 kinds of bad run |
| **Direct-attached 4-chip T1 rerun from this package** | **NOT RUN** — the probe host's TPU was busy |
| DP gradient/update probe on CPU | **PASS** — fixed placement repeats; regrouped samples change bits |
| **Generic DP16×TP4 diagnostic on Pathways/GKE** | **COMPLETE, DIRTY** — 18/18 TP2/4/8 rows completed; replicated, stock and F4 arms all show third-program drift. This is platform evidence, not a production-Qwen verdict. |
| **Canonical Qwen operator P1b on Pathways/GKE** | **TARGET PASS** — Attempt 0 produced zero differing bytes with live gradients at depths 1/2/4/8. |
| **Same-session toy T2 on Pathways/GKE** | **TARGET PASS** — all 7 fixed-placement checks passed on the topology-aware `(16,4)` mesh; arbitrary regrouping remains bitwise-sensitive. |
| **Round-trip: install from this package, reproduce signed numbers** | **NOT RUN** |
| Pathways topology/operator/toy-update admission | **TARGET PASS** — single-slice 64-chip Attempt 0, clean package provenance, P1a/P1b and T2 all passed; the raw log is machine-classified and hash-pinned. |
| DP16×TP4 Qwen3 adapter and abstract state inventory | **LOCAL CPU PASS** — fixed placement, grouped segmented adapter, deterministic DP16 tree and exact Qwen3-8B state shapes; no target materialization claim. |
| Bounded Qwen3-8B model initialization, backward and three optimizer commits | **TARGET SYSTEMS PASS** — dense-reference RC only; zero-TIM alignment was not measured. |
| P33 DP16×TP4 rank-local reducer and workload contracts | **LOCAL PASS, PUBLISHED** — 256 reverse calls/update correctness vehicle; target workload not run. |
| P33 FrozenLake and GSM8K production workloads | **R18 BOUNDARY 1 PASS; BOUNDARY 2 FAIL** — action-only serving decode and prefill are bitwise exact, while trainer-old is red. Do not rerun unchanged; use `cluster/P35_ENVELOPE_HANDOFF.md` to separate packing/metadata from wrapper/program context. |
| P35 three-arm envelope discriminator | **MULTI-CHUNK REPAIR PUBLISHED; TARGET NOT RUN** — r24 confirmed the response-256 Splash repair but exposed a diagnostic-only one-chunk assumption before B. The repaired producer reconstructs complete multi-chunk B metadata; no 64-chip carrier verdict exists yet. |
| P34 Qwen3-32B DeepSWE DP16×TP8 per role | **LOCAL PASS, TARGET NOT RUN** — role split, canonical M256 adapter, fixed DP16 reducer, renderer and classifier exist; no 4×8×8 workload artifact. |

**Do not spend another 64-chip run repeating the bounded P32 admission.** Its model-init,
backward and three-update systems gates have already passed, but they used a dense synthetic
objective and did not measure A=B=C. The next target work is the P33 queue in
`cluster/P33_QUEUE.md`. For FrozenLake specifically, r15 stopped before the boundary reporter;
follow `cluster/P35_ENVELOPE_HANDOFF.md`; the next target work is a three-arm pre-backward
envelope discriminator, not another unchanged short diagnostic. GSM8K remains independently
classified. TP8 remains a generic
platform diagnostic until it has a separate Qwen8B production contract.

---

**Pathways regime note (2026-08-10).** P36 proved the excess-precision flag never reached
the server-side compiler before `envon1` (KNOWN_FOOTGUNS #13). All Pathways renderers and both
static manifests now deliver `XLA_FLAGS=--xla_allow_excess_precision=false` through the
`pathways-proxy` container environment, and contract tests lock it. Every Pathways number
recorded before `envon1` is a flag-off baseline. The active target is P36.3: rerun the P35
three-arm envelope discriminator under the flag-on regime.

## What can your machine do?

| You have | Run | Time |
|---|---|---|
| Any machine with Python + JAX (CPU is fine) | **Task A** — CPU gates | < 1 min |
| ≥ 2 TPU chips, no model needed | **Task B** — topology probes | seconds + compile |
| A GKE cluster with Pathways | **Task C** — cluster ladder | minutes, staged |
| The pinned image + a 4-chip v5p + a checkpoint | **Task D** — round-trip | ~20 min / ~1 h |

Steps in `RUNBOOK.md`. Do not skip ahead: each task's output is the input to deciding whether
the next one is worth running.

---

## Three things that will bite you

**1. A green run can mean nothing.** The engine chain is loaded by file path and by module
name. A missing member does not raise — the engine silently uses its stock module, every
switch still reads "on", and everything passes. `[PATHTRACE]` lines are the only evidence the
intervention ran. **The exit code is not evidence.**

**2. Absence is not a pass.** A gate that printed no measurement line did not run. Treat a
missing line as red.

**3. `grep` lies on these logs.** Progress-bar control characters make `grep` treat a log as
binary and silently report nothing — indistinguishable from "it never happened". Always
`grep -a`.

More of these, all of them real, in `KNOWN_FOOTGUNS.md`.

---

## What to report back

Artifacts, not conclusions. A verdict without its raw log cannot be re-read later.

```
1. what you ran      task letter, exact command, machine (chip count, kind, single/multi slice)
2. raw output        file path + sha256sum -- not a paste of the interesting lines
3. the numbers       verbatim, including the ones that looked boring
4. any override      CANON_ALLOW_IMAGE_DRIFT / CANON_ALLOW_UNVERSIONED / edited thresholds
5. exit codes        of every step, not just the last
```

Do not paste tokens or `.git/config` contents. Do not summarise a red gate as "mostly working".
If a threshold had to move to make something pass, that is the finding — report it as one.

---

## Where to read more

| Question | File |
|---|---|
| What is the actual mechanism? Why bitwise? | `README.md` |
| What versions is this pinned to, and can I reach them? | `ANCHORS.md` |
| What is proven, and where is the evidence weak? | `EVIDENCE.md` |
| What should I measure before trusting a new topology? | `CLUSTER_ADMISSION.md` |
| What traps produce green runs? | `KNOWN_FOOTGUNS.md` |
| How do I run it on GKE? | `cluster/README.md` |
| How were the signed numbers produced? | `recipes/README.md` |
| How was this package built, phase by phase? | `docs/` |

`docs/` is provenance — the record of how each piece was verified, written during the work. It
is useful for "why is it like this", not as a starting point. This page is the starting point.

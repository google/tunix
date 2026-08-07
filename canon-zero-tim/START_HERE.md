# Start here

You have this package and no context. This page gets you to a useful action in five minutes.

---

## What this is

Rollout engines and training code disagree about logprobs — the same weights and tokens give
different numbers, and that difference is what corrupts the importance ratio in RL. This
package makes them **bit-identical** and ships the gates that prove it.

It is packaging of work already done, not a new result. What is signed and what is not is in
`EVIDENCE.md`; read that before repeating any claim from here.

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
| **Generic DP16×TP4 diagnostic on Pathways/GKE** | **COMPLETE, DIRTY** — 12/12 full-slice rows completed; replicated, stock and F4 arms all show third-program drift. This is platform evidence, not a production-Qwen verdict. |
| **Canonical Qwen operator P1b on Pathways/GKE** | **NOT RUN** — the new hard gate directly executes promoted RMSNorm/projections/SwiGLU/F4 at model dimensions. |
| **Same-session T2 on Pathways/GKE** | **NOT RUN** — the old second Python process connected a second IFRT client and the log ended without a T2 marker. |
| **Round-trip: install from this package, reproduce signed numbers** | **NOT RUN** |
| Pathways numerical admission | **NOT PASSED** — requires canonical P1b PASS followed by same-session T2 PASS |

**The next useful action on the 64-chip target is Task C in `RUNBOOK.md`** — rerun the staged
cluster ladder through `dp-gate-only` with the canonical P1b and same-session T2 changes.  A P1b
red stops T2; a P1b green permits the DP fixed-placement gate without starting another client.

---

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

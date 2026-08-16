---
name: manage-canon-zero-tim-branch
description: Operate the yuxzhang/canon-zero-tim delivery branch — the zero-TIM RL stack, its diagnostics, its evidence, and its multi-agent workflow. Use for evaluating runs, landing CLs, launching diagnostics, managing flags, and packaging evidence. Use abc-bitwise-repro for the numerical A/B/C derivations themselves. This in-package copy (canon-zero-tim/.claude/skills/) is the canonical version; the repository-root and outer .claude copies are pointer stubs.
---

# Operate canon-zero-tim (v2)

## 0. Sources of truth — read these before this skill

Facts live in registries; this skill defines procedures only. **On any conflict, the
registry wins over this skill and over the references/ files.**

| File (package root) | Holds | Mutation rule |
|---|---|---|
| `THREADS.md` | six-thread status board: state, next gate, owner | single writer (the evaluator), updated per pull |
| `FLAGS.md` | every settable `CANON_*` flag: tier, lifecycle, sunset, vetoes | append-freely / retire by sunset; veto section append-only |
| `EVIDENCE.md` | claim → run → artifact path + SHA | append-only |
| `KNOWN_FOOTGUNS.md` | numbered incident lessons | append-only |

Reading order for any task: `THREADS.md` → the thread's `tasks/<thread>/state.md` and
`log.md` → the lane's runbook under `cluster/` → `FLAGS.md` for every switch you touch.

## 1. Non-negotiable boundaries

- `main` is read-only. Published history is immutable; pay debt with additive commits.
- Never print a remote URL, `.git/config`, or any credential/W&B key/HF token/K8s secret.
  Credentials and their wiring are user-owned; read-only presence checks only.
- **Never commit or push without the user's explicit approval for that specific action.**
  Executing and maintenance agents build CLs locally and stop; the user approves each push.
- Code, exceptions, markers, and program output are English.
- Never alter precision, loss, sampling, gradient, optimizer semantics, or a production
  default to make a numerical gate pass, unless the user explicitly approves.
- Preserve failed logs and failed run directories. Infrastructure failure is
  `INCONCLUSIVE`, never numerical red, never deleted.
- Long jobs run in background with short polls; never wait blindly.

## 2. Resolve the repository, revision, and worktree

1. Never assume the current directory or branch. Record repo, branch, full HEAD, dirty
   state with read-only `git -C` commands (no remotes printed).
2. Run `python3 canon-zero-tim/.claude/skills/manage-canon-zero-tim-branch/scripts/preflight_runtime.py
   --repo <worktree> --require-clean` before editing.
3. Worktree conventions: new worktrees go under `worktrees/` at the outer-repo root
   (`sequence_packing/` is the deprecated legacy lot; the primary clone
   `sequence_packing/tunix` stays where it is — it holds the shared object store every
   worktree points into, and relocating it is deferred surgery). Prefix meaning:
   `yuxzhang/*` = shared pushed branches; `local/*` = local staging branches;
   `backup/*` = immutable anchors. **Multi-day work gets a named `local/*` branch, not a
   detached HEAD** — detached commits die with their worktree. Integration is
   rebase-then-fast-forward onto `yuxzhang/canon-zero-tim`; the shared history is a
   straight line, never a merge.
4. Fetch before editing only when remote state matters; never pull/rebase a dirty tree;
   never clean or reset another agent's worktree.

## 3. Shape ledger — before touching any numerical boundary

Keep five row counts as separate named variables; collapsing any two has caused real
incidents (details: `references/shape-contracts.md`):

caller-global M · shard-local M · canonical-kernel M · semantic valid rows ·
scheduler capacity (`MIN_TOKEN_BUCKET` is global; `max_num_batched_tokens`/`max_num_seqs`
are per-DP-rank). The signed DP16 contract: rollout global 256 → local 16 → pad to
kernel 256; prefill global 4096 → local 256 → the same kernel-256 program. Reject every
unregistered global shape fail-closed. `co-batch=1 ≠ shape 1`: a single-active call still
runs the full fixed-M program (verified in production, P38s16); a replay at DP1/batch-1 is
a different executable and inadmissible as strict evidence.

## 4. Claim layers and status vocabulary

Choose the cheapest layer that answers the question; never promote across layers:
static → host CPU → pinned exact-image → direct TPU → Pathways Attempt-0 → real recipe.
Status words: `IMPLEMENTED`, `STATIC PASS`, `CPU PASS`, `TARGET NOT RUN`, `TARGET PASS`,
`INCONCLUSIVE`. Evidence grades: **signed** (full terminal package + classifier PASS) vs
**analysis-grade** (core numbers internally verified, formal package incomplete) — an
analysis-grade run may inform decisions but never closes a target gate.

## 5. Run contract (every launch, success or failure)

1. Every launch gets a run id and, on return, one immutable directory under the owning
   thread's `evidence/` — bootstrap deaths included.
2. Before launch: render from an exact SHA + pinned image; run the launch-preflight trio
   (resolved-env truth table, contract sweep, intent-diff — the intent-diff exists because
   a mislabeled s12b ran at concurrency 256).
3. During: the GCS incremental snapshot worker is the durability layer
   (`PREFLIGHT.json` = bucket writable, `LIVE.json` = in progress, `COLLECTED.json` =
   evidence saved, `COMPLETE.json` = postflight passed). Terminal markers are written by
   the **surviving worker**, never by post-exit shell steps (four runs died in that window).
   `PREFLIGHT` is not a numerical result; `COLLECTED` is not a completed run.
4. After download: `scripts/package_run.sh <src> <run_dir>` — dedup, compress (never
   truncate), self-excluding `SHA256SUMS`, five-piece completeness check that synthesizes
   an INCONCLUSIVE `verdict.json` when pieces are missing. Run directories are write-once.
5. Append the run to `EVIDENCE.md`; update `THREADS.md`.
6. Judge with the classifier before reading numbers; a zero exit code is not evidence —
   marker lines are; completeness is judged **before** instrument reachability; INIT
   proves a module loaded, only OBSERVE proves the production path calls it; Attempt-0
   only — a retried JobSet launders determinism claims.

## 6. Gate ladder for code changes

`git diff --check` + syntax + secret scan → host-CPU units + negative controls → pinned
exact-image overlay/manifest tests → forced-device CPU mesh (if sharding changed) → direct
TPU (if kernel/mesh changed) → Pathways Attempt-0 → real recipe + classifier + postflight.
Stop at the first hard failure; downstream numbers are VOID. Every classifier change
re-runs its negative controls; a gate that cannot fire is not a gate (a bf16 +0 low-bit
negative stayed green because the device flushes subnormals — flip a normal value).

## 7. Flag lifecycle

- New flag: register in `FLAGS.md` **with a sunset condition** before the code lands.
  Creation is cheap; deletion is ordered.
- Lifecycle: experimental → certified (per workload) → default-on → welded / retired.
- Rollout of observational flags: first enable as `=verify` riding a routine run
  (bitwise dual-compute, raise on mismatch); flip to `=1` after the first green.
  Certification never transfers across workloads or geometries (program identity).
- **Welding a numerical flag removes a code path = program change**: it needs the same
  certification ladder as enabling it. Diagnostic families sunset with their case.
- The veto/retirement section is append-only; re-proposing a vetoed item requires new
  evidence and the user's explicit reversal.

## 8. Diagnostic regimes (P38-class investigations)

- Frozen-weight multi-round runs: N rollout/alignment rounds, zero backward, zero
  optimizer commits; nonterminal rounds use the dedicated control flow, the final round
  exits 42; per-round capsules are immutable and the latest-red alias may not overwrite
  an earlier red round.
- Every observer passes the **observer-neutrality gate** first: instrumented and
  uninstrumented endpoints bitwise equal on one host; device reads must not add program
  boundaries; per-token device fetches are forbidden.
- Lanes: production training runs **warning-only** (A−B demotes to warning; `B−C ≠ 0`,
  non-finite, replica inequality, transaction faults stay fatal). Diagnostics run
  **strict**. The A−B sentinel is not removable while the carrier case is open
  (user ruling 2026-08-15); campaigns under warning-only are classified
  alignment-degraded and never claim zero-TIM.
- Correction discipline: any claim must be reproducible from committed inputs by re-running
  the official classifier; a withdrawn claim stays in the log as a superseded entry
  (the s17 "differs→equal" correction is the model).

## 9. Multi-agent protocol

Roles: **evaluator** (reviews every pull; owns the board's structure, cross-thread
arbitration, and claim-wording demotions; approves nothing); **thread executors** (own
their thread's worktrees and CLs, and write their own `THREADS.md` row and their own
runs' `EVIDENCE.md` rows); **maintenance agent** (governance CLs). Row-level write
permission on the registries; one writer per any other mutable file. Push queue: hot-thread pushes take
priority; before any push, re-fetch and prove the remote tip equals your base — if it
moved, rebase and re-run the focused gates; after pushing, verify local HEAD equals the
remote-tracking SHA. Handoffs are files, not chat: a task is transferable when its
state.md answers "what is true, what is next, what is the gate".

## 10. Commits and CL discipline

One concern per CL. Description: imperative complete-sentence first line; body states the
problem, the approach, **the downsides**, and the background with artifact paths (raw log
paths on disk, not perishable links). No authorship trailers. Refactors ship separately
from behavior changes. Evidence commits are append-only; documentation must match the
committed artifacts (stale handoff numbers are a named incident class).

## 11. Report format

Lead with gate status, then: branch/HEAD/dirty state · first failing boundary + shape
ledger · implementation vs validation vs claim (three ledgers, never collapsed) ·
TARGET NOT RUN debt · exact next command + expected artifact · rollback · external
mutations performed. Never claim zero-TIM from static, CPU, exact-image, or
admission-only evidence; never call an analysis-grade run signed.

## References

`references/` holds domain background (shape contracts, runtime/compilation, alignment
diagnostics, recipe lifecycle, review rubric). They are periodically snapshotted and may
lag; **registries and raw evidence override them**.

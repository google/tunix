---
name: manage-canon-zero-tim-branch
description: Review, validate, and safely advance the additive yuxzhang/canon-zero-tim delivery branch and its self-contained canon-zero-tim package. Use when auditing the original six-CL stack, checking CL boundaries and descriptions, reconciling documentation with executable evidence, paying TPU/P6a/GKE validation debt, changing NOT RUN claims, or preparing follow-up commits without rewriting published history or touching main.
---

# Manage the canon-zero-tim Branch

Use this skill for delivery-branch integrity and evidence management. Use the project's
`abc-bitwise-repro` skill instead when the task is to explain or modify the numerical A/B/C
mechanism itself.

## Non-negotiable boundaries

- Treat `main` as read-only. Never merge, rebase, reset, or checkout over it.
- Treat the published clean-branch history as immutable unless the user explicitly requests a
  history rewrite. Pay debt with additive follow-up commits.
- Do not infer validation from additive-only changes, zero deletions, `py_compile`, CPU tests,
  a successful import, or an honest `NOT RUN` note.
- Do not print remote URLs or `.git/config`; credentials have appeared in a remote URL before.
- Do not push, open a PR, apply a JobSet, or allocate/change cloud resources without explicit
  approval.
- Keep code docstrings and program output in English.
- An explicit user-approved operational exception does not turn a red numerical boundary green.
  Encode it as a default-off, scope-locked policy with preregistered bounds, a distinct verdict
  and claim level, negative controls for every rejected scope, and a one-switch rollback. Preserve
  the original strict gate for all other workloads, stages, and boundaries.

## 1. Resolve the exact object under review

Never assume the current directory or branch.

1. Locate the repository containing `canon-zero-tim/`.
2. Record the branch name, full HEAD, worktree status, and requested commit range with read-only
   `git -C` commands. Do not display remotes.
3. Distinguish the original six-CL package endpoint (`3f037d8d`) from later additive commits.
   Use current files for current status and `git show <commit>:<path>` for historical claims.
4. Preserve unrelated dirty files. A clean delivery branch does not authorize cleaning another
   worktree.

For a mechanical inventory of the original stack, run:

```bash
python3 .claude/skills/manage-canon-zero-tim-branch/scripts/audit_commit_stack.py \
  --repo sequence_packing/canon_zero_tim_p32
```

The script emits facts only. It does not award a quality or validation verdict.

## 2. Read the branch's sources of truth

Read these from the revision being evaluated, in order:

1. `canon-zero-tim/START_HERE.md`
2. `canon-zero-tim/EVIDENCE.md`
3. `canon-zero-tim/ANCHORS.md`
4. `canon-zero-tim/RUNBOOK.md`
5. `canon-zero-tim/CLUSTER_ADMISSION.md`
6. `canon-zero-tim/KNOWN_FOOTGUNS.md`
7. `canon-zero-tim/docs/plan.md`, `docs/design.md`, and the relevant `docs/phase*.md`
8. The full commit messages and diffs for the CLs being judged

`docs/` is provenance, not the operator entry point. Use it to explain why a decision was made,
not to override a newer signed gate or runbook.

## 3. Maintain three separate ledgers

Never collapse these ledgers into one:

| Ledger | Question | Typical evidence |
|---|---|---|
| Implementation | What code or packaging exists? | diff, manifest, installer, scripts |
| Validation | What actually ran, where, and with what result? | exact command, raw log, SHA-256, exit status |
| Claim | What may now be said? | signed evidence row and its stated scope |

Use only these status words:

- `IMPLEMENTED`: code exists; no execution claim.
- `STATIC PASS`: syntax, manifest, byte identity, or other non-runtime check passed.
- `CPU PASS`: CPU gate passed; no TPU claim.
- `TARGET NOT RUN`: required TPU, Pathways, GKE, or round-trip gate has not run.
- `TARGET PASS`: the intended target gate ran, fail-closed checks passed, and raw evidence exists.
- `INCONCLUSIVE`: missing measurement, missing control, infrastructure failure, or ambiguous scope.

Read `references/review-rubric.md` before changing a status or evaluating CL quality.

## 4. Audit each CL

For every CL, report:

1. concern and dependency;
2. files and insertions/deletions;
3. whether mechanical vendoring/refactoring is separated from behavior;
4. whether the CL can be checked out and inspected independently;
5. verification actually performed at that revision;
6. unverified paths and operational drawbacks;
7. whether any path escapes `canon-zero-tim/`.

Apply the repository's CL rule: one concern per CL; roughly 1,000 lines is a review warning,
not an automatic failure. A large generated or vendored move can be acceptable when isolated.
Mixing that move with an installer or behavioral change is review debt because it hides the
small semantic delta inside mechanical volume.

For an already-published oversized CL, prefer recording the debt and adding validation or a
follow-up clarification. Do not force-push merely to make the history prettier unless the user
explicitly accepts the coordination cost.

## 5. Promote evidence only through the matching gate

Use the branch's current runbook commands; do not reconstruct them from memory. The baseline
debt ladder is:

1. patch application and byte-identity checks;
2. shim/install manifest checks plus live import markers;
3. T0 CPU oracle and its negative control;
4. T1 on real TPU, including every expected measurement line;
5. P6a/D1 fresh-install round-trip against signed numbers;
6. JobSet rendering/schema or server-side dry-run where available;
7. staged GKE execution: `probe-only` -> `install-only` -> `gate-only` -> `dp-gate-only`;
8. training only after admission and separate user approval.

Before reading numerical output, run the gate's postflight/classifier. Missing target lines,
hard-gate failure, wrong backend, an override, or a negative control that does not reject makes
the result red or inconclusive. Never upgrade `TARGET NOT RUN` from a code review.

Every promoted claim must include:

- exact reproducible command;
- machine/topology and pinned image or source revision;
- raw artifact path and SHA-256;
- all relevant exit codes and overrides;
- targeted rollback (normally unset the gate or revert the additive CL).

## 6. Prepare safe follow-up changes

When the user asks for a fix:

1. Start from the intended clean branch, never `main`.
2. Re-read current remote state before editing, but do not pull/rebase unless asked.
3. Freeze the validation entry point before a long run.
4. Keep validation-debt repayment separate from functional changes where practical.
5. Update evidence/status prose only after the raw gate exists.
6. Run secret scans that exclude neither `.git/config` nor generated run trees, but never print
   the secret-bearing values.
7. Show the proposed commits, tests, residual `NOT RUN` items, and rollback before requesting
   push approval.

If a user has explicitly approved a bounded training exception, additionally require the runtime
record and final classifier to distinguish `reported_reds` from `blocking_reds`. A successful
process exit under that exception must use a non-zero-TIM verdict name and must never be promoted
to `TARGET PASS` for the original bitwise claim.

## 7. Report the result

Lead with the honest branch status, then give:

```text
Branch / full HEAD / dirty state
CL audit: concern, size, dependency, verified scope
Evidence matrix: IMPLEMENTED / STATIC PASS / CPU PASS / TARGET NOT RUN / TARGET PASS
Debt: exact missing gate and why current evidence cannot replace it
Next action: cheapest discriminating gate, command, expected artifact, rollback
Mutations: files changed, commits created, pushes or external actions performed
```

Never say “the package is verified” when only its packaging is verified. Say which layer is
green and retain every remaining `NOT RUN` item verbatim.

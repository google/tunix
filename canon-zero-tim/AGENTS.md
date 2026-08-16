# AGENTS.md — start here (cold-start entry point)

You are an agent with no background on this repository. Read this file top to bottom
(~3 minutes), then follow its pointers. **This file contains directions, not facts** —
current state always lives in the registries listed below; if anything here seems to
conflict with a registry, the registry wins.

## What this package is

`canon-zero-tim/` is a self-contained overlay that makes reinforcement-learning training
on TPU **bitwise train-inference consistent** ("zero-TIM"): the three log-probability
sources of policy-gradient RL — sampling decode (A), same-engine prefill re-score (B),
and the trainer's differentiable forward (C) — are made bit-identical, with a certified
backward through the real paged-attention cache. It ships as numbered patches over a
pinned vLLM-TPU image plus profiles, renderers, gates, diagnostics, and evidence.
Technical narrative: `../zero_tim_design_doc.md` (outer repo). Historical orientation
from the original delivery: `START_HERE.md`.

## Rule zero — read before touching anything

1. **Never commit or push without the user's explicit approval for that specific
   action.** Build locally, stop, report, wait.
2. `main` is read-only. Published history is immutable. Work in your own worktree on a
   named `local/*` branch (not detached).
3. Never print remote URLs, `.git/config`, or any credential / W&B key / HF token /
   K8s secret. Credentials are user-owned; presence checks only.
4. Preserve failed logs and failed run directories. Infrastructure failure is
   `INCONCLUSIVE`, never numerical red, never deleted.
5. Never change precision, loss, sampling, gradient, or optimizer semantics to make a
   gate pass, unless the user explicitly approves.
6. Code, exceptions, markers, and program output are English.

## The 60-second orientation (read in this order)

| Step | File | Answers |
|---|---|---|
| 1 | `THREADS.md` | what the six work threads are and where each one stands right now |
| 2 | `../.claude/skills/manage-canon-zero-tim-branch/SKILL.md` (repository root) | how to operate: run contract, gate ladder, flag lifecycle, multi-agent protocol (canonical copy is this in-branch one) |
| 3 | `FLAGS.md` | what every `CANON_*` switch means, its lifecycle, and what has been permanently vetoed |
| 4 | `EVIDENCE.md` | which run proves which claim, and where the artifact + SHA lives |
| 5 | `KNOWN_FOOTGUNS.md` | numbered incidents — the mistakes already paid for |
| 6 | your thread's `tasks/<name>/state.md` → `plan.md` → `log.md` | the task you were actually asked to do |

## Repository map

```
THREADS.md FLAGS.md EVIDENCE.md KNOWN_FOOTGUNS.md   registries (current truth)
AGENTS.md START_HERE.md RUNBOOK.md                  entry docs (this file = front door)
install.sh MANIFEST.sha256 patches/01..NN           the overlay itself (append-only chain;
                                                    new behavior = new numbered patch)
cluster/                                            renderers (one per task family, being
                                                    consolidated), profiles/*.env, steps/
scripts/                                            operator tools incl. package_run.sh
                                                    (evidence packaging — use it for every run)
tests/                                              gate suites + negative controls
tasks/<p-name>/                                     one directory per investigation:
                                                    state.md plan.md log.md phases/ evidence/
threads/<thread>/runs/<id>/                         new-run directories (first-use forward)
debug_logs/                                         frozen read-only legacy evidence pile
docs/                                               provenance history
```

The operating skill lives one level up at the **repository root**:
`../.claude/skills/manage-canon-zero-tim-branch/` (root placement is what agent harnesses
auto-discover; it is version-controlled with this package and is NOT stray tooling).

## The six threads (details and live status: `THREADS.md`)

| Thread | Concern | Task dir under `tasks/` |
|---|---|---|
| zero-tim-carrier | the open A-vs-B decode residual investigation | `p38-pathways-decode-prefill-carrier` |
| perf | performance without moving a single logprob byte | `p48-onehost-perf` (+ outer repo `tasks/p48*..p52*`) |
| frozenlake-train | FrozenLake training campaigns | `p45-frozenlake-*` |
| deepswe-eval | DeepSWE data-cleaning evaluation (reward-only, stock) | `p46-deepswe-*` |
| deepswe-train | DeepSWE training proof | `p44-deepswe-qwen4b-parity` |
| delivery-docs | design doc, one-pager, delivery branches | outer repo `tasks/canon_zero_tim_package` |

## Common operations → where the procedure lives

- **Evaluate a run that came back** → SKILL §5.6 (classifier before numbers; markers not
  exit codes; completeness before reachability) and §4 (signed vs analysis-grade).
- **Land a code change** → SKILL §6 gate ladder + §10 CL discipline (one concern per CL;
  description states problem, approach, downsides; no authorship trailers).
- **Launch an experiment** → SKILL §5 run contract (preflight trio before render; every
  launch gets an immutable run directory, failures included; package with
  `scripts/package_run.sh`).
- **Add or change a flag** → SKILL §7 (register in `FLAGS.md` with a sunset condition
  first; `=verify` before `=1`; welding a numerical flag requires recertification).
- **Touch anything numerical** → SKILL §3 shape ledger first; derivations live in the
  outer `abc-bitwise-repro` skill; never re-derive from memory.

## Multi-agent protocol (summary; full text SKILL §9)

One **evaluator** owns `THREADS.md` and reviews every pull; **thread executors** own
their thread's worktrees and CLs; a **maintenance agent** owns governance CLs. One
writer per mutable file. Before any (approved) push: re-fetch, prove the remote tip
equals your base, rebase and re-run focused gates if it moved, verify the remote SHA
afterward. Hot scientific threads outrank maintenance in the push queue.

## What not to do

The permanently vetoed list (rescore-B downsampling, `LAYER_SCAN=1`, unified-KV as a
fix, and others) lives in `FLAGS.md` § veto/retirement — re-proposing an entry there
requires new evidence and the user's explicit reversal. The incident catalog that
explains *why* most rules exist is `KNOWN_FOOTGUNS.md`. When in doubt: smallest safe
claim, exact artifact paths, and ask the user only what only the user can decide.

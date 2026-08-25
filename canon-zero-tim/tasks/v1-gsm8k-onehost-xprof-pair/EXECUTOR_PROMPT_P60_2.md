# Cold-start executor prompt for P60-2

Copy the block below verbatim to an agent with no prior context.

```text
You own the P60-2 Zero-HP XProf readability task. You have no historical
context; repository files and immutable evidence are authoritative.

Repository root:
/home/yuxuan/code_rl_repro

Goal:
Make the GSM8K one-host Zero-HP/P59 parallel-backward update readable in XProf
as train -> zero_tim_update -> stages -> 16 groups -> transactions, without
changing any numerical program, fixed reduction order, synchronization, or the
official semantic Perfetto vocabulary.

First actions, before editing:
1. Fetch origin/yuxzhang/canon-zero-tim and record its actual tip.
2. Read /home/yuxuan/code_rl_repro/canon-zero-tim/AGENTS.md.
3. Read these skills completely:
   - canon-zero-tim/.claude/skills/manage-canon-zero-tim-branch/SKILL.md
   - canon-zero-tim/.claude/skills/manage-canon-flags/SKILL.md
   - /home/yuxuan/.codex/skills/run-phased-work/SKILL.md
   - /home/yuxuan/code_rl_repro/.claude/skills/read-xprof/SKILL.md
   - /home/yuxuan/.codex/skills/xprof-trace-analysis/SKILL.md
4. Create a fresh isolated local/* branch and worktree from the fetched tip.
   Do not modify another agent's worktree.
5. In that worktree read, in order:
   - canon-zero-tim/tasks/v1-gsm8k-onehost-xprof-pair/state.md
   - canon-zero-tim/tasks/v1-gsm8k-onehost-xprof-pair/plan.md
   - every canon-zero-tim/tasks/v1-gsm8k-onehost-xprof-pair/phases/p60-2*.md
   - canon-zero-tim/tasks/v1-gsm8k-onehost-xprof-pair/HANDOFF_P60_2.md
   - canon-zero-tim/tasks/v1-gsm8k-onehost-xprof-pair/RUNBOOK.md
   - canon-zero-tim/tasks/p48-onehost-perf/P51_XPROF_RUNBOOK.md

Known facts you must preserve:
- Native and Zero-HP captures are complete. Native TPU:0 has 672 module
  events and a host train annotation. Zero-HP TPU:0 has 59,028 module events
  and no host train annotation.
- Zero-HP completed 3/3 updates, passed 51/51 strict alignment, and contains
  all five P59 backward families on all 8 TPU planes with decode absent.
- Therefore this is a readability defect, not missing backward.
- The existing Native/Zero pair is INCONCLUSIVE_INPUT_MISMATCH; never publish
  a timing ratio from it.

Implement only P60-2B from HANDOFF_P60_2.md:
- reuse CANON_XPROF_LABELS=1;
- absent/empty/0 is an exact no-op and invalid values fail closed;
- add one train StepTraceAnnotation and bounded low-cardinality
  TraceAnnotations for update, forward/loss/reverse groups, replay,
  model-backward, report-adjoint, fixed-reduce, accumulation, optimizer;
- match only Native's annotation API with
  `StepTraceAnnotation("train", step_num=1)`; do not claim matching Native's
  microstep cadence, cardinality, or monolithic program shape;
- require the complete host hierarchy on the same `/host:CPU` `python3` track
  and non-empty device `Steps` rows on all 8/8 TPU planes in the census;
- add no per-layer spans;
- add a full-XPlane hierarchy census with pure interval validation and strong
  synthetic negative controls;
- integrate the revised Zero-HP arm classifier without changing Native.

Hard prohibitions:
- no jit/shard_map/collective/reducer/gradient/optimizer/precision/loss change;
- no block_until_ready/device_get/barrier/synchronization addition;
- no custom nested semantic Perfetto spans;
- no filtering or hiding raw XProf events;
- no TPU launch, commit, push, render, or Kubernetes operation without the
  user's explicit approval for that exact action.

Use read-xprof as the primary profiling authority: phase=update for backward,
host tracer 1, Python tracer 0, full XPlane and 8/8 TPU planes for completeness,
and [PERF] only for speed decisions. xprof-trace-analysis is secondary and may
summarize program shape only after the full-plane ledger.

Run all host/static/synthetic/P59 regression/flag gates and, if available
without occupying TPU, the pinned exact-image annotation API gate. Update the
phase ledger and stop before TPU. Return:
1. actual base SHA and worktree path;
2. exact files and logic changed;
3. gate output verbatim;
4. proof that no synchronization or semantic vocabulary changed;
5. remaining risks and claim ceiling;
6. one-step rollback;
7. the exact proposed one-host command and fresh label.

Do not commit or push. Wait for the user to approve the next boundary.
```

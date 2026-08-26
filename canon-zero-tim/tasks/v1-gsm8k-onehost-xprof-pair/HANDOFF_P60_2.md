# P60-2 execution handoff: make P59 backward readable in XProf

> **Current phase is P60-2G.** The original whole-update hierarchy below is
> retained as historical context, not the current target. P60-2G uses one
> Native API `train` span per real Zero-HP reverse/reduce/accumulate
> transaction, train steps 32..47 for warm update 2. The last real span owns
> optimizer commit; no synthetic terminal step is allowed. Read
> [`p60-2g-native-train-steps.md`](phases/p60-2g-native-train-steps.md) before
> operating this task.

## P60-2G remote-operator quickstart (authoritative)

The current implementation lives in
`/home/yuxuan/code_rl_repro/worktrees/p60_2g_native_steps_0825` on
`local/p60-2g-native-steps-0825`, based on `9f91d930`. The P60-2G change is
still uncommitted. The current remote-tracking tip is one evidence-only commit
ahead at `23e0bddfc4f742eecb66092db5bdc80def9571ca`; it changes only
`tasks/v1-apc-m15-target-debug/` and has no path overlap with this 22-path
concern. **Do not launch from either base SHA or from a reconstructed patch.**
After explicit local-commit approval, rebase that implementation commit onto
the then-current operator tip, rerun the required gates, and record the final
full 40-character SHA here before a remote agent treats the run as clean-source
acceptance evidence. Commit, push, and TPU launch remain separate approval
boundaries.

This is a Zero-HP-only fresh carrier. Do not rerun Native and do not run the
historical two-arm procedure in `HANDOFF.md`. The signed capture tuple is
fixed, not operator-tunable:

| Field | Required value |
|---|---|
| Host/topology | one direct four-chip v5p, DP4×TP1, mesh ids `0,2,1,3` |
| Work | Qwen3-1.7B GSM8K, 8 prompts, 8 generations, response 256, concurrency 1 |
| Updates | 3 real optimizer updates |
| Capture | `phase=update`, skip 2, steps 1, window `2->3` |
| Tracers | host 1, Python 0, TPU `TRACE_ONLY_XLA`, labels 1 |
| UI index | `train_32..train_47`; train 47 owns optimizer commit |
| XProf budget | soft warning `1,200,000,000`; hard maximum `1,500,000,000` logical bytes |

The runner generates both `train/xprof_size_census.txt` and
`train/xprof_size_receipt.json`. `PASS` is at or below 1.2 GB; `WARN` is above
1.2 GB but at or below 1.5 GB and remains admissible; `FAIL` or any total above
1.5 GB makes the arm RED. The budget counts every regular file under
`train/xprof` and never truncates, deletes, or rewrites an oversized capture.
The raw XPlane, compressed trace JSON, semantic Perfetto, receipt, censuses,
classification, alignment/update reports, and terminal logs enter the root
`SHA256SUMS` before its immediate verification.

On the remote v5p host, after receiving the approved implementation SHA:

```bash
cd <clean-worktree-at-the-delivered-p60-2g-sha>
test -z "$(git status --porcelain)"
git rev-parse HEAD
python3 canon-zero-tim/.claude/skills/manage-canon-zero-tim-branch/scripts/preflight_runtime.py \
  --repo . --require-clean

unset V1_GSM8K_XPROF_ALLOW_DIRTY
export V1_GSM8K_XPROF_EXPECT_HOSTNAME="$(hostname)"
bash canon-zero-tim/tasks/v1-gsm8k-onehost-xprof-pair/scripts/run_onehost_gsm8k_xprof_zero_hp.sh \
  'p60_2g_native_train_steps_zero_<unique-utc-date>'
```

With no evidence-root override, `<root>` below is
`/mnt/disks/tunix-data/gsm8k-onehost-xprof/v1_zero-hp_<label>`. The label and
root must be fresh; never reuse a failed or successful directory.

Inspect and return the complete immutable root, including failed roots:

```bash
cat <root>/train/xprof_size_census.txt
python3 -m json.tool <root>/train/xprof_size_receipt.json
python3 -m json.tool <root>/train/classification.json
sha256sum -c <root>/SHA256SUMS
```

The successful wrapper ends, in order, with `SHA_LEDGER_PASS` and the unique
Zero-HP `GREEN` marker. The target is not accepted unless the size census is
GREEN (`status=PASS|WARN`), classification is PASS, wrapper exit is zero, and
the independent manifest check succeeds. Preserve and return the entire root
on any RED, timeout, missing tail, size failure, or SHA failure.

## Mission

Repair the **observability hierarchy** of the existing one-host GSM8K Zero-HP
update capture. Do not redesign or optimize P59. The current capture is
complete and numerically green, but it is difficult to navigate because the
G6 path bypasses the stock trainer's `train` annotation and exposes 59,028
fine-grained module events without update/group parent intervals.

The intended result is a fresh Zero-HP `phase=update` capture with this host
schedule index:

```text
train(step_num=<global step>)
└── zero_tim_update
    ├── forward_groups
    │   └── forward_group ×16
    ├── loss_pullback ×1
    ├── reverse_groups
    │   └── reverse_group ×16
    │       ├── replay_forward ×1
    │       ├── model_backward ×1
    │       ├── report_adjoint ×1
    │       ├── fixed_dp_reduce ×1
    │       └── gradient_accumulate ×1 (`micro_step`, `is_last_accumulate`)
    └── optimizer_commit ×1 (`update_step`)
```

This hierarchy is an index over the existing TPU work. It must not alter that
work or conceal the raw decomposition.

`StepTraceAnnotation("train", step_num=...)` is compatible with Native only at
the annotation API boundary. Zero-HP intentionally retains one whole-update
parent; it does not reproduce Native's microstep cadence, cardinality, or
monolithic program shape. Device `Steps` rows are numeric derived rows in both
captures and are not semantic `train_{step_number}` labels.

## Read first

Read these files completely, in order, before editing:

1. `canon-zero-tim/AGENTS.md`
2. `canon-zero-tim/.claude/skills/manage-canon-zero-tim-branch/SKILL.md`
3. `canon-zero-tim/.claude/skills/manage-canon-flags/SKILL.md`
4. `/home/yuxuan/.codex/skills/run-phased-work/SKILL.md`
5. `/home/yuxuan/code_rl_repro/.claude/skills/read-xprof/SKILL.md`
6. `/home/yuxuan/.codex/skills/xprof-trace-analysis/SKILL.md`
7. this task's `state.md`, `plan.md`, and all `phases/p60-2*.md`
8. this task's existing `HANDOFF.md`, `RUNBOOK.md`, and `log.md`
9. `tasks/p48-onehost-perf/P51_XPROF_RUNBOOK.md`

The read-xprof skill is the primary operational authority. In particular:

- backward uses `P51_XPROF_PHASE=update`, not a whole-step window;
- the full XPlane and all eight TensorCore planes decide completeness;
- bounded trace JSON and UI screenshots do not decide completeness;
- host annotations are scheduling envelopes, not device busy time;
- profile-wall time is not an A/B speed measurement;
- performance decisions use matched, unprofiled `[PERF]` records.

## Immutable baseline

| Arm | XPlane | Bytes | SHA-256 | TPU:0 modules | Host `train` |
|---|---|---:|---|---:|---|
| Native | `/mnt/disks/tunix-data/gsm8k-onehost-xprof/v1_native_native_dev2_20260824/train/xprof/plugins/profile/2026_08_24_22_27_40/t1v-n-4a77ebd0-w-0.xplane.pb` | 159,449,612 | `a2367fc94d4fa3643b5895e9cef068383c8d464bfc8b945bc95e0ab14186e4a6` | 672 | present |
| Zero-HP | `/mnt/disks/tunix-data/gsm8k-onehost-xprof/v1_zero-hp_zero_dev3_20260824/train/xprof/plugins/profile/2026_08_24_22_10_24/t1v-n-4a77ebd0-w-0.xplane.pb` | 813,929,492 | `c247dbde98dab67510fa0e5e28d5c49fa79eb586a57d2763a86d7affc1a7e6f5` | 59,028 | absent |

Both arms completed 3/3 optimizer updates. Zero-HP passed 51/51 strict
alignment records and its full XPlane contains the required P59 layer, head,
norm, embed, and adjoint backward families on 8/8 TensorCore planes with decode
absent. Therefore this is not a missing-backward bug.

The historical pair remains `INCONCLUSIVE_INPUT_MISMATCH`: completion ids,
completion masks, and advantages differ. It can prove program shape but cannot
support a Native/Zero timing ratio.

## Source map and intended seams

Line numbers may move after rebase; locate the symbols with `rg`.

- `tunix/sft/peft_trainer.py`, `PeftTrainer.train`: Native already opens
  `jax.profiler.StepTraceAnnotation("train", ...)`.
- `tunix/rl/agentic/agentic_rl_learner.py`, `_run_p28_g6_update`: Zero-HP
  whole update and optimizer transaction. The caller around
  `_canon_xprof_update_entry()` and the official flat `PEFT_TRAIN` semantic
  span is the top-level annotation seam.
- `tunix/rl/canonical_qwen3_adapter.py`, `_xprof_jit`: existing parser and
  default-off module-label contract for `CANON_XPROF_LABELS`.
- `tunix/rl/canonical_qwen3_adapter.py`,
  `segmented_dp_grpo_value_and_grad`: forward groups, loss pullback, reverse
  groups, report adjoint, fixed reducer, and gradient sink.
- `tunix/rl/canonical_qwen3_adapter.py`, `_p32_reverse_group`: replay-forward
  followed by model backward.
- `tunix/rl/dp_training.py`, `FixedDPRankGradientReducer.finalize_staged`:
  fixed-order reduction implementation. Annotate its call site; do not edit its
  mathematics or rank order.

Prefer one small observation helper in the profiling layer that:

1. parses `CANON_XPROF_LABELS` exactly once;
2. returns `contextlib.nullcontext()` for absent/empty/`0`;
3. returns `jax.profiler.TraceAnnotation` or `StepTraceAnnotation` for `1`;
4. rejects every other value;
5. validates stable low-cardinality names and integer metadata.

Do not duplicate incompatible env parsing between learner and adapter.

## Prohibited changes

- No new JIT, `shard_map`, scan, fusion, collective, reducer, barrier,
  `optimization_barrier`, device readback, or synchronization.
- No new `block_until_ready()` or `device_get()`.
- No precision, tile, loss, sampling, optimizer, gradient, or fixed-order
  reduction change.
- No per-layer host span. Existing module names already identify layers.
- No renaming or filtering of the raw 59,028 module events.
- No custom nested semantic-Perfetto vocabulary. Keep the official flat
  `peft_train` span exactly as it is; P55's custom nested semantic spans were
  reverted because they made that view worse.
- Require a non-empty device `Steps` line on all 8/8 TPU planes, matching the
  stock Native capture. Do not use numeric `Steps` row events as semantic
  stage labels; the host hierarchy supplies those labels.
- If asynchronous host/device timing does not visually align, do not add a
  sync. Classify it and build a derived timestamp join in P60-2D.

## Implementation deliverables

1. Default-off host annotation helper and unit tests.
2. The bounded hierarchy in the learner/adapter call sites above.
3. A full-XPlane hierarchy census, separate from the existing all-plane module
   census. It requires the exact host `train(step_num=1)` API event, every
   hierarchy span on the same `/host:CPU` `python3` track, and non-empty device
   `Steps` lines on 8/8 TPU planes. Factor a pure interval validator for
   synthetic fixtures.
4. Positive and negative controls for missing parent, duplicate group, orphan
   child, wrong group count, and wrong optimizer count.
5. Zero-HP arm classifier integration that requires the new census only for a
   revised hierarchy carrier. Native's existing contract stays unchanged.
6. `FLAGS.md` wording update only if needed to state that
   `CANON_XPROF_LABELS=1` now enables host hierarchy annotations as well as JIT
   labels. Do not add a flag.
7. Updated task `state.md`, phase result, `log.md`, runbook, and handoff.

## Pre-TPU verification and stop point

Before asking to run TPU, complete and report:

1. `git diff --check`, Python/shell syntax, relevant unit tests.
2. flag registry audit.
3. hierarchy-census synthetic positive and all negative controls.
4. P59 DP2xTP2 and registered TP4/TP8 serial-versus-parallel exact-gradient
   regressions with labels off and on, including the existing one-ULP negative.
5. pinned exact-image validation of the JAX annotation API and the existing
   P59/V1 gates, if the local image is available without occupying TPU.
6. a diff audit showing zero new synchronization and zero semantic-Perfetto
   vocabulary changes.

Then stop. Do not commit, push, launch TPU, or render a cluster job. Return the
diff summary, exact gate outputs, risks, rollback, and the proposed fresh label.
The user decides whether to authorize the one-host canary.

## One-host canary after separate approval

Only after explicit approval:

```bash
export V1_GSM8K_XPROF_EXPECT_HOSTNAME="$(hostname)"
bash canon-zero-tim/tasks/v1-gsm8k-onehost-xprof-pair/scripts/run_onehost_gsm8k_xprof_zero_hp.sh \
  '<fresh-p60-readable-zero-label>'
```

For this uncommitted executor worktree, the exact proposed development-grade
command is:

```bash
export V1_GSM8K_XPROF_EXPECT_HOSTNAME="$(hostname)"
export V1_GSM8K_XPROF_ALLOW_DIRTY=1
bash canon-zero-tim/tasks/v1-gsm8k-onehost-xprof-pair/scripts/run_onehost_gsm8k_xprof_zero_hp.sh \
  p60_readable_zero_local_20260825
```

The approved development carrier was run once after the initial parent-order
failure was fixed, using label `p60_readable_zero_dev2_20260825`. It passed all
acceptance gates. Its immutable root is
`/mnt/disks/tunix-data/gsm8k-onehost-xprof/v1_zero-hp_p60_readable_zero_dev2_20260825`.
The dirty override makes this analysis-grade evidence, not a clean-SHA release
receipt. Do not copy the example above to launch another run without new user
approval.

The launcher must retain update phase, one captured update, host tracer 1,
Python tracer 0, and `CANON_XPROF_LABELS=1`. Do not rerun Native.

Acceptance requires all of the following:

- 3/3 optimizer updates;
- 51/51 strict alignment PASS and zero FAIL;
- one non-empty XPlane with recorded size and SHA-256;
- non-empty `Steps` rows on all 8/8 TPU device planes;
- P59 backward present and decode absent on exactly 8/8 TensorCore planes;
- on every TensorCore plane,
  `jit__precomputed_gradient_scaled_step` appears exactly 16 times and
  `jit__precomputed_gradient_commit` exactly once; this is the mechanical
  optimizer-tail/no-drop gate;
- exactly one `train` and one `zero_tim_update`;
- the `train` event has `step_num=1`, matching the Native API contract;
- every hierarchy span is on the same `/host:CPU` `python3` track;
- 16 forward groups and 16 reverse groups;
- one replay, model-backward, report-adjoint, fixed-reduce, and accumulator
  child per reverse group;
- accumulator `micro_step` values exactly 0..15 and exactly one
  `is_last_accumulate=1` at 15;
- one optimizer transaction with `update_step=1`;
- the official semantic Perfetto counts unchanged;
- raw log, full XPlane tree, semantic Perfetto, trace JSON, all censuses,
  classification, and SHA ledger preserved.
- exactly one terminal GREEN marker frozen in `driver.log` before
  `SHA256SUMS` is generated;
- wrapper exit 0 followed by `SHA_LEDGER_PASS`, plus independent
  `sha256sum -c SHA256SUMS` success. A driver GREEN alone is insufficient.

## Claim ceiling and rollback

The second canary passed the P60-2C whole-update contract, so its strongest
allowed claim remains:

`ONE-HOST XPROF READABILITY + NUMERICAL NEUTRALITY PASS`

It is dirty-tree analysis-grade evidence, not a signed clean-SHA receipt. It
is not a Native/Zero speed claim and not a 64-chip certification.

P60-2E was added afterward to expose truthful accumulator microsteps and the
separate optimizer update. Its first clean-SHA run passed the core target
gates, but the old runner produced its SHA ledger before appending the terminal
GREEN line. That immutable root remains packaging RED. P60-2F fixes
finalization order additively in historical clean source
`5549b5b6046f91406d1897b47618fca83c5fad7d`. Fresh root
`v1_zero-hp_p60_readable_zero_p60_2f_ledger_clean_20260825_r1` passes 3/3
updates, 51/51 alignment, the exact same-track hierarchy and metadata, 8/8
backward plus optimizer-tail planes with decode absent, classifier PASS with
no reasons, `SHA_LEDGER_PASS entries=9`, and an independent 9/9 manifest
check. That historical source therefore has CLEAN-SHA TARGET PASS. The
latest-tip integration `c87838d8a77ddca33800df024b3fef9edc503327` passes
host and pinned exact-image admission only; it was not target-rerun and must
not inherit the TARGET PASS label. Neither result promotes the earlier
packaging-RED root or creates a Native/Zero speed claim.

Rollback is the single annotation/census/classifier CL. With
`CANON_XPROF_LABELS` absent or `0`, runtime behavior must already be an exact
no-op, so rollback must not touch P59, reducer, or optimizer code.

## P60-2G current handoff

The immutable P60-2F JSON has exactly 1,000,448 events and mechanically fails
the new UI gate: `train=1`, `forward_group=16`, while loss, all reverse stages,
all accumulators, and optimizer are absent. Its full XPlane remains complete.
The first captured reverse group also contains first-use compiler work; update
2 is selected because update 0 changes sharding at commit and update 1 compiles
the stable post-commit identity.

Current implementation contract:

- `zero_tim_update(update_step=2)` owns forward, loss, train 32..47, and
  optimizer;
- every train owns its matching complete reverse transaction; train 47 stays
  open through optimizer;
- `TRACE_ONLY_XLA` is the low-density navigation carrier;
- full XPlane separately requires 8/8 backward families, scaled-step x16,
  commit x1, decode absent, non-empty Steps rows, and zero compiler events
  inside the captured update;
- streaming trace JSON separately requires all train/reverse spans and the
  optimizer tail on the same host track;
- the size census inventories all regular XProf files, permits a soft
  `WARN` only through 1,500,000,000 bytes, and rejects a stale or oversized
  receipt without deleting the run;
- 3/3 updates, 51/51 alignment, semantic Perfetto, classification, and the
  immutable SHA ledger remain mandatory; the ledger directly includes raw
  XProf and semantic Perfetto.

Current local receipt after the budget addendum: P60 13/13 and document set
15/15, P59 37/37, V1/P64 67/67, and the complete pinned exact-image ladder all
PASS. The pinned container reported zero TPU chips. A read-only replay of the
historical P60-2F artifact totals 802,001,091 XProf bytes and returns size
`status=PASS`. These are construction/tooling gates only; no fresh P60-2G v5p
profile exists.

No commit, push, or TPU launch is implied by this handoff. The current source
line in the quickstart deliberately has no implementation SHA. After local
commit approval, record that full SHA before a remote operator uses this
one-host command; push and launch remain separate approval boundaries:

```bash
export V1_GSM8K_XPROF_EXPECT_HOSTNAME="$(hostname)"
bash canon-zero-tim/tasks/v1-gsm8k-onehost-xprof-pair/scripts/run_onehost_gsm8k_xprof_zero_hp.sh \
  'p60_2g_native_train_steps_zero_<unique-date>'
```

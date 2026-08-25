# GSM8K one-host Native vs Zero-HP XProf runbook

> **P60-2 readability follow-up is active.** The P60-2C second Zero-HP canary
> proved the whole-update hierarchy at dirty-tree analysis grade. P60-2E adds
> accumulator microstep and optimizer-update metadata locally; that increment
> is `TARGET NOT RUN`. Before any new run, read
> [`HANDOFF_P60_2.md`](HANDOFF_P60_2.md) and all `phases/p60-2*.md`. Do not
> rerun Native. A fresh Zero-HP canary is allowed only after P60-2B's
> host/static/exact-image gates and explicit user approval. The revised census
> also requires the Native API-compatible host `train(step_num=1)` event, the
> complete hierarchy on one `/host:CPU` `python3` track, non-empty device
> `Steps` rows on 8/8 TPU planes, `micro_step=0..15`, exactly one last
> accumulator at 15, optimizer `update_step=1`, and, on every device plane,
> `jit__precomputed_gradient_scaled_step`×16 plus
> `jit__precomputed_gradient_commit`×1. API-compatible does not
> mean Native-compatible cadence, cardinality, or monolithic program shape:
> Zero-HP intentionally retains one whole-update `train` parent.

The primary analysis method is
`/home/yuxuan/code_rl_repro/.claude/skills/read-xprof/SKILL.md`: trainer work
uses `phase=update`; complete XPlane plus 8/8 TensorCore planes decides capture
completeness; `[PERF]` from comparable unprofiled steps decides speed. The UI,
host hierarchy, semantic Perfetto, and trace JSON are attribution/navigation
views, not interchangeable clocks.

P60-2B/P60-2E local implementation gates are green on branch
`local/p60-2e-microstep-latest-0825` at base
`cdd3987caa648e6112ee8fc184b2e3421de3a4b2`. The latest registry contains
372 flags because P63 is included. P60-2C passed on immutable dev2 evidence;
P60-2E remains `TARGET NOT RUN`. Do not interpret its local annotation API
output as proof that the new metadata appears in a TPU XPlane.

Run from the exact worktree on the direct four-chip v5p. Do not run the two
arms concurrently. During development only, the dirty-tree override is
explicit; acceptance evidence requires a clean committed tree.

```bash
export V1_GSM8K_XPROF_EXPECT_HOSTNAME="$(hostname)"
export V1_GSM8K_XPROF_ALLOW_DIRTY=1

bash canon-zero-tim/tasks/v1-gsm8k-onehost-xprof-pair/scripts/run_onehost_gsm8k_xprof_native.sh \
  '<fresh-native-label>'
bash canon-zero-tim/tasks/v1-gsm8k-onehost-xprof-pair/scripts/run_onehost_gsm8k_xprof_zero_hp.sh \
  '<fresh-zero-label>'
```

Each arm must end in:

```text
[V1.GSM8K.XPROF] GREEN arm=<arm> backward_xprof=1 root=<absolute-root>
```

The runners set `CANON_P60_DETERMINISTIC_AB=1` in both arms. This is a
diagnostic input-control flag: it pins engine seed 42, serial scheduling,
concurrency one and the registered 1024/256 tensor widths. It does not enable
canonical numerical kernels in Native, and it does not guarantee identical
sampled completions when the two inference programs produce different logits.
Do not remove it or hand-add a P32 workload to Native.

The device census is intentionally arm-aware:

- Native stock training is one XLA `jit__train_step` per trajectory group,
  so every TPU plane must contain exactly 16 and no decode module.
- Zero-HP must contain P59 parallel backward for layer, head, final norm,
  embedding and mapping adjoint on exactly eight planes, plus no decode
  module. Every plane must also contain
  `jit__precomputed_gradient_scaled_step` exactly 16 times and
  `jit__precomputed_gradient_commit` exactly once; either mismatch is a RED
  optimizer-tail/drop verdict.

Do not apply the old P55 `pullback`-only/19-track census to Native: it was
certified for a different trainer and topology and produces a false RED here.
The task-specific semantic census instead requires the exact one-update event
counts for each arm.

Then run the complete pair postflight. It resolves exactly one trace per arm,
runs the pair hash classifier, runs the installed `xprof-trace-analysis`
summary, and hashes the compact outputs and both arm census records:

```bash
bash canon-zero-tim/tasks/v1-gsm8k-onehost-xprof-pair/scripts/analyze_gsm8k_xprof_pair.sh \
  '<absolute-native-root>' \
  '<absolute-zero-root>' \
  '<fresh-absolute-pair-output-dir>'
```

Require `verdict=PASS`, `matched_profiled_work=true`, and both arm verdicts
`PASS`. `INCONCLUSIVE_INPUT_MISMATCH` means the two valid profiles consumed
different rollout tokens/advantages and therefore cannot support a causal
performance comparison; inspect `mismatched_profiled_work_arrays` and do not
average their timings. This does not invalidate either arm's standalone
backward-capture proof. The postflight returns 0 for a matched PASS and 3 for
a scientifically valid input mismatch; both write the complete output
directory. Any other nonzero return is a tooling/capture failure.

The arm classifier already requires one non-empty XPlane, one non-empty trace,
all TPU planes arm-specific-backward-present/decode-absent, and a valid
one-update semantic Perfetto. The summary is additional attribution, not a
substitute for those gates.

Artifact authority is ordered as follows:

1. the full XPlane plus the task-specific all-plane census decides whether the
   complete backward was captured;
2. the semantic Perfetto decides whether the requested update window opened
   and closed around the intended training transaction;
3. `trace.json.gz` is a convenient operation-attribution view only. Its trace
   buffer can omit modules even when the XPlane is complete, so never use a
   raw module count from this file as the backward-completeness gate.

Certified one-host development evidence from 2026-08-24 demonstrates this
distinction: Native has 16/16 `jit__train_step` modules on every one of eight
TensorCore planes in XPlane, while its compressed trace JSON exposes only 11
on the selected plane. The XPlane census is the authoritative result.

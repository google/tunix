# GSM8K one-host Native vs Zero-HP XProf runbook

> **P60-2G supersedes the historical whole-update navigation contract.**
> Historical clean source `5549b5b6` remains TARGET PASS for P60-2F's original
> full-XPlane contract, but its exported UI trace has only one 62.66-second
> `train(1)` and omits the reverse/optimizer tail. Under P60-2G it is
> `NUMERICAL/FULL-XPLANE PASS / NATIVE-LIKE UI FAIL / PERFORMANCE
> INCONCLUSIVE`. The current local change captures warm update 2 and emits the
> 16 real reverse/reduce/accumulate transactions as Native API train steps
> 32..47; train 47 owns the real optimizer commit and there is no synthetic
> train 48. A fresh Zero-HP run requires clean local and exact-image gates,
> then separate explicit launch approval. Do not rerun Native.

P60-2G fixes the signed artifact budget as well: every regular file under
`train/xprof` contributes its logical byte size, with a soft warning at
`1,200,000,000` bytes and a hard RED above `1,500,000,000` bytes. The runner
does not truncate an oversized artifact. It writes a machine-readable
`xprof_size_receipt.json`, verifies it against the current files in the arm
classifier, and directly hashes every raw XProf file plus semantic Perfetto in
the final root manifest.

The primary analysis method is
`/home/yuxuan/code_rl_repro/.claude/skills/read-xprof/SKILL.md`: trainer work
uses `phase=update`; complete XPlane plus 8/8 TensorCore planes decides capture
completeness; `[PERF]` from comparable unprofiled steps decides speed. The UI,
host hierarchy, semantic Perfetto, and trace JSON are attribution/navigation
views, not interchangeable clocks.

Historical P60-2F source
`5549b5b6046f91406d1897b47618fca83c5fad7d` passed the fresh clean-SHA target
on root
`/mnt/disks/tunix-data/gsm8k-onehost-xprof/v1_zero-hp_p60_readable_zero_p60_2f_ledger_clean_20260825_r1`.
It ended with `SHA_LEDGER_PASS entries=9` and Zero-HP GREEN; independent
`sha256sum -c SHA256SUMS` passed all entries. The prior P60-2E packaging-RED
root remains preserved and is not retroactively accepted. The latest-tip
integration `c87838d8a77ddca33800df024b3fef9edc503327` passed the complete
host gates and has pinned exact-image admission but was not target-rerun. Base
`a909fda1` includes M15/P64 runtime and evidence changes, so the complete
aggregate-plus-P60 pinned exact-image ladder was rerun on the final rebased
three-commit tree before publication rather than admitted by byte identity.
The registry contains 378 flags, including P64.

Run P60-2G from the exact worktree on the direct four-chip v5p. The target is
Zero-HP only; do not run Native. During development only, the dirty-tree
override is explicit; acceptance evidence requires a clean committed tree.

```bash
export V1_GSM8K_XPROF_EXPECT_HOSTNAME="$(hostname)"
unset V1_GSM8K_XPROF_ALLOW_DIRTY

bash canon-zero-tim/tasks/v1-gsm8k-onehost-xprof-pair/scripts/run_onehost_gsm8k_xprof_zero_hp.sh \
  'p60_2g_native_train_steps_zero_<unique-date>'
```

Each successful arm must end on stdout in this order:

```text
[V1.GSM8K.XPROF] SHA_LEDGER_PASS entries=<count> root=<absolute-root>
[V1.GSM8K.XPROF] GREEN arm=<arm> backward_xprof=1 root=<absolute-root>
```

The wrapper must return 0, and `sha256sum -c <root>/SHA256SUMS` must
independently return 0. `driver.log` contains exactly one terminal GREEN or
RED marker and is never changed after manifest construction. A standalone
GREEN marker without `SHA_LEDGER_PASS` is not acceptance evidence.

The remote operator must also require:

```text
V1_GSM8K_XPROF_SIZE_CENSUS_GREEN status=<PASS|WARN> ... hard_max_bytes=1500000000
```

`PASS` means at most 1.2 GB; `WARN` is admissible only while the exact total is
at most 1.5 GB. Inspect `train/xprof_size_receipt.json` for per-file sizes. Any
size census RED, stale receipt, symlink, missing raw profile, or total above
the hard maximum forces arm classification FAIL while preserving the root.

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

The Zero-HP arm classifier requires one non-empty XPlane, one non-empty trace,
all TPU planes backward-present/decode-absent, a valid one-update semantic
Perfetto, the full-XPlane hierarchy/warm-compile gate, and the independent UI
trace-JSON gate. The summary is additional attribution, not a substitute for
those gates.

Artifact authority is ordered as follows:

1. the full XPlane plus the task-specific all-plane census decides whether the
   complete backward was captured;
2. the semantic Perfetto decides whether the requested update window opened
   and closed around the intended training transaction;
3. `trace.json.gz` decides the P60-2G UI-navigation claim: train 32..47, all
   reverse transactions, and optimizer containment must be visible. It still
   cannot replace the full-XPlane module/backward completeness gate.
4. `xprof_size_receipt.json` decides the transfer-size contract and must match
   every current regular file under `train/xprof`; the manifest directly
   covers those raw files rather than relying only on hashes copied into the
   classification record.

Certified one-host development evidence from 2026-08-24 demonstrates this
distinction: Native has 16/16 `jit__train_step` modules on every one of eight
TensorCore planes in XPlane, while its compressed trace JSON exposes only 11
on the selected plane. The XPlane census is the authoritative result.

# P60-2A — Freeze the XProf readability baseline

- Status: passed

## Finding

- Confirmed: both historical arm captures are numerically and structurally
  valid for their own programs. Native and Zero-HP each completed 3/3 updates;
  Zero-HP passed 51/51 strict alignment records; all eight Zero-HP TensorCore
  planes contain the five required P59 backward families and no decode family.
- Confirmed: Native TPU:0 contains 672 `XLA Modules` events. Zero-HP TPU:0
  contains 59,028. The difference is a real decomposition/dispatch difference,
  not evidence that XProf failed to capture backward.
- Confirmed: Native `/host:CPU` has a `train` event on the Python thread because
  `PeftTrainer.train` enters `StepTraceAnnotation("train")`. The G6 Zero-HP
  path bypasses that loop and has no equivalent XProf parent event.
- Confirmed: Zero-HP's existing `peft_train` is written to a separate semantic
  Perfetto artifact. It cannot parent or group the XLA modules in the XProf
  Trace Viewer.
- Confirmed: `CANON_XPROF_LABELS=1` gives individual JITs stable names such as
  `zt_tr_dp_parallel_bwd_layer_27`; it does not annotate the containing update,
  group, report-adjoint, reducer, or optimizer host schedule.
- Hypothesis: bounded JAX host annotations will make the decomposition
  navigable without changing any JAX computation or synchronization.

## Evidence

| Arm | XPlane | Size | SHA-256 | TPU:0 module events | Host `train` |
|---|---|---:|---|---:|---|
| Native `native_dev2_20260824` | `.../v1_native_native_dev2_20260824/train/xprof/plugins/profile/2026_08_24_22_27_40/t1v-n-4a77ebd0-w-0.xplane.pb` | 159,449,612 | `a2367fc94d4fa3643b5895e9cef068383c8d464bfc8b945bc95e0ab14186e4a6` | 672 | present |
| Zero-HP `zero_dev3_20260824` | `.../v1_zero-hp_zero_dev3_20260824/train/xprof/plugins/profile/2026_08_24_22_10_24/t1v-n-4a77ebd0-w-0.xplane.pb` | 813,929,492 | `c247dbde98dab67510fa0e5e28d5c49fa79eb586a57d2763a86d7affc1a7e6f5` | 59,028 | absent |

The full path prefix for both artifacts is
`/mnt/disks/tunix-data/gsm8k-onehost-xprof/`. Existing classification and
census files remain the authority for capture completeness.

## Execution

1. Applied the repository-external
   `/home/yuxuan/code_rl_repro/.claude/skills/read-xprof/SKILL.md` workflow and
   loaded the full XPlanes with `xprof.profile_data.ProfileData`; the bounded
   trace JSON was not used as a completeness oracle.
2. Counted plane names, host thread events, `Steps`, `XLA Modules`, `XLA Ops`,
   and `XLA TraceMe` lines without rewriting either artifact.
3. Correlated the absence/presence of `train` with the current code paths.

## Exit gate

- Command: direct read-only ProfileData inspection of the two immutable paths
  above; no TPU launch.
- Pass: counts reproduce as Native=672, Zero-HP=59,028 on TPU:0; Native host
  `train` present; Zero-HP host `train` absent; existing arm classifiers remain
  PASS.
- Fail: if a future reinspection disagrees, preserve both artifacts and reopen
  this phase before changing instrumentation.

This phase deliberately makes no Native/Zero-HP speed claim. The historical
pair has different completion, mask, and advantage arrays, and the read-xprof
rules reserve performance decisions for matched, unprofiled `[PERF]` steps.

## Result

Passed on 2026-08-25. The readability defect is localized to missing parent
annotations plus genuine program fragmentation. No source was changed.

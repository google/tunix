# P58.6 — One-host Native versus optimized Zero-HP XProf pair

Status: implementation complete; host and pinned-image gates PASS; direct TPU
pair NOT RUN because no remote-host execution was authorized.

## Goal

Produce two matched Qwen3-4B-Instruct-2507 operation-attribution packages on
one explicitly named direct-attached four-chip host:

- `native`: stock numerical runtime plus the independently signed P58 stock-B
  observer;
- `zero-hp`: strict Zero-TIM serving optimizations supported by DP1 x TP4.

The carrier is intentionally update-shaped but performs no optimizer commit.
It first warms one backward, then captures an identical in-memory repeat.
Fixed diagnostic advantages `[-1, 1]` prevent the historical one-row DeepSWE
all-zero reward from eliminating backward. This is a backward-shape carrier,
not a training-quality sample.

## Frozen carrier

| Field | Value |
|---|---|
| Topology | exactly four TPU devices, DP1 x TP4 colocated |
| Work | one prompt x two generations, max concurrency 1 |
| Limits | prompt 3,584; response 512; turns 2 |
| Seed | 42 with serial scheduling |
| Optimizer commits | 0 |
| Update calls | one warmup plus one identical profiled repeat |
| APC / fixed head / P59 | off / off / off |
| Capture | update XPlane + trace.json.gz + semantic Perfetto |

P59 and the registered 4B fixed head are deliberately absent because DP1 x
TP4 cannot represent their DP8 x TP8 target geometry.

## Hard gates

1. A fresh absolute artifact root and a first-use label are mandatory.
2. Source commit, source-diff digest, local branch, hostname, model snapshot,
   R2E SHA, Docker task-image ID, and runner digest are signed in the manifest.
3. Both repeats hash the exact backward inputs and preserve model, reference,
   optimizer, accumulator, and train-step fingerprints.
4. Native boundaries must be shape-valid and finite. Zero-HP must have exact
   boundaries and no real `verdict=FAIL`.
5. Both arms require nonzero finite repeat-exact gradients, complete device/UI
   and semantic captures, and a sealed `SHA256SUMS` package.
6. The pair is causal only when source/provenance/work hashes match. Token or
   work drift becomes `INCONCLUSIVE_INPUT_MISMATCH`, never a speed claim.

XProf is used for operation attribution. Timing claims must come from `[PERF]`
outside the profiled repeat; overlapping op durations are not wall time.

## Result log

- Implemented by
  `scripts/run_onehost_deepswe_xprof_common.sh:1`, with thin arm wrappers at
  `scripts/run_onehost_deepswe_xprof_native.sh:1` and
  `scripts/run_onehost_deepswe_xprof_zero_hp.sh:1`.
- Arm and pair decisions are implemented at
  `scripts/classify_onehost_xprof.py:1` and
  `scripts/classify_onehost_xprof_pair.py:1`.
- Verified by host tests `test_onehost_xprof.py` 5/5 and
  `test_onehost_xprof_pair.py` 2/2, flag audit 366/366, shell/Python syntax,
  and pinned image
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
  terminal `P58_EXACT_IMAGE_CPU_PASS ... onehost_xprof=1 ... regressions=1`.
- Direct one-host XPlane/Perfetto package is **not verified** because this turn
  had no authorization to execute on the other direct-attached TPU host and
  the current container exposes no TPU devices.
- DP8 x TP8, Pathways, P59, 4B/TP8 fixed head, and APC are **not verified** by
  this carrier because its signed topology is DP1 x TP4 and those switches are
  intentionally off.

## Rollback

The selector, provenance fields, no-commit carrier, wrappers, and classifiers
are additive and default off. Revert the isolated P58.6 concern to restore the
pre-carrier P58 runtime; production profiles remain unreachable from it.

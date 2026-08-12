# P38.2g7: DP-divisible diagnostic consumer batch

- Status: local implementation complete; target P38s8 not run.

## Evidence correction

P38s7 proved the standard serving hook is reachable, the installed overlay has
the expected identity, and the DP16xTP4 adapter registers. It then passed 40
trajectories to an adapter which requires a DP16-divisible leading dimension.

The rendered P38s7 command did not configure five global prompts. It configured
32 global prompts, a 32-prompt consumer mini-batch, and eight generations. The
agentic queue consumer accepts a final partial list before its sentinel; in
P38s7 that list contained five completed prompt groups, producing 40
trajectories. The committed evidence does not contain the raw terminal log, so
the reason production ended after five groups is not claimed.

## Repair

Only P38 serving capture changes its consumer mini-batch:

```text
global dataset batch:       32 prompts (unchanged)
P38 consumer mini-batch:     4 prompts
generations per prompt:      8
P38 diagnostic trajectories: 32
engine data size:            16
trajectories per DP rank:      2
```

The diagnostic stops after the durable pre-backward record. It therefore does
not need to wait for all 32 prompt groups before producing the evidence unit.
FrozenLake full training, P45 DP8xTP8, evaluation, backward mathematics,
precision, optimizer placement, and sampling are unchanged.

The renderer parses the final shell command and fails closed unless it contains
the exact 32/4/8/DP16 geometry. A five-prompt negative control is rejected. The
recipe independently admits `mini_batch_size=4` only while
`CANON_P38_PRECHECK_ONLY=1`, derives a 32-trajectory mini-batch, and prints a
runtime contract marker before model execution.

## Local evidence

- P38 renderer unit gate: 6/6 PASS, including the five-prompt negative
  control.
- P38 outer postflight shell gate: PASS.
- Pinned image
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`:
  Qwen3-1.7B and Qwen3-8B overlays each passed 20/20 tests with all 29
  manifest entries exact.
- The adjacent P45 pinned-image gate passed 83 workload tests, 31 alignment
  tests, the Qwen3-8B TP8 seven-site contract, overlay import, and canonical
  forward/VJP probe. This proves that the default 32-prompt mini-batch used by
  full training and P45 was not changed.
- `git diff --check` and Python/shell syntax checks: PASS.

## Target exit gate

Run P38s8 stock only using the superseding top section of `HANDOFF.md`. Require
the runtime `DIAGNOSTIC_BATCH_CONTRACT` marker, four standard-path pre/post
records, the run-specific mismatch capsule, classifier PASS, serving archive,
at least one exact request/token-history join, and outer postflight. Backward
and optimizer commits remain zero. Do not rerun unified KV.

## Claim ceiling

This repair removes the P38 diagnostic batching failure. It does not identify
or repair the decode-versus-prefill numerical carrier and does not admit
FrozenLake training.

# Plan

## Outcome

Test one controlled variable: delivery of `--xla_allow_excess_precision=false` to the remote Pathways proxy compiler. Do not load a checkpoint, run training, change declared dtypes, or reinterpret prior flag-off evidence.

## Phases

| Phase | Deliverable | Exit gate | Status |
|---|---|---|---|
| P36.1 | Experimental gate-only renderer and fail-closed proxy-argv contract | `bash canon-zero-tim/tests/p36_proxy_xla/run_cpu.sh` | passed |
| P36.2 | One source-pinned 64-chip flag-on way-count result | Complete expected way-count rows plus proxy manifest/log evidence | active |
| P36.3 | Conditional P35 envelope-only recheck | Run only if P36.2 materially changes the replicated arm | pending |

## Decisions

- Confirmed: Client-side `XLA_FLAGS` does not attest the Pathways proxy compiler configuration.
- Confirmed: The current 64-chip and 256-chip proxy args omit the excess-precision flag.
- Hypothesis: The omitted proxy flag is load-bearing for the replicated cross-program drift.
- Decision: Use a dedicated gate-only renderer before changing shared workload defaults.
- Decision: Preserve old Pathways results as valid flag-off baselines; do not call them canonical flag-on results.

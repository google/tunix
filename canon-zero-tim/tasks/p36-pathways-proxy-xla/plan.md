# Plan

## Outcome

Test one controlled variable: delivery of `--xla_allow_excess_precision=false` to the remote Pathways proxy compiler. Do not load a checkpoint, run training, change declared dtypes, or reinterpret prior flag-off evidence.

## Phases

| Phase | Deliverable | Exit gate | Status |
|---|---|---|---|
| P36.1 | Initial proxy-argv delivery contract | Local renderer gate plus target proxy startup | failed on target; superseded |
| P36.1a | Direct-attached one-host OFF/ON sensitivity control | Eight paired depths with fixed image/probe | passed |
| P36.1b | Proxy-container `XLA_FLAGS` delivery contract | `bash canon-zero-tim/tests/p36_proxy_xla/run_cpu.sh` | passed locally; target pending |
| P36.2 | One source-pinned 64-chip flag-on way-count result | Complete expected way-count rows plus proxy manifest/log evidence | passed (`envon1`, Section 34: replicated 0/262144 across widths at depth 8; f4-fixed 0 at widths 4/8) |
| P36.3 | Conditional P35 envelope-only recheck | Run only if P36.2 materially changes the replicated arm | active (condition met: replicated ~34% -> 0; P35 renderer now inherits proxy env delivery from the shared P33 path) |

## Decisions

- Confirmed: Client-side `XLA_FLAGS` does not attest the Pathways proxy compiler configuration.
- Confirmed: The pinned proxy rejects `--xla_allow_excess_precision=false` as a raw top-level
  command-line argument, before any numerical probe runs.
- Decision: Deliver the flag through the proxy container's `XLA_FLAGS` environment and require
  that no raw excess-precision argument is present.
- Hypothesis: The omitted proxy flag is load-bearing for the replicated cross-program drift.
- Confirmed: In the direct-attached generic THIRDPROG probe, the flag removes 88% to 94% of differing bytes but leaves a nonzero residual.
- Decision: Use a dedicated gate-only renderer before changing shared workload defaults.
- Decision: Preserve old Pathways results as valid flag-off baselines; do not call them canonical flag-on results.
- Boundary: A rendered environment contract and successful proxy startup do not prove compiler
  consumption; only the registered target way-count result can do that.

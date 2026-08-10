# P36.1 — Pathways proxy flag delivery

- Status: failed on target; superseded by P36.1b

## Finding

- Confirmed: The client profile exports `--xla_allow_excess_precision=false`.
- Confirmed: The remote Pathways proxy manifest does not receive that flag.
- Hypothesis: The missing server-side flag causes all or part of the replicated `jit(f)` versus `jit(value_and_grad(f))` drift.

## Execution

1. Render a strict attempt-zero `gate-only` JobSet from the reviewed 64-chip base.
2. Add exactly one `--xla_allow_excess_precision=false` argument to `pathways-proxy` only.
3. Isolate Pathways scratch and disable reuse of the shared client GCS compilation cache.
4. Reject missing, duplicate and `true` flag controls.
5. Run local renderer gates before publishing or launching the manifest.

## Exit gate

- Command: `bash canon-zero-tim/tests/p36_proxy_xla/run_cpu.sh`
- Pass: Renderer tests pass, all three flag negative controls reject, and the generated manifest contains one proxy flag with `CANON_MODE=gate-only`.
- Fail: Keep P36.2 blocked and fix only the delivery contract; do not launch a target run.

## Result

`bash canon-zero-tim/tests/p36_proxy_xla/run_cpu.sh` passed 6/6. The generated manifest contained
exactly one proxy flag, used `gate-only`, isolated Pathways scratch and disabled the shared client
GCS compilation cache. Missing, duplicate, `true` and wrong-container flag controls were rejected.
The adjacent P35 renderer gate passed 7/7. The P33 renderer gate passed 6/6 in the pinned
frozenlake image; the host Python environment lacks `metrax`, so the host-only import attempt was
not used as evidence.

That result proved only the initially specified manifest contract. Target Attempt `flagon1`
subsequently proved that contract wrong: the pinned proxy exited with
`Unknown command line flag 'xla_allow_excess_precision'`. No numerical row was produced. The raw
argument path is therefore failed and preserved here as superseded evidence, not reported as a
numerical failure.

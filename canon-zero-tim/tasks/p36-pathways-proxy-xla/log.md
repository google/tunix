# Log

## 2026-08-10 UTC — P36.1: task binding

- Type: decision
- Fact: The checked-in Pathways proxy args omit `--xla_allow_excess_precision=false`; the existing environment guard checks only the JAX client environment.
- Hypothesis: Delivering the flag to the proxy will materially reduce the replicated way-count drift.
- Action: Bound a dedicated P36 gate-only experiment rather than changing shared production manifests.
- Command: omitted
- Result: Implementation active; no target numerical result exists.
- Files/artifacts: `state.md`, `plan.md`, `cluster/P36_PROXY_XLA_HANDOFF.md`
- Rollback: Remove the isolated P36 renderer, tests and task records; no shared runtime path is changed.
- Next: Run the local P36 gate.

## 2026-08-10 UTC — P36.1: proxy delivery contract passed

- Type: code change
- Fact: The dedicated renderer adds exactly one false excess-precision flag to the pinned Pathways proxy and no other Pathways container.
- Action: Added a gate-only renderer, six unit tests, four proxy-flag negative controls, isolated scratch/cache settings and target handoff instructions.
- Command: `bash canon-zero-tim/tests/p36_proxy_xla/run_cpu.sh`
- Result: PASS, 6/6. Adjacent P35 renderer PASS, 7/7. P33 renderer PASS in the pinned image, 6/6.
- Files/artifacts: `cluster/render_p36_proxy_xla_jobset.py`, `tests/p36_proxy_xla/`, `cluster/P36_PROXY_XLA_HANDOFF.md`
- Rollback: Remove the isolated P36 files; no shared P33, P34, P35, model, optimizer or credential path changed.
- Next: Publish the source and run one source-pinned Attempt-0 64-chip gate-only JobSet.

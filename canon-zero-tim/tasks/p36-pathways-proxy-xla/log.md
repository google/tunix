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

## 2026-08-10 UTC — P36.1a: one-host OFF/ON sensitivity control

- Type: experiment
- Fact: The authorized one-host VM was READY, HEALTHY and idle before the run. Both arms used four TPU devices, image ID `418dc632edd8` and probe SHA `faf65c53223c8ccf1b7d5545084aefe1eabb0918d88ea43127e61ecc577b602f`.
- Action: Ran the same eight-depth THIRDPROG probe serially with the excess-precision flag absent and present.
- Command: `bash canon-zero-tim/tests/p36_proxy_xla/run_onehost_pair.sh`
- Result: PASS as a sensitivity control. OFF differed at 70,556 to 116,607 of 262,144 bytes; ON differed at 8,438 to 7,234 bytes, an 88.04% to 93.80% reduction. ON was not bitwise exact.
- Files/artifacts: `phases/p36-1a-onehost-sensitivity.md`, `artifacts/p36_onehost_excess_off.raw.log`, `artifacts/p36_onehost_excess_on.raw.log`, `artifacts/p36_onehost_excess_pair.driver.log`
- Rollback: Diagnostic containers exited with `--rm`; no model, checkpoint, training or canonical VM file was modified. The copied raw artifacts can be omitted from a later CL without affecting runtime code.
- Next: Run the proxy-side P36.2 gate; do not substitute this direct-attached result for Pathways evidence.

## 2026-08-10 UTC — P36.2 `flagon1`: raw-argument delivery failed

- Type: target evidence
- Fact: The pinned proxy rejected the raw argument before any numerical probe ran.
- Command: `sed -n '1,40p' canon-zero-tim/debug_logs/p36_flagon1/pathways_proxy.raw.log`
- Result: INCONCLUSIVE infrastructure/delivery failure:
  `Unknown command line flag 'xla_allow_excess_precision'`.
- Files/artifacts: `debug_logs/p36_flagon1/pathways_proxy.raw.log`
- Rollback: The failed JobSet was an isolated attempt-zero gate; shared P33/P34 manifests and
  training defaults were unchanged.
- Next: Replace raw proxy argv delivery with proxy-container `XLA_FLAGS` and repeat under a new
  source pin and run ID.

## 2026-08-10 UTC — P36.1b: proxy-environment contract passed locally

- Type: code change
- Fact: The corrected renderer emits no raw excess-precision proxy argument and exactly one
  proxy `XLA_FLAGS=--xla_allow_excess_precision=false` entry.
- Action: Hardened the renderer and negative controls against missing, duplicate, `true`, raw and
  wrong-container delivery.
- Command: `bash canon-zero-tim/tests/p36_proxy_xla/run_cpu.sh`
- Result: PASS, 7/7. P35 renderer regression PASS, 7/7. Host P33 tests were not evidence because
  optional `datasets` and `metrax` dependencies are absent.
- Files/artifacts: `cluster/render_p36_proxy_xla_jobset.py`,
  `tests/p36_proxy_xla/test_render_p36_proxy_xla_jobset.py`,
  `phases/p36-1b-proxy-env-delivery.md`
- Rollback: Revert only the isolated P36 renderer/tests/docs; no shared workload, model, loss,
  optimizer, credential or precision dtype path changed.
- Next: Publish after review, then run `envon1` on the authorized 64-chip cluster.

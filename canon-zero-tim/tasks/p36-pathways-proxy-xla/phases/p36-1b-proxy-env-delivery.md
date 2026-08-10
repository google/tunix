# P36.1b — Pathways proxy environment delivery

- Status: passed locally; target pending

## Finding

- Confirmed: The pinned Pathways proxy rejects the excess-precision setting as a raw top-level
  command-line argument.
- Confirmed: Google Cloud's Pathways documentation requires XLA configuration in the proxy
  container and documents `XLA_FLAGS` as a proxy custom environment mechanism.
- Boundary: Local YAML validation cannot prove that the remote compiler consumed the setting.

## Execution

1. Remove the raw excess-precision argument from proxy `args`.
2. Add exactly one literal proxy environment entry:
   `XLA_FLAGS=--xla_allow_excess_precision=false`.
3. Reject missing, duplicate and `true` values, any raw occurrence, and placement on the resource
   manager or worker.
4. Preserve the pinned proxy image, attempt-zero policy, isolated scratch, gate-only mode and
   source pin.

## Exit gate

- Command: `bash canon-zero-tim/tests/p36_proxy_xla/run_cpu.sh`
- Pass: Seven tests pass; rendered proxy args contain no raw occurrence and proxy env contains
  exactly one false setting.
- Target boundary: Proxy startup plus a complete way-count table is still required before this
  delivery path has a numerical verdict.

## Result

Local gate PASS, 7/7. The P35 renderer regression also passed 7/7. The host-only P33 suite was
not used as evidence because this host lacks `datasets` and `metrax`; its previously reviewed
pinned-image result remains unchanged.

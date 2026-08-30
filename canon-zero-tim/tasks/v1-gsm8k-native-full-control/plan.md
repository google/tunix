# GSM8K Native/mismatch full-control plan

| Phase | Status | Exit gate |
|---|---|---|
| P0 — bind and audit | `DONE` | P56 stock Native and V1-HP Zero identities, command, seed, and W&B routing are source-proven; warning-only P33 draft is rejected |
| P1 — render-only carrier | `DONE` | Renderer uses the stock profile, strips raw Zero selectors and proxy precision pin, preserves the P33 command/restart shell, and never launches |
| P2 — offline validation | `DONE` | Real `00_env.sh`, mixed-arm negatives, stock-file hashes, driver import, adjacent suites, flag audit, and exact-image gate pass |
| P3 — handoff | `DONE` | Native and Phase4 handoffs describe the stock path, exact receipts, target gates, and unrun work without stale warning-only claims |
| P4 — target admission recovery | `IMPLEMENTED / TARGET NOT RUN` | Real Splash-kernel negative reproduces Attempt 03; Explicit-only reshard and Auto-mesh negative pass in the pinned image; a fresh DP16xTP4 Native target must cross the first learner forward and optimizer commit |

## Rejection rules

- Reject any change to the original P33 scientific command, seed, geometry,
  update horizon, or optimizer residency. Native must keep the untreated stock
  lm head; Zero retains its fixed lm head.
- Reject any Native manifest containing a Zero-only selector, including
  checked-VMA, first-update gate, P70 receipt scheduling, P71 scan, P67
  scoping, overflow-safe clip, or the collective reducer.
- Reject `CANON_P32_WORKLOAD`, any alignment observer/report, a canonical
  engine overlay, the canonical proxy excess-precision pin, or a changed stock
  engine file in Native.
- Reject project/group drift between Native and Zero; use distinct run names
  rather than distinct projects.
- Reject a wrapper that launches, accepts a dirty/wrong source, reuses an
  output directory, or emits more than one manifest.
- Offline passes establish construction only; do not claim target parity,
  performance, or convergence without a real approved full run.

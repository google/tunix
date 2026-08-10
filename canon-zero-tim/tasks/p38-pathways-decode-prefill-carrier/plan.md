# Plan

## Outcome

Preserve the bitwise zero-TIM contract. First make every red A-B observation durable and
localizable, then reproduce it without backward, run a matched proxy-flag causal pair, and use a
strict no-commit backward only to measure the remaining T-current boundary. Do not introduce a
tolerance, recompute old logprobs as a release fix, change precision, or commit optimizer state
while A-B is red.

## Phases

| Phase | Deliverable | Exit gate | Status |
|---|---|---|---|
| P38.1 | Durable mismatch coordinates, exact bits, bounded stdout JSON, and corrected units | P33 CPU suite plus injected one-bit, invalid-shape, truncation, and failed-run artifact controls | complete |
| P38.2 | One flag-on production-boundary reproduction | A-B coordinates and bit patterns survive in raw logs; B-C remains a hard gate | pending |
| P38.3 | Same-source proxy flag ON/OFF pair | Fresh proxies, isolated caches, matching token hashes, and pre-registered A-B verdicts | pending |
| P38.4 | One diagnostic backward with optimizer forcibly skipped | B-C and T-old/T-current stay hard; model and optimizer arrays do not change | pending |
| P38.5 | Carrier-specific numerical repair | A-B returns to zero under the source-pinned flag-on regime | pending |
| P38.6 | GSM8K and FrozenLake full campaigns | All step-0 boundaries are zero and the unchanged hard gates automatically admit training | pending |

## Decisions

- Confirmed: r35 is not a low-amplitude-only observation. FrozenLake reported a sparse maximum
  logprob difference of `0.10390`, so byte density cannot justify continuing optimization.
- Confirmed: decode logprob rows were padded from 16 to canonical local M256. The remaining
  carrier is not simply an M1-versus-M256 logprob-tail comparison.
- Hypothesis: The proxy precision flag changes a serving program boundary, but r19 and r35 are
  not a single-variable pair and do not establish causality.
- Decision: A-B may be report-only only in a dedicated no-commit diagnostic mode; committing
  training remains bitwise fail-closed.
- Decision: The final full campaigns may be single launches. Their existing step-0 hard gates
  determine whether training continues, so separate one-update and three-update cluster jobs are
  not required after the carrier is repaired.

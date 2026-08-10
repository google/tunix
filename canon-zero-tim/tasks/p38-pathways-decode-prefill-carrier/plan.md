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
| P38.1b | One-host production-tail construction gate | One real-Qwen DP1xTP4 precheck-only record plus a same-input tail control; no Pathways claim | complete |
| P38.2a | Model-free tail aval and sharding discriminator | The actual sampling transform and canonical scorer execute at the registered compact/local/global shapes, emit complete shape/sharding evidence, and detect a one-bit negative control | active |
| P38.2b | Flag-on GSM8K production-boundary reproduction | A-B coordinates, exact bits, raw/processed-target decomposition, and logical sequence metadata survive in raw logs; B-C remains a hard gate | pending |
| P38.2c | Flag-on FrozenLake production-boundary reproduction | The same fields plus turn, chunk, and logical-KV coordinates survive for the multi-turn workload; B-C remains a hard gate | pending |
| P38.3 | Same-source proxy flag ON/OFF pair | Fresh proxies, isolated caches, matching token hashes, and pre-registered A-B verdicts | pending |
| P38.4 | One diagnostic backward with optimizer forcibly skipped | B-C and T-old/T-current stay hard; model and optimizer arrays do not change | pending |
| P38.5 | Carrier-specific numerical repair | A-B returns to zero under the source-pinned flag-on regime | pending |
| P38.6 | GSM8K and FrozenLake full campaigns | All step-0 boundaries are zero and the unchanged hard gates automatically admit training | pending |
| P38.2d | User-approved bounded GSM8K full campaign plus strict FrozenLake backward-no-commit | Renderer, preflight, runtime gate, classifier, and negative controls enforce the workload-specific policy | active |

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
- Confirmed: r35 printed `runner_sampling_adapter_same_object=True`; a new
  shared Python function object would repeat an existing mechanism. P38.1b
  instead tests the production boundary and the tail inputs/compiled envelope.
- Decision: a green one-host result is a construction gate only. P38.2 remains
  the first admissible target reproduction.
- Correction: GSM8K and FrozenLake no longer share one pre-registered carrier.
  r35 measured `logp_diff_max<5e-6` for GSM8K but `0.10390` for FrozenLake.
  A tail-aval result may explain the former without explaining the latter.
- Correction: do not require one ULP or uniformly distributed mismatches. Those
  are observations to measure, not prerequisites to assume.
- Decision: do not implement F1b until P38.2a reports the actual global avals,
  shard layouts, processed-target values, target logprobs, and implied
  normalizers. A shared Python callable already exists; only equal compilation
  signatures can support a one-executable claim.
- Amendment (2026-08-10): the user explicitly authorized one committed GSM8K
  full campaign with bounded A/B drift reported rather than blocked. This does
  not replace P38.5 or P38.6 and cannot support a zero-TIM completion claim.
  FrozenLake remains fail-closed and is limited to backward-no-commit.

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
| P38.2a | Model-free tail aval and sharding discriminator | The actual sampling transform and canonical scorer execute at the registered compact/local/global shapes, emit complete shape/sharding evidence, and detect a one-bit negative control | deferred unless capsule replay does not reproduce |
| P38.2b | Flag-on GSM8K production-boundary reproduction | A-B coordinates, exact bits, raw/processed-target decomposition, and logical sequence metadata survive in raw logs; B-C remains a hard gate | pending |
| P38.2c | Flag-on FrozenLake production-boundary reproduction | The same fields plus turn, chunk, and logical-KV coordinates survive for the multi-turn workload; B-C remains a hard gate | pending |
| P38.3 | Same-source proxy flag ON/OFF pair | Fresh proxies, isolated caches, matching token hashes, and pre-registered A-B verdicts | pending |
| P38.4 | One diagnostic backward with optimizer forcibly skipped | B-C and T-old/T-current stay hard; model and optimizer arrays do not change | pending |
| P38.5 | Carrier-specific numerical repair | A-B returns to zero under the source-pinned flag-on regime | pending |
| P38.6 | GSM8K and FrozenLake full campaigns | All step-0 boundaries are zero and the unchanged hard gates automatically admit training | pending |
| P38.2d | User-approved bounded GSM8K full campaign plus strict FrozenLake capsule capture | Renderer, preflight, runtime gate, classifier, and negative controls enforce the workload-specific policy | ready; target not run |
| P38.2e | Schedule-aware GSM8K optimizer transaction | LR-zero and positive-LR commit controls plus target update-0 evidence | ready for independent parallel target run |
| P38.2f | FrozenLake KV-threshold mismatch capsule | Attempt 0 reproduces the hard pre-backward red, and two bounded rows survive pod deletion and pass transport/array SHA checks | complete; rows 191/199 verified at `036e845a` |
| P38.2g | FrozenLake single-row causal replay | Stock multi-turn replay reproduces the red before single-turn, MIXED-only KV-unified, and all-distribution KV-unified counterfactuals are interpreted | one-host target row 191 complete: local serving envelope not reproduced; R2/R3 stay gated; move shadow arms to Pathways |
| P38.2g2 | Source-pinned Pathways serving capture and combined KV arm | Real continue-decode metadata is durable and stock is reproduced before the default-off all-cache-read arm is interpreted | locally gated; target not run |
| P38.2h | Candidate target backward-no-commit | The selected candidate first makes all forward boundaries exact, then passes actual-model gradient/DP-reducer gates with zero optimizer commits | pending; forbidden before P38.2g selects a candidate |

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
- Correction (2026-08-11): P38d5 GSM8K update 0 applied an effective LR of
  exactly zero, not merely a small update below a bf16 threshold. Adam moments
  changed and all 16 microbatches were active; the old G6 gate incorrectly
  required a model mutation at LR zero.
- Observation (2026-08-11): the 25 P38d5 FrozenLake mismatches begin only at
  logical KV prefix 1791 or later. This is a localization signal, not yet a
  causal attribution to a page or attention tile boundary.
- Correction (2026-08-11): Phase 13 did not establish `CANON_KV_UNIFIED` as a
  repair. Its PATHTRACE-proven two-pass arm produced the same per-token values
  as stock, and Phase 14 separately showed full-fresh versus cache-plus-fresh
  equality inside one MIXED kernel. The new long-context/multi-turn/Pathways
  domain permits a retest but not a prior claim of efficacy.
- Correction (2026-08-11): the GSM8K completion-length summary does not prove
  that any action reached logical KV prefix 1792. Compute valid prompt length
  plus completion position before using GSM8K as a depth negative control.
- Decision: the first refreshed FrozenLake `backward-no-commit` JobSet is a
  capsule-capture run because the known A-B hard gate precedes backward. Do not
  report its label as evidence that backward executed.
- Decision: do not mix an unverified KV-unified implementation into the first
  target capture. After a verified capsule exists, run stock and single-turn
  controls first; add default-off MIXED-only and all-distribution two-pass arms
  with isolated cache inputs. Prefix cache stays disabled.
- Observation (2026-08-11): one-host synthetic prompt lengths 256 and 1788
  both produced R0=R1 bitwise while R0/REF and R1/REF were red at every scored
  action. The shallow maximum was larger. This rejects a depth-1791
  interpretation for the synthetic probe and keeps R2/R3 gated on a verified
  production capsule plus an exact serving-envelope control.
- Observation (2026-08-11): verified target row 191 gives R0=R1 exactly, while
  both differ from REF at 395 of 517 action logprobs. REF exactly reproduces
  captured `S_prefill`/`T_old`; R0/R1 do not reproduce captured decode. The
  local mask-derived serving envelope therefore fails its prerequisite and
  cannot be used to interpret KV-unified counterfactuals.
- Source audit (2026-08-11): production decode uses the donated-cache
  `continue_decode` loop, so prompt-only P18/P35 capture misses the real A
  program. RPA v3 exposes only a combined `update_kv_cache` switch: false both
  skips the fused write and forces all-cache reads. A write-only `W` arm is not
  constructible from the public v3 API and must not be claimed. P38.2g2 starts
  with a real continue-decode capture and the combined historical `U` arm.

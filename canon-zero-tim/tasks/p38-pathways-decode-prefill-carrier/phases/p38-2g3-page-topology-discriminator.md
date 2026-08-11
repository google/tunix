# P38.2g3: exact-state page-topology discriminator

- Status: pending; blocked on a complete P38.2g2 stock capture.

## Finding

- Confirmed: the real Pathways decode output differs from canonical rescore in
  FrozenLake step 0 and GSM8K update 1, while clean mask-derived one-host
  replay does not reproduce the production decode output.
- Confirmed: the current artifacts do not isolate page allocation. The
  GSM8K update transition changes weights, sampled tokens, scheduler state,
  and cache history together.
- Hypothesis H1: logical-equivalent KV content produces different output under
  a fragmented/non-contiguous physical page topology.
- Hypothesis H2: an invalid or stale padding-row block-table entry leaks across
  a request/DP boundary. The observed `row % 16` enrichment is only a clue;
  its slot/rank meaning must be captured explicitly.
- Hypothesis H3: cache page contents already differ before attention because
  the fused writer or page reuse is wrong.
- Hypothesis H4: the proxy precision flag is a confounded code-epoch axis, not
  the page-state carrier.

## Prerequisite hardening

Before the 64-chip stock capture:

1. record only requests present in `num_scheduled_tokens` and reject an empty
   selected request rather than indexing a live-but-unscheduled request;
2. emit and validate `request_id`, exact token-history hash, `global_row`,
   `dp_rank`, local scheduler slot, input-batch index, selector range, and
   physical block IDs;
3. validate that request lists, DP mappings, selector rows, sequence lengths,
   and block-table rows describe the same requests;
4. require stock to report zero `KV_UNIFIED_two_pass` PATHTRACE hits and U to
   report a positive hit; and
5. require the captured request to join the durable A-B mismatch record by
   request/token history, not row ordinal.

## Execution

### E0: production reproduction

Run P38.2g2 stock on Attempt 0 with backward disabled and zero optimizer
commits. Preserve the pre-dispatch record even if the backend later exits.

Pass only if the replayed A action vector is bitwise equal to the complete
captured production A action vector for the selected request: identical action
count, `np.array_equal` over every action position, zero differing elements and
bytes, and identical masked SHA-256. Matching only the previously red
coordinates is insufficient. The same request's B vector must also reproduce
the captured B vector and the source A-B comparison must remain red. A complete
record that cannot join the mismatch request is `INCONCLUSIVE`.

If E0 fails, stop before every counterfactual. Perform one bounded recapture
revision that adds the missing state from this preregistered list:

- complete per-call co-batch request membership and scheduled-token counts;
- request-to-DP-rank, input-batch-index, local-slot, and selector mappings;
- per-call request-distribution metadata, query starts, sequence lengths, and
  physical block tables;
- page allocation/free generation and cache-buffer identity where available;
- sampling RNG/leaves, generated-token history, and decode-loop call/step
  ordinals; and
- prefix-cache-disabled and source/mesh/weight attestations.

Do not add numerical repair arms while E0 is not exact.

### E1-E4: exact-state counterfactuals

Using identical weights, token history, positions, valid lengths, logical KV
content, and compiled shapes, compare:

| Arm | Variable changed | Required invariant |
|---|---|---|
| E1 | relocate valid logical pages to a different valid physical-ID permutation | reconstructed per-layer logical page-content hashes equal E0 |
| E2 | relocate the same content to a contiguous physical table | reconstructed per-layer logical page-content hashes equal E0 |
| E3 | keep all valid rows/pages unchanged and sanitize only inactive/padding block-table rows to an immutable zero sentinel | valid-row tables and logical page-content hashes equal E0 |
| E4 | keep the E3 table fixed and replace only the padding-only sentinel contents with a deterministic finite poison pattern | the sentinel appears in no valid row; all valid page contents and every non-sentinel input equal E3; the same executable fingerprint is reused |

Before using a zero sentinel, prove that it is initialized, immutable during
the call, and cannot alias a live page. E4 is a causal poison control, not a
production candidate. Before injection, assert that the sentinel physical page
does not appear in any valid row and that no valid logical page aliases it.
Use a finite deterministic pattern rather than NaN/Inf. E3 exact plus an E4
output change establishes that padding-only page contents influence the stock
kernel; E3 exact without an E4 response is insufficient to declare a data
leak, because table sanitization may have changed control flow instead.
Inject the poison as runtime data with unchanged shape, dtype, sharding, and
compile signature; a poison constant that creates another executable voids the
causal verdict.

For E0-E4, record logical page-content hashes after every cache-mutating event,
not only at final state. Each checkpoint carries call ordinal, decode step,
turn/chunk or append event, request ID, logical/physical page IDs, sequence
length, and per-layer K/V hashes. The first temporal divergence determines
whether the carrier begins in page write/reuse or only during attention read.

Do not call E1/E2 a topology experiment if page-content equivalence is absent.
Do not interpret an arm that fails to reproduce E0 before changing its single
variable.

### E5: same-source flag control

Run flag OFF only in an isolated diagnostic proxy with the same source,
weights, prompt corpus, and registered workload. Read A-B before downstream
B-C. This separates the flag axis from the source epoch but is not a release
configuration.

## Exit gate

- Local construction gate: focused P38 serving classifier/renderer/transport
  tests plus the complete pinned-image P33 CPU gate all pass, including
  live-but-unscheduled, mapping-inconsistency, missing-join, missing-PATHTRACE,
  and poisoned-padding negative controls.
- Target reproduction gate: Attempt 0, exact source, complete stock capture,
  exact request/token-history join, known A-B red, B-C exact, no backward, and
  zero optimizer commits.
- Counterfactual gate: E0 is reproduced before E1-E4, every purported
  page-topology arm proves logical page-content equality at each registered
  write event, and E4 proves its sentinel is padding-only before poisoning it.

## Decision table

| Observation | Verdict | Next repair |
|---|---|---|
| E3 becomes exact and E4 poison changes output | padding-page data dependency causal | fix valid bounds/index masking in the kernel; sentinel routing remains a diagnostic rather than the preferred production fix |
| E3 becomes exact but E4 poison has no effect | sanitization changes behavior but data leak is unproved | inspect table validity/control-flow differences; do not declare padding contents causal |
| E1/E2 change output with equal logical content | physical traversal/order causal | canonicalize gather and reduction in logical page order |
| temporal page-content hashes first differ after a write/reuse event | write/reuse corruption | isolate that event and the first differing layer/page; do not blame read topology |
| E0 reproduces; combined U becomes exact | combined cache-write/read mechanism causal | treat U as a candidate only; current v3 API cannot distinguish writer from read-source |
| E0 reproduces; U remains red | combined KV unification falsified | inspect page gather, first differing hidden/attention layer, and boundary indexing |
| E0 does not reproduce | `INCONCLUSIVE` | repair capture/replay fidelity; no numerical fix admitted |

A pool dose-response experiment is optional and does not enter the main ladder.
Use it only if the stock capture does not contain enough affected requests to
select an exact E0 row.

## Rollback

All capture and counterfactual mechanisms stay default-off. Leave
`CANON_P38_SERVING_CAPTURE_DIR` and `CANON_KV_UNIFIED` unset and omit the
P38-only page-table arm. Stock attention, precision, loss, and optimizer paths
remain selected.

## Result

Not run.

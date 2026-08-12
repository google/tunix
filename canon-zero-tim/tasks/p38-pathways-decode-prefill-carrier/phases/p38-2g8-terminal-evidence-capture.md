# P38.2g8: terminal standard-path evidence capture

- Status: operator protocol complete; target P38s9 pending.

## Evidence correction

P38s8 did not return a complete standard-path diagnostic. The committed file
is a 1,437-line, 173,137-byte excerpt that begins inside a device-memory report
and ends during initial canonical model compilation. It contains one capture
INIT marker but no byte-zero source/Attempt-0 preamble, OBSERVE, pre/post
capture, alignment record, child exit, classifier, archive, or outer
postflight. It is `INCONCLUSIVE_PARTIAL_EXCERPT`.

The archived s5/s6 “head full” files in `42139ffa` are byte-for-byte duplicates
of their already audited counterparts. They do not turn either nonterminal run
into new evidence.

The Section 55 claim that FrozenLake prompts stayed below 1536 is withdrawn.
The installed runner calls `_p38_observe_scheduled_prefixes` before applying
the prefix-stratum filter. Therefore a terminal full log with INIT but no
OBSERVE would indicate a reachability/wiring failure, not a range miss. The
partial s8 excerpt cannot establish either outcome.

## Deliverable

One stock-only P38s9 attempt returns a terminal byte-zero log and the exact
JobSet/pod/proxy/RM/event bundle. Its interpretation is preregistered:

| Terminal evidence | Classification |
|---|---|
| INIT=1, OBSERVE=0 | `INCONCLUSIVE_STANDARD_HOOK_NOT_REACHED` |
| OBSERVE>0, observed maximum <1536 | `INCONCLUSIVE_PREFIX_RANGE_MISS` |
| OBSERVE crosses a stratum, capture=0 | `INCONCLUSIVE_SELECTION_OR_MAPPING` |
| four pre/post pairs, missing classifier/archive | `INCONCLUSIVE_POSTFLIGHT` |
| complete records, exact join, classifier/archive/postflight | `CAPTURE_ADMITTED` |

The run remains pre-backward with zero optimizer commits. Unified KV is a
completed negative and is forbidden. FrozenLake full training and P45 use
different manifests and are outside this phase.

A local RoPE decode-shape/prefill-shape screen may proceed independently. Its
claim ceiling is operator/aval evidence only: neither a red nor exact result
may bypass exact E0 whole-vector reproduction. Attaching capture to P45 is not
an operator action because the current environment and postflight contracts
are intentionally fail-closed and diagnostic-only; that option requires a
separate reviewed shadow-mode implementation.

## Exit gate

The operator follows `HANDOFF.md` and
`cluster/P38_FROZENLAKE_DEBUG_RUNBOOK.md` from one immutable source commit.
The returned evidence directory must include the full non-timestamped head log
from byte zero plus final JobSet and pod state. An admitted capture must also
include the run-specific mismatch capsule and serving archive and pass an exact
request/token-history join.

## After the gate

- `CAPTURE_ADMITTED`: proceed to P38.2g3 exact E0 whole-vector replay, then
  measure the first divergent seam before choosing a repair.
- Any `INCONCLUSIVE_*`: make the single change named by the decision table and
  repeat stock only. Do not infer page, scheduler, RoPE, or kernel causality.

## Claim ceiling and rollback

This phase changes documentation only. It does not establish a numerical cause
or repair and does not admit FrozenLake training. Rollback is removal of this
operator protocol; runtime behavior is unchanged.

# P38.2g4: decode-envelope seam localization

- Status: active; D0 is published at `b89435ca` and D1 is not run.

## Objective

Localize the remaining FrozenLake `S_decode_vs_S_prefill` carrier without
assuming that RoPE, page ownership, KV contents, or another decode-only seam is
causal. The decisive prerequisite is an exact replay of one production decode
request. No counterfactual or repair is admissible before that replay is
bitwise exact over the complete action vector.

The convergence campaigns are operationally independent. Warning-only
training may collect evidence, but it cannot satisfy this phase or promote a
zero-TIM claim.

## Known facts and claim limits

- The projection, RMSNorm, SwiGLU, RPA primal, and log-softmax/gather sites use
  the canonical operator chain.
- The whole programs remain different: native continue-decode uses a live
  cache and scheduler metadata, while prefill rescore and trainer replay use
  different outer envelopes.
- RoPE is not numerically canonicalized by the current patch set. That makes it
  a candidate seam, not a proven cause.
- Stock and combined KV-unified target runs both remained red. Because they
  sampled different trajectories, their mismatch counts are not a paired
  effect measurement. Do not rerun the combined U arm.
- Existing one-host mask-derived replays reproduce the rescore vector but not
  the production decode vector. They cannot select a repair.
- Existing production artifacts contain alignment values but no admitted
  serving block-table archive. Page ownership and stale-cache hypotheses remain
  unproven.

## D0: local capture hardening

Replace the single `min_prefix=1788` trigger with four bounded capture strata:

1. `[1536, 1792)`;
2. `[1792, 2048)`;
3. `[2048, 2304)`; and
4. `[2304, 2560)`.

Capture at most one continue-decode call per stratum. The upper value `2560`
is an exclusive boundary, not a fifth record. Emit the observed prefix, stratum
index, exact bounds, callable implementation identity, request/DP/slot mapping,
token-history hash, attention metadata, physical page IDs, and pre/post arrays.

The classifier must fail closed when record counts, strata, source identity, or
array attestations drift. At least one record must join a mismatch-capsule row
by exact token-history prefix; zero joins is inconclusive and an ambiguous join
is fatal. A missing stratum is inconclusive rather than evidence of absence.

Before a target run, estimate the capture archive size and require at least five
times that size as free space in both the capture directory and stdout/archive
transport path.

### D0 exit gate

- focused renderer and classifier tests pass;
- negative controls reject a missing stratum, duplicate stratum, out-of-range
  prefix, missing implementation identity, zero exact joins, and ambiguous
  joins;
- both pinned model overlays install and compile with the updated patch; and
- the complete frozen-image P33 CPU gate passes.

## D1: one stock-only target capture

Run one Attempt-0 FrozenLake stock diagnostic on 64 chips. Prefix cache stays
disabled. Backward stays disabled and optimizer commits remain zero. Preserve
the complete outer log through postflight, the run-specific mismatch capsule,
the serving archive, classifier JSON, and final PATHTRACE.

The numerical prerequisite is the known production signature: A-B red and B-C
exact. The serving prerequisite is four admitted strata and at least one exact
request/token-history join. Do not launch U or a numerical repair arm.

## D2: E0 exact production replay

Replay the joined request with the captured scheduler inputs, live metadata,
cache state, RNG, and callable identities. Pass only when:

- the complete replayed A action vector equals captured A with
  `np.array_equal`, zero differing bytes/elements, and equal masked SHA-256;
- replayed B likewise equals captured B; and
- the source A-B comparison remains red.

Matching only previously red coordinates is insufficient. If direct-attached
one-host replay cannot reproduce E0, allow one bounded same-session Pathways
exact replay. If that also fails, stop and extend capture fidelity; do not run
counterfactuals.

## D3: first-divergence localization

Only after E0 passes, run two independent branches against the same E0 record.

### D3a: decode-envelope seam

Compare A and B at ordered checkpoints:

1. layer input;
2. Q/K/V projections;
3. Q/K normalization;
4. post-RoPE Q/K;
5. RPA output;
6. output projection and residual;
7. post-attention normalization;
8. MLP output; and
9. layer output, final norm, raw target logit, normalizer, and logprob.

The first differing checkpoint selects the repair site. In particular,
Q/K-normalization exact followed by post-RoPE red implicates RoPE; it must not
be assumed in advance. Observer instrumentation is valid only if observer-off
and observer-on endpoints are bitwise equal. The existing graph-cut controls
are initial falsifiers only because they change the graph.

### D3b: page and cache state

Run E1-E4 from P38.2g3 using identical weights, tokens, valid lengths, logical
KV contents, shapes, and executable fingerprints. Require temporal logical-page
K/V hashes after each mutating event. The padding-poison control remains
diagnostic-only and must prove its sentinel is absent from every valid table.

## D4: smallest repair

Change only the first proven divergent seam:

- RoPE: canonicalize the exact position/rotation operation used by both
  envelopes;
- another operator seam: route both envelopes through one registered
  implementation and compiled contract;
- page mapping or padding leak: repair validity/indexing while preserving valid
  logical page content; or
- write/reuse divergence: repair the first corrupting mutation.

Do not change precision, loss, sampling semantics, optimizer math, prefix-cache
policy, or the fixed-M/fixed-reduction contracts.

## D5: promotion ladder

1. one-host E0 plus negative controls;
2. 64-chip FrozenLake backward-no-commit with A=B=C-old;
3. actual-model T-old=T-current/THIRDPROG gate;
4. gradient, fixed-order DP reducer, replica, and optimizer transaction gates;
5. GSM8K regression;
6. DeepSWE backward-no-commit; and
7. strict FrozenLake full training.

No later rung can replace an earlier one.

## Rollback

Leave `CANON_P38_SERVING_CAPTURE_DIR` and the new prefix-strata variable unset.
The diagnostic remains default-off and stock training, attention, precision,
loss, optimizer, and cache behavior remain unchanged.

## Result

D0 passed locally on 2026-08-11. The capture now selects one concrete
scheduled request in each interval rather than admitting a call merely because
some unrelated row reached that interval. The record binds that anchor to its
request ID, exact prefix, token history, physical page mapping, source commit,
and callable identities. The classifier rejects an invalid anchor, incomplete
or duplicate strata, identity drift, an invalid five-times storage margin,
zero exact joins, and ambiguous joins.

Focused classifier tests pass 25/25, renderer tests pass 5/5, shell postflight
passes, and both pinned Qwen overlays match all 29 manifest entries and pass
14/14 exact-image tests. The complete frozen-image P33 CPU gate passes with 78
workload tests, 29 alignment tests, and all adjacent suites. The reconstructed
runner overlay SHA-256 is
`fe81622996a1c73bbd17187ee603e6a191165202da40d07b5e428fe41b5db516`.

This is a construction result only. Docker had no `/dev/vfio`, and no target
TPU, Pathways, backward, optimizer, repair, or cloud launch was performed. The
implementation was subsequently published as `b89435ca`. D1 remains blocked
on separate resource approval.

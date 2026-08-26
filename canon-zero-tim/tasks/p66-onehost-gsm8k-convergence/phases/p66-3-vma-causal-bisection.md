# P66.3: TP backward VMA/collective causal bisection

## Objective

Locate the first cause of the target-only gradient explosion without using
clipping, optimizer commits, or an infeasible whole-model monolithic oracle.
The observed target failure is a backward failure: strict A/B/C forward
alignment remains necessary but cannot certify this phase.

## Superseding evidence

- Historical one-host gradients have ordinary scale, while the Attempt-7
  DP16xTP4 GSM8K run reports every update at roughly `1e20`-`1e22` and P45/M15
  reach non-finite rank-local gradients. This is a regression, not a legitimate
  finite-gradient envelope.
- In the complete P62 target receipt the loss cotangent is sane
  (`max_abs=0.0112169`, `stable_norm=0.689952`), while group-0 `engine_vjp` is
  already red (`max_abs=5.79228e21`, `stable_norm=5.38142e22`). Therefore the
  DP reducer, streamed multiplier, accumulator, norm, clipping, and optimizer
  are downstream symptoms, not the first cause.
- The fetched P64 DP8xTP8 target artifact at remote commit `1406cc2d` makes the
  same boundary sharper on Qwen3-8B. Strict pre-alignment passes for 46,276
  actions; loss cotangent (`stable_norm=0.430281`) and group-0 input cotangent
  (`stable_norm=0.0698096`) are finite. The first non-finite is group-0
  `engine_vjp`, leaf 1, rank 3. Ranks with exact-zero input cotangent
  (0/1/2/5) emit exact zero, while the four ranks with nonzero input cotangent
  (3/4/6/7) emit NaN. This rejects a downstream reducer/clip explanation and
  unconditional constant contamination, but it does not distinguish VMA
  ownership from another engine-pullback defect because P64 has no S/U/P/R
  arms.
- The largest leaves are the embedding and earliest transformer layer. The
  depth profile is consistent with repeated transpose/collective amplification
  and must be measured directly before any repair claim.
- P66 ordinary whole-model `nnx.value_and_grad` attempt
  `p66o1_20260825t1903z` passed strict pre-alignment and captured the exact
  model-before tree, then failed as `INCONCLUSIVE_CARRIER`: XLA requested
  `450.29 GiB` of HLO temporaries on a `95.74 GiB` device. It emitted no
  gradient and cannot judge segmented math.

## Primary hypotheses

1. **H1 — P59 VMA/TP composition.** P59 makes both data and model manual at
   TP>1, then nests/localizes engine shard maps while the outer map uses
   `check_vma=False`. JAX documents that this disables out-spec replication
   checks and efficient reverse-mode VMA tracking, falling back to defensive
   `psum`s. A falsely replicated TP output can therefore be silently accepted
   or redundantly reduced.
2. **H2 — fixed TP gather transpose.** `CANON_FIXED_AR_GATHER=1` replaces the
   forward fixed ppermute tree by all-gather plus local ordered sum. Its
   transpose/custom-VJP interaction inside P59 may add a TP factor at every
   layer.
3. **H3 — padding-row cotangent reaches a small-RMS residual.** Token-id-0
   padding can have residual RMS around the numerically sensitive part of
   RMSNorm, but that is causal only if a masked padding row first receives a
   nonzero `dhidden`. Every G1 arm therefore records, for every real Qwen
   layer and chunk, input residual RMS plus post-pullback `dhidden` max split
   by real/padding rows. Small padding RMS with exactly zero padding cotangent
   rejects this mechanism for the replay. Nonzero padding cotangent supports
   it only after comparing S/U/P/R; the residual scale alone is not evidence.
4. **H4 — another segmented engine endpoint.** If H1/H2/H3 separate cleanly
   but the full reverse remains red, inspect the first layer boundary rather
   than returning to norm/clipping.

## Gates and arm order

### G0 — structural VMA probe

- Add default-off `CANON_P66_P59_CHECK_VMA=1` to the existing real installed
  DP2xTP4/TP8 composition probes.
- A replication/out-spec error is `STRUCTURAL_RED`, not a numerical verdict;
  record the exact value path and manual axis.
- A scan/custom-call VMA limitation is `INCONCLUSIVE_VMA_TOOLING`; do not call
  it the root cause.
- A clean compile only rejects the narrow H1 consistency claim; it does not
  prove target gradients sane.

### G1 — one-host full-Qwen TP discriminator

Use one v5p host and zero optimizer commits. Since DP2xTP4 needs eight chips,
the four-chip carrier uses the same physical TP4 engine with a diagnostic unit
data axis. It must execute the real 1.7B fixed head, all 28 layers, projection
and attention shims, and group-0 engine VJP. It is a causal proxy, never target
certification.

| Arm | P59 outer TP context | VMA check/transpose | fixed AR gather | Purpose |
|---|---:|---:|---:|---|
| S | off | stock inner semantics | on | serial segmented TP4 scale control |
| U | on | historical off/manual TP sums | on | reproduce the unsafe P59×TP interaction |
| P | on | checked, VMA-owned TP transpose | on | candidate P59 semantic repair |
| R | on | checked, VMA-owned TP transpose | off | isolate gather transpose after the VMA repair |

All arms require the same seven input hashes, exact model-before sample,
strict pre-alignment, finite loss cotangent, layerwise max-abs profile, the
28-layer real-vs-padding residual/cotangent profile, and no optimizer
transaction. `grad_norm > 1e6` is diagnostic-fatal; it is not clipped.

### G2 — target replay, only after G0/G1

The signed P64 capsule now exists (SHA-256
`af0dc4fc2f8dfb592682b70f752779b970fe9f47713f7fb0e05a5079d982e041`,
model binding SHA-256
`71d7a3775656a4f58762ff97946a0c223dfbe1cef90b8a09a7a49e1129c70053`).
After G1 passes, replay group 0 with current P59, P59-off serial, and
fixed-gather-off arms. This requires target launch approval and is
not substituted by the one-host proxy. A fix is admitted only if the target
full boundary chain is finite/plausible and the repaired arm matches its
serial/FP64 envelope.

### G1.5 — same-evaluation-point pullback oracle, before G2

G1 proves causality and restores an ordinary full-tree scale, but its P/S
comparison is a whole-gradient summary rather than a same-call derivative
oracle. Before spending a DP8xTP8 target allocation, run one additional
default-unreachable `tp4-vma-oracle` arm on the same one-host carrier.

Shape ledger:

- caller-global M: 256 rows;
- shard-local M: 256 rows because the diagnostic data axis is unit sized;
- canonical-kernel M: 256 rows;
- semantic valid rows: the exact `host_n_real` vector from the frozen group-0
  payload, never inferred from padding token ids;
- scheduler capacity: 16 sequences / 256 batched tokens.

Program identity remains Qwen3-1.7B, DP1xTP4, real fixed head, real RPA,
all 28 layers, `CANON_P59_RANK_PARALLEL_BACKWARD=1`, checked VMA, fixed gather,
one reverse group, and zero optimizer commits. The oracle arm changes no
production profile or default.

For head, final norm, layers 27/14/0, and embedding:

1. compute the checked-VMA candidate pullback first;
2. call the ordinary serial pullback with the exact same state, primal inputs,
   cache input, and output cotangents;
3. never feed the serial result into the candidate reverse;
4. compare every parameter cotangent plus the activation/cache cotangents that
   exist at that endpoint;
5. emit only bounded scalar receipts; no per-token device fetch is admitted.

The frozen P61 absolute caps apply independently to every endpoint:
`rel_l2 <= 4e-2`, `one_minus_cos <= 3.2e-4`,
`norm_ratio_error <= 4e-2`, and `sign_mismatch_rate <= 2e-2`. Shapes must be
exact, both trees must be finite, every live reference leaf must remain live,
and the endpoint count must be exactly six. A normal-value perturbation must
be rejected by the same comparator before any model receipt is accepted.

Observer neutrality is a separate hard gate: the oracle arm must retain the
same frozen input hashes/model-before sample and reproduce P13's candidate
engine/gradient sample, layer profile, and row-cotangent summary. Any strict
alignment failure, comparator failure, missing endpoint, negative-control
miss, state mutation, or optimizer commit stops G1.5 and forbids G2.

| G1.5 observation | Decision |
|---|---|
| Six endpoint receipts pass, negative fires, candidate remains neutral | admit source freeze, then request G2 target launch |
| Head/norm endpoint is first red | repair that endpoint's VMA/transpose at G1.5; do not launch target |
| Layer 27/14/0 is first red | localize to that real layer pullback and its cache/hidden boundary |
| Embedding is red | repair the fixed-vocab completed-sum transpose boundary |
| Candidate differs from P13 while oracle side result is unused | observer/program-identity red; remove or redesign the observer |
| Inputs or model sample differ | `INCONCLUSIVE_INPUT`; freeze a fresh same-input pair |

## Decision matrix

| Observation | Decision |
|---|---|
| VMA structural red, U huge, and P/R ordinary like S | H1 supported: repair P59 manual-axis/out-spec/transpose semantics |
| U/P huge, R ordinary, S ordinary | reject `CANON_FIXED_AR_GATHER` in P59 backward and repair its transpose |
| S, U, P, R all huge | first red predates P59 interaction; bisect engine endpoints/layers |
| S/P/R ordinary and U does not reproduce | proxy is non-discriminating; require signed target capsule replay |
| S ordinary, P/R ordinary on one host but repaired target red | proxy did not close the target gap; require signed target capsule replay |
| Padding residual RMS is small but padding `dhidden` is zero in every layer | reject H3 for this replay; RMS scale alone is non-causal |
| Padding `dhidden` first becomes nonzero only in U and P/R restore S | H3 is downstream of the historical P59 transpose defect; retain VMA repair and separately fix masking only if S is also affected |
| S/U/P/R all receive comparable padding `dhidden`, but only U explodes | padding may be a common stressor, not the optimization regression; H1 remains the discriminator |
| Any strict alignment FAIL | hard reject that arm; fresh label after repair |

## Forbidden shortcuts

- Do not use P63 scaled-L2 as a correctness repair.
- Do not commit an optimizer update in a diagnostic arm.
- Do not infer target safety from forced-CPU slices or one-host TP4 alone.
- Do not enable/disable more than one causal variable per arm.
- Do not commit or push without explicit user approval.

## Result log

- Verified by one-host S/U: the padding hypothesis is rejected for the frozen
  replay. S is ordinary (`engine_vjp stable_norm=6.0506024`); U uses identical
  input and grows real-row `dhidden` to `4.2658e19`, with full engine-gradient
  stable norm `1.5402378e21`. Both keep every observed padding cotangent zero.
- Verified by P64 remote artifact at commit `1406cc2d`: DP8xTP8 first becomes
  non-finite inside group-0 engine VJP only on ranks receiving nonzero
  cotangent. Artifact log SHA-256 is
  `43a262d6d57b4ac9cda077490460def511d86cfa6ba979ebab2cf671f70245cd`;
  receipt SHA-256 is
  `77134fb176e092b4a173690a77f7177fefbd80559b2f0c8753126bed39832635`.
- Not verified: P64 does not contain a serial or VMA-repaired arm and therefore
  does not certify the current repair.
- Verified by pinned image before the newest bridge edit: checked-VMA installed
  TP4/TP8 fixed-head/projection/attention composition passes with
  `manifests=2x36/36` and zero commits.
- P arm `p66p8_20260825t2155z` is a structural red, not a numerical verdict.
  It passed strict pre-alignment then rejected a `{V:data}` final-norm
  cotangent at a nested engine map whose output contract had erased VMA. Raw
  evidence remains under
  `/mnt/disks/tunix-data/logp_probe_1host/p66_tp4_tp4-p59_p66p8_20260825t2155z/`.
- Implemented but not yet pinned-image/TPU verified: under the P66 checked-VMA
  flag, TP>1 nested engine bodies directly reuse the already-local outer map
  instead of entering a second map with erased data/model specs. Historical
  VMA-off and production-default behavior are unchanged.
- Verified by P attempts `p66p9`, `p66p10`, and `p66p11`: checked VMA next
  exposed, in order, a duplicate varying pcast, missing RPA custom-call output
  VMA, and a real layer-27 hidden output typed `{V:(data,model)}` against the
  correct upstream cotangent `{V:data}`. Every attempt retained strict
  pre-alignment PASS and zero optimizer commits; none reached a numerical
  verdict.
- Evaluated, not assumed: the layer-27 mismatch is produced by the completed
  fixed TP all-gather/ring sum. Its value is identical on every model rank but
  local additions leave its manual-axis type varying. Relabeling the incoming
  cotangent varying would duplicate the logical loss cotangent and is rejected.
- Implemented and verified by host plus pinned image: P66-only TP pmean on the
  already-identical completed sum registers the hidden output invariant and
  supplies the matching transpose. P66 4/4, P59 37/37, and VMA-on installed
  TP4/TP8 composition `manifests=2x37/37` pass with zero optimizer commits.
  Full-Qwen TPU numerical behavior is not yet verified because P11 stopped at
  the pre-fix structural boundary.
- Verified by P12: the projection invariant boundary lets checked VMA traverse
  all 28 real transformer pullbacks. P12 then found the same ownership defect
  at the input embedding's completed fixed vocab sum; strict pre-alignment
  remained byte-zero and optimizer commits remained zero.
- Implemented and host/pinned-image verified: the P66-only embedding fixed-ring
  result receives the same invariant boundary. Both model installs remain
  37/37 and the VMA-on TP4/TP8 terminal gate passes. P13 subsequently supplied
  the full-Qwen numerical receipt below.
- Verified by one-host P13: the complete checked-VMA full-Qwen TP4 reverse is
  finite and ordinary. Engine norm is `6.05732584`, mapped norm is
  `0.37858307`, all 310 leaves are finite/nonzero, 17/17 strict alignment
  passes, all 56 padding cotangent observations are zero, and optimizer commits
  are zero. This is `0.1112%` from serial S by both engine and mapped norm,
  versus historical U at `1.5402378e21`.
- Not verified: one-host P13 is not P64 DP8xTP8 target replay and does not yet
  certify P64 DP8xTP8 target behavior.
- Verified by one-host R: gather-off produces exact-equal P input/pre-state,
  full engine and mapped gradient summaries, sampled gradient hashes, and
  profiles. R is 17/17 strict with zero commits. H2 fixed-gather transpose is
  rejected as the regression cause once VMA ownership is correct.
- Verified by final current-source S: 17/17 strict, zero commits, engine norm
  `6.050602436`, mapped norm `0.378162891`, exact P/R input hashes and
  model-before sample.
- Verified by final current-source U: strict pre-alignment passes with zero
  differing bytes and the same four-arm input hashes; every padding-row
  cotangent remains zero, while real-row `dhidden` reaches `4.2658096e19` and
  the 310-leaf engine gradient reaches stable norm `1.5402378e21`. The
  pre-registered `1e6` fatal threshold stops the arm before any optimizer
  commit. Classification is `EXPECTED_RED`; no failed alignment is present.
- Verified by the pre-registered four-arm classifier: verdict
  `H1_VMA_SUPPORTED`, empty contract-reason list, same four-arm pre-alignment,
  same S/P/R group hashes and model-before sample, and zero optimizer commits.
  P and R mapped-gradient norm ratio to S is `1.0011111163`; P and R are exact
  to each other in captured gradient evidence. The generated receipt SHA-256
  was `be5a160396474666fa214658faadd120dd337ccf16a2705132a3bfcec8c67c8a`;
  the durable sources are the four immutable run directories registered in
  `log.md`, and the receipt is reproducible with
  `tests/p66_backward/classify_tp4_campaign.py`.
- Not verified because no target launch was authorized: checked-VMA P59 on the
  signed P64 DP8xTP8 capsule. G1 therefore supports the repair mechanism but
  does not certify DP8xTP8, DP16xTP4, a real optimizer update, convergence, or
  production performance.
- Verified by final-source one-host G1.5 run `p66o2_20260826t0010z`: the
  default-unreachable oracle arm implemented at
  `tunix/rl/canonical_qwen3_adapter.py:6276` computes the candidate first and
  calls the pure comparator at `tunix/rl/p66_vjp_oracle.py:51`. All six
  endpoints pass the frozen P61 caps: head rel-L2 `5.7114e-7`, norm `0`, layer
  27 `9.4928e-4`, layer 14 `3.3325e-3`, layer 0 `5.2568e-3`, and embedding
  `0`; the cap is `4e-2`. Norm and embedding are array-exact, every tree is
  finite, and no live reference leaf becomes dead. The normal-value negative
  at `tunix/rl/p66_vjp_oracle.py:230` fires before model receipts.
- Verified by the hard arm classifier at
  `canon-zero-tim/tests/p66_backward/classify_tp4_arm.py:128`: final-source
  `p66o2` is 17/17 strict Zero-TIM, zero FAIL, engine norm `6.05732584`, mapped
  norm `0.37858307`, all 310 leaves finite/nonzero, and zero optimizer commits.
  The complete diagnostic backward is `179.736s` but is performance-ineligible.
- Verified by the independent P13 pair classifier at
  `canon-zero-tim/tests/p66_backward/classify_tp4_oracle_pair.py:51`: frozen
  pre-alignment hashes and model-before sample match, and candidate alignment,
  engine/gradient summaries, sampled gradient hashes, layer profile, and
  row-cotangent summary are exact after removing only the arm label. Verdict is
  `PASS`, with zero input and observer reasons.
- Verified after the final runtime cleanup: P66 host 16/16, P59 host 37/37,
  flag/syntax/diff gates pass; pinned image
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
  emits `P59_TP_SHIM_EXACT_IMAGE_PASS`, `manifests=2x37/37`, and executes both
  P66 unit-data installed-attention arms. The pinned-image raw stream was not
  durably saved, so it is registered as a reproducible admission receipt, not
  a signed raw artifact.
- Verified evidence provenance: immutable root
  `/mnt/disks/tunix-data/logp_probe_1host/p66_tp4_tp4-vma-oracle_p66o2_20260826t0010z/`
  has a passing `SHA256SUMS`; the classification SHA is
  `cfc78137211609997860cb6ad251be1464b10ffde28e736ace720e14ebc6b8b5`
  and observer-neutrality SHA is
  `d50d093754d07e9bc9bbe2d6d8429664808484e45586c0b0953d8530ba2be366`.
  The checked-in receipt is
  `tasks/p66-onehost-gsm8k-convergence/evidence/p66-g15-onehost-20260826/receipt.json`.
- Superseded but retained: `p66o1_20260825t2357z` produced the same numerical
  PASS before duplicate dead comparator code was removed from the adapter. Its
  runtime bytes differ from the final tree, so it is not used as the final
  source-freeze receipt.
- Not verified because no target launch was authorized: G2 signed P64 DP8xTP8
  replay, any real optimizer commit, convergence, or production performance.
  G1.5 admits source-freeze review only; it does not promote P59 to production.

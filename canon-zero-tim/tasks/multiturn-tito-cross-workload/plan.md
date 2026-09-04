# Plan

## Outcome

Determine whether text rendering and re-tokenization changes later-turn
FrozenLake prompts and provide a single explicit, closed treatment selector that
can apply exact token-in/token-out independently to P45, M15, both, or neither.
The legacy default remains neither; production DeepSWE remains independently
TiTO. No phase may infer DP8xTP8 certification from host or one-host evidence.

## Phases

| Phase | Deliverable | Exit gate | Status |
|---|---|---|---|
| T0 | Real-tokenizer transcript oracle for DeepSWE tool turns and FrozenLake user turns | exact per-turn hashes, role boundaries, first mismatch, and a poison negative that the oracle catches | complete |
| T1 | Isolated M15 one-host DP1xTP4 carrier and delivery chain | host positives plus production/P45/DeepSWE leakage negatives; no production renderer change | complete |
| T2 | Legacy rendered-text observer run | one real M15 trajectory with at least three nonterminal turns, per-turn `TOKEN_STREAM_*`, explicit B-full-reset receipt, no prompt override, no backward/commit | complete — r7 17/17 equal, 3/3 strict |
| T3 | Exact-token matched replay and strict alignment | same bounded carrier as r7; actual prompt equals reconstructed prompt; cross-arm prompt/trajectory hashes match; strict A=B=C; zero backward/commit | complete — r8 17/17 exact-equal, 3/3 strict, r7/r8 cross-arm MATCH |
| T4 | Shared-runtime DeepSWE regression control | rerun existing Qwen3-4B DP1xTP4 controlled carrier only if shared token helper/runtime changed | pending — scoped host/pinned DeepSWE regressions pass, but the shared engine changed and the DP1xTP4 controlled carrier has not been rerun |
| T5 | Paired performance attribution | correctness-green legacy/exact captures report prompt tokens, host tokenization, prefill/decode and compile separately | deferred — bounded wall time is 489s legacy versus 499s exact, but no component-level causal performance claim |
| T6 | Target decision | decision table records one-host TiTO health separately from production/DP8xTP8 admission | complete — one-host healthy; production remains selector-absent pending DP8xTP8 target |
| T7 | Latest-tip repair and treatment preregistration | P67 renderer has one performance-bundle writer; the four-arm truth table and claims are recorded before runtime edits | complete |
| T8 | Generic FrozenLake exact-token contract | one default-off enum derives P45/M15 exact admission, prompt reconstruction, and workload-labelled receipts while preserving the historical M15 selector | complete |
| T9 | Host and pinned-image certification | P45/M15 positives and absence, malformed, mixed, profile, topology, neighbor, poison, and missing-receipt negatives pass | complete — host and full pinned image green; target remains separate |
| T9a | Bounded first-diff trajectory diagnostics | default-off renderer flag; only a selected exact P45/M15 arm may emit one complete token-ledger dump before the unchanged fatal mismatch | complete — log-only reconstruction, integrity negatives, and pinned runtime green |
| T9b-0 | Frozen-trajectory construction oracle | for every later turn, the shared reconstruction equals the exact B/C prompt prefix; first-turn prompt provenance and a one-token poison are explicit | complete for production-shape fixture; real-r7 token arrays were not persisted and remain a target evidence item |
| T9b-1 | Engine-consumption witness | request-ID-joined submit, RequestOutput echo, and TPU-runner input length/SHA agree; observer-neutrality and missing/duplicate/wrong-request negatives pass | complete for host/installed-overlay construction; TPU runner unrun |
| T9b-2 | Bounded multi-diff diagnostic collection | `collect-64` is admitted only on rollout-only/no-backward/no-commit P45/M15 diagnostics; a diff closes only its trajectory, never enters loss, and the run continues up to a process-wide bound | complete for host construction; target unrun |
| T9b-3 | Durable capsule and witness return | complete capsules and runner-journal shards upload no-clobber to the registered protected evidence root with local/remote SHA receipts and a complete manifest | complete with fake-remote gates; real GCS unrun |
| T9b-4 | Host, image, and one-host admission | focused negatives, full host suites, immutable-image gate, then matched one-host observer-neutrality and three-way token join pass | host/image release complete — the existing Perf-v2 DP1xTP4 three-update carrier has a closed off/on identity and host-tested judge; the matched TPU pair is unrun |
| T9b-5 | DP8xTP8 rollout-only target diagnostic | P45 and M15 each return a complete bounded data set at production topology with zero backward and zero optimizer commits | pending — separate launch approval required |
| T9c-0 | Full-record contract and truthful accounting | `record-full` is a closed exact P45/M15 full-training policy; token red continues unchanged training but cannot claim Zero-TIM; coverage and runtime counters are measured | complete for host and immutable image; target unrun |
| T9c-1 | Trajectory/request/numerical join | token capsules and batch rows carry stable trajectory/request/step identities; missing or ambiguous joins fail classification | complete for host and immutable image; target unrun |
| T9c-2 | Incremental durable extraction | only new immutable files are hashed and uploaded with retry, heartbeat, idempotent finalization, and final inventory proof | complete with fake-remote gates; real GCS unrun |
| T9c-3 | P45/M15 full record carriers | explicit 300-update record arms preserve existing topology/resources and return training curves plus four separate verdicts | implementation and render gates complete — target launch requires separate approval |
| T9d-0 | Crash-durable append journals and strict poison controls | immutable byte-range deltas reconstruct all four journals; a nominal PASS containing any red list is rejected | complete for host and pinned-image construction; real abrupt-exit/GCS unrun |
| T9d-1 | Replay-complete all-update A/B/C sidecars | every alignment update returns an atomic host-only NPZ with exact row joins and per-array hashes | complete for host and pinned-image construction; target volume unmeasured |
| T9d-2 | Bounded red-policy actor snapshots | first-any and first-`>=1`/`>=8`/`>=32` A-B policy versions save at most four actor-only, pre-update, non-resumable snapshots | complete with fake manager and installed-code gate; real Orbax/GCS save unrun |
| T9d-3 | Observer-neutral and durable target admission | matched exact-TiTO off/on carrier, host suites, real-path Orbax startup probe, single-writer/update-0 gates, renderer, and pinned image pass before a separately approved DP8xTP8 pair | release committed — P57 232/232, V1 102/102, APC 31/31, flags 422/422, full pinned image PASS; carrier/judge implemented, matched one-host/real GCS unrun |
| T10 | DP8xTP8 treatment run | explicit P45-exact and M15-exact full runs return complete token receipts and their existing alignment/training classifications | pending — separate launch approval required |

## Decisions

- Confirmed: DeepSWE TiTO is a common transport invariant selected by the DeepSWE workload identity and has real DP1xTP4 evidence.
- Confirmed: M15 `verify|exact` currently admits only the production DP8xTP8 full identity, so it cannot honestly be used on one host without a dedicated diagnostic identity.
- Confirmed: both workloads use the Qwen parser, whose assistant-end updater appends zero tokens; FrozenLake differs by `enable_thinking=False` and user-role environment messages, while DeepSWE uses tool/user messages.
- Decision: the available direct-attached host and the existing certified rehearsal carrier expose four devices, so use M15 DP1xTP4 locally. This proves token transport and strict local alignment only; target TP8 remains explicitly unverified.
- Decision: compare live arms only internally. Cross-arm causal comparison uses a frozen trajectory/turn capsule because identical seeds do not guarantee identical sampling after prompt IDs diverge.
- Decision: diagnostics are strict, APC-off, max-concurrency one, zero backward, and zero optimizer commit. The production M15 full recipe remains untouched.
- Hypothesis: M15 may or may not reproduce DeepSWE retokenization drift; the legacy observer must decide before exact TiTO is described as a repair.
- Finding: a real 23-turn DeepSWE trajectory yields 10/11 later-turn drift receipts, beginning at turn 2/token 2242; a short synthetic FrozenLake user-only transcript yields 2/2 equality. The role geometry therefore matters and the synthetic M15 fixture cannot authorize production TiTO.
- Finding: live M15 r4 yields 17/17 later-turn equality across three rounds and strict A=B=C, but the formal classifier correctly rejected the run because its explicit B-full-reset marker was absent. The fail-closed observation is repaired and host-gated; fresh target evidence remains required.
- User decision: legacy equality is insufficient; certify M15 exact TiTO itself on the matched one-host carrier. This reopens T3/T6 without changing the production default or claiming DP8xTP8.
- Finding: exact r8 is healthy on the matched one-host carrier: 17/17 exact-token receipts, three strict A=B=C rounds, and zero backward/commit. Ordered prompt receipts and per-round token/action-mask hashes match legacy r7 exactly.
- Decision: this closes one-host TiTO correctness. It authorizes keeping an
  explicit exact M15 full option in the renderer, default off; using that
  option is itself the DP8xTP8 target admission boundary.
- Finding: latest source `6842edae` moved full-system optimization injection
  into the base P57 renderer, while the P67 wrapper still injects the same
  seven keys. The existing wrapper therefore fails before writing a usable
  manifest; this is independent of token continuity.
- User decision: prepare a paired exact-token wave for both P45 and M15.
  Represent treatment selection as one closed renderer enum with legacy as the
  default; do not silently make either workload exact.
- Decision: preserve `CANON_M15_TOKEN_CONTINUITY` only as a historical M15
  compatibility selector. New full treatments use
  `CANON_P57_TOKEN_CONTINUITY=exact`; simultaneous old and new selectors are
  fatal.
- User decision: add an explicit debug flag for a mismatching multi-turn
  trajectory. It remains absent by default, may accompany only a selected
  generic exact P45/M15 arm, emits bounded chunked token-ID/segment evidence
  for the first mismatch, and cannot downgrade the immediate exact-mode fatal.
- User decision: extend TiTO diagnostics so one bounded run can collect more
  than one independent mismatching trajectory and return the evidence
  durably. Production exact remains first-red fatal. The collection mode is a
  zero-backward, zero-optimizer diagnostic; it must not mask bad trajectories
  and continue ordinary GRPO training.
- Decision: host reconstruction, vLLM RequestOutput echo, and TPU-runner input
  are three distinct evidence layers. The existing receipt proves only the
  first. Target admission requires a request-ID-joined three-way length/SHA
  comparison; RequestOutput alone is not described as a runner witness.
- Decision: the first target does not combine the new token observer with an
  S0 finite A-B numerical capsule. That observer is a later, independent CL so
  a target red remains attributable.
- Decision: asynchronous TPU-runner capture order is not required to equal
  sampler submission order. Identity is proved by request-ID joins; runner
  record indices need only be unique and contiguous. The sampler independently
  rejects a `RequestOutput` whose ID differs from its submitted future.
- Decision: collect-mode stdout contains no reversible token chunks. Durable
  recovery is the atomic mode-0600 file plus verified GCS snapshots. On abrupt
  pod loss, only files included in the most recent successful 30-second poll
  are guaranteed; worker logs are not a token recovery path.
- Decision: GCS availability is a pre-workload condition. A no-clobber probe
  must upload, download, and hash-verify before the exact READY receipt permits
  rollout. Finalization is content-addressed and idempotently reuses an
  already-uploaded final snapshot after a worker retry.
- User correction: the next collector must run inside the 300-update P45/M15
  full identities rather than stop after an initial-policy rollout pass. A
  token difference is recorded and the unchanged trajectory continues through
  ordinary GRPO; it is neither masked nor replaced. The completed run is then
  labelled `NON_ZERO_TIM_DATA_COLLECTION`, never a Zero-TIM PASS.
- Decision: request identity remains a fatal infrastructure boundary. A
  same-ID token echo difference is scientific data and may continue, but a
  missing, duplicate, swapped, or foreign request ID cannot be safely joined.
- Decision: single-turn trajectories are unexercised, not equal. Runtime
  backward/update/checkpoint claims must be measured rather than literal.
- Decision: periodic evidence return uses immutable deltas with retry and
  health receipts instead of repeatedly hashing and archiving the full tree.
- User decision: make the full-record pair replay-complete. Persist one
  host-only A/B/C sidecar for every alignment update and retain bounded actor
  weights at the exact first-any and first-`>=1.0`-nat A-B policy versions.
  These actor-only artifacts are evidence, not resumable training checkpoints;
  the ordinary checkpoint-write count remains zero.
- Decision: mutable row-map/alignment/update JSONL files cannot enter the live
  immutable-file glob directly. Derive ordered immutable byte-range chunks at
  complete-line boundaries and require their concatenation to equal the final
  source journal byte-for-byte.
- Decision: an instrumented record run is correctness/debug evidence, not a
  performance run. Sidecar/snapshot bytes and wall time are reported, and the
  mandatory one-host off/on gate must preserve A/B/C plus gradient/update hashes
  without adding a JAX module.
- T9d-3 correction: retain four bounded threshold categories—`first-any`,
  `first-ge-1`, `first-ge-8`, and `first-ge-32`—rather than only the original
  two. A single policy step may satisfy multiple categories. The matched
  one-host pair reuses Perf-v2 DP1xTP4 and treats cross-arm input-hash drift as
  inconclusive, never numerical PASS.

# T9b — engine witness and bounded multi-diff collection

- Status: active

## Motivation

The existing exact-TiTO receipt compares the reconstructed integer ledger
with `SamplerOutput.padded_prompt_tokens`. The sampler constructs that array
from the same submitted IDs after generation, so this is a useful host-path
check but not independent evidence that the vLLM TPU runner consumed those
IDs. The current first-diff diagnostic also writes at most one capsule per
trajectory-engine lifetime and then preserves the production fatal, which is
correct for training but cannot collect a useful population in one diagnostic
run.

## Non-negotiable split

- Production P45/M15 exact full training remains first-diff fatal. No warning,
  mask, drop, retry, or replacement path is added to training.
- Multi-diff collection is a separate rollout-only diagnostic with zero
  backward and zero optimizer commits. A mismatching trajectory is closed
  after its evidence is captured; the next independent trajectory may run.
- The diagnostic never claims Zero-TIM when it observes a diff. A completed
  evidence run may be mechanically healthy while its numerical/token verdict
  remains red.
- S0 finite A-B numerical capture is not enabled in the first engine-witness
  target. It remains a separate observer and later CL.

## Evidence layers

1. `submitted`: the exact continuation ledger and the token IDs given to the
   sampler.
2. `engine_echo`: `RequestOutput.prompt_token_ids`, joined by request ID. This
   proves the vLLM request object retained the submitted IDs but is not called
   a TPU-runner witness.
3. `runner_input`: a minimal diagnostic-only installed-engine observer hashes
   the prompt portion of `runner.input_batch.token_ids_cpu` and records the
   same request ID. This is the runner-consumption witness.

The classifier requires equal lengths and SHA256 across all three layers for
every witnessed request, with no missing, duplicate, or foreign request. The
sampler separately proves that each returned `RequestOutput.request_id` equals
the ID of the request future it submitted. TPU-runner capture order may differ
from submit order under asynchronous scheduling; runner records therefore join
by request ID and require their own record indices to be unique and contiguous.
Full token IDs exist only in mode-0600 bounded mismatch capsules. Collect-mode
stdout carries token-free receipts, never reversible token chunks.

## Flag contract

- Extend `CANON_P57_TOKEN_CONTINUITY_DEBUG` from the closed value
  `first-diff` to `first-diff|collect-64`.
- `first-diff` retains its current meaning and full-training admission.
- `collect-64` is diagnostic tier, default absent, and is legal only with
  `CANON_P57_TOKEN_CONTINUITY=exact`, exact P45 or M15 identity, rollout-only,
  no backward, no optimizer commit, and the registered diagnostic profile.
- The bound is fixed at 64 per process and is allocated before formatting or
  I/O under a process-wide lock. It is not a user-settable free integer.
- Neighboring workloads, legacy mode, full training, evaluation, Native/IS,
  DeepSWE, malformed values, and partial delivery are fatal.

## Phases and gates

### T9b-0 — frozen construction oracle

- Refactor the shared segment builder so reconstruction and capsule metadata
  use one ordered source of truth.
- For every later turn in a frozen real trajectory, prove that reconstruction
  equals the exact prefix used to build B/C: initial unpadded prompt plus the
  completed assistant/environment token stream.
- Prove first-turn provenance is the sampler-submitted prompt IDs rather than
  a later text re-encode. A one-token substitution and missing environment
  segment must fail.

Result: the shared typed segment ledger and production-shape three-turn
fixture pass 15/15 focused controls. The fixture proves the code-level B/C
construction and first-turn padding boundary. It is not described as a real
r7 trajectory replay: r7 logged only length/hash receipts and explicitly
disabled trajectory logging, so its raw per-turn token arrays do not exist in
the preserved evidence directory. A real-trajectory instance of this oracle
will be emitted by the first approved diagnostic target.

### T9b-1 — engine witness

- Preserve request IDs and RequestOutput prompt IDs through sampler and
  rollout outputs without changing generated tokens or logprobs.
- Add a minimal append-only runner journal in a new overlay patch. It reads
  host-owned input-batch metadata only; no device fetch or synchronization.
- Observer-neutrality compares witness off/on outputs bitwise. Missing,
  duplicate, wrong-request, wrong-length, wrong-hash, and ordinary serving
  leakage controls must fail.

Result: implemented and host/image construction-gated. The sampler rejects a
returned request whose ID differs from the corresponding submitted future; the
host witness preserves submitted and `RequestOutput` lengths/hashes; overlay
patch 38 observes the prompt slice of `runner.input_batch.token_ids_cpu` and
emits only its request ID, length, and SHA256. The installed-overlay execution
gate proves two A-path requests are captured from the real patched function,
B/rescore is excluded, files are mode 0600, and missing/duplicate/cap negatives
fail. This has not yet been exercised on a TPU runner.

### T9b-2 — bounded diagnostic collection

- On a `collect-64` mismatch, allocate one global slot, persist the capsule,
  print only a token-free receipt, mark only that trajectory as
  diagnostic-different, and stop that
  trajectory without raising through the orchestrator.
- Continue with independent trajectories until the registered carrier horizon
  ends. Emit totals for trajectories, equal, different, capsules, budget
  exhausted, emission failures, backward calls, and optimizer commits.
- Any evidence-emission failure is recorded and makes final classification
  fail. It must not cause an unbounded retry or duplicate capsule.

Result: implemented and host-gated. The fixed process-wide reservation bound
is 64. A red trajectory terminates with `TOKEN_CONTINUITY_DIFFERENT`, is
excluded from every loss/update path, and the rollout-only carrier continues
with independent trajectories. The final classifier checks scalar accounting,
zero backward/optimizer/checkpoint receipts, unchanged step counters,
contiguous per-trajectory turns, complete typed segment topology, and every
capsule field/hash. Mechanical execution health and the scientific
`EQUAL|DIFFERENT` token verdict are reported separately.

### T9b-3 — GCS durability

- A lightweight shell worker watches only atomically completed capsule and
  runner-journal artifacts, uploads each object no-clobber to a registered
  protected evidence root, verifies the remote hash, and writes an append-only
  receipt.
- Before the workload starts, the worker no-clobber uploads and independently
  downloads a non-sensitive probe. `90_run.sh` waits for its exact READY ACK;
  missing credentials, a dead worker, a malformed ACK, or a remote hash
  mismatch prevents rollout from starting.
- Normal postflight requests final synchronization and writes a manifest whose
  object count and hashes equal the local classifier input. Every uploaded tar
  is reopened and checked for the exact regular-file member set, mode-0600
  metadata, sizes, and SHA256 values. A retry after final-manifest upload reuses
  the same snapshot/final identity rather than creating a divergent successor.
- Abrupt-exit recovery is limited to complete artifacts already uploaded by the
  preceding periodic poll. A file killed before atomic rename, or created after
  the last successful 30-second poll, can be lost. Collect-mode logs contain no
  raw token chunks and are not a recovery channel. Upload/readback mismatch,
  duplicate object, symlink escape, tampered archive, partial manifest, and
  missing GCS credentials have negative controls. Raw token evidence is mode
  0600 and never committed.

Result: implemented and host-gated with a fake remote. Probe/readback, periodic
snapshot, finalization, identical retry, symlink-parent escape, payload-tamper,
and shell READY/final-ACK controls pass. No real GCS object has been written and
abrupt pod loss inside the documented poll window remains intentionally
unrecoverable.

### T9b-4/T9b-5 — admission

- Run focused tests, P57/V1, flag audit, syntax/diff checks, and the complete
  immutable-image gate.
- Then run one matched one-host witness-off/on carrier with zero backward and
  zero commits. Only a separate approval may allocate the host.
- Finally prepare, but do not launch without approval, independent P45 and M15
  DP8xTP8 rollout-only diagnostics. Their classifier must report three-way
  witness completeness and the bounded diff population separately from the
  token verdict.

Local result: focused classifier 6/6, GCS 4/4, renderer 3/3, P57 209/209,
V1 101/101, flag audit 420/420, Python/shell syntax, and diff hygiene pass.
The complete immutable-image gate passes on image
`sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
with terminal `V1_HP_EXACT_IMAGE_PASS` and
`frozenlake_tito_engine_witness=1 frozenlake_tito_collect64=1
frozenlake_tito_gcs=1`. One-host and DP8xTP8 execution remain unrun and require
separate user approval.

## Rollback

The offline oracle, engine witness, collection policy, and GCS return are
separate CL concerns. Reverting all T9b behavior restores the already-gated
T9a implementation: exact full remains fatal with one reconstructable
first-diff capsule and legacy remains the default. No published evidence or
failed artifact is deleted.

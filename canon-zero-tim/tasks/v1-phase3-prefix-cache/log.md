# Phase3 prefix-cache log

## 2026-08-23 — P3.1 carrier construction started

- Created an isolated worktree at `local/v1-phase3-apc@bc5d1141`; the older
  detached `tim_c2_wt` and its two user-owned dirty paths were not touched.
- Registered a default-off FrozenLake APC reader.  No engine arithmetic,
  cache writer/reader, RoPE implementation, B rescore reset, alignment policy,
  or production default has changed.
- Validation status: not yet run.  No TPU launch, evidence directory, commit,
  push, or external mutation exists for this phase.

## 2026-08-23 — APC-off one-host v5p control green

- User authorized the directly attached one-host v5p.  Preflight in the pinned
  image observed four `TPU v5` JAX devices; no training/JAX process or `p51_*`
  container was active.
- Run `p3c1_20260823T0157Z` used DP1xTP4, Qwen3-8B, APC off, three diagnostic
  rounds, and zero backward/optimizer commits.  Action rows were 643, 617, and
  908.  Both A-B and B-C differing-byte vectors were `[0,0,0]`; every
  `CANON_ALIGN_PRE` verdict was `PASS`.
- Fail-closed classification is `CONTROL_GREEN`.  Evidence is append-only at
  `/mnt/disks/tunix-data/logp_probe_1host/p3_apc_p3c1_20260823T0157Z_control/`;
  final raw SHA-256 is
  `3326afa2f9f17c6a97951abb7e96ff9db4b9226ddf9431e264f21c1794e5c70c`.
  `sha256sum -c SHA256SUMS` passed for all five recorded files.
- The workload's inner `docker_exit=42` is its pre-registered controlled exit;
  the runner and classifier exited 0.  This proves only the APC-off control
  carrier.  APC-on, cache hits, first-red localization, repair, and all
  certification gates remain unverified.

## 2026-08-23 — APC-on real-hit reproduction is inconclusive

- Run `p3r1_20260823T0203Z` changed only APC from 0 to 1 on the same one-host
  v5p vehicle.  The runtime marker, engine kwargs, and vLLM config all attested
  `enable_prefix_caching=True`; reported cache-hit rates reached 86.3%.
- Three rounds had action rows 565, 615, and 965.  A-B differing bytes were
  `[0,0,0]`, B-C differing bytes were `[0,0,0]`, and all three alignment
  verdicts were `PASS`.  Prefix ranges were 956...1467, 970...1499, and
  1010...1561; no action row reached the 1686-token diagnostic boundary.
- The fail-closed classifier exited 1 with status `INCONCLUSIVE` and error
  `APC-on did not reproduce an A-B byte difference`.  This is not a zero-TIM
  failure: the byte gate stayed green.  It is a P3.1 reproduction failure, so
  no seam observer or numerical fix was launched.
- Evidence remains append-only at
  `/mnt/disks/tunix-data/logp_probe_1host/p3_apc_p3r1_20260823T0203Z_repro/`.
  SHA-256: raw `d926c38b6696d1fa408f721849b39194395f29517eace1ddee847634faffa89f`,
  report `39052bd15d99334fe49c63f352b230ffd0e7d0622cbb3a356f410aa9e0ca7ab7`,
  classification
  `db0d90deabda0a67df004e0a2286d2ed4b6ffc744dce5ab5988d578e54e0c112`.
- Two live explanations are preserved without choosing between them: (1) the
  current canonical/vLLM stack already neutralizes the historical APC issue for
  this production shape; (2) the bounded FrozenLake vehicle is too shallow and
  missed the historical depth/partial-cache-block seam.  User direction is
  required before changing the reproduction geometry.

## 2026-08-23 — Deep-prefix control attempt 1 failed before numerics

- `p3bc1_20260823T0233Z` exited before tokenizer/model construction because the
  new boundary runner set `HF_HOME=/mnt/disks/tunix-data` instead of the pinned
  cache root `/mnt/disks/tunix-data/hf`.  Offline Transformers therefore could
  not resolve Qwen3-8B.  No engine, cache request, A/B value, backward, or
  optimizer commit ran; this is vehicle infrastructure, not a numerical gate
  result.
- The failed evidence directory and raw log remain append-only.  Raw SHA-256 is
  `e39737b60f9e441e55163f689f0e336567daaa52b7a457f116e8177da33555b5`.
- Repair is one runner-only path correction.  The retry must use a fresh label;
  no APC, RoPE, cache, B-rescore, model, input, or classifier semantics change.

## 2026-08-23 — Deep-prefix control attempt 2 failed before A/B

- `p3bc2_20260823T0236Z` passed tokenizer/model construction, engine startup,
  weight loading, and actor/engine weight attestation.  Its first cache-prime
  request then failed at `tunix/rl/rollout/vllm_rollout.py:534`: the pinned
  vLLM JAX backend rejects a per-request `SamplingParams.seed`.
- No boundary case, cached A value, full-reset B value, backward, or optimizer
  commit ran.  This is a diagnostic-vehicle API mismatch, not a zero-TIM/APC
  numerical result.  The failed raw log remains append-only at
  `/mnt/disks/tunix-data/logp_probe_1host/p3_apc_boundary_p3bc2_20260823T0236Z_control.raw.log`;
  SHA-256 is
  `baab07ebb6c99a8b53de4b1ea8aec4437dc22d89630a338bffc590653ad93616`.
- The probe scores fixed prompt/target tokens and ignores its one generated
  token, so per-request sampling RNG is outside the measured value.  Remove
  only that unsupported argument and retry under a fresh label; all fixed
  inputs, APC treatment, B full reset, and byte classifier remain unchanged.

## 2026-08-23 — Deep-prefix APC-off control green

- `p3bc3_20260823T0239Z` exercised fixed prefix lengths
  `1535,1536,1537,1685,1686,1687,1788,1792,2047,2048,2049` with 16 fixed
  target tokens on DP1xTP4 Qwen3-8B, APC off.  All 11 A-B comparisons had
  zero differing bytes/elements; all A and B cached-token counts were zero.
  Classification is `BOUNDARY_CONTROL_GREEN`.
- B attested `reset_prefix_cache=True` and zero cached tokens in every case.
  The run performed zero backward calls and zero optimizer commits.  This
  proves only the deep-prefix control carrier, not APC-on correctness or full
  Phase 3 certification.
- Evidence is append-only at
  `/mnt/disks/tunix-data/logp_probe_1host/p3_apc_boundary_p3bc3_20260823T0239Z_control/`.
  `sha256sum -c SHA256SUMS` passed 5/5; final raw SHA-256 is
  `9d381ace5001a8c6d1e6d819310f6cc9e79597f5cb28ee66e0f4c73765a40f72`.

## 2026-08-23 — Deep-prefix APC-on attempt did not enter the cache path

- `p3br1_20260823T0244Z` attested APC enabled in the reader, engine kwargs,
  and live engine config, but all 11 A requests returned
  `num_cached_tokens=0`; engine cache metrics also stayed at 0.0%.  A-B was
  zero bytes in all cases, but the fail-closed classifier correctly returned
  `INCONCLUSIVE`, not green.
- Read-only source localization explains the vehicle failure: pinned vLLM
  `/usr/local/lib/python3.12/site-packages/vllm/sampling_params.py:500-504`
  sets `skip_reading_prefix_cache=True` whenever `prompt_logprobs` is requested.
  The diagnostic A used `prompt_logprobs=0` to score pre-fixed target tokens,
  so it explicitly bypassed APC.  This is not an APC numerical result and not
  a first-red interval.
- Alternative provenance-loss explanation is contradicted by both the vLLM
  source contract and engine-wide 0.0% hit metrics: `RequestOutput` propagation
  is not the primary cause.  The production-congruent correction is to make A
  a decode request with sampled-token `logprobs=1` and no prompt logprobs, then
  full-reset B re-scores the exact 16 token IDs returned by A.  Forcing cache
  reads on a prompt-logprob request would violate the pinned upstream safety
  default and is not proposed.
- Evidence remains append-only at
  `/mnt/disks/tunix-data/logp_probe_1host/p3_apc_boundary_p3br1_20260823T0244Z_repro/`;
  `sha256sum -c SHA256SUMS` passed 5/5 and final raw SHA-256 is
  `ddde9c7302c2312d3c9ea8debb3a90b6aff305556862dece01acfb463abf6995`.
  Await user approval before changing the carrier and retrying with a new
  label.

## 2026-08-23 — Production-decode boundary carrier v2 frozen

- User approved the proposed carrier correction and allowed a commit, while
  explicitly prohibiting push.  No RoPE, attention, KV-cache, B-rescore reset,
  alignment policy, backward, optimizer, or default flag semantics changed.
- Probe schema v2 keeps the same 11 fixed prefixes, but A now performs the
  production-congruent operation: cache-readable decode with sampled-token
  `logprobs=1`, no prompt logprobs, `ignore_eos=True`, and exactly 16 returned
  tokens.  B full-reset re-scores those exact A-returned IDs.  The classifier
  rejects any A contract that requests prompt logprobs or skips cache reads.
- Host static gates are green: 16/16 tests, flag registry 324/324, Python/shell
  syntax, and `git diff --check`.  Because the source diff and A semantics
  changed, both APC-off control and APC-on treatment require fresh labels; v1
  evidence is preserved and cannot certify v2.

## 2026-08-23 — Production-decode v2 APC-off control green

- `p3bc4_20260823T0301Z` ran the frozen v2 vehicle on DP1xTP4 Qwen3-8B with
  APC off.  All 11 fixed-prefix cases returned exactly 16 A decode tokens;
  A-B had zero differing bytes/elements in every case, and A/B cached-token
  counts were all zero.  Classification is `BOUNDARY_CONTROL_GREEN`.
- The report attested sampled-token logprobs enabled, prompt logprobs absent,
  `skip_reading_prefix_cache=False`, B full reset, zero backward, and zero
  optimizer commits.  This releases only the matching v2 APC-on treatment.
- Evidence is append-only at
  `/mnt/disks/tunix-data/logp_probe_1host/p3_apc_boundary_p3bc4_20260823T0301Z_control/`.
  `sha256sum -c SHA256SUMS` passed 5/5; final raw SHA-256 is
  `d2340179e2f0a5426e9c37b8eab063ac6ef185c33049d7469cff4f97aac3f70c`.

## 2026-08-23 — Production-decode v2 APC-on is deep exact, no historical red

- `p3br2_20260823T0306Z` changed only APC from off to on against the matching
  v2 control.  All 11 A requests genuinely consumed cached prefixes:
  `1280,1280,1536,1536,1536,1536,1536,1536,1792,1792,2048` tokens.  Every B
  request attested full reset and zero cached tokens.
- A-B differing bytes/elements were zero in all 11 cases through prefix 2049;
  classification is `BOUNDARY_DEEP_EXACT_NO_RED`.  This is strong bounded
  evidence that current canonical APC decode is byte exact at the historical
  depth seams, but it does not satisfy P3.1's pre-registered historical-red /
  first-red requirement and is not G-A...G-E certification.
- Two explanations remain.  Most likely, the historical “APC-on corruption”
  label conflated APC with the older at-scale serving KV-read carrier; the P38
  red itself ran APC off.  Less likely, the corruption requires the historical
  vLLM revision, at-scale multi-request/multihost scheduling, or another
  topology condition absent from DP1xTP4.  Do not invent a RoPE/cache repair
  without a red.
- Evidence is append-only at
  `/mnt/disks/tunix-data/logp_probe_1host/p3_apc_boundary_p3br2_20260823T0306Z_repro/`.
  `sha256sum -c SHA256SUMS` passed 5/5; final raw SHA-256 is
  `135a5e58ad129e6b97b355255e1077526dc93df7e6a304135fb270e58663dcb5`.

## 2026-08-23 — Production-decode v2 deterministic repeat is bitwise exact

- After the user approved moving from bug hunting to release validation,
  clean commit `d1de4d52` reran the identical APC-on v2 vehicle as
  `p3br3_20260823T0327Z`.
- Its complete `boundary.report.json` is byte-for-byte identical to p3br2;
  both report SHA-256 values are
  `fa80dfd5f8be52017164169c7f0747a194fd90d0ef3b458ebdd1e1fb24fd7fec`.
  For all 11 cases, prefix, target token IDs/hash, input hash, A hash, B hash,
  A/B cached-token counts, and difference count match exactly.  A-B remains
  zero bytes in every case.
- New evidence is append-only at
  `/mnt/disks/tunix-data/logp_probe_1host/p3_apc_boundary_p3br3_20260823T0327Z_repro/`;
  `sha256sum -c SHA256SUMS` passed 5/5.  Final raw SHA-256 is
  `e04e6602fb660b99ad2c4327dcaf113e135735dbb0f7002682e085e8cbeee61d`.

## 2026-08-23 — Certification launch attempt 1 failed at hostname preflight

- Label `p3gbc1_20260823T0334Z` was rejected before evidence creation because
  `run_p3_apc_onehost.sh` still defaulted to the retired v4 hostname.  The
  authorized directly attached host is `t1v-n-4a77ebd0-w-0`; no container,
  engine, A/B/C values, or evidence path was created.
- This is an infrastructure preflight result, not a zero-TIM/APC numerical
  result.  The runner default was corrected at
  `scripts/run_p3_apc_onehost.sh:51`; the label remains retired and was not
  reused.

## 2026-08-23 — One-host multi-round G-B/G-C certification green

- The certification-purpose classifier at
  `scripts/classify_p3_alignment.py:96-102` requires a real APC hit, exactly
  three ordered rounds, and zero A-B bytes; the common loop at `:69-94` also
  requires every B-C comparison to be valid, finite, and zero.  This is a
  stricter release-validation branch of the original fail-closed reproducer,
  not a relaxation of the byte gate.
- Run `p3gbc2_20260823T0335Z` completed DP1xTP4 Qwen3-8B with action rows
  `565/615/965`, A-B bytes `[0,0,0]`, B-C bytes `[0,0,0]`, and a maximum
  measured prefix-cache hit rate of `86.3%`.  Classification is
  `GB_GC_CERTIFICATION_GREEN`; B remained full reset and the vehicle performed
  zero backward calls and zero optimizer commits.
- Evidence is append-only at
  `/mnt/disks/tunix-data/logp_probe_1host/p3_apc_p3gbc2_20260823T0335Z_cert/`.
  `sha256sum -c SHA256SUMS` passed 5/5.  SHA-256 values are classification
  `ca1a402c5f5c402169afd69d4365153c53a4bc5bfe859af523434762afb35fd8`,
  pre-alignment
  `014471735e12cbaaf28be6d8eb6128c163ed410fd76b9a4390740185fffccfe7`, and
  final raw `ae5a60c4bec4ba8a6d1f379d716f17f4cd8759f5b3547e74e10a3142734cec8d`.
- **Verified by** the persisted classification/report and checksum manifest.
  **Not verified because** dirty-page negative control, matched APC-off/on
  performance, XProf/Perfetto, and G-E at-scale canary remain outstanding.

## 2026-08-23 — Matched greedy APC performance pair is a one-host proxy KEEP

- Added default-off runner modes at `scripts/run_p3_apc_onehost.sh:12-13`
  that hold DP1xTP4, Qwen3-8B, three rounds, `max_concurrency=1`, data, seed,
  scheduler, canonical stack, B full reset, and zero updates fixed; both use
  greedy `temperature=0.0`, and APC is the only cross-arm treatment.  The
  fail-closed pair classifier at `scripts/compare_p3_perf_pair.py:110-210`
  requires equal per-round `N_action` plus six hashes (`tokens`, `action_mask`,
  `policy_version`, A, B, C) before reading timing.
- Attempt label `p3pc1_20260823T0356Z` was rejected before evidence/container
  creation by the execution sandbox's `sudo/no-new-privileges`; it is retired
  and is not a numerical result.  Formal control `p3pc2_20260823T0357Z` is
  `CONTROL_GREEN`; formal treatment `p3pa1_20260823T0406Z` is
  `GB_GC_CERTIFICATION_GREEN`, with APC hit rate up to `85.5%`.  Both are
  3/3 A-B=0 and B-C=0 and passed their checksum manifests 5/5.  Their frozen
  `source.diff` SHA-256 values are identical:
  `e412cd58e694a404ae37fa53d92aeece21b4865104a5860f860b216d0f01f154`.
- Pair attempt `p3pp1_20260823T0414Z` was rejected by sandbox write policy and
  created no output.  Formal comparison `p3pp2_20260823T0414Z` is
  `MATCHED_INPUTS / KEEP_ONEHOST_PROXY`: all six cross-arm hashes match in all
  three rounds, action rows are `248/616/952`, and rollout call counts are
  `4/10/10` in both arms.  Round 0 (retained but excluded from decision) is
  `31.241s -> 30.801s` (`+1.408%`).  Steady rounds are
  `21.183s -> 19.661s` (`+7.185%`) and `31.969s -> 30.338s` (`+5.102%`), for
  aggregate `53.152s -> 49.999s`, a `5.932%` speedup.  B-rescore is invariant
  within measurement noise (`0.885/1.119s` vs `0.884/1.109s`), as expected.
- Controlled-run wall time was `483s -> 482s`; model initialization and the
  one-time C compile dominate this metric, so it is recorded but not used for
  KEEP.  **Verified by**
  `/mnt/disks/tunix-data/logp_probe_1host/p3_apc_perf_pair_p3pp2_20260823T0414Z/comparison.json`,
  SHA-256
  `5248d0608517ca54c07e02a828afdba30685ad5e2b97a61f7ec32d708dcf97d7`.
  **Not verified because** this is a greedy one-host proxy, not XProf shape,
  dirty-page sensitivity, production stochastic long-run, or G-E at scale.

## 2026-08-23 — Matched diagnostic XProf and semantic Perfetto are green

- `scripts/run_p3_apc_onehost.sh:14-15,102-110` adds separate default-off
  `xprof-control` and `xprof-apc` modes.  Both capture only diagnostic round 1
  (`skip=1, steps=1`) with Python tracing disabled, export the official
  `tunix.perf` semantic timeline, execute zero backward/update, and retain the
  unchanged three-round A-B/B-C gate.  The learner boundary is implemented at
  `tunix/rl/agentic/agentic_rl_learner.py:2586,3603-3630`; the recipe export is
  wired at `examples/frozenlake/train_frozenlake_qwen3.py:1439-1450`.
- APC-off `p3xc1_20260823T0420Z` and APC-on
  `p3xa1_20260823T0433Z` are both `PROFILE_GREEN`.  All three rounds have the
  same N_action and all six cross-arm hashes (`tokens`, `action_mask`,
  `policy_version`, A, B, C); A-B/B-C are zero in both arms.  The frozen
  `source.diff` SHA-256 is identical in both runs:
  `cd64e7818277fd615e9ed790dfb1bb9e53bf22bc807657b8f350dd688524841a`.
- Control/treatment XPlane sizes and SHA-256 values are
  `1,737,428,217 / 919cda4d...` and `1,676,523,924 / de733de9...`;
  trace-json values are `43,084,785 / c23fbacf...` and
  `42,774,289 / 1a9cf5e7...`; semantic Perfetto values are
  `3,776 / 36f282bb...` and `3,776 / d9c12ad0...`.
- The persisted cross-arm summary at
  `/mnt/disks/tunix-data/logp_probe_1host/p3_apc_xprof_pair_p3xp2_20260823T0453Z/summary.json`
  (SHA-256 `307c2f4ae46d1f40886d4d307ffb15fbe3d0353ef40f29bcef9844147c63f7ab`)
  finds 112,394 representative TPU events and 100,834 host events in each arm.
  Main module counts are identical: `jit_run_model` 35/35,
  `jit_compute_and_gather` 34/34, and `jit_run_compute_logits` 34/34; principal
  HLO/op counts are also identical.  Structural protobuf parsing finds 92
  packets, 8 tracks, and 84 begin/end events in each semantic trace, with the
  same `rollout`, `environment`, and `advantage_computation` labels including
  step/group/pair tags.
- **Verified by** both profile classifiers, manifests, XProf census, and
  structural Perfetto parse.  The profile's short device/host span deltas are
  shape evidence only and do not replace the matched non-profile timing pair.
  The unchanged graph counts rule out a different compiled operator graph as
  the primary speedup mechanism; host/scheduler/prefill bookkeeping is the
  remaining inference.  **Not verified because** no finer semantic span yet
  localizes that host work and G-E/64-chip was not run.

## 2026-08-23 — G-D dirty-page negative control proves the gate has teeth

- The explicit default-off negative at
  `tunix/rl/rollout/vllm_rollout.py:481-694,697-809` is available only when
  `CANON_P3_APC_DIRTY_PAGE=1`, APC is on, and the gate-only carrier is active.
  It selects an actual cached prefix block under the idle in-process engine
  lock, replaces one layer-0 BF16 page, and records before/after hashes.  B is
  unchanged and still calls the full-reset prefill rescore.  The fail-closed
  classifier is at `scripts/classify_p3_boundary.py:40-188`.
- Formal run `p3gd1_20260823T0506Z` changed physical block 1, logical extent
  256, shape `[256,8,2,128]`: 1,046,314 page bytes / 524,288 BF16 elements
  changed.  Page SHA-256 changed from `4aae07b1...` to `30e14955...`.
- On the targeted prefix 1535, A consumed 1280 cached tokens and A-B became
  30 bytes / 13 elements red, first at token 0.  The remaining ten independently
  re-primed cases stayed A-B zero; every B attested `reset_prefix_cache=True`
  and zero cached tokens.  Classification is `DIRTY_PAGE_GATE_CAUGHT` with
  vector `[30,0,0,0,0,0,0,0,0,0,0]`.
- **Verified by** append-only evidence at
  `/mnt/disks/tunix-data/logp_probe_1host/p3_apc_boundary_p3gd1_20260823T0506Z_dirty/`;
  checksum manifest 5/5 passed.  Report/classification/raw SHA-256 values are
  `ff08495e...`, `5aa8f21c...`, and `86906d2c...`.  This intentional red is the
  pre-registered negative control, not a zero-TIM failure.  **Not verified
  because** G-E/64-chip and production stochastic long-run remain outstanding.

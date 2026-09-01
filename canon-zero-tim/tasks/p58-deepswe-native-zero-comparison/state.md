# State

## P58.31 K23 gradient-accumulation geometry repair (2026-09-01)

- K23 crossed P58.30 on the real 128-device DP8xTP8 plus DP8xTP8 target. The
  complete log proves 128 trajectories, 8 reward-one trajectories, 47 final
  nonzero advantages, 3 effective prompt groups, and strict A=B=C over
  396,233 action tokens. Six `MODEL_TIMEOUT` rows were compact-filtered. The
  immutable incident report's 393,135-token summary is stale; do not rewrite
  it or repeat it as the authoritative complete-log value.
- K23 emitted `[P59.DP8] gradient_reducer_ready dp_axis=dp dp_size=8`,
  completed group 1/16 across eight ranks, and reported exact replicas plus a
  finite nonzero gradient. It then failed before accumulator mutation with
  `segmented update accumulation changed: 8 != 16`. This closes P58.30's
  axis-identity scope and proves one real grouped pullback/reduction, not an
  optimizer update.
- Root cause was a launcher/learner geometry alias. P58 has 128 global
  trajectories, eight trajectories in each DP8 streamed group, and sixteen
  groups. The launcher used the sixteen local groups as the trajectory
  microbatch width, making `RLTrainingConfig` derive eight accumulation steps
  while the segmented trainer expected sixteen.
- `DeepSWEWorkload` now exposes `train_trajectory_micro_batch_size=dp_size`
  and `gradient_groups=local_trajectories`. Launcher and learner both consume
  that 128/8/16 contract and fail before backward unless derived accumulation
  steps equal registered groups. Required receipts are
  `[DEEPSWE.ACCUMULATION] PASS ... trajectory_micro_batch=8 ...
  gradient_groups=16 ... gradient_accumulation_steps=16` and
  `[CANON_P34_DP8] accumulator_contract_ready trajectories=128 micro=8
  groups=16 gradient_accumulation_steps=16`.
- P34 static passes ten suites; focused contract/script tests and the complete
  digest-pinned P58 image gate pass through `RLTrainingConfig`, agentic
  geometry, `PeftTrainer` precomputed accumulation, and first-update
  boundaries. Development replay `p58k23accumdev_20260901T004406Z` then
  passed the official direct-v5p Qwen3-4B DP1xTP4 classifier: strict A=B=C
  over 1,254 action tokens, finite/nonzero/repeat-exact gradients, device
  optimizer state, unchanged model/optimizer/accumulator/reference state, and
  zero commits. The cached profiled repeat took 12.565 seconds. This dirty-diff
  development result cannot certify DP8/TP8 accumulation.
- Source commit/push is explicitly approved for this delivery. No flag,
  recipe, optimizer placement, image publication, Kubernetes mutation,
  target launch, optimizer commit, checkpoint, or 1,000-update completion
  occurred in this local repair.
- Immutable incident package:
  `canon-zero-tim/evidence/p58_k23_gradient_accumulation_mismatch_incident/`.

## P58.30 K22 grouped-trainer axis identity repair (2026-08-31)

- K22 ran the 128-device disaggregated DeepSWE target far enough to cross the
  P58.29 lazy-scan failure and reach the post-pullback P59 reducer boundary.
  The immutable raw tail shows the reverse path reaching layer 0 before
  `FunctionalMappingError: P59 report and grouped trainer data axes differ`.
- The incident report additionally records 128 trajectories, 4 solved tasks,
  393,135 action tokens, exact A=B=C, and layers 35 through 0. Those earlier
  receipts are analysis-grade only in the committed package because
  `RAW_ERROR.log` contains just the final 100 lines rather than the complete
  run log. Do not promote them to independently reproducible evidence.
- Root cause was an identity mismatch, not gradient math: the DeepSWE branch
  kept the serving-facing adapter alias `data`, while the report adjoint
  correctly derived `dp` from the actual trainer state's `("dp", "tp")`
  `NamedSharding`. The safety check therefore compared two names for the same
  replicated trainer role and failed closed after backward work.
- Operator HEAD `110146c6f48e997fd426226333d2f39cb3486840` removes the P34
  special case and always derives the grouped reducer axis from trainer state.
  Local hardening makes that boundary explicit and adds forced-four-device
  regressions: stale adapter alias plus `dp/tp` resolves to `dp`, ordinary
  `data/model` remains `data`, and `fsdp/tp` is rejected.
- The repair changes no flag, model, data, sampler, loss, precision,
  optimizer, topology, deadline, TiTO, or Zero-HP program. Focused pinned-image
  tests pass 3/3; P34 static passes ten suites; the flag audit passes 409/409
  with `changed_names=0`; and the complete P58 image gate exits zero with
  `grouped_trainer_axis=3` and `P58_EXACT_IMAGE_CPU_PASS`. K23 later emitted
  reducer axis `dp` and completed group 1/16, closing this phase's axis scope.
- K23 did not complete the remaining 15 reductions, optimizer commit,
  checkpoint, or 1,000-update campaign. Its later accumulator-cadence failure
  is owned by active P58.31, so neither K23 nor this phase is a training PASS.
- Immutable incident package:
  `canon-zero-tim/evidence/p58_k22_data_axis_mismatch_incident/`.

## P58.29 K15 disaggregated scan execution-mesh repair (2026-08-31)

- K15 ran on 128 TPU v5p devices split into rollout 64 (DP8xTP8) and trainer 64 (DP8xTP8). The incident package's `DP32xTP4` prose is a stale label; raw lines 3–6 are authoritative. K15 completed all 128 multi-turn R2E trajectories (116 finished naturally, 12 max-turn, 0 timeouts), solved 3 SWE tasks in Step 0 (`Reward = 1.0`), generated 31 non-zero advantage samples (24.2%), and produced 407,262 action tokens.
- Rescore-B passed and strict pre-alignment passed 100% with exact A=B=C (0 differing bytes, 0 differing elements, hash `1ef8b0406cb2...`).
- Segmented backward crashed at `canonical_qwen3_adapter.py:8100` -> `run_layers_fwd_tape_scan:3687` -> `_p71_fwd_scan_fn` with `ValueError: Received incompatible devices for jitted computation` due to JIT tracing reading `linear._CANON_MESH` (serving mesh `[0, 4, 8, 12...]`) while arguments were on trainer execution mesh `[2, 3, 18, 19...]`.
- Root cause: four lazily created scan JITs invoked by `run_layers_fwd_tape_scan`, `run_layers_scan`, `run_layers_tape_scan`, and `run_layers_rev_scan` bypassed the execution-mesh scope used by eager segmented callables. The local repair applies the same trainer-mesh binding to all four and leaves colocated callables unchanged by identity.
- Local validation on unpublished parent `55553dfe0c3c895de81c66191e5082ed9ec41a32` passes the disjoint positive and colocated negative (2/2), P34 static (10 suites), the 409/409 flag audit, and the complete digest-pinned image gate with `disaggregated_scan_mesh=2` and `P58_EXACT_IMAGE_CPU_PASS`.
- No repaired target, backward, optimizer commit, checkpoint, commit/push, image publication, Kubernetes mutation, or TPU launch occurred. K16 remains separately gated and must cross the former scan trace, complete segmented reverse, produce finite nonzero gradients, and commit exactly the intended first optimizer transaction.
- Immutable incident package: `canon-zero-tim/evidence/p58_k15_disaggregated_mesh_scan_incident/`.

## P58.28 K11 prompt-only grouped-reverse repair (2026-08-30)

- K11 source `2f61f8fc7cf073964a9adbd30e78de872426a4d2` proves
  the P58.27 workload-interface repair on the real 128-device target. It
  completed 128 multi-turn trajectories, 427,594 action tokens, Rescore-B,
  and strict Step-0 pre-alignment with exact A=B=C. It then stopped at the
  first segmented reverse because one DP8 group contained three prompt-only
  environment-failure/timeout rows and `_p32_group_spec` required every rank
  to have at least one completion-valid token.
- Latest operator parent is
  `9f6b9c7eb6c32792604a966a7c0b8d9efa4072aa`. The local repair adds an
  explicit, default-false group-builder admission and enables it only under
  the validated P34 DeepSWE identity. Prompt-only rows retain empty action
  masks and therefore contribute zero loss and zero gradient; no fake token,
  dropping, resampling, or algorithm/topology change occurs.
- Pinned-image gates pass: the loss contract is 6/6; the grouped
  forward/reverse regression proves zero output/cotangent behavior; and the
  exact K11 DP8 length replay preserves the observed counts and 20 M256
  chunks. P34 static passes ten suites, the flag audit passes 409/409 with
  `changed_names=0`, and the complete P58 gate exits zero with
  `p34_empty_completion=2` and `P58_EXACT_IMAGE_CPU_PASS`.
- K11 remains a signed strict-alignment target PASS only through
  pre-backward. Segmented backward, optimizer commit, checkpoint, and the
  1,000-update campaign remain unproved. No commit, push, image publication,
  Kubernetes mutation, or TPU launch is authorized by this local repair.

## P58.27 K10 common workload identity repair (2026-08-30)

- K10 source `0e954153cdfd21ee79ebf57eaa6afb4bf273aff0` proves the
  P58.26/K09 startup repair on the real 128-device target. It completed 128
  multi-turn R2E trajectories, 404,028 action tokens, Rescore-B, and strict
  Step-0 pre-alignment with A-B and B-C both zero. It then stopped before the
  first segmented forward/backward because `DeepSWEWorkload` exposes
  `contract_name` while the shared DP adapter expects `.name`.
- Latest operator parent is
  `98d102eb27fe05fcee327688d0aa6d236b32be4a`. The local repair adds a
  read-only `name` property returning the existing `contract_name`; it does
  not add a dataclass/recipe field. All P34/P39/P43/P44/P46/P58 DeepSWE
  contracts therefore satisfy the common adapter identity, and the exact P58
  token-width path returns 4096/16384 rather than raising `AttributeError`.
- Host P34 static passes ten suites; DeepSWE contract passes 6/6; GSM8K
  renderer passes 12/12 in the pinned image; flag audit passes 409/409 with
  `changed_names=0`. The complete pinned P58 gate exits zero with
  `deepswe_workload_identity=1`, installed TP4/TP8 shims, and
  `P58_EXACT_IMAGE_CPU_PASS`.
- No flag, model, data, sampler, loss, precision, optimizer, topology,
  deadline, TiTO, or Zero-HP bundle changed. K10 is a signed strict-alignment
  target PASS only through pre-backward. The repair has not been rerun on the
  128-device target; segmented forward, backward, optimizer commit,
  checkpoint, and 1,000-step completion remain unproved.

## P58.26 K09 full-startup scope repair (2026-08-30)

- Latest operator tip `0d224e4a0e8c278f1bf9f699af235fdea83ef327`
  contains the immutable K09 incident but no runtime repair. The local repair
  was reconciled without conflict over both shared Qwen explicit-mesh
  resharding changes (`0b62b6bb` and `0d224e4a`). K09 source
  `0b62b6bbd3d9fa44268c7640047d4b60047cb4d5` passed TiTO admission, clean-data
  filtering (4,578 to 1,012), 128-device inventory, and rollout/trainer
  DP8xTP8 mesh construction, then failed before rollout with an unbound
  one-host-only `P58_Q4_TP4_TRAJECTORY_REPLAY` name.
- The local repair initializes that selector to `False` before the one-host
  admission block and additionally gates replay geometry on
  `ONEHOST_SMOKE`. Full training therefore cannot call or inherit one-host
  replay geometry; the admitted one-host replay path is unchanged.
- A new executable AST regression covers the exact K09 full-mode branch and
  the positive one-host branch. It also rejects any future uppercase selector
  assigned only inside the one-host block and loaded later without a top-level
  binding.
- Final validation on `0d224e4a0e8c278f1bf9f699af235fdea83ef327`:
  P34 static passes ten suites, focused P58 passes 49/49, the script contract
  passes 10/10, Python/diff hygiene passes, and the flag audit passes 409/409
  with `changed_names=0`. The complete digest-pinned image gate exits zero
  with `P58_EXACT_IMAGE_CPU_PASS ... regressions=1` on image
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`.
- No model/data/sampling/loss/precision/optimizer/mesh/timeout/TiTO/Zero-HP
  setting changed. K09 remains `INCONCLUSIVE_PRE_ROLLOUT`; no fresh target,
  backward, optimizer commit, checkpoint, or training-completion evidence
  exists. No image publication or Kubernetes/TPU mutation occurred.

## P58.25a default-full YAML TiTO admission and one-host proof (2026-08-30)

- The P58 JobSet renderer now places `CANON_P34_DEEPSWE=1` in the raw
  container environment for every Native/Native+IS/Zero stage, rather than
  relying only on the later profile source.  Every rendered JobSet also has
  `canon.zero-tim/token-transport=tito` provenance.  Missing/wrong provenance
  or a raw DeepSWE identity of `0` fails closed; the paired recipe signature
  includes both fields.
- This adds no new settable flag.  TiTO remains selected by the existing
  DeepSWE workload identity and is common to Native and Zero.  The full
  Zero-HP prepare marker now states `transport=token-in-token-out`.
- Focused renderer/sampler/one-host/wrapper tests pass 50/50; P34 static passes
  ten suites; syntax/diff hygiene and the 409/409 flag audit pass.  The
  digest-pinned complete gate on image ID
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
  ends with `P58_EXACT_IMAGE_CPU_PASS ... zero_hp_full=1 ... regressions=1`.
- Direct-v5p run `p58s25titoctl_20260830t0713z` is a real Qwen3-4B-Instruct
  DP1xTP4/R2E TiTO proof.  It emitted one admission plus 23 continuation
  receipts and returned `EXACT_TOKEN_CONTINUITY_ALIGNMENT_PASS` over 2,413
  action tokens with zero A-B and B-C differing elements.  KV fingerprints
  were equal.  Controlled exit 42 proves backward and optimizer commits were
  not reached.  Return bundle SHA-256 is
  `a68925aa95aaeddcdc9f3f0be625aa92418b221959e1ef11cdc8c7f0ebbbcb35`.
- The first development carrier `p58s25tito_20260830t0700z` reached the same
  exact pre-alignment but was intentionally stopped after it fell into an
  irrelevant long backward compile.  It is not acceptance evidence; only the
  controlled, classified `...titoctl...` run is admitted.
- The hardware evidence remains bound to source
  `18f29c56daf471cc0ac011396d7c7a09f35d695b` plus its recorded dirty diff.
  Publication preparation then preserved the local changes in a named stash,
  cleanly rebased onto exact operator parent
  `cd32949e9b63b927e99f3cfba724f4f5f6d03cda`, restored without conflict, and
  reviewed the intervening M15/GSM8K/FrozenLake commits.  The shared M15 exact
  token path remains mutually exclusive with DeepSWE and does not change the
  `CANON_P34_DEEPSWE=1` TiTO selector.  A final non-overlapping Qwen3 embedder
  sharding commit advanced the operator parent to
  `e89272d1d6c99b8f3c5014f0974b4fe57f2a4156`; the P58 changes were rebased
  again without conflict and the focused/P34/flag/exact-image gates passed on
  that parent.  Executors must use the final remote readback SHA containing
  this entry.  No image publication or Kubernetes mutation occurred.

## P58.25 DeepSWE token-in/token-out continuity (2026-08-30)

- DeepSWE continuation is now treated as TiTO for every admitted DeepSWE
  profile, including Native, Native+IS, Zero, Qwen3-32B, diagnostics, and
  one-host. It is a common transport invariant and no longer a TP4-Zero-only
  intervention.
- Later turns concatenate the rollout worker's actual initial prompt IDs,
  exact sampled assistant IDs, and once-tokenized R2E environment IDs. The
  integer prompt is sent with `apply_chat_template=False`; sampled assistant
  text is never re-tokenized.
- Native and Zero therefore differ only in their registered numerical /
  algorithm bundles, not token transport. Non-DeepSWE agentic workloads keep
  their previous path.
- Startup and each real continuation emit `[DEEPSWE.TITO]` receipts.
  P58 postflight requires exactly one admission receipt and, for ordinary
  training, at least one multi-turn continuation receipt.
- Host syntax, selector, one-host, sampler, and postflight contracts pass.
  The full renderer test is unavailable on the bare host because `metrax` is
  absent, but the digest-pinned complete image gate passes and emits
  `P58_EXACT_IMAGE_CPU_PASS ... regressions=1`; it observes a real focused
  `[DEEPSWE.TITO] CONTINUATION` receipt. P34 static passes ten suites and the
  flag audit passes 409/409. No target has run.
- User authorized this source commit and operator-branch push. The TiTO concern
  was rebased onto exact operator parent
  `509d3866b39228ce7df29d4eb3e5394591c69de0`. Its collector overlap with the
  upstream observer-only M15 token verifier was reconciled by sharing the
  strict reconstruction helper while keeping M15 observer-only and DeepSWE
  exact input separately admitted. Post-rebase focused, P34 static,
  flag-audit, and digest-pinned complete gates pass. Executors must use the
  final remote readback SHA that contains this phase. `main` remains untouched;
  no image publication, Kubernetes mutation, or TPU launch occurred.

## P58.24 K03 JobSet exclusive-topology repair (2026-08-30)

- Operator tip `ae1e92f7660eb0ad73b20b47b8a4d7703aaea57c` preserves the K03
  incident package. K03 reached Kueue admission and CPU-head startup, then
  `vpod.kb.io` rejected indexed worker followers because their Pod template
  lacked `cloud.google.com/gke-nodepool`. Nothing numerical ran; K03 is
  infrastructure `INCONCLUSIVE` with no resumable state.
- Root cause is annotation scope, not a missing hard-coded pool: the manifest
  put JobSet's exclusive-topology annotation on the worker Pod template.
  P58.24 moves it to `JobSet.metadata.annotations` and forbids the Pod-level
  copy. Kueue-managed sentinels again omit literal nodepool affinity; explicit
  real pools remain exact.
- Host construction passes: renderer and annotation-scope negatives,
  system-optimization workload tests, Bash/Python syntax, and a full CLI
  sentinel render with top-level exclusive topology. The digest-pinned
  complete gate also passes with
  `P58_EXACT_IMAGE_CPU_PASS ... system_optimization=1 ... regressions=1`.
- Model, data, B8xG16, DP8xTP8 roles, 1,000 updates, device optimizer, strict
  alignment, and the P59/P67/P63/P70/P71 system tuple are unchanged. No flag
  was added or modified. No image publication or target launch occurred.

## Publication checkpoint for P58.23 (2026-08-30)

- User approval for commit/push was granted for this delivery. Implementation
  commit `fb178803d53ff562cefdfdc8e7b3fac3563d9d6e` is rebased onto exact
  operator tip `4ce03fad6e10466acece308a3fe05b41af3825c2`; executors must use the
  final remote readback commit that contains it.
- The final digest-pinned gate passes with `system_optimization=1`,
  `trajectory_replay_b2g2=1`, `p59_tp4_tp8=2`, `m15_token=1`, and
  `regressions=1`. P34 static passes 10 suites, flag audit passes 408/408,
  and focused P58/M15 post-rebase gates pass.
- Upstream M15 runner patch 36 and P58 patch 37 compose into runner SHA-256
  `dae6dfa8a45bfd0a34b41baa9ec7c258229e8824c427a2fb863b620add074f98`.
  P58 is a single round-zero observer and does not require M15's diagnostic
  round file; M15 retains its strict explicit-round contract.
- Clean render-only verification resolves Qwen3-4B-Instruct-2507 Zero/full,
  DP8xTP8 per role, B8xG16, 1,000 updates, resident optimizer, P59 checked VMA,
  P67 serving scope, first-update gate, P63 stable clip,
  fingerprint-hybrid compare, first-group warmup, batched finite fetch, and
  P71 forward scan. DP collective reduce remains absent. Render artifact
  SHA-256 was
  `61b837dbc9915373c931eebfbbee0fc67c75f9726d7db3893b108c67eac1331c`;
  it was not applied.
- Publication does not certify DP8xTP8 performance, strict Zero-TIM, P59 target
  behavior, or optimizer commits. No image publication or cluster launch
  occurred.

## Completed P58.23 optimized B2xG2 one-host backward (2026-08-30)

- P58.22 real-R2E evidence remains valid: Qwen3-4B-Instruct-2507 DP1xTP4,
  continue-decode 8, prefix cache off, and strict A=B=C over 2,413 action
  tokens.  Its missing receipt was backward rather than rollout/alignment;
  P58.23 closes that local trainer receipt with immutable replay evidence.
- P58.23 replaces the hours-long serial compilation carrier with the current
  P28/P30/P71-forward optimized train path.  It uses global B2xG2 (four real
  trajectory rows), K=2,560, two mixed `[1,0]` groups, TPU-resident optimizer,
  and backward-no-commit.  `batch_size` and `mini_batch_size` are both 2;
  batch size one is forbidden for this replay.
- The deterministic combined replay source is
  `/mnt/disks/tunix-data/deepswe-replay-sources/p58-q4-b2g2-k2560-v2`.
  Manifest/journal SHA-256 are `482d7934a95207d0d77bb4857fbb200d7b367cbf437dda6585937b20909afa8f`
  and `091a9273c2067876fbee1996ee853e3c8e861352e307cd5fb94fea2563aec456`.
  It repeats the strict-exact Scrapy `[1,0]` pair as two physical groups.  This
  validates B=2 shape/math, not prompt diversity.  v1 is rejected and retained
  only as evidence because its Coverage group was historically alignment-red.
- Construction is green on base `07a427612bf34c1910436cecb3d4deafdaa71015`:
  exact-image installation matches 37/37, P34 static passes 10 suites, the
  flag audit passes 408/408, Python compilation passes, and the combined
  replay loader contract returns exactly B2xG2.  The operator ref later gained one
  unrelated APC status-document commit; it has not been merged into this
  dirty local worktree.
- The one-host cold bound is 1,800 seconds with cache namespace
  `/mnt/disks/tunix-data/jax-compilation-cache/p58-q4-tp4-systemopt-b2g2-k2560`.
  P59 remains off on DP1; no serial-reference run is allowed.
- Target `p58s23optb2g2g_20260830t0132z` is PASS.  It admitted four replay
  rows and 1,254 action tokens with byte-exact A=B=C, then ran the current
  P28/P30/P71-forward optimized backward as two trajectory microsteps.  Both
  warmup and profiled repeat produced gradient norm `8.544539451599121` on
  each microstep; the repeat was exact, finite, and nonzero.  The profiled
  repeat took 12.418 seconds (`forward=1.462`, `reverse=10.790`).
- Model, reference, optimizer, accumulator, and train step were unchanged;
  optimizer commits were zero and optimizer memory was device-resident.  Peak
  HBM was 56,370,843,648 bytes (52.5 GiB) per device at the observed maximum.
  The classifier outcome is
  `ZERO_TIM_RECORDED_TRAJECTORY_BACKWARD_NO_COMMIT_PASS`.
- Immutable artifact root:
  `/mnt/disks/tunix-data/deepswe-onehost-xprof/p58_zero-hp_p58s23optb2g2g_20260830t0132z`.
  Return bundle SHA-256 is
  `7d33ee791146d2309c16866d8e30f15f0f012e05e88f6c795b587938f973f795`.
- Final construction rerun passes the digest-pinned exact-image terminal
  (`trajectory_replay_b2g2=1`, `system_optimization=1`, `regressions=1`), P34
  static 10 suites, flag registry 408/408 with `FLAG_AUDIT_PASS`, focused
  replay/one-host 7/7, Python/Bash syntax, and `git diff --check`.
- P58.23 is complete. There is no active target launch; final remote readback
  and the later TP8 promotion remain separate gates. Commit/push is approved
  for this delivery, but image publication and every launch remain unapproved.

## Completed P58.22 Qwen3-4B continue-decode repair (2026-08-29)

- `p58s22kv9d_20260829t0846z` proves the repaired one-host forward path:
  continue-decode remains `8`, prefix cache remains off, and both strict
  boundaries are byte-exact over 2,413 real action tokens.  Its controlled
  alignment-only exit has no backward and no commits.
- `p58s22bw3_20260829t0931z` has one usable real trajectory, one official
  max-context/compact-filtered trajectory, and strict A=B=C over 2,413 action
  tokens.  Its 5,120-token backward compile exceeded 7,200 seconds; it is
  incomplete, not a numerical failure.  Three later 4,096-width clean-census
  candidates each clipped both fixed-seed rows and were rejected before
  backward by the new rollout-only carrier screen.
- The active short backward carrier is now 1,792 prompt plus 2,880 response
  tokens (train width 4,672), the minimal signed padding over the observed
  Pillow prompt/completion lengths 1,737/2,862.  Target
  `p58s22bw6_20260829T123125Z` adds a 21,600-second process bound and exact
  persistent JAX cache path solely for compile completion.  No backward PASS
  or TP8 promotion is claimed until its classifier succeeds.

- `p58s22kv6_20260829t0802z` ruled out an immediate KV comparison.  Live A
  and the durable trainer sequence encode identical tool-call text with
  different BPE tokenizations beginning at position 2242 (`97183` versus
  `28,1725`).  The exact-prefix observer correctly refused clean B.  This
  explains why prefix cache off was still RED: the fault is cross-turn token
  continuity before cache comparison, not cached-prefix reuse.
- The local, uncommitted repair keeps exact sampled/environment token IDs
  across later agent turns and passes them to vLLM without re-tokenizing.
  Scope is the existing P58 Qwen3-4B DP1xTP4 Zero-admission selector only; no
  new independent flag exists.  Focused agentic/sampler contracts and the
  continue-KV 8/8 overlay probe pass.  Full pinned-image and fresh real strict
  alignment evidence are the next gates.
- The real discriminator now atomically derives the existing P38
  precheck-only/controlled-exit pair.  A green run must exit 42 before
  backward and is alignment-only evidence, not backward admission.

- The one-host discriminator has passed its final construction rerun.  Runner patch
  35 and `CANON_P58_Q4_TP4_CONTINUE_KV_DIAGNOSTIC=1` select one bounded
  serial request in `[2280,3072)`, join live-A and clean-B integer KV
  fingerprints at the exact same token prefix, and classify cache write/state
  versus read/program without changing sampling or alignment.
- Four target attempts remain preserved and inconclusive: the first exposed
  P38 serving-capture coupling, the second exposed the 16-page inherited
  instrumentation bound, and `p58s22kv3_20260829t0647z` produced an A/B pair
  at target prefix 2270 but stopped on raw sharding-repr drift.  Read-only
  comparison found that pair's KV fingerprints equal, but 2270 precedes the
  actual first RED at 2286, so it cannot classify the defect.  The active
  repair records canonical device-to-slice sharding and moves the lower bound
  to 2280 so the next pair must span the RED seam.
- `p58s22kv4_20260829t071538z` selected the intended prefix 2285 and captured
  A through 2472, beyond first RED 2286, but wrote no B because the inherited
  B hook required full prompt completion.  Prompt-logprob rescore can stop one
  input token short.  P58 now captures the first exact-prefix clean chunk that
  covers `A.target_seq_len`; ordinary P38 remains unchanged.
- Focused static evidence is green: all 37 Qwen3-4B TP4 overlay files match
  the new runner digest, and the partial-rescore positive plus generic P38
  negative probe passes 8/8.  The complete pinned-image terminal marker has
  `continue_kv_observer=1` and `regressions=1`; a fresh real A/B pair remains
  pending.  This does not yet prove a target fingerprint or a numerical
  repair.

- P58.21 target `p58s21std_20260829t0357z` is a valid single-variable causal
  control.  With standard decode it admitted 2,553 action tokens and returned
  exact A-B and B-C: zero differing elements/bytes and `max_abs=0.0` on both
  boundaries.  This makes `CANON_CONTINUE_DECODE=8` a necessary cause of the
  P58.20 environment-seam RED on the one-host carrier.
- Immutable control artifact:
  `/mnt/disks/tunix-data/deepswe-onehost-xprof/p58_zero-hp_p58s21std_20260829t0357z`.
  The process hit its 7,200-second bound while compiling the first complete
  8,192-token backward, so `backward_no_commit` is absent and the classifier
  correctly reports `ZERO_TIM_BACKWARD_INCOMPLETE`.  This is not a backward
  PASS and not a backward numerical RED.
- P58.22 kept continue-decode value `8`,
  joins bounded integer KV fingerprints for the live A cache and clean B
  cache at an identical environment seam, repairs the identified cache/state
  path, then reruns strict alignment plus backward-no-commit.
- The backward rerun uses a separately attested direct-host-only 1792+2880
  sequence carrier.  The actual prompt/completion lengths are 1,737/2,862 and
  the historical first RED at logical prefix 2,286 remains covered.
  Production 16K/50-turn geometry and all TP8 profiles remain unchanged.
- No commit or push is authorized.  TP8 remains blocked until repaired
  continue-decode=8 passes one-host A=B=C and finite nonzero repeat-exact
  backward with zero commits.

## Completed P58.21 Qwen3-4B environment-seam discriminator (2026-08-29)

- P58.20 construction passed the complete digest-pinned image gate, including
  `qwen4b_tp4` 37/37 installation, five TP4 shape self-tests, and terminal
  `P58_EXACT_IMAGE_CPU_PASS`.
- Direct-v5p target `p58s20dev_20260829t0330z` then ran two real Pillow/R2E
  trajectories with the full seven-target overlay. It admitted 3,300 action
  tokens and returned exact `S_prefill=T_old` (0 elements/0 bytes), but
  `S_decode` differed in 1,307 elements/2,694 bytes. The first mismatch was
  the first action token after an environment result at logical prefix 2,286
  (`abs_delta=0.009761810302734375`); initial action tokens were exact and the
  +/-1 token shift controls were substantially worse. The strict gate stopped
  before backward; optimizer commits remained zero.
- Immutable development artifact:
  `/mnt/disks/tunix-data/deepswe-onehost-xprof/p58_zero-hp_p58s20dev_20260829t0330z`.
  The classifier outcome is `ZERO_TIM_ALIGNMENT_RED`; trajectory,
  pre-alignment, batch metrics, and manifest SHA-256 values are recorded in
  its classification output.
- P58.21 added one default-off matched-control
  selector whose sole numerical change is baseline `CANON_CONTINUE_DECODE=8`
  versus an empty value (`standard-decode`). All model/data/sampling/overlay/
  fixed-head/prefix-cache/alignment contracts stay identical. A green control
  does not certify the high-performance arm; it promotes a continue-decode
  cache repair and a fresh baseline=8 rerun.  The target was exact, so P58.22
  owns the bounded KV/state repair; no TP8 promotion occurred.
- No commit or push is authorized.

## P58.20 construction checkpoint (2026-08-29)

- User decision: one-host Qwen3-4B strict Zero-TIM is now a hard prerequisite
  for any further TP8 or 128-chip P58 work.  P58.19 is preserved but deferred;
  P58.20 was the only active phase at this checkpoint.
- Work proceeds in clean named worktree
  `/home/yuxuan/code_rl_repro/worktrees/p58_q4_tp4_onehost_0829`, branch
  `local/p58-q4-tp4-onehost-0829`, source base
  `16c224aa80eb6b3a544be19f693c0542ab4b0dcb`.  The older P58.19f dirty
  worktree is untouched.
- Confirmed construction gap: the existing one-host Zero-HP carrier installs
  only the topology-generic TPU runner.  The existing Qwen3-4B projection and
  fixed-head registrations are TP8-only, so its finite B-C RED is not evidence
  that Qwen3-4B cannot reach Zero-TIM on TP4.
- Active deliverable: an additive `qwen4b_tp4` model variant and single
  `CANON_P58_Q4_TP4_ZERO_ADMISSION=1` selector install all seven engine
  targets, retain explicit sampling `0.7/0/1.0`, and require strict byte-exact
  A=B=C before backward.  P59 and every production/TP8 selector stay off.
- Target exit gate: one direct-attached v5p-4 real-R2E G2 package with exact
  A=B=C over all valid action tokens, finite nonzero repeat-exact backward,
  TPU-resident unchanged optimizer/model state, zero commits, and a verified
  immutable artifact checksum.  Until that target passes, TP8 is not run.
- Publication status: no commit or push is authorized.  Direct one-host
  execution is authorized only after the construction ladder passes.

## Current P58.19 three-round coarse seam-localization checkpoint (2026-08-28)

- Current status: target incident `canon-p58-seamcoarse-full-p58s19d` (128 TPU
  v5p) is sealed in `evidence/p58s19d_byte_bound_incident/`.  On source base
  `af006872b64c2d6327588b4d4cef757242ddc222`, P58.19e is implemented locally
  but not committed or published: the selector now derives 4 GiB per round,
  and append-only runner patch 34 extends the existing monotonic observer
  budget reset from `m15-wide-v1` to exact `p58-seam-v1`.  Continue-decode
  bypass remains verified with 635+ records across `[1686,4096)`; the target
  has not been rerun.
- Target Incident (p58s19d): Attempt `canon-p58-seamcoarse-full-p58s19d` (128 TPU v5p)
  executed Step 0 rollout and tool actions up to Step 4, covering bands `[12, 15]`,
  before hitting `P38 seam evidence exceeded its registered output byte bound`.
- Sealed incident package (p58s19d): `evidence/p58s19d_byte_bound_incident/`
  with `RAW_ERROR.log`, `INCIDENT_REPORT.md`, and verified `SHA256SUMS`.
- Prior Incident (p58s19c): Attempt `canon-p58-seamcoarse-full-p58s19c` (128 TPU v5p)
  terminated due to `expected=standard actual=continue_decode`.
- Sealed incident package (p58s19c): `evidence/p58s19c_continue_decode_incident/`.
- Prior Incident (p58s19b): Attempt `canon-p58-seamcoarse-full-p58s19b` (128 TPU v5p)
  terminated with `records=0` under the old `[3072, 4608)` window.
- Sealed incident package (p58s19b): `evidence/p58s19b_seam_observer_contract_incident/`.
- Repair decision: keep signed `CANON_CONTINUE_DECODE=8`.  Exact
  `p58-seam-v1` admits continue-decode scheduler chronology, emits an explicit
  `tensor_capture=0` bypass receipt, and returns before incident/tensor
  payload construction.  Standard remains the only tensor-strata source;
  foreign profiles and unknown program paths fail closed.  Production and
  neighboring workload defaults are unchanged.
- P58.19d construction status: focused P58 suites pass 52/52, P34 static
  passes 10 suites, deterministic flag audit passes 394/394/394, syntax and
  diff hygiene pass, and the complete digest-pinned dependency-image gate
  emits `P58_CONTINUE_DECODE_OVERLAY_PASS cases=5
  tensor_capture=standard-only` plus `P58_EXACT_IMAGE_CPU_PASS ...
  continue_decode_observer=1 ... regressions=1`.  All 37 installed Qwen3-4B
  overlay files match MANIFEST.  No Pathways or TP8 target claim exists.
  Publication does not authorize an image build, Kubernetes mutation, or
  target rerun.
- P58.19e construction checkpoint: after reconciling upstream M15 patch 33
  with P58 patch 34, pinned-image overlay assembly installs all 37 Qwen3-4B
  files.  The dynamic probes prove P58/M15 per-round budget reset, M15 replay
  round provenance, round-jump rejection, and a foreign-profile no-op.  The
  complete digest-pinned suite exits zero with
  `P58_CONTINUE_DECODE_OVERLAY_PASS cases=5 tensor_capture=standard-only
  round_budget=p58+m15` followed by `P58_EXACT_IMAGE_CPU_PASS ...
  continue_decode_observer=1 ... m15_token=1 regressions=1`.  This remains
  construction evidence; the 128-chip target has not been rerun.
- P58.19c validation: Python compilation and diff hygiene pass; focused
  renderer/profile/classifier tests pass 45/45; P34 static passes 10 suites;
  deterministic flag audit passes declared/actual/unique 394/394/394; and the
  complete digest-pinned dependency-image gate exits zero with terminal
  `P58_EXACT_IMAGE_CPU_PASS ... coarse_seam=1 ... regressions=1`. The bare-host
  environment-contract module cannot import optional dependency `metrax`, so
  it is not claimed as a host PASS; that contract passes inside the pinned
  dependency image. These are construction results, not a Pathways/TP8 target
  result.
- Prior P58.19 implementation status:
  `f58a97748a8895835fba4944f5c5a34ba8bee352` is published on
  `yuxzhang/canon-zero-tim` and immediately read back at ahead/behind `0/0`.
- Reconciled target fact: sealed P58.18 `p58aba01` returned finite A-B RED and
  exact B-C in ON-A/OFF/ON-B, with controlled exit and zero backward/optimizer
  commits.  Its classifier decision is `CHECKED_VMA_NOT_SUFFICIENT`.
- Interpretation correction: the result proves only that checked-VMA is not a
  sufficient cause of the DeepSWE decode/prefill seam.  It does not prove the
  seam is independent of checked-VMA and does not authorize a P67 repair.
- Active deliverable: one default-off P58 selector prepares a single 128-chip
  DP8xTP8 rollout + DP8xTP8 trainer JobSet containing three sequential
  frozen-weight coarse layer-observer rounds.  Every round must be classified,
  sealed, uploaded, read-back verified, and acknowledged before the next.
- Signed carrier stays Qwen3-4B-Instruct-2507, clean 1,012 tasks, B8xG16,
  16K/50 turns, seed 42, concurrency 128, fixed lm-head, continue-decode 8,
  prefix cache off, strict B-C, resident optimizer but zero commits.
- Phase: `phases/p58-19-three-round-coarse-seam.md`.
- Validation: host renderer 30/30, profile 11/11, classifier 4/4, shared fake
  GCS persistence PASS (three-round collection plus missing-selector
  negative), deterministic flag audit 394/394/394, and the complete pinned
  dependency-image gate PASS.  Its terminal marker includes
  `checked_vma_aba=1 coarse_seam=1 ... regressions=1`.
- Exact-image recovery: the gate found that the fixed-head sub-check treated
  unset `CANON_KV_UNIFIED` differently from the global unset-is-zero contract;
  the check now uses the same default while still rejecting explicit `1`.
  The new flag also advanced the prefix-cache adjacency count from 393 to 394.

## Current P58.18 checked-VMA matched-triplicate checkpoint (2026-08-28)

- Status: ON-A/OFF/ON-B exact-geometry implementation and construction gates
  are complete; source publication is approved for this delivery. The remote
  executor must fetch and record the final operator-branch tip. Target not run;
  no image, Kubernetes object, or TPU work was created.
- Each independent JobSet keeps the signed Qwen3-4B-Instruct-2507 carrier:
  128 chips split into rollout DP8xTP8 and trainer DP8xTP8, clean 1,012 tasks,
  B8xG16, 16K, 50 turns, seed 42, concurrency 128, fixed lm-head,
  continue-decode 8, prefix cache off, strict Step-0 A/B/C, full trajectory
  journal, then controlled exit before VJP/backward/optimizer.
- Selector contract: `on` derives checked-VMA/P66/P67=`1/1/1`; `off` derives
  `0/0/0`. Both diagnostic arms force first-update/P63=`0/0` because backward
  is forbidden. Selector absent leaves production Zero-HP `1/1/1/1/1`
  unchanged.
- Parallel plan: three JobSets request 384 TPU chips, three anti-affined CPU
  heads, and aggregate sandbox concurrency 384 (768 requested CPU and 1,536
  GiB at the signed per-sandbox requests). Render PASS is not aggregate
  capacity or Kueue admission evidence.
- Interpretation: concurrent ON-A/OFF/ON-B is one matched OFF control plus two
  ON replicates, not a temporal ABA sandwich. Cross-run token identity is not
  required. Each arm must independently return exact B-C and valid finite or
  exact A-B evidence with zero training activity.
- Validation: Python/shell syntax; focused renderer 27/27, profile 9/9,
  per-arm classifier 7/7, and wave-contract 4/4; deterministic flag audit
  `393/393/393`; and the complete pinned dependency-image gate all pass. The
  terminal image marker includes `checked_vma_diagnostic=1
  checked_vma_aba=1 ... regressions=1`. This is construction evidence, not a
  Pathways/TP8 target result.
- Phase: `phases/p58-18-checked-vma-aba-wave.md`.

## Current p58z08 intake correction (2026-08-27)

- Status: analysis complete; P58.17 exact-geometry checked-VMA-off target is
  still NOT RUN. A fresh pull left the operator worktree at
  `5d4f2fceb6996bb0a5e2149a21c8fd846d89dcb5` with no newer target artifact.
- `p58z08` used source `395c0e0de8626c96e85457b997efddd2dd2dec48`
  and the ordinary `zero-hp-full` job identity. It did not contain the P58.17
  selector, diagnostic job name, precheck marker, controlled exit, or
  diagnostic classification. Checked VMA, the first-update gate, and P63 were
  enabled. Therefore it is not a failed checked-VMA-off arm.
- Rollout/data fact: 128 rows; 120 succeeded, five model-timeout, three
  context-limit; four solved trajectories; two effective prompt groups; 30
  admitted nonzero advantages. This is not an all-zero or sandbox-capacity
  failure.
- Numerical fact: `N_action=389067`; B-C (`S_prefill_vs_T_old`) is exact; A-B
  has 17,507 differing elements and 39,031 differing bytes. First delta is
  `0.02544403076171875` at an environment-to-action boundary; later maximum is
  `9.499740600585938`. The incident report incorrectly labels the byte count
  as token differences; preserve the report and use this correction.
- First failing boundary: strict pre-backward A-B gate. No VJP, backward,
  optimizer update, or checkpoint commit executed. The archived raw
  log/report lack a complete packaged run and classifier, so the evidence is
  analysis-grade.
- Next action: render and, only after separate approval, run the exact P58.17
  128-chip `zero-hp-vmaoff-precheck` discriminator. Exact A-B implicates the
  topology-shaped P67/VMA scope and triggers an explicit pullback-identity
  repair; finite-red A-B with exact B-C promotes seam replay. Do not launch
  another ordinary 1,000-update Zero-HP job before this gate.

## Current P58.17 decode-vs-prefill seam checkpoint (2026-08-27)

- Status: one-host diagnostic complete; exact-geometry checked-VMA-off
  discriminator implemented and host-tested, target not run. Implementation
  `b54bd81a26e418ef3ff32f34d25ae8d81d9ac3f9` is published on the operator
  branch and its first remote readback matched HEAD/FETCH_HEAD/tracking with
  ahead/behind `0/0`. No image publication or Kubernetes mutation occurred.
  One real direct-attached four-chip v5p carrier ran locally.
- Immutable target fact: `p58z07` ran source
  `ef46b0b3a5d8754160f0cce323ec3861b04dccdc` on disjoint rollout/trainer
  DP8xTP8 roles. It returned 121 `SUCCEEDED` and seven `MODEL_TIMEOUT` rows,
  six solved trajectories, and 31 admitted nonzero advantages. The
  P58.16 four placement/state/JIT/scorer contracts passed. The strict
  pre-backward gate then stopped before VJP/AdamW/checkpoint.
- Numerical fact: `N_action=379496`; `S_prefill_vs_T_old` is exact;
  `S_decode_vs_S_prefill` has 32,952 differing elements and 71,797 differing
  serialized bytes. The earlier log entry calling 71,797 a token count is
  corrected here. First mismatch absolute delta is `0.00435257`; maximum is
  `11.87498` later in the trajectory.
- Artifact join: all 1,024 reported mismatches exactly match durable token ID,
  action mask, and decode logprob. They map to trajectory rows 49 and 62, the
  same signed Pillow task. Shift-0 median absolute delta is `0.0040245`; -1/+1
  medians are about `0.4952/0.4922`, refuting a simple token offset.
- Implementation: `classify_decode_prefill_probe.py` fails closed on missing,
  non-finite, count-drifted, or unjoinable evidence. The default-off seam
  selector extends only the Zero-HP DP1xTP4 backward-no-commit carrier to the
  frozen task, G2, 4K response, 16 turns, serial scheduling, strict alignment,
  durable trajectory output, and automatic return bundle.
- Local result: `p58s17` returned two successful real R2E trajectories,
  `N_action=4808`, and diagnostic `PASS / FINITE_RED_REPRODUCED` with zero
  optimizer commits. A-B differs at 2,488 elements (`max_abs=1.3662147522`),
  and B-C differs at 988 elements. Shift 0 is decisively closer than shifts
  -1/+1, so a simple token displacement is rejected.
- Evidence: bundle
  `/mnt/disks/tunix-data/deepswe-onehost-xprof/p58_zero-hp_p58s17_20260827t1045z/P58_SEAM_PROBE_RETURN.tar.gz`
  has SHA-256
  `6285b5d2e8958ee85bd4b4190beaa240c7239ad6d07165a0948d7ba7f2b32eee`.
- Repair intake: the one-host trainer now uses topology-aware device order;
  the Zero path installs the generated runner as a real package overlay;
  alignment normalizes inactive `top_k/top_p=None`; and the final runner pins
  the frozen whitelist SHA in its manifest. TP8-only model/kernel overlays are
  excluded from TP4 by construction.
- Validation: focused probe/one-host tests pass 11/11; the pinned-image
  alignment suite passes 43/43; and the complete P58 exact-image gate ends in
  `P58_EXACT_IMAGE_CPU_PASS ... onehost_xprof=1 ...
  disaggregated_trainer_mesh=4 ... regressions=1`.
- Claim boundary: this carrier does not match `p58z07` exactly because local
  B-C is also RED. It is not forced-token replay and cannot certify
  TP8, DP8, disaggregated Pathways, backward, optimizer correctness, or
  convergence. The next decisive experiment is an admitted, Step-0/no-commit
  checked-VMA-off selector on the exact DP8xTP8+DP8xTP8 carrier; do not
  improvise the selector against the production full profile.
- Exact-geometry implementation: renderer flag
  `--checked-vma-off-diagnostic` creates a 128-chip Zero/full HP carrier and
  sets the single selector `CANON_P58_CHECKED_VMA_DIAGNOSTIC=off`. The profile,
  `00_env.sh`, authoritative `env.sh` reload, Python contract, runtime marker,
  controlled-exit postflight, durable classifier, flag registry, and
  render-only preparation wrapper all fail closed. The selector derives
  checked VMA/P66 alias/P67/first-update gate/P63 to zero and preserves every
  other signed P58 Zero-HP serving/data/geometry field. Production default is
  unchanged when absent.
- Target evidence contract: 128 durable rows; finite positive action count;
  exact B-C; either exact or finite-red A-B classification; exactly one Step-0
  precheck and code-42 exit; no fixed-head VJP, P59/P66 backward, global step,
  or optimizer update. The full run root is returned. Target remains NOT RUN.
- Phase: `phases/p58-17-decode-prefill-seam-probe.md`.

## Historical P58.16 pre-target NNX loader-metadata checkpoint (2026-08-27)

- Status: implementation commit
  `dba5211ac4945fefb50337603c800d9f8e3d37b5` is published and read back on
  `yuxzhang/canon-zero-tim`; pinned dependency-image CPU gates PASS. No
  matching image was published and no 128-TPU retry has run. `main` is
  untouched.
- Source intake: the clean isolated worktree fast-forwarded from
  `a04b65febcb5e163bf1f30bf33065decbe29651f` through five remote commits to
  `959a3258fe70230c483cec9a25b191b7b3d4ab4b`. The incoming P58 artifact is
  the 7,958-line `p58z06` raw log; the other incoming runtime changes belong
  to P68/DP-collective and M15 concerns and do not repair this failure.
- Immutable target fact: `p58z06` loaded the exact 1,012-task clean list,
  admitted 128 TPU devices with disjoint DP8xTP8 rollout/trainer roles, and
  completed vLLM model warmup. It then failed during adapter initialization
  after the first placement receipt, before rollout. No trajectory, trainer
  logprob, alignment, backward, AdamW, commit, or checkpoint exists. The raw
  log lacks a source SHA; none is inferred. Evidence checksums pass under
  `evidence/p58z06_nnx_loader_metadata_error/`.
- Root cause: Pathways dummy loading adds `_is_loaded=True` to every one of
  the 398 live NNX parameters. Flax includes variable metadata in State
  treedefs, while the weight-free trainer `nnx.eval_shape` reconstruction has
  no loader marker. P58.15 compared raw treedefs and rejected this provenance
  difference; segmented backward contained the same latent comparison.
- Repair: compare logical NNX treedefs after removing only exact
  `_is_loaded=True` from copied Variables. False/non-boolean values fail;
  every other metadata field, path/type, leaf count, shape, and dtype remains
  exact. The same check guards segmented backward. The full classifier now
  requires exactly one 398-leaf state-contract receipt.
- Validation: Python compilation and diff hygiene PASS; forced four-CPU-device
  nested-JIT forward/backward, segmented pullback, partial-overlap negative,
  and false-marker negative PASS; Zero-HP classifier 7/7 PASS. The complete
  pinned-image gate exits zero with `P58_EXACT_IMAGE_CPU_PASS ...
  disaggregated_trainer_mesh=4 ... regressions=1`. No TPU is exposed locally,
  so this is construction evidence only.
- Next action: source publication is complete. Matching-image publication,
  Kubernetes apply, and target launch remain separately approval-gated. Once
  the matching image is published/read back and sandbox admission passes, use
  fresh `p58z07`; never resume/overwrite `p58z01`-`p58z06`. Require four
  placement/state/JIT/scorer receipts, trainer old/current logps, strict
  A=B=C, finite nonzero 16-group backward, and one coherent update-0 commit.
- Phase: `phases/p58-16-nnx-loader-metadata.md`.

## Current P58.15 nested-JIT trainer-mesh checkpoint (2026-08-26)

- Status: implementation commit
  `f60cdd569c2737df6cb2968125c8e42680938981` published and
  dependency-image CPU gates PASS; 128-TPU retry not run.
- Source intake: `git pull --ff-only` first advanced the isolated P58 worktree
  to `a36cbd1b156e013a75af4071e91a238be49bc95b`, then reconciled the local
  repair over `98a2dfd9e8ece301374fdfb55518b3bc9ebef4d4`. Immediately before
  publication the remote advanced again to exact base
  `be758e68faa9db5b06be153a0656c4c861e3119f`; that incoming commit contains
  M15 evidence only. Neither intervening commit overlaps P58. The repair was
  rebased, revalidated, committed as the exact implementation SHA above, and
  published only to the operator branch; `main` is untouched.
- Immutable target fact: `p58z04`, built from
  `3f159250c4781b3faafde238f768457a0478446b`, emitted both P58.14 placement
  receipts and completed all eight prompt groups / 128 Step-0 trajectories in
  1,709 seconds. Eight `MODEL_TIMEOUT` and one
  `MAX_CONTEXT_LIMIT_REACHED` rows were compact statuses. The first hard
  failure was the first trainer old-policy-logprob call: trainer state occupied
  one 64-device role while a `jit inside jit` still named the disjoint rollout
  role. No trainer logprob, alignment, backward, AdamW, commit, or checkpoint
  completed. Evidence checksum passes.
- Root cause: P58.14 rebound explicit adapter shardings but reused vLLM
  `model_fn` and `compute_logits_fn`. Both nested JITs were created at engine
  initialization with rollout-mesh output shardings. Its prior CPU mock used
  plain Python functions and therefore missed this captured-device closure.
- Repair: disaggregated trainer execution reconstructs the same live NNX graph
  weight-free with `nnx.eval_shape`, validates exact state tree/shape/dtype,
  and rebuilds both nested JITs on the trainer execution mesh. The segmented
  trainer forward/backward uses the same reconstructed graph. The fixed-AR
  mesh global is rebound only inside a locked trace context and restored.
  Native, colocated, serving, sampling, loss, strict A=B=C, optimizer, and the
  signed B8xG16 recipe are unchanged.
- Validation: forced 2-rollout + 2-trainer CPU tests execute the real
  nested-JIT `value_and_grad` and segmented layer pullback with finite nonzero
  gradients; partial overlap still fails closed. The complete pinned-image
  gate exits zero with `P58_EXACT_IMAGE_CPU_PASS ...
  disaggregated_trainer_mesh=4 ... regressions=1`. No `/dev/vfio` is present,
  so this is construction evidence, not target proof.
- Next action: fetch/read back the final operator tip and prove it contains
  implementation `f60cdd569c2737df6cb2968125c8e42680938981`, build/read back
  a matching image, rerun the complete
  gate, pass sandbox admission, obtain separate launch approval, and render
  fresh `p58z05`. Require exactly the original two placement receipts plus
  `trainer model callables rebuilt ... mesh_bound_jits=2`, then trainer
  old/current logps, strict A=B=C, finite nonzero 16-group backward, and one
  coherent update-0 transaction. Never resume/overwrite `p58z01`-`p58z04`.
- Evidence/phase: `evidence/p58z04_disaggregated_mesh_error/` and
  `phases/p58-15-nested-jit-trainer-mesh.md`.

## Current P58.14 disaggregated trainer-mesh checkpoint (2026-08-26)

- Status: source implementation
  `dce0e93777548b7623e4f41702144f8d00f242f5` published and dependency-image
  CPU gate PASS; no 128-TPU retry has run.
- Source intake: clean worktree
  `/home/yuxuan/code_rl_repro/worktrees/p58_fixed_seed_0824`, branch
  `local/p58-fixed-seed-0824`, initially pulled at
  `3820b168457830112e6ce4b505fcedc9691bd705` and finally reconciled to exact
  operator tip `bde8f4c6e055ff077b24af716857786ce967f422`, then publication-time
  tip `9ae21d22c2c096d4c2b39724b40e87768ece8934`. The intervening
  FrozenLake source and M15 evidence commits did not overlap the P58 repair.
  `main` is untouched.
- Immutable target fact: `p58z03`, built from
  `8eb65480d3705d96ab282799ad5a6c1901596248`, returned all 128 Step-0
  trajectories and proved the P58.13 fixed-head M=`2048/256` admission. It
  then failed while compiling the first canonical trainer old-policy-logprob
  forward because trainer-state arrays and adapter sharding constraints named
  disjoint 64-device role sets. It did not complete trainer logprobs,
  alignment, forward execution, backward, AdamW, a commit, or a checkpoint.
  Earlier Pallas/VJP markers were tracing receipts, not execution proof.
- Root cause: the canonical adapter was constructed from the rollout runner
  only, so its differentiable input/cache/output/sample constraints and
  mesh-bound log-softmax scorer retained rollout devices while consuming
  trainer-resident state.
- Repair: adapter registration now supplies trainer state; the adapter derives
  the exact trainer `dp,tp` mesh, preserves the engine axis topology on those
  devices, and binds only the differentiable trainer forward there. Serving
  remains rollout-bound. Disaggregated serving/trainer scorers are separate
  mesh-bound instances from the same factory/math. DP/TP mismatch and partial
  device overlap fail closed; colocated and Native paths retain their prior
  behavior.
- Validation: Python compilation and diff hygiene pass. A forced four-CPU
  disaggregated `jax.jit(value_and_grad)` executes with finite primal,
  finite/nonzero gradient, and a partial-overlap negative. Existing colocated
  adapter regressions pass. The complete local dependency-image gate exits
  zero with `P58_EXACT_IMAGE_CPU_PASS ... disaggregated_trainer_mesh=3 ...
  regressions=1`. The image has no `/dev/vfio`, so no TPU/Pathways claim is
  made. An unrelated pulled stale flag-count assertion was corrected from 385
  to the authoritative 386; its 31-test suite passes.
- Next action: fetch the final operator tip and prove it contains implementation
  commit `dce0e93777548b7623e4f41702144f8d00f242f5`. Then build/pin a matching
  image, rerun the complete gate, pass sandbox admission, obtain separate
  launch approval, and render fresh `p58z04`.
  Require the two `[CANON_ADAPTER.PLACEMENT]` receipts, completed trainer
  logprobs, strict A=B=C, finite nonzero backward, and the coherent update-0
  transaction before continuing the same 1,000-update job. Never resume or
  overwrite `p58z01` through `p58z03`.
- Evidence/phase: `evidence/p58z03_device_sharding_error/` and
  `phases/p58-14-device-sharding-mismatch.md`.

## Current P58.13 Qwen3-4B M2048/P59-only VMA checkpoint (2026-08-26)

- Status: completed source repair; implementation published and pinned-image
  construction PASS. Target `p58z03` proved M=2,048 admission and exposed the
  P58.14 trainer-mesh bug before trainer execution.
- Source: worktree
  `/home/yuxuan/code_rl_repro/worktrees/p58_fixed_seed_0824`, branch
  `local/p58-fixed-seed-0824`. Implementation commit
  `bea1aabde39c43c13ca4eaefab989301c6e8b46c` is published on
  `yuxzhang/canon-zero-tim`, rebased over the latest FrozenLake P67 full-run
  promotion `c73c9a6c3676c9a1ba27e9b871b0f2e14ff6adb4`, and exact
  local/FETCH_HEAD/remote-tracking readback matched at `0/0`. `main` is
  untouched.
- Immutable target fact: `p58z02` proved the P58.12 global JAX seed route and
  returned all 128 Step-0 rows in one 1,514.2-second wave. It contained one
  `MODEL_TIMEOUT` and two `MAX_CONTEXT_LIMIT_REACHED` rows; those were compact
  statuses, not the crash. The first hard failure was later in trainer
  canonical per-token-logprob forward, before alignment completion, backward,
  AdamW, or any optimizer commit.
- Root error: Qwen3-4B TP8 produced caller-global fixed-head semantic M=2,048
  (`data=8`, local M=256), but the fixed-head registry admitted M=2,048 only
  for Qwen3-8B TP8. Qwen3-4B `(hidden=2560,tp=8)` fell back to learner
  M=`(4096,)` and rejected shape `(2048,2560)`.
- Fixed-head repair: register learner M `(2048,4096)` only for exact Qwen3-4B
  TP8 and retain the existing exact Qwen3-8B TP8 registration. Every other
  geometry retains `(4096,)`; Qwen3-32B TP8 remains a negative for M=2,048.
- Shared FrozenLake repair: Wave 5 proved strict A-B/B-C `0/0` with
  checked-VMA scoped to P59 backward. Exact P58 Zero-HP now derives
  `CANON_P67_P66_VMA_P59_ONLY=1` together with checked-VMA. Native raw,
  Native+IS, non-HP Zero, Qwen3-32B, and unrelated profiles remain off. This
  preserves the serving graph; it does not relax the strict A=B=C gate.
- Validation: 50/50 focused host tests, P34 static 10 suites, P57 146/146,
  and the flag-registry regression pass. The Qwen3-4B installed overlay
  matches 37/37 files and reports `learner_M=2048,4096`; the independent
  Qwen3-32B image gate reports `learner_M=4096`. The complete pinned image
  exits zero with `P58_EXACT_IMAGE_CPU_PASS ... qwen4b_fixed_head=1
  checked_vma=1 vma_p59_only=1 first_update=1 ... regressions=1`. No
  `/dev/vfio` is visible, so no TPU target is claimed.
- Historical transition: the matching target was rendered as fresh `p58z03`.
  Preserve it as immutable P58.14 trigger evidence; never resume or overwrite
  `p58z01`, `p58z02`, or `p58z03`.
- Evidence/phase: `evidence/p58z02_backward_fixed_lm_head_error/` and
  `phases/p58-13-backward-fixed-lm-head-m2048.md`.

## Current P58.12 JAX engine-seed/cleanup checkpoint (2026-08-26)

- Status: completed source repair; `p58z02` target proved the engine-global
  seed route and exposed the later P58.13 trainer-logprob fixed-head failure.
- Source: worktree
  `/home/yuxuan/code_rl_repro/worktrees/p58_fixed_seed_0824`, branch
  `local/p58-fixed-seed-0824`. The repair was built on exact pulled base
  `7f6fc071082f291bf926b1c5bc79021733628c2e`; implementation commit
  `c10fbe0487d1f6635975b84806f1efdce6bc95c1` was pushed only to
  `yuxzhang/canon-zero-tim` and immediate local/FETCH_HEAD/remote-tracking
  readback matched with ahead/behind `0/0`. The published history preserves
  immutable `p58z01` Attempt-0 evidence. `main` is untouched.
- Last verified target fact: `p58z01` admitted 128 TPU devices, the exact
  DP8xTP8 roles, 1,012 clean tasks, 128 R2E sandboxes, and a compiled vLLM
  engine. The first Step-0 generation failed because P58.10 routed seed 42 to
  JAX `SamplingParams.seed`, which is unsupported. Abort cleanup then hit the
  kubernetes-client empty-body `None.decode` defect. No trajectory, backward,
  optimizer commit, or resumable checkpoint exists.
- Repair: seed 42 remains common to dataset and rollout, but JAX receives it
  through global `EngineArgs.seed`; per-request JAX seed now fails early rather
  than being silently dropped. Runtime/postflight require the exact
  `[P58.SEED] ... scope=engine-global` and `[VLLM.JAX_SEED] ...` receipts.
  W&B/manifests/classifiers use the same bounded scope. R2E cleanup handles
  only the exact empty-body decode defect, confirms 404 within the existing
  deadline, and retries the exact Pod DELETE when its first outcome is
  ambiguous; unrelated errors remain fatal.
- Frozen workload: Qwen3-4B-Instruct-2507, clean 1,012 tasks, B8 x G16,
  16K/50 turns, rollout DP8 x TP8 plus trainer DP8 x TP8, 128 chips,
  TPU-resident optimizer, strict A=B=C, and 1,000 commits. No numerical or
  algorithmic flag changed.
- Validation: syntax/compile/diff hygiene pass; focused sampler 7/7, one-host
  artifact 5/5, and bounded cleanup regression pass; P34 static emits
  `suites=10`; P57 passes 146/146; flag audit passes 385/385. The complete
  pinned image exits zero with `P58_EXACT_IMAGE_CPU_PASS ... zero_hp_full=1
  checked_vma=1 first_update=1 stable_clip=1 ... regressions=1`. It reports no
  `/dev/vfio`, so no TPU target is claimed.
- Latest-tip reconciliation: the operator branch advanced during validation
  by three P4.10/P66 commits. The local branch fast-forwarded to
  `7f6fc071082f291bf926b1c5bc79021733628c2e`; shared `90_run.sh` retained both
  the new FrozenLake diagnostics and the P58 seed receipts without conflict.
  Final gates are rerun on that exact tip.
- Next action: the execution agent fetches the final operator tip, proves it
  contains implementation commit `c10fbe0487d1f6635975b84806f1efdce6bc95c1`,
  and builds/pins the matching image. Image publication, Kubernetes
  application, and a fresh `p58z02` target each require their separate
  approval. Never resume or overwrite `p58z01`.
- Phase: `phases/p58-12-jax-engine-seed-cleanup.md`.

## Current P58.11 checked-VMA Zero-HP checkpoint (2026-08-26)

- Status: implementation published and construction gates passed; target
  Attempt 0 (`p58z01`) reached vLLM initialization but failed before first
  generation on the P58.10 JAX per-request seed route. Superseded as the
  active repair by P58.12; its numerical gates remain the retry contract.
- Source intake: clean worktree
  `/home/yuxuan/code_rl_repro/worktrees/p58_fixed_seed_0824`, branch
  `local/p58-fixed-seed-0824`, constructed at fetched operator tip
  `644beb38cee2388862941019269ad264a581064f`, then fast-forwarded without
  overlap over V1-only evidence tip
  `4003f61cabb6f2d5e43d4c217cebb4dca2c3d217` before publication.
- Objective: admit the shared P66 checked-VMA P59 backward repair,
  `CANON_V1_HP_FIRST_UPDATE_GATE`, and P63 overflow-safe global-norm clipping
  into exactly the strict P58 Qwen3-4B Zero-HP full profile. Native raw,
  Native+IS, ordinary Zero, P44/P46, and diagnostics remain unchanged.
- Shape correction: P58 has eight outer 16-trajectory chunks but sixteen
  rank-major DP8 gradient groups. The real accumulator denominator and
  first-update microstep count are 16. Global canonical M is 2,048 and the
  shard-local/kernel M is 256.
- Frozen recipe: Qwen3-4B-Instruct-2507, clean 1,012 tasks, B8 x G16,
  16K/50 turns, rollout DP8 x TP8 plus trainer DP8 x TP8, 128 chips,
  optimizer resident, prefix cache/APC/sampler IS/group filtering off, strict
  A=B=C, and exactly 1,000 committed updates.
- Implementation: the exact Zero-HP profile derives checked-VMA, the internal
  P66 compatibility alias, the first-update gate, and P63 max-norm 1.0. The
  runtime uses `contract_name` as DeepSWE workload identity, executes the
  16-group precommit/commit gate, persists P63 evidence, and exports stable
  norm, naive-norm-finite, fallback, and clip-factor W&B metrics. Postflight
  requires exactly 1,000 P63 commit receipts, P59/checked-VMA receipts for
  every ordered attempt, and exactly two update-0 first-update receipts. A
  legal all-compact attempt is reconciled as zero-commit and removed from
  committed-step timing. Native, Native+IS, and non-HP Zero remain isolated.
- Additional commit-boundary repair: P58 intentionally carries the shared P33
  launch-admission bit, but `DeepSWEWorkload` has `contract_name`, not `name`.
  The old unconditional schedule check would therefore fail at the first real
  P58 optimizer commit. Schedule identity now uses the normalized workload
  identity; a real CPU P58 16-group/0-to-1 commit regression covers it.
- Validation: syntax, Python compilation, and diff hygiene pass. Focused
  profile 7/7, classifier 5/5, first-update 6/6, stable-clip source 3/3,
  exact-image environment 12/12, P63 validator/commit 10/10, and P58 real CPU
  first-commit integration 1/1 pass. Adjacent P34 emits
  `P34_STATIC_PASS suites=10`; P59 host 37/37, V1 Phase4 76/76, and P57
  146/146 pass. Flag audit is 383/383 with `FLAG_AUDIT_PASS`. Complete pinned
  image `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
  exits zero with `P58_EXACT_IMAGE_CPU_PASS ... zero_hp_full=1 checked_vma=1
  first_update=1 stable_clip=1 ... regressions=1`.
- Claim ceiling: the pinned container reports no `/dev/vfio`; no TPU,
  Pathways, R2E sandbox, Kubernetes, or 128-chip target was run. This is local
  construction and one CPU optimizer transaction only, not target evidence.
- Next action: P58.12 source is published. After matching-image readback,
  sandbox-capacity admission, and separate launch approval, use the corrected
  engine-global seed route for a fresh `p58z02` JobSet. The P58.11
  first-update and 1,000-commit gates remain unchanged.
- External effects: source commit/push only. No image publication, Kubernetes
  mutation, TPU run, model download, credential change, or artifact deletion.
- Phase: `phases/p58-11-qwen4b-zero-checked-vma.md`.

## Current P58.10 fixed-seed checkpoint (2026-08-24)

- Status: implementation published and exact-read back; pinned-image
  construction PASS. Worktree
  `/home/yuxuan/code_rl_repro/worktrees/p58_fixed_seed_0824`, branch
  `local/p58-fixed-seed-0824`, was built on operator tip
  `687b2bd6d0815b5628af39e7adbf949e429e72ae`, then replayed without conflict
  over latest fetched tip `ff646a4d76f58e9f328bc640f44d362637eb1432`.
  Implementation commit `9597de3d99fbf65c87f4fea3d86e639cca0b7abe`
  was pushed only to `yuxzhang/canon-zero-tim`; immediate local/FETCH_HEAD/
  remote-tracking readback matched with ahead/behind `0/0`. The older P58
  worktree was already dirty with unrelated P59/V1 work and was not modified.
- Contract: every P58 Native-raw, Native+IS, and Zero-HP render contains
  exactly one `--seed=42`. Any missing, duplicate, or different seed is
  rejected. The training entry point requires 42 for P58, uses it for both
  dataset shuffle and `RolloutConfig.seed`, and emits `[P58.SEED] PASS`.
- Provenance: W&B records `seed`, `rollout_seed`, and `seed_scope`; the durable
  run manifest records `dataset_seed=42`, `rollout_seed=42`, and the same
  scope. P58 classifiers require those fields.
- Claim boundary: this is configuration-level reproducibility. vLLM/R2E
  generation and sandbox collection remain asynchronous, so bitwise-equal
  trajectory order or identical end-to-end artifacts across independent jobs
  is not claimed.
- Validation: Python compilation and `git diff --check` pass; focused
  renderer/sampler/one-host tests pass 33/33. Bare-host artifact/classifier
  imports are unavailable because `metrax` is absent. The complete pinned
  image `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
  exits zero with `P58_EXACT_IMAGE_CPU_PASS ... paired_renderer=1 ...
  onehost_xprof=1 zero_hp_full=1 ... regressions=1`.
- Next action: fetch the final operator tip containing implementation commit
  `9597de3d`, render the fresh Native+IS full JobSet, and verify both its
  unique `--seed=42` argument and first `[P58.SEED] PASS dataset_seed=42
  rollout_seed=42 ...` marker. Launch remains separately user-gated.
- Source was committed and pushed only under explicit approval. No image
  publication, Kubernetes mutation, live-job stop, or TPU execution occurred.
- Phase: `phases/p58-10-fixed-seed.md`.

## Current P58.9 publication checkpoint (2026-08-24)

- Status: Native-IS target selected and source publication complete.
  Implementation commit `2aedd73c957abba29d21d05b866a996af2f66dfd`
  was pushed only to `yuxzhang/canon-zero-tim`; first remote readback matched
  local/FETCH_HEAD/remote-tracking refs with ahead/behind `0/0`. Worktree
  `/home/yuxuan/code_rl_repro/worktrees/p58_is_zero_refine_0824`, branch
  `local/p58-is-zero-refine-0824`, was originally based on
  `614156c1ab067192ab65b2969543e23904f192be` and replayed over exact operator
  tip `7b85b42d0a019d70f32a7dc9712c538ad42f5cb5` before publication.
- Scope: add a third executable recipe, Native+token-IS, without changing the
  existing Native-raw or Zero-HP numerical programs. Native+IS is selected
  only by renderer `--arm native --sampler-is`, resolves the existing disable
  tuple to `0:0`, uses threshold `2.0`, and requires trainer-old logps plus
  materialized TIS weights. Native-raw and every Zero recipe remain `1:1`.
  Partial tuples and IS on Zero fail closed. Group filtering remains absent.
- Retry correction: P58 is restored to exact Attempt-0
  (`maxRestarts: 0`, `restartStrategy: Recreate`). The prior JobSet retry used
  the same persistent run root without attempt isolation. Five renderer-only
  keepalive environment names had no exact-image code consumer and were
  removed rather than represented as recovery controls.
- Shared recipe remains Qwen3-4B-Instruct-2507, clean 1,012 tasks, B8 x G16,
  16K/50 turns, 128-chip synchronous disaggregated DP8 x TP8 per role,
  one-hour batch deadline, TPU-resident optimizer, prefix cache off, and full
  1,000 committed updates.
- Latest target fact: the operator reports a sharp training-reward drop in the
  live Native/no-IS run and judges the run collapsed. The onset update is not
  established and must not be recorded as a fixed optimizer step. Its
  exact run id, log, W&B series, and checkpoint receipts are not yet present
  locally, so the root cause is not independently classified here. The
  execution decision is final: preserve that run as immutable Native-raw
  failure evidence, stop its exact JobSet, do not resume its optimizer
  checkpoint, and do not launch Native raw again.
- Evidence after replay: focused renderer/profile/sampler/observer tests 40/40,
  Python/Bash syntax, and diff hygiene pass. Bare-host environment import
  is `INCONCLUSIVE` because `metrax` is absent. The complete pinned-image gate
  passes with `P58_EXACT_IMAGE_CPU_PASS ... paired_renderer=1 ...
  zero_hp_full=1 ... p59_real_shim=4 p59_rpa=2 ... m15_token=1
  regressions=1`.
- Claim ceiling: implementation plus pinned-image construction only. No direct
  TPU run, Pathways target, optimizer commit, full training, or performance
  result exists.
- Next action: the remote executor identifies and archives the exact live
  Native-raw JobSet, then deletes only that JobSet and verifies cleanup. After
  fetching the final operator tip and proving it contains implementation
  commit `2aedd73c`, render and launch a fresh `--arm native
  --sampler-is` full run from the original frozen base checkpoint using a new
  run id/root/W&B/checkpoint. Never resume the collapsed Native-raw state.
- Blocker: source publication is not blocked. Remote execution must still
  preserve/stop the exact Native-raw run, pin the final read-back source SHA
  and image digest, pass render/admission checks, and never resume the
  collapsed optimizer state.
- Phase: `phases/p58-9-native-is-attempt-zero-refine.md`.

## Current P58.6/P58.7/P58.8 checkpoint (supersedes the legacy P58.5N snapshot below)

- Status: active; P58.6/P58.7 are implemented and the approved four functional CLs plus one audit-only release CL on latest tip `ccbcf572` pass the post-barrier host and pinned-image admission ladder. Publication is authorized; exact remote readback and hardware targets remain open.
- Objective: produce a matched one-host Qwen3-4B Native versus optimized Zero-HP XProf/Perfetto pair, then run the optimized strict-Zero Qwen3-4B DeepSWE-derived DP8 x TP8 full 1,000-commit campaign after separate publication and launch approval.
- Worktree: `/home/yuxuan/code_rl_repro/worktrees/p58_zero_hp_release3_0823`, branch `local/p58-zero-hp-release3-0823`, based exactly on latest fetched operator tip `ccbcf572dc903bb1cce12f897cbdb05aec94922a`. It retains all earlier immutable failure evidence and P58 restart/keepalives plus the new P57 evaluation-cycle, final-only checkpoint, and lazy NumPy host-render fixes. Older dirty and release worktrees remain untouched.
- Current phase: P58.8 latest-tip release publication. The CL-A/B/C/D/E scope and reverse rollback order are in `RELEASE_CL_PLAN.md`; P58.6 direct-host pair, P58.7 target, and P59 hardware target remain `NOT RUN`.
- Implementation: P58.6 has two thin wrappers, one fail-closed common driver, signed provenance/work/state hashes, a warmup plus identical no-commit update capture, XPlane/trace/semantic Perfetto requirements, arm classifier, cross-arm classifier, and immutable package sealing. P58.7 has an additive default-off Zero-HP profile, renderer selector, DeepSWE exact admission, Qwen3-4B TP8 fixed-head receipts, P59 receipts, update XProf/Perfetto wiring, and a 1,000-update strict postflight/performance ledger. P58.8 repairs the fetched Phase4 P59 TP4 nested-mesh first red with an exact-device two-axis engine carrier covering TP4/TP8, and scopes the P57 Zero/full W&B project to its signed profile.
- Numerical policy: any real Zero alignment FAIL kills the candidate. P59 is accepted under ordinary-JAX FP64 gradient correctness; serial-AdamW weight trajectory identity is explicitly not claimed. APC stays off for P58.7.
- Latest-tree host validation: P59 30/30, current P57 136/136 including the wrong-profile W&B negative control, V1 12/12, APC 31/31, flag registry 366/366, Python/Bash syntax, and `git diff --check` PASS. Bare-host P58 discovery executes 51 tests but cannot collect four modules because this shell lacks `metrax`; this is `INCONCLUSIVE` dependency coverage, superseded for those modules by the complete pinned-image gate.
- Pinned-image validation: image `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a` passes `P59_TP_SHIM_EXACT_IMAGE_PASS fixed_head=2 installed_projection=2 report_adjoint=2 fixed_reducer=2 topologies=DP2xTP4,DP2xTP8 optimizer_commits=0 manifests=2x36/36`. Both P58 and V1 complete exact-image suites terminate with `p59_real_shim=4`. Positive paths execute the modified P59-local fixed-head/projection VJPs and emit `all_gather_rank_order_f32_barrier`; negative paths keep ordinary global outputs and exact device-index maps. TP8 initially exposed BF16 per-rank-add error (`0.5` max abs versus FP64 while the serial probe was exact); fixed-rank TP partials now accumulate in barrier-pinned FP32 and cast once, after which TP4 and TP8 serial/parallel probes are byte-exact.
- Release-tree correction: the unrelated APC-on availability decision from the older dirty tree is excluded. A first complete P58 exact-image rerun then failed only because its vLLM test double omitted the stock `num_cached_tokens` field; the mock now supplies zero. P58 and V1 exact-image reruns both terminated PASS on that first release tree before the operator tip advanced. No real run, numerical gate, or optimizer commit was involved in the construction red.
- Latest-tip reconciliation: the earlier `24b1bbcf` tree preserved P58 `maxRestarts=3` and Pathways/IFRT/GRPC keepalives. The new `ccbcf572` reconstruction additionally preserves all three later P57 fixes while migrating only P58/P59 dirty hunks. FP32 TP accumulation now uses fixed-reducer-style operand barriers, and both complete exact-image gates passed after that change.
- Target not run: no direct TPU XPlane/Perfetto pair, DP8 x TP8 Pathways run, 4B/TP8 fixed-head target, P59 target, full evaluation/checkpoint horizon, or 1,000-commit result exists. The fifth CL only prevents immutable log markers from masquerading as settable flags during changed-base auditing; it changes no runtime. No image publication, YAML apply, or TPU launch occurred.
- Hardware limitation: the available direct one-host v5p exposes four chips, while every registered production fixed-head geometry is TP4 or TP8. A real P59 DP>1 composition therefore needs at least DP2 x TP4 (eight chips); the requested DP2 x TP2 fixed-head program would be an unregistered artificial geometry. No one-host TPU composition PASS is claimed.
- Next action: push the approved five-CL stack and require exact remote readback. After publication, keep P58.6 and P58.7 deferred while the separately approved V1 GSM8K DP16 x TP4 full run performs the first repaired target certification; P45 and M15 follow only after their preceding full postflight.
- Latest evidence: `evidence/p58rel3-p58-exact-image-20260823/`, `evidence/p58rel3-v1-exact-image-20260823/`, and `evidence/p58rel3-release-tree-20260823/`; committed runtime/test delta manifest SHA `babc1c708f7cee01c14e465058991013fd5483e6a0a75b7c367a22cd44e329da`.
- Runbook: `RUNBOOK_P58_6_7.md`. Phase files: `phases/p58-6-onehost-native-zero-xprof.md`, `phases/p58-7-qwen4b-zero-hp.md`, and `phases/p58-8-p59-tp-mesh.md`.
- Updated: 2026-08-23 UTC, publication closeout.

## Legacy P58.5N snapshot (historical; no longer current)

- Status: active
- Objective: run the untreated native DeepSWE-derived Qwen3-4B-Instruct path as a full 1,000-update 128-chip campaign on the frozen clean-data recipe. Preserve the zero arm for the eventual causal comparison, but do not launch it until its optimization work is complete and the user explicitly reactivates it.
- Definition of done for the active phase: the published native source and image pass preflight; one 128-chip native JobSet runs rollout DP8 x TP8 plus trainer DP8 x TP8; exactly 1,000 optimizer commits complete; full trajectory, signal, mismatch, optimizer-placement, cleanup, evaluation, and checkpoint evidence is durable; at least one finite nonzero Native serving-path mismatch is observed across A-B or B-C; every Native numerical boundary is shape-valid and finite; and the native full-stage classifier says `PASS`. The first three commits are online monitoring milestones, not an early-stop canary. This does not complete the deferred two-arm comparison.
- Task directory: `canon-zero-tim/tasks/p58-deepswe-native-zero-comparison`
- Directory state: source intake was fast-forwarded from exact operator tip `5f449cc8def801b4a61387ef664b2cb1f7ab05cf`, including immutable p58f12 evidence and a later P57-only checkpoint change. During publication the remote advanced by P57-only evidence commit `e7958a27851931ab9bcff232088efd95bbc12021`; the first normal push was safely rejected, that commit was reviewed as non-overlapping, and the P58 commits were rebased without conflict. The ordinary all-compact no-commit path, sandbox-capacity circuit breaker/probe, tests, and reconciled handoff are implementation commit `135867f04bfa0fc90ea1d4528ba59f365573a78b`. This publication-evidence checkpoint follows it; executors must fetch and exactly read back the final operator tip rather than pinning the implementation commit. No zero-TIM flag was enabled, deleted, or repurposed; `main` remains untouched.
- Current phase: P58.5N — native 128-chip full 1,000-update campaign.
- Completed/closed phases: P58.1 loss/compact-filter/accumulation contract; P58.2 paired profile, renderer, full-trajectory journal, classifier, negative controls, and exact-image regression gate. P58.3 one-host sanity was explicitly `WAIVED` by the user on 2026-08-21; it is not a PASS. P58.4N three-update was superseded by the user's direct-full decision after p58c05 failed before execution; it is not a PASS.
- Last verified fact: `p58f12` proved the p58f11 normalized-prompt repair and wrote a valid durable 128-row Step-0 journal, but all 128 RepoEnv pods remained Kueue `scheduling_gated` until sandbox-start timeout. Every row became signed compact-filtered `ENV_TIMEOUT`; no rollout generation occurred, so no sampling-transform provenance existed. Learner preprocessing nevertheless requested processed `S_prefill` and raised `RuntimeError: processed S_prefill must follow generate()` before alignment/backward/update/checkpoint. Effective R2E throughput was zero; this is CPU sandbox scheduling/capacity evidence, not a model/vLLM-throughput diagnosis.
- Local repair: a signed P58 all-compact batch now takes an explicit empty-completion observer path: it validates prompt/completion structure and the observer signature, skips the engine because there are zero scored targets, and records `engine_called=false` rather than fabricating log probabilities. Alignment admits zero action tokens only with durable all-compact P58 provenance and still rejects unsigned/partial zero-action input, nonfinite values, or nonzero gradients. Ordinary all-compact batches caused by model/context/runtime outcomes complete the existing zero-gradient/no-commit transaction, suppress weight sync plus `policy_version`/RL/trainer/optimizer-step advance, increment only durable `batch_index`, and consume the next prompt batch without resampling. A full `all_sandbox_start_timeout_batch` is different: after its journal and bounded metrics are durable, the learner emits `[P58.SANDBOX_CAPACITY] BLOCKED ... optimizer_commits=0 prompts_consumed_after_batch=0` and raises `BLOCKED_SANDBOX_CAPACITY` before rescore/alignment/trainer or the next prompt batch. Inconsistent infrastructure evidence fails closed.
- Validation: shell syntax, Python compilation, `git diff --check`, and the production-shaped probe/verifier regression pass 4/4 on host. The complete pinned-image gate at `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a` exits zero after the new three circuit-breaker tests, probe tests, empty-rescore tests, full P58 suites, 60 shared-common tests, 42 alignment tests, trainer accumulation/compact tests, P34/P44 adjacency, and stock observer; terminal marker is `P58_EXACT_IMAGE_CPU_PASS loss_oracle=1 weighted_accumulation=1 compact_filter=1 durable_journal=1 paired_renderer=1 alignment_policy=1 stock_observer=1 regressions=1`. Host-only full import remains unavailable because this shell lacks `metrax`; the pinned image supplies it. No target TPU/Pathways/Kueue execution was run. Prior direct-one-host metadata/device exposure remains blocked and is not a TPU PASS.
- Next action: fetch and exactly read back the final `yuxzhang/canon-zero-tim` tip. Before fresh native full `p58f13`, a remote operator must render one real-task-image sandbox probe, separately approve/apply it, and obtain `P58_SANDBOX_CAPACITY_PASS scope=one-sandbox-admission-only`; also confirm Kueue/node capacity for the 128-Pod request floor (256 requested CPU and 512 GiB requested memory, plus head/cluster overhead). Retain concurrency 128, B8 x G16, all deadlines, topology, TPU-resident optimizer, and every Native/Zero flag. Require a later effective batch to reach finite Native boundaries/backward and the first TPU optimizer commit. Do not bypass Kueue, enable SandboxFleet, or render/apply zero.
- Blockers: p58f08 through p58f12 have no optimizer checkpoint and are immutable `INCONCLUSIVE`; p58f12 has a diagnostic trajectory journal but no resumable trainer state. Exact live ClusterQueue/ResourceFlavor/node capacity is not present in the returned log and must be confirmed remotely. Main `c7d8950f12a9c55a976bf2e1a0d8b447d71c20b3` contains Agent Sandbox/SandboxFleet commit `e789573964b6f695ded85fe519040bd06a2b9f37`, but it is deliberately not integrated: it cannot create quota, is not yet P58 Kueue/fail-closed, and its lookahead can request 256 sandboxes for B8 x G16. Zero launch remains deferred.
- Key artifacts: `plan.md`; `HANDOFF.md`; `log.md`; `phases/p58-5-native-full.md`; `../../debug_logs/p58_p58f12_deepswe_empty_batch_rescore.raw.log` (SHA-256 `10f718fb6221e3bfb3ae509ff394fbf6ea44caab1a9388c3ae1033f6410e109a`); its classification JSON (SHA-256 `e0831acd814b5398726e3a24f5063a73090e51bcbf9bcbf3d9b39614d9b626e3`); target journal `/mnt/disks/linchai_data/deepswe_zero_tim/canon-p58-ds4b-native-full-p58f12/debug/batch-000000.trajectories.jsonl.gz` (SHA-256 `d4453eb0873a89933ebd1ccd281fd97c22f86f2870f3ac76dff7fefecae8986c`); prior immutable evidence listed in `log.md`; `../../cluster/P58_DEEPSWE_TIM_RUNBOOK.md`; `../../tests/p58_deepswe_native_zero/`.
- Updated: 2026-08-23 UTC

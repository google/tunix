# V1 Phase4 three-full handoff

## 2026-08-25 superseding status — G5b full-log carrier ready for commit review

The preserved six-line P62 target excerpt is not a complete G5 result. It
shows a finite loss cotangent and finite-but-extreme group-0 engine/rank-local
tree, but omits strict pre-alignment, groups 1-15, fixed-DP and scaled seams,
the final accumulator, and the zero-commit discard terminal. Under the repaired
classifier it is `FATAL_CONTRACT`; it does not admit stable clipping, an
optimizer transaction, or a production full run.

On current operator base `41a2043c`, the uncommitted G5b repair makes the
complete evidence path fail-closed. P62 seeds the exact resolved-profile
receipt into its unique `$CANON_STATE/run.log`, appends all workload output,
and automatically classifies that exact file before the pod exits. The
postflight receipt binds the full-log SHA/size/line count and classification
SHA. The classifier requires all 16 reverse groups plus every registered
boundary and discard; a partial finite naive-L2 overflow is
`INCONCLUSIVE_INCOMPLETE`, never a successful finding. The renderer records
the exact run-log and classification paths.

Final-tree validation is green: V1 38/38, P57 144/144, P59 37/37, APC 31/31,
M15 APC target 9/9, flags 371/371, Bash syntax and diff hygiene. The complete
pinned image exits zero with terminal `V1_HP_EXACT_IMAGE_PASS ...
p62_numeric=6 ... apc_m15_carrier=39 ... manifests=3`. The latest full image
run was observed on parent `bdfa50e1` but was not sealed as a new signed raw
artifact. The only incoming delta to `41a2043c` is a one-line M15 APC
zero-commit checkpoint exemption; its focused host and pinned-image target
gates pass 9/9. P62 runtime blobs are unchanged. No TPU, JobSet, optimizer
transaction, commit, or push occurred.

Next boundary: obtain explicit commit/push approval for this one G5b carrier
concern, read back the exact remote 40-character SHA, then render and separately
approve one fresh GSM8K DP16xTP4 `backward-no-commit` P62 JobSet. Do not launch
the GSM8K full recipe until G5b explains whether the `5.38e22` magnitude is a
real backward/scaling fault or only a valid finite tree whose naive norm
overflows. Never launch through a pipe and never reuse a run ID.

## 2026-08-25 superseding status — P62 first-red carrier admitted through one host

This is the current boundary. Attempt 7 is strict Zero-TIM through the full
forward and all 16 GSM8K reverse groups, but `norm=inf` is not explained by
the saved log. The earlier max-scaled production clipping proposal is
withdrawn: GSM8K, FrozenLake, and DeepSWE again use historical stock
`optax.clip_by_global_norm`. `stable_global_norm` remains an observer only.

The default-off `CANON_P62_BACKWARD_NUMERIC_DEBUG=1` carrier is admitted only
for strict GSM8K DP16xTP4, global trajectories 256, global/local M 4096/256,
16 reverse groups, fixed head and P59 enabled, `backward-no-commit`, and
`CANON_P58_NO_OPTIMIZER_COMMIT=1`. It prints the first-red boundaries from
loss scale through final accumulator, then discards the accumulator. Any
non-finite boundary is fatal and every valid run requires
`optimizer_commits=0`.

Verified by host/forced-CPU and pinned image:

- V1 host 34/34, P59 host 37/37, and post-rebase flag audit 371/371;
- complete exact-image raw SHA
  `604c95e5953f97fa8465e03f38b15589bd38fbf618b04c5652be0328b446689e`,
  unique terminal `V1_HP_EXACT_IMAGE_PASS ... p62_numeric=6 ... manifests=3`;
- focused G2 installed-shim raw SHA
  `8fb3720e3ac39cf80535833e1786585950ab13bd7015b4c9c9aa66da0dc60b92`:
  TP4/TP8 fixed-head, report adjoint, fixed reducer, installed projection and
  installed attention all green, with 10 P62 receipts and two caught NaN
  first-red negatives. Failed carrier r1 is preserved beside it;
- real one-host v5p DP2xTP2 run
  `a7_numeric_dp2tp2_20260825_r2`, 54 seconds, real RPA and staged-spec
  carriers green, FP64 oracle relative-L2 `3.77417983e-08`, cosine `1`, both
  wrong-scaling negatives separated, zero optimizer commits;
- durable evidence under
  `evidence/v1_hp_attempt7_p62_numeric_exact_image_20260825_r1/` and
  `evidence/v1_hp_attempt7_p62_onehost_v5p_20260825_r2/`; focused G2 is under
  `evidence/v1_hp_attempt7_p62_g2_exact_image_20260825_r1/` and `..._r2/`.

Claim ceiling: the one-host carrier proves the DP2xTP2 reduction/accumulation
algebra and real installed RPA mechanism. It does not execute the full Qwen
DP16xTP4 target and therefore does not explain the historical `norm=inf`.
Full recipes and all optimizer commits remain blocked. The next hardware step
is one fresh user-run P62 GSM8K DP16xTP4 diagnostic, not a production full
recipe. Classify its earliest red using
`phases/v1-p4-5-attempt7-numeric-localization.md`; only then design a bounded
one-commit fix. Publication of this diagnostic stack was explicitly approved
on 2026-08-25, but it does not authorize a JobSet or optimizer transaction.

Publication audit: the scoped P62 stack was rebased on operator runtime tip
`eb58954f`, then through the publication-time M15 evidence/documentation tip,
preserving its APC target/replay flags, tests, and Attempt-0 failure receipt. The merged
pinned image `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
exited zero with the unique combined terminal
`V1_HP_EXACT_IMAGE_PASS ... p62_numeric=6 ... apc_m15_carrier=33 ... manifests=3`.
This final merged-tree execution was observed in the release terminal but was
not saved as a new signed raw artifact; the durable P62 r1 and G2 r2 artifacts
and their checksums remain the evidence sources above. The classifier's
alignment/update strings are log markers, not environment flags, and are split
lexically so the deterministic flag audit remains 371/371 without inventing
two false flags.

## Historical superseded status — Attempt 7 stable-clipping proposal

This section supersedes older publication/launch-ready wording below. The
active worktree is now rebased on pulled operator tip
`ff913a84ec9aa66bfd152415688bc431ca1d1a1b`; its relevant immutable logs are:

- GSM8K Attempt 7:
  `canon-zero-tim/debug_logs/v1hp_att7_gsm8k_g64s_p28_g6_norm_inf_error.raw.log`,
  SHA-256
  `68aa10263bed8343623ef48d933d4bb1fbca367cc3df01745a03cd108316425a`;
- one-host native XProf:
  `canon-zero-tim/debug_logs/v1_gsm8k_onehost_native_20260824_v2_exit137.raw.log`,
  SHA-256
  `3312c56e74ef1cc7d10072791993ee47fc72ec6d7931b1d73ec8641b17496128`.
- FrozenLake P45 Attempt 7:
  `canon-zero-tim/debug_logs/v1hp_att7_fl_f45s_dp_reduction_unequal_replicas.raw.log`,
  SHA-256
  `41d2dd0cb4810cbe3e0f434c18558575f48033d6eb428d951b222772598584e8`.

Attempt 7 is not a Zero-TIM red. Step 0 has 191,439 action tokens,
`S_decode==S_prefill==T_old` byte-for-byte, and zero alignment FAIL. All 16
P59 reverse groups finish and report replica equality before the old P28 G6
activity guard stops with `active=True norm=inf`; no optimizer commit occurs.
The guard used Optax's naive FP32 sum of squares. It also failed before the
commit path's independent per-leaf finite evidence and did not serialize the
adapter's per-group finite bit, so the saved log cannot distinguish these two
cases:

1. every element is finite but squaring a value above about `1.84e19` (or the
   aggregate sum) overflows FP32;
2. at least one gradient element is genuinely NaN/Inf.

The uncommitted repair intentionally handles both without weakening a gate:

- P28 precomputed microgradient and commit diagnostics use max-scaled L2;
- P28 production optimizers use the same stable clipping transform, so a
  finite overflow no longer becomes an all-zero Optax update;
- the G6 gate separately consumes each adapter report's element-finiteness bit
  and remains fatal for any genuine NaN/Inf;
- full postflight now requires exactly one
  `[P28.G6] STABLE_GLOBAL_NORM ... algorithm=scaled-l2` runtime receipt.

P45 independently passes strict step-0 pre-alignment for 48,082 actions with
both byte deltas zero, enters the real DP8xTP8 P59 fixed-head/projection
backward, then stops before its first reverse-group receipt and before any
optimizer commit at
`fixed DP gradient reduction produced unequal replicas: flags=[0,...,0]`.
That old message was ambiguous: `jnp.array_equal` is false for identical NaNs,
so it could not distinguish genuinely unequal finite replicas from a common
non-finite gradient. The repair now checks the staged DP table for finiteness
before reduction, reports the first bad rank/leaf/tree path, checks the reduced
tree again, and only then runs the unchanged finite-replica equality gate.
NaN/Inf remains fatal; no `equal_nan=True` admission was introduced.

Validation on the latest dirty repair tree is host V1 30/30, P57 144/144, P59 34/34,
APC 31/31, flag audit 368/368, and `git diff --check`. Pinned-image focused
norm tests are 16/16. A forced-CPU DP8xTP8 gate proves finite fixed reduction,
common-NaN rejection, and finite replica-mismatch rejection 3/3. The complete
pinned-image gate rerun exits zero with exactly
one terminal
`V1_HP_EXACT_IMAGE_PASS ... p59_fused_linear=2 ... manifests=3`. Durable logs,
receipt, and checksums for the superseding gate are under
`evidence/v1_hp_attempt7_norm_dp_diagnostic_exact_image_20260825_r3/`;
complete raw SHA is
`fa4960bed7f7d94250c59d683aeb89dd7fc7edd81fdbcbe367b30c3a7c5017ee`.
Claim ceiling is `HOST PASS / FORCED-CPU DP8xTP8 PASS / EXACT-IMAGE PASS /
POST-FIX TPU TARGET NOT RUN`.

The one-host native exit 137 is separate: the OS killed Python during serial
rollout generation after 377 seconds, with no traceback or numerical verdict.
Treat it as `INCONCLUSIVE_RESOURCE_KILL`; do not use it to judge Zero-TIM or
the norm repair. Its carrier needs memory telemetry before any resource fix.

Next boundary: review the dirty repair. Do not commit or push without a fresh
explicit user instruction. After publication and exact remote readback, a
separately approved launch may render and start GSM8K/P45/M15 together. The
first real optimizer transaction of each recipe is the target discriminator.
GSM8K must distinguish finite norm overflow from a true non-finite leaf. P45
must now report either a precise non-finite rank/leaf/path or retain the
finite-replica mismatch; only a finite, exact reduction may proceed. Preserve
every Attempt-7 and post-fix artifact.

## Mission and current boundary

Prepare exactly three strict optimized Zero-TIM full-training recipes from one
approved immutable source: GSM8K Qwen3-1.7B DP16xTP4 for 200 updates,
FrozenLake P45 Qwen3-8B DP8xTP8 for 300 updates, and FrozenLake M15-main
Qwen3-8B DP8xTP8 for 300 updates. M15 is a production/scientific recipe, not a
canary. The original three-recipe stack is published in the operator history.
The active repair worktree is
`/home/yuxuan/code_rl_repro/worktrees/v1_attempt6_p59_restore_0824`, branch
`local/v1-attempt6-p59-restore-0824`, based on pulled operator tip
`0a68e1f705b6b63ca4dc86e5713e4785cb73e7d1`. The branch was cleanly
fast-forward rebased from `f2dd9d90` after fetching the three P60 GSM8K XProf
carrier commits `ad972daa`, `56c6a6d4`, and `0a68e1f7`; no local commit was
rewritten.
The earlier tip archives immutable Attempt-6 logs from source `85f45c21`.
GSM8K `g64r` passes strict step-0
pre-alignment for 193,146 actions with both byte deltas zero, traverses all
previous P59 TP4 repairs, then stops before its first optimizer commit because
the staged-spec metadata restorer still rejects every non-TP1 difference.
The local repair admits only same-mesh, leading-DP metadata normalization whose
`devices_indices_map` is exactly equal to the expected trainer placement; it
continues to reject a TP-sharded parameter gradient that has become physically
TP-replicated. Focused DP2xTP4 and DP2xTP8 forced-CPU gates are green. The
dependency-complete pinned-image gate and a real-v5p DP2xTP2 staged-spec
mechanism gate are now green; DP16xTP4/DP8xTP8 optimizer commit and performance
remain unverified. After publication,
render only from the exact 40-character SHA read back from the operator branch
and require a clean worktree.

After exact-image admission, publication, exact remote readback, rendering,
and separate launch approval, start all three full JobSets in one wave. Do not
gate P45 or M15 launch on GSM8K's first optimizer commit. Every recipe still
owns an independent first-commit admission and strict zero-TIM verdict; a red
freezes and kills only that recipe while the other healthy full runs continue.

Do not push, rerun the pinned image, publish an image, apply a JobSet, or occupy
TPU resources without the separate user approval for that boundary. The
one-time pinned-image and bounded one-host approvals were consumed by the green
runs below. The 2026-08-24 publication approval is scoped to local runtime CL
`26b8a36d`, carrier CL `ef481f02`, the following evidence/ledger CL, and their
single operator-branch push; it does not authorize another image/TPU run or a
JobSet launch. Never launch through a pipe. Run IDs, campaign roots, and
evidence directories are first-use only; preserve every failed run.

Post-rebase host admission is V1 29/29, P57 144/144, P59 34/34, APC 31/31,
P60 GSM8K XProf 4/4, and flag audit 368/368. The saved exact-image and real-v5p
evidence hashes still verify byte-for-byte. Those historical hardware runs
certify the Attempt-6/APC-off/cache runtime they executed; they do not
retroactively certify the newly inherited P60 learner/XProf runtime additions.

The current production decision supersedes the earlier P45-only APC readiness
choice: all three Phase4 full recipes now force
`CANON_VLLM_ENABLE_PREFIX_CACHING=0`. This disables only cross-request prefix
reuse; ordinary request-local prefill/decode KV state and B's independent
`reset_prefix_cache=True` full recomputation remain unchanged. Phase3 APC code
and diagnostic carriers stay default-off for a separate debugging thread.
The three manifests already inherited the P33 JAX persistent-cache directory
and GCS bucket. The local hardening makes those values an exact renderer and
postflight contract and emits durable `restore`/`save` receipts instead of
silencing GCS failures. A cache miss/error is performance evidence, never an
alignment verdict; a missing receipt or wrong directory/bucket/profile is a
release-carrier failure. V1 full runs save immediately after the training
command, before any fail-closed postflight can exit. This APC/cache hardening is
host- and pinned-image-green; GCS cache hit/JIT reduction remains target-unrun.

The earlier bootstrap failures and their P58.8 repairs remain historical. The
new immutable attempt-1 logs are under
`evidence/v1_hp_three_full_attempt1_20260823/`: GSM8K `g64f` stopped
pre-optimizer when a DP-only `[256,151936]` cotangent was not localized to the
TP4 fixed-head width `[256,37984]`; P45 `f45g` stopped in C-forward because
the Qwen3-8B/TP8 fixed-head contract omitted learner M2048. Neither is a real
alignment FAIL. The current repair restores `P(data,model)` before the
P59 head VJP and admits M2048 only for the 8B/TP8 geometry. Host/static gates
and the dependency-complete post-fix pinned-image gate are green.

Attempt 2 is immutable under
`evidence/v1_hp_three_full_attempt2_20260824/`. GSM8K `g64k` and P45 `f45i`
both passed strict step-0 pre-alignment, then stopped before optimizer because
the P59 local projection shim treated engine fused-layout `n_shards=1` as if it
were the mesh TP degree. M15 `m15i` is a genuine numerical red: APC-on decode
differs from full prefill on 760 elements / 1389 bytes with max abs
`0.998443603515625`, while prefill and independent B rescore are exact. Per the
hard rule, APC is dead for M15/main and is reverted there; no warning or
tolerance was introduced. P45 remained APC-on for Attempts 3-6; the uniform
APC-off production decision below supersedes that historical choice. The local P59 repair admits the legitimate q_proj
one-layout-shard boundary while retaining invalid-layout and width negatives.
The full classifier now requires exactly one explicit APC-off runtime receipt
for M15 and rejects a missing, duplicate, or opposite-arm receipt.

Attempt 3 is immutable under
`evidence/v1_hp_three_full_attempt3_20260824/`. GSM8K `g64m` passed strict
step-0 pre-alignment for 194,633 action elements with both canonical byte deltas
zero, completed all 16 forward groups, and crossed the P59 head and q/k/v
projection-local boundaries. Before any optimizer commit, the stock attention
entry mistook already TP-local K/V (`2` heads on TP4) for global GQA and
expanded them again to `4`; the correctly localized cache remained `2`, so RPA
rejected `(9,256,2,2,128)` versus the erroneous expected
`(9,256,4,2,128)`. This is `INCONCLUSIVE_PRE_OPTIMIZER_SHAPE_CONTRACT`, not an
alignment FAIL. Patch 25 skips that repeat only under the exact two-manual-axis
P59 context, validates local Q/K/V/cache shape, and leaves ordinary serving GQA
unchanged. Full postflight now requires its exact local-KV runtime receipt.

P45 `f45m` independently passed strict step-0 pre-alignment for 45,074 action
elements with both byte deltas zero and completed all 32 forward groups. Its
first backward then expanded already TP8-local K/V from one head to eight, so
RPA rejected `actual_num_q_heads=4` versus `actual_num_kv_heads=8`. This is the
same patch-25 seam at the target TP8 geometry. M15 `m15m` passed strict
pre-alignment for 124,867 action elements, then stopped before forward/backward
because its signed physical 4096/8192 prompt/completion buffers were compared
against the original P45 4096/2048 contract. CL `aa84c147` admits 4096/8192
only for the registered `m15/selection` and `m15/main` DP8xTP8 tuples; partial,
foreign, and m10 tuples remain negative. All three Attempt-3 runs have zero
alignment FAIL, zero optimizer commits, and no performance claim.

Attempt 4 is immutable under
`evidence/v1_hp_three_full_attempt4_20260824/`; all four `SHA256SUMS` entries
verify. GSM8K `g64p`, P45 `f45p`, and M15 `m15p` passed strict step-0
pre-alignment for 190,635, 47,329, and 122,754 action elements respectively,
with both byte deltas zero and no alignment FAIL. The repaired TP4/TP8 RPA
boundary emitted its exact `P59_RPA_LOCAL_KV_READY` receipt in all three runs.
The first fatal then occurred at the final decoder layer's `gate_proj`:
installed `linear_p22xf.py:106` compared the already TP-local output width
1536 against the globally declared width 6144 on TP4 or 12288 on TP8 because
the engine config legitimately retained `n_shards=1`. The raw terminals are
`gsm8k_g64p_error.log:12179`, `p45_f45p_error.log:21910`, and
`m15_m15p_error.log:19955`. This is one pre-optimizer shape-contract seam,
not a numerical verdict; all three runs have zero optimizer commits.

Attempt-4 runtime CL `5bd90bff` validates every local projection's flattened
feature width against the model-exact `site.n_local`. Only gate/up, whose last
axis is physically TP-local under the outer P59 map, divide global
`output_sizes` by the live TP degree; q/k/v continue using their independent
layout-shard count. It emits `P59_LOCAL_FUSED_LINEAR_READY`, and full
postflight requires exact TP4 `6144->1536` or TP8 `12288->1536` gate and up
receipts with `layout_shards=1`. Missing/wrong receipt and wrong-width controls
are fatal. Host gates pass P59 34/34 and V1 23/23. The focused pinned-image
gate passes installed TP4 and TP8 projection plus RPA carriers, 2x36/36
manifests, ordinary-global negatives, and zero commits. The complete V1 image
gate exits zero with additive terminal `p59_fused_linear=2`. Durable raw SHA is
`9d50ec495c189a77dfdab92b8496580a58a55d101ed03cd2b977728a69ef5001`;
receipt SHA is
`62995bb94a849602eeb2390d8e83b75bb1bf6b082d7044d47912d8b9e694b205`.
Claim ceiling remains `HOST PASS / EXACT_IMAGE PASS / ATTEMPT-4 TARGET REDS
PRESERVED / POST-FIX TARGET NOT RUN`.

Attempt 6 is immutable under
`evidence/v1_hp_three_full_attempt6_20260824/`; all four `SHA256SUMS` entries
verify. GSM8K `g64r` records strict PASS at
`gsm8k_g64r_error.log:11060` and the first fatal at
`gsm8k_g64r_error.log:13553-13555`. The sharding inventory at line 11064
contains 113 replicated parameter leaves with `P(None,)`. Report-adjoint
normalizes their staged form to `P(dp)`, while the trainer-derived expected
form is `P(dp,None)`: these `NamedSharding` objects compare unequal but have
identical per-device index maps. The old helper rejected the difference only
because TP=4, before reaching its existing physical-equivalence check. Local
`canonical_qwen3_adapter.py:345-413` removes only that TP1 restriction and
renames the helper to describe its actual invariant. The TP4/TP8 installed
fixed-head composition test now includes this replicated-parameter leaf and
continues through production report-adjoint and fixed reduction; a separate
negative proves TP-replicated staged data cannot replace a TP-sharded expected
placement. P45 `f45r` and M15 `m15r` archives end mid-computation and have no
terminal traceback or completed update, so they receive no numerical or
runtime classification. Claim ceiling:
`HOST PASS / EXACT_IMAGE PASS / ONEHOST_TPU_MECHANISM PASS / TARGET NOT RUN`.
The exact-image terminal records `staged_spec_restore=2`; the two invocations
cover DP2xTP4 and DP2xTP8 positive plus wrong-placement negative. Durable
image evidence is
`evidence/v1_hp_attempt6_apcoff_cache_exact_image_20260824_r1/`, raw SHA
`8d8d776451615de58a749c0be0200d28107b86cc44504200afde4f5acffc712a`.
The real-v5p run
`/mnt/disks/tunix-data/logp_probe_1host/p59_rpa_a6restore_dp2tp2_20260824_2256utc/`
passes the replicated-leaf seam at DP2xTP2, fires the wrong-placement negative,
retains RPA/ordinary-TP4 controls, and makes zero optimizer commits. Its
raw/driver SHAs are `432bc6ae015d3b325ebeb5e06fff412ce6e53d1108cc7aa6d09b3c6d8a837d` /
`2bb7ff5409ed404fa13261a2a3934bb6baef4b7e21f456ec1038bfccd98f33e7`.

The operator's appended `SHA256SUMS` contains an impossible stale self-hash and
is preserved unchanged. `SHA256SUMS.artifacts` is the additive verification
source for the three logs and `receipt.json`; all four entries pass. See
`MANIFEST_NOTE.md` in the same evidence directory.

The approved direct-attached v5p mechanism gate is green at
`/mnt/disks/tunix-data/logp_probe_1host/p59_rpa_a3dp2tp2tp4_20260824_0648utc_r2`.
On the same four physical chips it executed real RPA forward plus VJP2 backward
under P59 `DP2xTP2`, caught a wrong local-cache negative, then rearranged the
mesh as ordinary `DP1xTP4` and proved the stock global GQA expansion still
works. The run made zero optimizer commits. This is real-hardware mechanism
evidence, not DP16xTP4 production certification.

## Resolved bundle

- All recipes: automatic P47a, continue-decode K=8, fixed-AR gather,
  DP-aware gathered logprobs, logprob step fusion, fixed LM head, resident
  trainer placement, batched report, and P59 rank-parallel backward.
- GSM8K only: batched evidence on.
- FrozenLake: batched evidence off.
- All three production recipes: APC off. B rescore remains an independent full
  recomputation with `reset_prefix_cache=True`; normal request-local decode KV
  caching remains enabled.
- All three manifests: JAX cache directory
  `/tmp/jax_compilation_cache`, minimum compile time `0`, XLA caches `all`, and
  GCS root `gs://yuxzhang-tunix-models/cache/p33_compilation_cache`. The
  profile basename is the remote namespace.
- Explicitly off: batched reverse, fused tree ops, norm-matmul, sample-split,
  engine-logprob-readback, anchor overlap, and vanilla/non-Zero paths.
- FrozenLake held-out rollout-only eval runs at pre-update policy steps
  `0,50,...,250` and after training at 300. Runtime receipts map the first six
  to enclosing global timing rows `1,51,...,251`; final 300 maps to `none`.
  Eval never enters trainer forward, backward, or optimizer.

## Claim provenance and ceilings

- The published operator history supplies the P56 serving, P59/APC foundation,
  three-recipe integration, and P57 300-update signed in-process-eval setup.
  The current release adds the P59 TP4/TP8 and signed P57 W&B repairs; target
  certification remains pending after publication.
- P56 knives have one-host KEEP evidence. Their complete current profiles and
  DP8/DP16 target geometries have not run at target scale.
- P59 is accepted under ordinary-JAX FP64 gradient correctness: the oracle is
  rel-L2 `3.91e-16`, the frozen real-Qwen gradient gate records `1.582%`, and
  DP4 reverse measured 3.605x. Serial and parallel AdamW first-step deltas
  differ by rel-L2 `9.976%`; do not claim trajectory identity.
- Attempt-1 overturned the old TP4/TP8 construction ceiling: the prior test
  pre-sharded `dlogits` and did not cover the production DP-only full-vocab
  carrier. The new test starts from that production placement and the full
  postflight requires `head_cotangent_partition_ready`; it still needs the
  dependency-complete pinned image and real target before promotion.
- APC passed Phase3 one-host G-A through G-D, including the dirty-page negative
  control and matched performance/XProf. Attempt-2 M15/main failed G-E and its
  APC knife is VETOED. P45 did not record the same numerical red, but the user
  chose the lower-risk uniform production policy, so P45 is also APC-off.
  Strict A(APC)-B(full-reset)=0 bytes remains mandatory for the separate APC
  debugging thread and was not relaxed.
- JAX persistent cache is configured in all three rendered manifests and its
  host restore/save carrier is tested. Attempt-6 restored zero entries under
  the old silent script, so no target cache hit or JIT reduction is claimed.
- Qwen3-8B TP8 fixed-head code/overlay construction and pinned-image
  construction gate are green. The DP8xTP8 target is pending; TP4
  certification does not transfer to TP8.
- The current supported bundle passed a Qwen3-1.7B DP4xTP1 one-host v5p proxy:
  3/3 optimizer transactions, 51/51 strict alignment PASS, 0 FAIL. This proxy
  excludes APC and fixed LM head because those registered geometries are not
  represented by 1.7B/TP1. It does not certify any 64-chip topology or
  performance.

## Admission commands

From the worktree root, run the host gates exactly:

```bash
bash canon-zero-tim/tests/v1_phase4/run_cpu.sh
bash canon-zero-tim/tests/p57_frozenlake_tim/run_cpu.sh
python3 -m unittest discover -s canon-zero-tim/tests/p59_backward -p 'test_*.py'
python3 -m unittest discover -s canon-zero-tim/tests/p3_prefix_cache -p 'test_*.py'
python3 canon-zero-tim/.claude/skills/manage-canon-flags/scripts/audit_flag_registry.py \
  --repo . --changed-base origin/yuxzhang/canon-zero-tim
git diff --check
```

The historical pinned-image gate was executed against image
`sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`:

```bash
bash canon-zero-tim/tests/v1_phase4/run_exact_image.sh \
  sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a
```

Require the exact terminal marker:
`V1_HP_EXACT_IMAGE_PASS dp16_gathered=1 dp2tp2_parallel=2 p59_tp4_tp8=2 p59_real_shim=4 p57_wandb=1 perfetto_window=1 manifests=3`.
This is an exact-image admission receipt for the pre-attempt-1 tree. The
historical receipt is not a signed raw-log artifact: the
stdout/stderr log was not durably preserved, so no raw-log path or SHA exists.

The current Attempt-3 repair must instead end with additive `p59_rpa=2` and
`m15_token=1` fields. They prove the installed-attention DP2xTP4/DP2xTP8 VJP2
carriers with wrong-cache and ordinary-serving negatives, plus the signed M15
4096/8192 positive and partial/foreign negatives:
`V1_HP_EXACT_IMAGE_PASS dp16_gathered=1 dp2tp2_parallel=2 p59_tp4_tp8=2 p59_real_shim=4 p59_rpa=2 p57_wandb=1 m15_token=1 perfetto_window=1 manifests=3`.

That Attempt-3 gate is now green on tested commit
`f0af2d9b31d3ca1324549df3660ebc6894856b74`, tree
`24675392adee620ab36b87f9a0c4f7e8111f4839`. Durable logs and the signed
receipt are under
`evidence/v1_hp_attempt3_fix_exact_image_20260824_r1/`: P58 raw SHA-256
`a07f05631373c13c54f03906dbda5b07b3d9981ab50148b7e48d23f88037534e`,
V1 raw SHA-256
`d9fe0af37025abd20a6027027ed995849a301ef9b5a2c69fecb00fcfa028861d`,
and receipt SHA-256
`16bc0f85921b40e1a0e6dbcbd6187329199c6833c99d5f1b280eca14e58305cb`.
Both scripts exited zero; both include `P59_TP_SHIM_EXACT_IMAGE_PASS` with
`installed_attention=2`, and the P58/V1 terminals include `p59_rpa=2` plus
`m15_token=1`. This is dependency-complete CPU/image admission, not a target
optimizer or performance result.

The current post-fix gate passed on the same immutable image. Its raw log is
`evidence/v1_hp_postfix_exact_image_20260824_r3/run.log` with SHA-256
`7ef23c9b7f4997a1855a16e99e348e4c981a1f80f9614cc95be1703771338264`;
its receipt SHA-256 is
`4c99f542ea6907ad48f7d716e8bb9db2db77865a3fec136e3cf88bcd5ec82f5f`.
It contains one required V1 terminal plus
`P59_TP_SHIM_EXACT_IMAGE_PASS ... topologies=DP2xTP4,DP2xTP8 ...`, and no
unittest failure terminal. Failed r1/r2 carrier logs are preserved beside it.
This is dependency-complete CPU/image admission, not target execution.

The attempt-2 repair passed the complete gate again on that image. The raw log
is `evidence/v1_hp_attempt2_fix_exact_image_20260824_r4/run.log`, SHA-256
`281c13a6c0b4dd84a3a19505b1f147ee8e4aaaeff9161738a9a2c521f6813dbc`;
receipt SHA-256 is
`3db65eef408e92534ee0759437800b79c445fd8fb556ac2447309d3618ea9364`.
The focused r3 additionally records real q_proj `layout_shards=1` under TP4
and TP8, exact serial/parallel gradients, fixed TP input reduction, fused-layout
positive control, wrong-width negative, and ordinary-serving global negative.
Failed r1/r2 carrier logs remain immutable. This is still not a target or
optimizer-commit result.

The separately approved one-host v5p integration proxy is frozen at evidence
root
`/mnt/disks/tunix-data/logp_probe_1host/p59_dp4_v1_v1hp_20260823_0824utc`.
Its terminal marker is
`[P59.DP4] GREEN kind=v1 zero_tim=51/51 fail=0`; all six entries in its
`SHA256SUMS` verify.

## Launch and postflight

The approved target plan uses direct full trains, not separate short canaries.
After the repair is committed, pushed, and exactly read back, render from that
new exact source SHA using `RUNBOOK.md`, require three manifest PASS receipts,
and freeze every YAML hash. With separate launch approval and all three
64-chip allocations confirmed, apply GSM8K, P45, and M15 in one wave. Each
first real optimizer commit is that recipe's independent early admission
checkpoint, not a shortened run: require zero real alignment FAIL plus its
registered P59-local, fixed-head, token/APC-off, and optimizer receipts, then let
the same JobSet continue to its full horizon. A red freezes only that recipe;
it does not stop another healthy full run. Each must receive its own complete
strict-alignment, P59/APC-off/fixed-head, cache, timing, XProf, Perfetto, eval,
and horizon postflight. A GSM8K green does not certify the experimental APC
path, TP8 fixed head, DP8xTP8,
FrozenLake evaluation, or M15 workload geometry.

Any real `CANON_ALIGN` or `CANON_ALIGN_PRE verdict=FAIL` kills that recipe.
Missing horizon, receipts, trace, checkpoint, or artifacts is INCONCLUSIVE,
not PASS. Performance judgment comes from `[PERF]`; XProf/Perfetto provides
operation attribution and never overrides the bitwise gate.

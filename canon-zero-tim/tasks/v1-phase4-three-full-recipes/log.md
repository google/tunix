# Log

## 2026-08-24T07:00:22Z — Attempt-3 repair committed locally; push gate remains closed

- The user authorized commit and push. A fresh fetch proved the operator tip still equals the immutable Attempt-3 base `614156c1ab067192ab65b2969543e23904f192be`; no rebase is required.
- Runtime, append-only patch, manifest, classifiers, installed-image carriers, and one-host carriers are isolated in local CL `8a9c8019` (`Fix P59 local RPA head handling`). Its English imperative commit body records the added boundary check/receipt and the unverified DP16xTP4 target as the drawback.
- This evidence/registry/handoff update is isolated as the following local CL. Push is not executed yet because the new installed-attention DP2xTP4/TP8 pinned-image gate is explicitly separate-approval-bound and has not run. The real-v5p DP2xTP2/DP1xTP4 mechanism PASS does not silently substitute for that gate.
- Rollback: revert the ledger CL first, then `8a9c8019`; preserve both one-host attempt directories and all target failure evidence.

## 2026-08-24T06:39:21Z — Attempt-3 RPA local-KV seam isolated and repaired on host

- Incoming source/evidence: fast-forwarded to `614156c1ab067192ab65b2969543e23904f192be`; both Attempt-3 artifacts pass `SHA256SUMS`. In `evidence/v1_hp_three_full_attempt3_20260824/gsm8k_g64m_error.log:11082`, step-0 pre-alignment passes for 194,633 action elements with `S_decode_vs_S_prefill=0` and `S_prefill_vs_T_old=0`; there is no alignment FAIL and no optimizer commit.
- First red interval: forward completes all 16 groups, then the P59 head boundary passes at `gsm8k_g64m_error.log:12066`. The first fatal is installed `rpa_diff_chunked.py:204`, recorded at `gsm8k_g64m_error.log:12192`; `gsm8k_g64m_error.log:12205` reports actual local cache `(9,256,2,2,128)` versus expected `(9,256,4,2,128)`.
- Root cause: P59's outer manual DP×TP map already produced TP-local K/V (2 KV heads on TP4) and a matching local cache. The stock attention prelude saw `2 < tp_size=4` and executed its global GQA repeat a second time, so RPA derived a false four-head cache contract. This is a pre-optimizer shape-contract stop, not a numerical verdict.
- Repair: append-only patch `25-attention-p59-local-kv.patch` skips the stock repeat only under the exact P59 two-manual-axis context, validates local Q/K/V/cache and emits `P59_RPA_LOCAL_KV_READY`. Ordinary serving retains the stock global GQA repeat even when the P59 flag is present. The full classifier requires exact TP4/TP8 local head/cache/packing receipts and has missing/wrong-shape negatives.
- Host result: V1 21/21, P57 144/144, P59 34/34, APC 31/31, flags 366/366, syntax and diff hygiene pass. The patch applies to pinned stock; Qwen3-1.7B overlay construction/manifest is 36/36 with generated attention SHA `58d102e8c385368e7d1b47ce81ff3e866a8a1c43ba0b370a5da4aea729fb52f7`.
- Claim ceiling: `IMPLEMENTED / HOST+PINNED-STOCK CONSTRUCTION PASS / ONEHOST MECHANISM PASS / EXACT-IMAGE EXECUTION NOT RUN / TARGET NOT RUN`. The new installed-attention DP2×TP4/TP8 VJP2 carrier, wrong-cache negative, and ordinary-global-GQA control are wired into the exact-image gate but require separate execution approval. The bounded four-chip TPU mechanism gate ran as recorded below; no commit, push, render, Kubernetes mutation, production target, or optimizer commit occurred.
- One-host Attempt 1 `p59_rpa_a3dp2tp2tp4_20260824_0645utc` stopped in 18 seconds before RPA compilation because the carrier required a device-kind string containing `v5p`, while this fixed v5p host reports the exact JAX string `TPU v5`. The immutable raw log and `SHA256SUMS` remain under `/mnt/disks/tunix-data/logp_probe_1host/`; this is `TEST_CARRIER_IDENTITY_RED`, not an RPA or numerical result. The repaired identity gate retains the pinned hostname, exact four-TPU count, and exact `TPU v5` kind, and requires a fresh label.
- One-host Attempt 2 `p59_rpa_a3dp2tp2tp4_20260824_0648utc_r2` passed on the pinned four-chip v5p in 32 seconds. It ran the real installed RPA kernel and VJP2 backward under P59 `DP2xTP2`, emitted the exact local receipt (`q=4, kv=1, cache=1, packing=2`), produced four finite gradient norms, caught the deliberately wrong cache shape, then rearranged the same devices as ordinary `DP1xTP4` and passed the unchanged stock global-GQA expansion. Terminal count is 1, local marker count is 1, traceback count is 0, optimizer commits are 0, and both `SHA256SUMS` entries verify. Raw/driver SHAs are `c28af46bda81e262a7de282d75d5f809dd0813eb364ff98a2153eba47b9f5826` / `7e35d4c4f0056b94ea64e4131a08ea1a2d6d89d625f81ce187e7a82542aac99b`.
- Rollback: revert the additive Attempt-3 repair CL only; never delete Attempt-3 or earlier failure evidence.

## 2026-08-24T05:13:22Z — attempt-2 reds classified and repaired locally

- Incoming source/evidence: fast-forwarded to `238ca28cf6eb642429de66c0da58b68ea659309f`; all four entries in `evidence/v1_hp_three_full_attempt2_20260824/SHA256SUMS` verify.
- GSM8K `g64k` and P45 `f45i`: strict step-0 pre-alignment PASS, then no optimizer commit. Both hit `P59 local fused-linear split ... sizes=(128,) tp=1` in real q_proj backward. Root cause: the shim confused engine fused-layout `config.n_shards` with the live mesh TP degree; a non-fused q_proj legitimately has one layout shard inside TP4/TP8.
- M15 `m15i`: hard numerical FAIL before backward. `S_decode_vs_S_prefill` differs on 760 elements / 1389 bytes with max abs `0.998443603515625`; `S_prefill_vs_T_old` is exact. This isolates the red to APC-on decode, so APC is target-VETOED and reverted only for M15/main. P45 remains APC-on. No gate tolerance changed.
- Repair: the local P59 layout helper admits positive `n_shards=1`, retains invalid/divisibility/width failures, and updates `MANIFEST.sha256`. The installed-shim carrier now executes real q_proj `(128,),1` under DP2xTP4 and DP2xTP8, while retaining fused-layout, wrong-width, and ordinary-global controls. Full postflight requires exactly one explicit APC-off runtime receipt for M15 and rejects an unexpected APC-on marker.
- Final admission: after adding missing/duplicate/opposite APC-off marker negatives, host gates pass V1 19/19, P57 144/144, P59 31/31, APC 31/31, and flags 366/366. Full pinned-image r5 passes with raw-log SHA `90affa9db1ca8ba4df6d7334aa7897aa9bd77492d93fd1378753396ff531556e` and terminal `V1_HP_EXACT_IMAGE_PASS ... p59_real_shim=4 ... manifests=3`. Post-fix target remains unrun.
- Host PASS: V1 18/18, P57 144/144, P59 31/31, APC 31/31, flags 366/366, syntax, manifest, and diff hygiene. Bare-host P33 broad discovery still lacks optional `datasets`/`metrax`; 29 other tests pass and the dependency-complete image gate supersedes it.
- Exact-image evidence: r1 is an immutable sandbox/sudo infrastructure red; r2 is an immutable trailing-`None` sharding-assertion carrier red; focused r3 passes with SHA `a556a808e7ba5a40e7b8f4d45e8398af8b9ec3a216286ca642e91985da9af50d`; complete r4 passes with raw-log SHA `281c13a6c0b4dd84a3a19505b1f147ee8e4aaaeff9161738a9a2c521f6813dbc` and receipt SHA `3db65eef408e92534ee0759437800b79c445fd8fb556ac2447309d3618ea9364`.
- Claim ceiling: `HOST PASS / EXACT_IMAGE PASS / ATTEMPT2 TARGET REDS PRESERVED / POST-FIX TARGET NOT RUN`. No commit, push, manifest render, Kubernetes mutation, TPU run, or post-fix optimizer commit occurred.
- Rollback: revert the evidence/ledger CL, then the M15 APC profile/classifier CL, then the P59 layout/manifest/carrier CL. Never delete attempt-2 or r1-r4 evidence.

## 2026-08-24T03:57:06Z — direct-full launch order confirmed

- Decision: all three target jobs are direct full trains; there is no separate short canary.
- Order: render only from published/read-back SHA `71d889a32f4668353c758d5c00df88299e6c0d35`, start the 200-update GSM8K full train first, and treat its first real optimizer commit as an early admission checkpoint rather than a stopping horizon.
- Gate: require zero real alignment FAIL plus the registered P59-local, fixed-head, and optimizer receipts at that checkpoint. Keep GSM8K running toward its full horizon; after this checkpoint P45 followed by M15 may start as 300-update full trains from the same source SHA.
- Claim boundary: this records the approved execution sequence only. No target manifest was rendered and no TPU or Kubernetes resource was started by this checkpoint.

## 2026-08-24 UTC — post-fix pinned-image r3 admitted

- Focused r3 ran the revised non-head DP2xTP2 carrier and the retained rank-2
  generic-head carrier: 2/2 PASS. Focused raw-log SHA-256 is
  `eae6378df8339481b93a60592b2808a74cd4a4cf9c1093536e19ff6f2f04e71a`.
- The complete V1 pinned-image script then exited zero and emitted exactly one
  `P59_TP_SHIM_EXACT_IMAGE_PASS ... topologies=DP2xTP4,DP2xTP8 ...` plus
  exactly one `V1_HP_EXACT_IMAGE_PASS ... p59_real_shim=4 ... manifests=3`.
  No unittest FAILED or traceback terminal is present. Raw-log SHA-256 is
  `7ef23c9b7f4997a1855a16e99e348e4c981a1f80f9614cc95be1703771338264`;
  receipt SHA-256 is
  `4c99f542ea6907ad48f7d716e8bb9db2db77865a3fec136e3cf88bcd5ec82f5f`.
- The old DP2xTP2 endpoint test is now honestly scoped to non-head endpoints.
  Generic head remains covered at rank-2 TP1, while actual P59 TP-local head
  VJP and fixed TP input reduction remain independently covered by installed
  fixed-head TP4/TP8 tests. Runtime rank-2 enforcement was not loosened.
- Result: `HOST PASS / EXACT_IMAGE PASS / TARGET NOT RUN`. Source publication
  is next; no render, Kubernetes object, TPU use, or optimizer commit occurred.

## 2026-08-24 UTC — post-fix pinned-image attempt 1 stopped at stale DP2xTP2 carrier

- Command: `bash canon-zero-tim/tests/v1_phase4/run_exact_image.sh`, with raw
  stdout/stderr written directly to
  `evidence/v1_hp_postfix_exact_image_20260824/run.log` and no launch pipeline.
  Raw log SHA-256 is
  `621edbe196233dd00bcefe68790d30b9c7fd929f9f18c8071627efa828d8c2b1`.
- Result: exit 1. The dependency-complete gate reached
  `P59_TP_SHIM_EXACT_IMAGE_PASS ... manifests=2x36/36`, then the older
  `CanonicalQwen3AdapterTest.test_p59_rank_parallel_endpoint_pullbacks_match_serial_dp2_tp2`
  supplied rank-3 `(2,3,3)` logits cotangents at
  `tests/rl/canonical_qwen3_adapter_test.py:763`. The production repair rejects
  that at `tunix/rl/canonical_qwen3_adapter.py:478` because its logical carrier
  is rank 2. The V1 terminal PASS was absent.
- Classification: `TEST_CARRIER_RED / EXACT_IMAGE FAIL / TARGET NOT RUN`.
  This is not an alignment verdict and no optimizer transaction, TPU, render,
  or JobSet occurred. The failed evidence is immutable.
- Repair under validation: retain the strict rank-2 production API; update the
  old DP2xTP2 test to use flattened logical rows and a TP-divisible toy
  vocabulary, with serial per-rank masks over row ranges. Do not add rank-3
  compatibility to runtime code.

## 2026-08-24 UTC — P59-local release contract hardened on host

- Exact remote readback completed for the first repair stack at
  `dfec27378bfdd9b73b7bf8f7930bacaa685b3a16`; the earlier review snapshot that
  reported remote `5f3e8ff9` is superseded.
- Both FrozenLake TIM and V1-HP runtime branches now select learner M2048. The
  fixed-head receipt classifier has a fail-closed P59-local mode for the exact
  DP16/M4096/M256 and DP8/M2048/M256 contracts, one local chunk, and the
  barrier-pinned fixed TP input reduction. Ordinary request receipts cannot be
  substituted for the P59-local learner receipt.
- Full postflight now requires the exact recipe profile, global/local head
  cotangent shapes, P59-local primal shape, one-chunk VJP, and TP reduction.
  Negative controls reject wrong global/local shape, wrong profile, wrong
  chunks, missing reduction, and invalid global-M/DP pairs.
- Host result: V1 17/17, P57 144/144, P59 31/31, APC 31/31,
  fixed-head/receipt 32/32, flags 366/366, Python/Bash syntax, and diff hygiene
  PASS. The follow-up stack remains uncommitted until its explicitly approved
  post-fix pinned-image gate completes.
- Rollback: revert the evidence CL, then the carrier CL, then the contract CL.
  The already-published attempt-1 runtime repairs and immutable failed logs
  remain intact.

## 2026-08-24 UTC — attempt-1 repair reconstructed for authorized publication

- The user explicitly authorized commit and push. The repair is split into the
  P59 head-cotangent placement CL, the FrozenLake M2048 fixed-head CL, and this
  registry/evidence/handoff CL so both runtime concerns remain independently
  reversible.
- A fresh fetch proved the operator tip still equals immutable evidence base
  `5f3e8ff95075642b5e660af8d1219e1c98e71c72`; the local stack is linear and
  fast-forwardable.
- This publication is source delivery only. The post-fix pinned-image gate is
  still separately approval-bound and remains mandatory before any render or
  launch; no TPU or Kubernetes action is authorized by this entry.

## 2026-08-23 UTC — attempt-1 target shape boundaries repaired on host

- Incoming immutable evidence: `g64f` and `f45g` under `evidence/v1_hp_three_full_attempt1_20260823/`; all four archive SHA checks pass. `g64f` has 1 alignment PASS, 0 FAIL, 0 completed updates; `f45g` has 0 alignment records and 0 completed updates.
- GSM8K first fatal: the processed-logprob VJP produced the logical `[256,151936]` cotangent with DP-only placement, while the P59 outer manual DP/TP map selected the fixed-head local-output boundary `[256,37984]`. The old forced-CPU test supplied `P(dp,tp)` in advance and therefore did not exercise this seam. The repair explicitly device-places the rank-2 cotangent as `P(data,model)` before constructing/invoking the cached P59 head map, emits `head_cotangent_partition_ready`, and rejects non-divisible vocabulary widths.
- FrozenLake first fatal: Qwen3-8B/TP8 C-forward legitimately uses learner M2048, but the fixed-head registry admitted only M4096. The repair admits M2048 only for the 8B/TP8 geometry, maps it to eight M256 chunks, makes VJP receipts dynamic, and passes `--learner-m 2048` only for the FrozenLake training profile. Other model/topology geometries continue to reject M2048.
- Postflight hardening: full-recipe classification now requires the new P59 head-partition receipt; the fixed-head receipt classifier rejects an M4096/VJP receipt substituted for FrozenLake M2048.
- Host results: fixed-head plus receipt 27/27, P59 31/31, P57 139/139, V1 13/13, APC 31/31, and 121 executable P38 serving tests pass. The broad P38 host discovery has one import-only error because optional `metrax` is absent; the production installed-shim TP4/TP8 test is therefore `INCONCLUSIVE` in bare host Python, not FAIL.
- Claim ceiling: `IMPLEMENTED / HOST+STATIC PASS / POST-FIX EXACT-IMAGE NOT RUN / TARGET NOT RUN`. No pinned image, TPU, Kubernetes object, optimizer transaction, commit, or push was created by this repair turn.
- Rollback: revert the adapter cotangent placement/postflight receipt concern independently from the geometry-specific M2048 fixed-head/receipt concern; keep the immutable failed logs and this superseding classification.

## 2026-08-23 UTC — target bootstrap reds consumed by P58.8 repair

- Incoming evidence: GSM8K DP16 x TP4 stopped in P59 head pullback on the trainer `dp/tp` versus six-axis engine shard-map context mismatch; FrozenLake DP8 x TP8 stopped earlier on the signed P57 Zero/full W&B project check. Neither run committed an optimizer update.
- Local follow-up: P58.8 adds an exact-device TP4/TP8 P59 two-axis engine carrier, barrier-pinned FP32 TP input-cotangent sums, and the exact-profile P57 W&B admission. On latest base `ccbcf572`, complete P58 and V1 pinned-image gates pass with `p59_real_shim=4 p57_wandb=1`; P59/P57/V1 host suites are 30/30, 136/136, and 12/12.
- Claim boundary: this repairs source admission only. The failed run directories remain immutable; DP16 x TP4 and DP8 x TP8 targets must use fresh run IDs after publication and still require strict alignment, P59, fixed-head, optimizer, XProf/Perfetto, and full-horizon evidence.

## 2026-08-23 UTC — V1.P4.1: bind three-recipe integration

- Type: decision / implementation
- Fact: the remote source contains P57 P45/M15 450-update contracts but not the local P56, P59, or APC performance implementations.

## 2026-08-23T06:42:26Z — host and real-env admission

- Integrated the default-off P56 serving chain, P59 rank-parallel trainer path, and Phase3 APC path.
- Added exactly three strict full profiles/manifests and a full postflight. Semantic Perfetto now serializes only training step 2 instead of every update; XProf captures the same warmed update. The classifier excludes that step from the steps2+ mean.
- PASS: V1 10/10; P57 120/120; APC 31/31; P59 28/28; flag audit 351/351; synthetic installed manifests 34+2+2.
- Negative PASS: partial bundle rejected, run-id/output reuse rejected, short/nonzero P57 high-performance recipe rejected, one real ALIGN FAIL rejected, missing XPlane rejected.
- Not run: pinned exact-image gate (separate approval); no TPU, Kubernetes apply, commit, or push.
- Action: created an isolated local integration branch; imported default-off implementations; added a DP-aware gathered-logprobs path and workload-scoped v1 profiles.
- Result: implementation in progress; target not run.
- Rollback: leave `CANON_V1_HP_FULL` unset and use the existing profiles; all new behavior remains unreachable.
- Next: render and verify the three manifests locally.

## 2026-08-23T06:52:19Z — exact-profile audit and regression closeout

- Audited every exact legacy P57 profile comparison before target launch. Added the V1 FrozenLake profile only to materialized-workload, signed full-horizon, zero-arm purity, and fixed-head receipt admission. The legacy request-only/eval and token-IS paths still reject the V1 profile.
- PASS: V1 renderer/classifier/real-env 10/10; P57 adjacency 120/120; P59 28/28; APC 31/31; flag registry 351/351 with `FLAG_AUDIT_PASS`; Python/shell syntax and `git diff --check`.
- The first flag audit exposed a runtime marker as a false settable name after a literal was rewritten. Restored the established split-literal form without changing emitted bytes; the deterministic registry audit is green.
- Not run: pinned exact-image gate (separate approval); no TPU, Kubernetes apply, commit, push, or image publication.
- Next: run `tests/v1_phase4/run_exact_image.sh` after explicit image-execution approval.

## 2026-08-23T07:15:43Z — merge latest P57 300-step setup

- Fetched remote `9c422bd2` and advanced the local integration branch base to that exact commit while preserving the uncommitted Phase3/P56/P59/V1 worktree.
- Integrated the signed P57 setup: P45/M15 full horizons are 300; seed contract is data=42/vLLM=0; rollout-only held-out evaluation runs at `0,50,...,300`; trainer eval forward/backward/optimizer remain excluded; milestone retention is zero.
- Updated both optimized FrozenLake profiles/manifests and the V1 postflight to require the P57 in-process evaluation classification. This is part of each uninterrupted full JobSet, not a short 64-chip canary.
- PASS after integration: V1 10/10; latest P57 127/127; P59 28/28; APC 31/31; flags 351/351; syntax and diff hygiene.
- Not run: pinned exact-image; no TPU, Kubernetes apply, commit, push, or image publication.

## 2026-08-23T07:19:09Z — separate evaluation cost from training performance

- Added dual timing views for FrozenLake: raw steady timing retains in-process evaluation cost, while training-only steady timing excludes policy-step `50,100,...,250` evaluation cycles and the profiled update.
- PASS: focused timing negative/positive test raises the V1 host suite to 11/11; flag audit remains 351/351.

## 2026-08-23T07:41:39Z — correct eval-cycle wall-row identity

- Type: bug fix / launch admission hardening
- Fact: FrozenLake evaluation uses pre-update policy step `s`, but the enclosing completed wall row is global step `s+1`. The earlier policy-step-based exclusion therefore removed the adjacent ordinary cycle and retained the actual eval cycle.
- Action: `tunix/rl/agentic/agentic_rl_learner.py:3308` now emits and asserts explicit `[P57.EVAL.CYCLE] policy_step=s enclosing_global_step=s+1` receipts; final policy step 300 emits `none`. `classify_inprocess_eval.py:106` validates the full mapping, and `classify_full_recipe.py:151` consumes it without inference. Renamed the derived timing view to `direct_eval_cycle_excluded` because APC cache occupancy prevents a counterfactual training-only claim.
- Registration: `FLAGS.md:19` now records Qwen3-8B TP8 fixed-head construction separately from TP4 certification; pinned-image and DP8xTP8 target remain unverified. Added a branch-local `HANDOFF.md` and exact P59/APC host commands.
- Result: PASS by V1 11/11, P57 128/128, P59 28/28, APC 31/31, flag audit 351/351, Python/shell syntax, and `git diff --check`.
- Not verified: pinned exact-image execution and all current-bundle TPU/64-chip targets; no commit, push, image execution, TPU run, or Kubernetes mutation occurred.
- Next: obtain separate pinned-image execution approval, run `tests/v1_phase4/run_exact_image.sh`, then freeze the intent diff for commit review.

## 2026-08-23T08:05:00Z — pinned-image failures repaired, final gate green

- Type: exact-image validation / fail-closed repair
- First red: the installed `tpu_runner_p21_l30.py` failed its manifest because patch 24 used a zero-context line-number insertion and patch 22 shifted the target. The XProf-label hunk landed inside `_p38_serving_begin`. Replaced it with a contextual insertion after `extract_lora_metadata()`; rebuilt output is syntax-valid and retains the registered SHA `0dc495ca...e0c` without rewriting the manifest.
- Second red: the V1 final gate imported `/app/tunix` instead of the read-only mounted worktree. Added `PYTHONPATH=/workspace` and a static negative control. The next fail-closed run then exposed a missing `CANON_P32_WORKLOAD=gsm8k` in the forced DP16 gathered-logprobs test; supplied the exact workload contract to that test and its latent DP16 sibling.
- Result: PASS against pinned image `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a` with terminal `V1_HP_EXACT_IMAGE_PASS dp16_gathered=1 dp2tp2_parallel=2 perfetto_window=1 manifests=3`. V1 host gate increased to 12/12.
- Limitation: this proves installation, overlay construction, forced DP16 gathered-logprobs, DP2xTP2 P59 structure, and Perfetto contracts; it is not target-hardware execution.

## 2026-08-23T08:24:12Z — current bundle one-host v5p proxy green

- Type: direct TPU integration / strict Zero-TIM gate
- Action: added a fail-closed V1 DP4xTP1 carrier that replaces the older P59 A/B recipe with the final supported serving/trainer flags. It explicitly keeps APC and fixed LM head off because the one-host 1.7B/TP1 proxy cannot represent their registered production geometries.
- Command: `bash canon-zero-tim/tasks/p59-dp16-parallel-backward/scripts/run_onehost_dp4.sh v1 v1hp_20260823_0824utc` on `t1v-n-4a77ebd0-w-0`, with no launch pipeline.
- Result: PASS by classifier: 3/3 committed optimizer transactions, 48/48 group alignments plus 3/3 pre-alignments, total 51/51 PASS and 0 FAIL, one rank-parallel pullback per transaction, exact DP replicas. Positive update evidence at step 1 records effective LR `4.000000330961484e-09` and `1,185,315,230` changed parameter elements.
- Performance: cold step 0 wall `469.22s`; step 1 wall `100.47s`; step 2 wall `83.75s`. Step 2 decomposes to forward `6.426s`, reverse `33.494s`, optimizer `0.303s`, weight sync `23.750s`; classifier stable sample count is 1, so this is diagnostic and not an 8-step tail verdict.
- Evidence: `/mnt/disks/tunix-data/logp_probe_1host/p59_dp4_v1_v1hp_20260823_0824utc`; all six `SHA256SUMS` entries verify. The container exited and released the TPU lane. Root-owned artifact chmod warnings were non-fatal; every judgment artifact is readable and hash-valid.
- Limitation: verified by Qwen3-1.7B DP4xTP1 one-host v5p only; not verified because APC, TP4/TP8 fixed head, DP16xTP4, and DP8xTP8 are outside this proxy geometry. This strict integration run used `capture=0`, so it produced semantic Perfetto evidence but no new XPlane; exact-image validated the XProf plumbing and each full target remains configured to capture its warmed update window. No commit, push, image publication, or Kubernetes apply occurred.

## 2026-08-23T09:09:24Z — committed-tree registry and runtime audit green

- Reconstruction committed the previously untracked P59/APC/V1 carriers and therefore exposed names that the dirty-tree-only audit could not inspect. The release registry is now 359/359 with `FLAG_AUDIT_PASS`; the earlier 351/351 entries above remain accurate historical receipts for the dirty-tree gates.
- Final host rerun: V1 12/12, P57 128/128, P59 30/30, APC 31/31, flags 359/359, and diff hygiene PASS.
- The final committed source differs from tested freeze tree `331ac60940b2c754fa516d94eaf039513b41dc11` only in release evidence/registry/count assertions and one host-script trailing blank. All runtime paths are byte-identical, so no TPU rerun is required.
- Fresh fetch found remote tip unchanged at `9c422bd224671a4ee0c6795223d0168debd4ca62`. No rebase, push, rendering, or launch occurred.

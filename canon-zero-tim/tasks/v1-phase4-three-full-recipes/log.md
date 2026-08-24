# Log

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

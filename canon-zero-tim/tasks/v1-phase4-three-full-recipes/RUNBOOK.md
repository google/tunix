# V1 high-performance three-full runbook

## Current — render the matched GSM8K Native/mismatch and Zero controls

Use the dedicated handoff at
`../v1-gsm8k-native-full-control/HANDOFF.md`. The Native preparation wrapper
is `../v1-gsm8k-native-full-control/prepare_gsm8k_native_full.sh`; the Zero
wrapper is `scripts/prepare_gsm8k_full_dp16tp4_p74.sh`. Both are render-only,
must receive the same clean published source SHA and fresh identities, and
must resolve the same W&B project/group
`zero-tim-gsm8k-dp16-tp4` / `qwen3-1p7b-dp16-tp4`.

The Native manifest must select `CANON_GSM8K_VANILLA=1`, keep
`CANON_P32_WORKLOAD` and all alignment/P59/V1/P70/P71/P63 selectors absent,
remove the proxy excess-precision pin, and enter the stock-engine branch with
`canonical_overlay=skipped alignment=off`. The Zero manifest must contain the
registered system tuple documented in the handoff, while
`CANON_DP_COLLECTIVE_REDUCE` remains absent. Neither wrapper launches.

## Current — render the P45/M15 fast Zero pair, never launch from here

The selected wave is direct P45 plus M15/main optimized Zero full training,
with in-process evaluation and checkpointing both disabled for speed. First
finish source review, host gates, explicit commit/push approval, remote SHA
read-back, and a clean checkout at that exact SHA. Then use fresh IDs and a
never-used output directory:

```bash
bash canon-zero-tim/tasks/v1-phase4-three-full-recipes/scripts/prepare_p67_frozenlake_two_full_wave.sh \
  <approved-40-character-sha> \
  /tmp/v1-p67-frozenlake-full-<fresh-wave-id> \
  <fresh-campaign-root> \
  <fresh-p45-run-id> \
  <fresh-m15-run-id>
```

The wrapper refuses a dirty tree, SHA mismatch, reused output root, or
duplicate IDs. It hashes exactly two immutable manifests, preserves the P57
64-chip autoscale/exclusive-topology carrier, emits
`V1_P67_FROZENLAKE_WAVE_READY ... manifests=2 ... launch=not-executed`, and
only prints two unpiped `kubectl apply` commands. It never executes them.

After reviewing `manifest-index.json`, confirm both YAMLs and their profile
resolution contain the same published source SHA, `CANON_P59_CHECKED_VMA=1`,
`CANON_P67_P66_VMA_P59_ONLY=1`, `CANON_V1_HP_FIRST_UPDATE_GATE=1`, strict
alignment, APC-off, full/300 updates, evaluation-off, checkpoint mode
`disabled`, empty checkpoint residual fields, and the correct P45 versus
M15/main identity. Both must also resolve the reviewed system tuple:

```text
CANON_DP_COMPARE_MODE=fingerprint-hybrid
CANON_DP_DISTINCT_SCHEDULE=first-group-warmup
CANON_DP_FINITE_FETCH=batched-commit
CANON_P71_SCAN=fwd
```

The token-input diff is mandatory and must be the only TITO admission:

```text
frozenlake-p45: CANON_M15_TOKEN_CONTINUITY absent
frozenlake-m15: CANON_M15_TOKEN_CONTINUITY=exact
```

Reload both generated `env.sh` files and recheck the same tuple. Reject an
empty P45 key as presence, a M15 `verify` value, or any neighboring workload
with the selector. Before accepting M15 postflight, require at least one
runtime `mode=exact verdict=TOKEN_STREAM_EQUAL first_mismatch=-1` receipt and
require every token-continuity receipt to have matching lengths and SHA256.
Any missing or unequal token receipt is fatal; the finite A-B warning policy
does not cover token-input inequality.

`CANON_DP_COLLECTIVE_REDUCE` must be absent. The command must contain
`--eval_every_n_steps=0` and no `--num_test_batches`. Check no conflicting
workload is live. The user may apply both YAMLs together; never append a pipe
to either launch command.

Watch both raw logs immediately. For each recipe, update 0 must pass strict
prealignment, one checked-VMA/P67 resolved-env contract, one first-update
precommit receipt (`microsteps=32`, denominator 32, finite, nonzero,
`stable_norm <= 1e6`), and one valid `0 -> 1` AdamW commit receipt before
weight sync. Any mismatch, non-finite gradient, missing/duplicate receipt, or
invalid optimizer transaction is fatal for that recipe. Preserve all evidence
and let the other independently healthy recipe continue.

Do not call either run certified until the in-container full classifier passes
all 300 updates, all strict alignment records, both evaluation/checkpoint
disabled runtime markers, P59/fixed-head/reduction evidence, JAX-cache
receipts, XProf, Perfetto, and artifact hashes. These fast runs intentionally
have no held-out evaluation JSON, no resume point, and no final checkpoint;
an interruption requires a fresh run from step 0. Return the complete bundle
listed in the first section of `HANDOFF.md`.

## Historical — render the P45 TP8 matched diagnostic pair, never launch from here

Never render launch YAML from a dirty or merely local worktree. After exact
remote read-back and checkout of the clean published P4.10 SHA, use fresh IDs
and a never-used output directory:

```bash
bash canon-zero-tim/tasks/v1-phase4-three-full-recipes/scripts/prepare_fl_tp8_ab_wave.sh \
  <approved-40-character-sha> \
  /tmp/v1-fl-tp8-ab-<fresh-wave-id> \
  <fresh-p66-off-run-id> \
  <fresh-serving-scope-run-id>
```

The wrapper refuses a dirty tree, SHA mismatch, reused output root, or
duplicate IDs. It hashes exactly two P45 DP8×TP8 YAMLs and only prints two
unpiped `kubectl apply` commands. It never executes them. Both jobs use the
complete P45 rollout geometry and stop after one strict precheck with zero
backward and zero optimizer commits.

If 128 chips are intentionally allocated, launch the pair together. With only
64 chips, launch `p66-off` first; later `serving-scope` must use a new run ID.
Preserve all run directories. For each arm require one exact profile marker,
one P38 round, controlled exit 42, sufficient prefix depth, finite A−B, and
B−C exactly zero. Interpret the two-arm matrix only as written in
`phases/v1-p4-10-frozenlake-tp8-ab-recovery.md`; no result authorizes a full
train until the follow-up TP8 trainer-forward and backward gates pass.

After both jobs finish, return the per-arm artifact hashes, exact classifier
terminals, JSON fields, and paired matrix verdict listed under `Required
return bundle` in `HANDOFF.md`. Do not report Kubernetes `Complete` as
`ZERO_TIM_RECOVERED`; only the persisted classification JSON supplies that
outcome.

## Current — render all three checked-VMA full jobs, never launch from here

P4.9 prepares exactly three independent full jobs. Final manifests do not
exist yet because the runtime is intentionally uncommitted. After explicit
commit/push approval, remote read-back, and checkout of the exact clean
published SHA, use fresh IDs and a never-used output directory:

```bash
bash canon-zero-tim/tasks/v1-phase4-three-full-recipes/scripts/prepare_checked_vma_three_full_wave.sh \
  <approved-40-character-sha> \
  /tmp/v1-hp-checked-vma-<fresh-wave-id> \
  <fresh-campaign-root> \
  <fresh-gsm8k-run-id> \
  <fresh-p45-run-id> \
  <fresh-m15-run-id>
```

The wrapper refuses a dirty worktree, `HEAD`/approved-SHA mismatch, reused
output root, or duplicate run IDs. It renders and hashes exactly three YAMLs,
emits `V1_HP_CHECKED_VMA_WAVE_READY ... launch=not-executed`, and only prints
the three `kubectl apply` commands. It never executes them. Review
`manifest-index.json`, confirm the published SHA by remote read-back, check
that no conflicting P51/P59/P62/P64/Phase4 workload is live, and obtain launch
approval. Apply the three YAMLs without any pipe. The user may apply all three
in one wave; no recipe waits for another recipe's first update.

For each job, inspect the raw log immediately. Before accepting update 0 it
must contain:

- no real `CANON_ALIGN_PRE` or `CANON_ALIGN verdict=FAIL`;
- exactly one `[P59.CHECKED_VMA]` receipt for the expected topology;
- one `canon-v1-first-update-precommit-v1` receipt with all elements finite,
  at least one nonzero, `0 < stable_norm <= 1e6`, `microsteps=16` and
  `denominator=16` for GSM8K, or `microsteps=32` and `denominator=32` for
  either FrozenLake recipe; and
- one `canon-v1-first-update-commit-v1` receipt proving valid `0 -> 1`, finite
  gradient/delta, and learning-rate/parameter-change coherence.

The existing P59 reduction, P63, optimizer, strict alignment, APC-off, XProf,
Perfetto, JAX-cache, evaluation, checkpoint, and full-horizon receipts remain
mandatory. A missing or duplicate first-update receipt is fatal. If one
FrozenLake job is red, freeze and retain its complete evidence but leave the
other healthy jobs running. Do not start performance attribution or XProf
comparison until the user supplies run IDs after launch.

## Historical — P64 P45 capture, then optional diagnostic replay

The recovery chain is published and exactly read back at
`548db7e9f014def3cb2b37e66c6f0e62c2041f1d`. Do not launch from a dirty,
merely local, or different tree. The user must first choose launch matrix A
(GSM8K full plus P64 capture, recommended) or B (all four jobs, accepting that
unchanged P45/M15 full are known Step-0 backward reds). After that choice,
render one fresh P45 capture; the renderer never launches:

```bash
python3 canon-zero-tim/tasks/v1-phase4-three-full-recipes/scripts/render_p64_p45_numeric_debug.py \
  --source-commit 548db7e9f014def3cb2b37e66c6f0e62c2041f1d \
  --run-id <fresh-p64-capture-id> \
  --output-dir /tmp/v1-p64-<fresh-p64-capture-id> \
  --capsule-mode capture
```

Review the YAML and receipt, require
`P64_P45_NUMERIC_RENDER_PASS ... capsule_mode=capture optimizer_commits=0`,
then obtain launch approval. Never pipe the apply command:

```bash
kubectl apply -f /tmp/v1-p64-<fresh-p64-capture-id>/jobset-p64-p45-numeric-debug.yaml
```

Capture must pass strict prealignment, upload the immutable capsule and model
binding, then either identify the first non-finite boundary or complete all 32
groups and discard with zero commits. Preserve the raw log, classification,
capsule URI, capsule SHA, and binding SHA even when backward stops at first red.

Only when another observer iteration is needed, use a new run ID and the exact
captured hashes:

```bash
python3 canon-zero-tim/tasks/v1-phase4-three-full-recipes/scripts/render_p64_p45_numeric_debug.py \
  --source-commit <approved-40-character-sha> \
  --run-id <fresh-p64-replay-id> \
  --output-dir /tmp/v1-p64-<fresh-p64-replay-id> \
  --capsule-mode replay \
  --capsule-gcs-uri <captured-gs-uri> \
  --capsule-sha256 <captured-capsule-sha256> \
  --model-binding-sha256 <captured-binding-sha256>
```

Replay must print transport, producer-bypass, live-model verification, and
`backward_scope ... groups=1/32 selected=group0 optimizer_commits=0` receipts.
Its classifier evidence kind is `diagnostic-replay-not-certification`; never
use it as a fresh strict target verdict or optimizer result.

## Historical — P63 publication boundary before Attempt-7 recovery

The P62 first-red campaign is complete. Real DP16xTP4 G5b proved that the
backward tree is finite and the old `norm=inf` is naive FP32 sum-of-squares
overflow. The default-off P63 hybrid clip has passed the complete host and
pinned-image ladders on the final dirty runtime tree. Its evidence is under
`evidence/v1_hp_p63_exact_image_20260825_r2/`; require the unique terminal
`V1_HP_EXACT_IMAGE_PASS ... p63_clip=1 ... manifests=3` and verify its
`SHA256SUMS` before publication review.

The reviewed P63 concern is published at exact operator SHA
`98be7b291ddb92391f71d360dd59b09f83edc118`. Do not render or launch from a
dirty tree or substitute a local SHA. Render three fresh full manifests from
that exact SHA, audit their hashes, and obtain separate explicit launch
approval. The launch remains one concurrent
wave with no short canary: GSM8K DP16xTP4 for 200 updates, P45 DP8xTP8 for 300,
and M15 DP8xTP8 for 300.

All three rendered environments must resolve
`CANON_P63_OVERFLOW_SAFE_CLIP=1`; neighboring and diagnostic profiles must
omit it. Every committed update must emit exactly one `[P63.STABLE_CLIP]
update=` receipt with an all-finite tree, finite selected norm and clip factor,
and exact max norm 1 for GSM8K or 100 for FrozenLake. GSM8K postflight also
requires at least one observed fallback because G5b proved its stock norm
overflows. A non-finite gradient is fatal and must never be converted into a
fallback. These checks do not relax any strict alignment, P59, APC-off, XProf,
Perfetto, JAX-cache, evaluation, checkpoint, or full-horizon gate.

## Historical — Attempt-7 P62 first-red diagnostic before any full recipe

The three production JobSets are currently blocked. First publish one reviewed
40-character SHA containing P62, then render exactly one fresh GSM8K
DP16xTP4 `backward-no-commit` diagnostic. The renderer never launches:

```bash
python3 canon-zero-tim/tasks/v1-phase4-three-full-recipes/scripts/render_attempt7_numeric_debug.py \
  --source-commit <approved-40-character-sha> \
  --run-id <fresh-p62-id> \
  --output-dir /tmp/v1-p62-<fresh-p62-id>
```

Require one `P62_NUMERIC_RENDER_PASS ... optimizer_commits=0`, review the YAML
and receipt, verify no active P51/P59/P62 or three-full JobSet, and obtain the
separate 64-chip launch approval. Do not pipe the apply command:

```bash
kubectl apply -f /tmp/v1-p62-<fresh-p62-id>/jobset-p62-gsm8k-numeric-debug.yaml
```

The workload may exit nonzero when it deliberately catches the first
non-finite boundary. The launcher itself classifies the complete
`$CANON_STATE/run.log` before exiting and must print exactly one
`[P62.NUMERIC.POSTFLIGHT]` receipt containing the full-log SHA, byte/line
counts, classification path and classification SHA. A hand-selected P62
excerpt is not evidence. Before deleting the pod, copy both exact paths from
that receipt into a fresh append-only evidence directory and verify both
hashes. Offline reclassification is a second check, not a substitute for the
in-pod full-log classification:

```bash
python3 canon-zero-tim/tasks/v1-phase4-three-full-recipes/scripts/classify_attempt7_numeric_debug.py \
  <raw.log> \
  --output <fresh-evidence-dir>/classification.json
```

`ROOT_LOCALIZED_NONFINITE` and `FINITE_NAIVE_L2_OVERFLOW` are useful G5
findings, not permission to update weights. `ALL_BOUNDARIES_FINITE_NO_COMMIT`
means the recorded seams are finite but their magnitudes still require review.
`FATAL_CONTRACT` or `INCONCLUSIVE_INCOMPLETE` does not localize the numerical
root. Any alignment FAIL or optimizer commit is fatal. Only after a classified
G5 finding may a separate one-commit repair phase be designed.

This renderer prepares exactly three JobSets and never launches them. Use one
approved, pushed 40-character source SHA. Run IDs and the campaign root are
single-use; failed attempts are never reused.

```bash
python3 canon-zero-tim/tasks/v1-phase4-three-full-recipes/scripts/render_three_full_recipes.py \
  --source-commit 548db7e9f014def3cb2b37e66c6f0e62c2041f1d \
  --output-dir /tmp/v1-hp-three-full-a \
  --gsm8k-run-id <fresh-id> \
  --p45-run-id <fresh-id> \
  --m15-run-id <fresh-id> \
  --campaign-root <fresh-campaign-root>
```

Require three `V1_HP_MANIFEST_PASS` lines and one terminal
`V1_HP_THREE_FULL_RENDER_PASS manifests=3`. Do not hand-edit YAML. Before any
launch, run the package local/exact-image gates, record all YAML hashes from
`manifest-index.json`, confirm no earlier JobSet remains live, and obtain
explicit approval for each apply.

Local admission from the exact source worktree:

```bash
bash canon-zero-tim/tests/v1_phase4/run_cpu.sh
bash canon-zero-tim/tests/p57_frozenlake_tim/run_cpu.sh
python3 -m unittest discover -s canon-zero-tim/tests/p59_backward -p 'test_*.py'
python3 -m unittest discover -s canon-zero-tim/tests/p3_prefix_cache -p 'test_*.py'
python3 canon-zero-tim/.claude/skills/manage-canon-flags/scripts/audit_flag_registry.py \
  --repo . --changed-base origin/yuxzhang/canon-zero-tim
git diff --check
```

After those host gates pass, the separately approved pinned-image command is:

```bash
bash canon-zero-tim/tests/v1_phase4/run_exact_image.sh \
  sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a
```

The pre-attempt-1 2026-08-23 pinned-image admission against image
`sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
passed with terminal marker
`V1_HP_EXACT_IMAGE_PASS dp16_gathered=1 dp2tp2_parallel=2
p59_tp4_tp8=2 p59_real_shim=4 p57_wandb=1 perfetto_window=1 manifests=3`. It includes the production-image install/hash
checks for Qwen3-1.7B TP4 and Qwen3-8B TP8, but its P59 test supplied an already
TP-sharded cotangent and did not cover the real full-vocabulary seam exposed by
`g64f`. It does not admit the current repair. A post-fix rerun requires fresh
approval and remains a separate evidence event; require both TP4/TP8 installed-
shim cases plus the 8B/TP8 M2048 overlay contract.

The Attempt-3 repairs additionally require terminal fields `p59_rpa=2` and
`m15_token=1`.
The P59 field is emitted only after installed-attention DP2xTP4 and DP2xTP8 VJP2
carriers pass, including the wrong local-cache shape negative and an ordinary
global-GQA control with the P59 flag present. The M15 field additionally
requires the signed 4096/8192 positive plus partial and foreign candidate/split
negatives; P45 remains 4096/2048.

The Attempt-4 repair additionally requires `p59_fused_linear=2`. The
installed projection carriers must emit both gate and up receipts at TP4
(`declared_width=6144 local_width=1536`) and TP8
(`declared_width=12288 local_width=1536`), always with
`layout_shards=1 pieces=1`. Full target postflight requires the matching
recipe-specific receipts; missing, wrong-width, wrong-TP, or partial-site
evidence is fatal.

The current Attempt-6 admission additionally requires
`staged_spec_restore=2` in `P59_TP_SHIM_EXACT_IMAGE_PASS`. The complete green
raw log and receipt are under
`evidence/v1_hp_attempt6_apcoff_cache_exact_image_20260824_r1/`, raw SHA
`8d8d776451615de58a749c0be0200d28107b86cc44504200afde4f5acffc712a`.
The bounded real-v5p follow-up is
`/mnt/disks/tunix-data/logp_probe_1host/p59_rpa_a6restore_dp2tp2_20260824_2256utc/`;
it proves the replicated-leaf staged-spec positive and wrong-placement negative
at DP2xTP2 with zero optimizer commits. Neither gate substitutes for the first
real DP16xTP4/DP8xTP8 optimizer commit.

The complete Attempt-3 admission passed on tested commit `f0af2d9b`, tree
`24675392adee620ab36b87f9a0c4f7e8111f4839`. Use the durable P58/V1 raw logs
and receipt under
`evidence/v1_hp_attempt3_fix_exact_image_20260824_r1/`; do not substitute the
earlier rawless or pre-Attempt-3 receipts.

The current supported bundle also passed a separately approved one-host v5p
DP4xTP1 three-update proxy with 51/51 strict PASS and 0 FAIL. Evidence is at
`/mnt/disks/tunix-data/logp_probe_1host/p59_dp4_v1_v1hp_20260823_0824utc`.
It exercises gathered logprobs, fixed-AR gather, logprob step fusion,
continue-decode 8, batched report/evidence, and P59 rank-parallel backward.
APC and fixed LM head are excluded by construction, so this is launch-risk
reduction rather than target-topology certification.

The runs are uninterrupted full training: GSM8K 200 updates; P45 and M15 300
updates. FrozenLake writes rolling recovery checkpoints every 10 and emits the
signed rollout-only held-out curve at policy steps `0,50,...,300`; it does not
retain seven redundant full-model milestones. Each pre-update evaluation emits
an explicit `policy_step -> enclosing_global_step` receipt: policy steps
`0,50,...,250` map to global timing rows `1,51,...,251`; final policy step 300
maps to `none`. Each run captures one warmed `phase=update` XProf window and a
semantic Perfetto trace. The profiled update is excluded from performance
means. FrozenLake additionally reports a `direct_eval_cycle_excluded` steady
mean using those receipts. This is not called training-only: evaluation can
perturb allocator/JIT/runtime state even with APC disabled, so the view is not
a counterfactual no-eval run.
The raw mean remains visible so direct evaluation cost is not hidden.

All three production profiles require APC off. Require exactly one FrozenLake
`[P3_APC_CONFIG] enabled=0` receipt and no enabled=1 receipt for both P45 and
M15. This disables cross-request prefix reuse only; request-local prefill/decode
KV state is unchanged and B rescore still uses `reset_prefix_cache=True`.

All three manifests must also contain the exact JAX persistent-cache contract:
local `/tmp/jax_compilation_cache`, minimum compile time `0`, XLA caches `all`,
and GCS root `gs://yuxzhang-tunix-models/cache/p33_compilation_cache`. Step 28
must emit one `[JAX_CACHE_SYNC] phase=restore ...` receipt and step 90 one
`phase=save` receipt under `CANON_STATE`; V1 saves before postflight. `hit` or
`saved` proves transport plus nonempty contents, while `empty`, `error`, or
`no-tool` is an explicit performance limitation and does not alter the strict
alignment verdict. Missing/malformed receipts or a wrong bucket/profile/local
path fail the full carrier. Record cache status alongside first-JIT timing; do
not claim a compilation speedup from configuration alone.

Judgment is fail-closed. Every expected alignment record must pass and any real
`CANON_ALIGN verdict=FAIL` kills that recipe. Missing XProf/Perfetto,
checkpoint, completion, update, or runtime performance receipts makes the run
`INCONCLUSIVE`; it is never silently rerun or shortened.

On the first backward, require
`[P59.DP<dp>] head_cotangent_partition_ready` with the target placement
`data,model`, followed by the exact
`[PATHTRACE] P59_RPA_LOCAL_KV_READY` receipt. GSM8K requires
`tp=4 local_q_heads=4 local_kv_heads=2 cache_heads=2 packing=2`; P45/M15
require `tp=8 local_q_heads=4 local_kv_heads=1 cache_heads=1 packing=2`.
FrozenLake must additionally emit the P59-local fixed-head primal
receipt with `semantic_M=256 ... chunks=1 p59_local=1 global_M=2048 dp=8` and
the VJP receipt with `semantic_M=2048 local_M=256 chunks=1` plus
`tp_input_reduction=all_gather_rank_order_f32_barrier`; M4096 or ordinary
eight-chunk receipts cannot substitute for them. Missing either receipt makes
the run `INCONCLUSIVE` before performance interpretation.

After a pushed approved SHA has rendered the three immutable manifests, and
only after separate launch approval, apply GSM8K, P45, and M15 in one launch
wave. Do not wait for one recipe's first optimizer commit before applying the
other two. Each run's first real optimizer commit remains its own early
admission checkpoint: require zero real alignment FAIL and that recipe's
registered P59-local/fixed-head/token/APC-off/optimizer receipts, but do not stop or
shorten another healthy full run. The direct-full commands are:

```bash
kubectl apply -f "$OUT/gsm8k/jobset-v1-hp-gsm8k-full.yaml"
kubectl apply -f "$OUT/frozenlake-p45/jobset-p57-frozenlake-zero-300.yaml"
kubectl apply -f "$OUT/frozenlake-m15/jobset-p57-frozenlake-zero-m15-main-300.yaml"
```

Do not pipe any apply or workload-launch command. These are three concurrent
full runs, not canaries; M15 is a production/scientific recipe, not a canary.
Apply only after checking all three 64-chip allocations and storage quota;
P45/M15 keep only the
rolling recovery checkpoint while their seven held-out evaluations stay inside
the same uninterrupted full JobSet.

The in-container postflight writes `v1_hp_<recipe>_full.classification.json`.
It requires the complete 200/300 horizon, zero real ALIGN FAIL, the signed
FrozenLake in-process evaluation classification, all P59
parallel/reduction receipts, exact APC-off resolved profiles, and exactly one
`[P3_APC_CONFIG] enabled=0` runtime receipt with no APC-on marker for both P45
and M15. It also records both JAX cache receipts and requires their exact
profile/bucket/local-path identity. Finally it requires one XPlane plus UI
trace and exactly one semantic Perfetto file. Use `xprof-trace-analysis`
after packaging each returned run; operation attribution is a separate claim
from the `[PERF]` timing verdict.

## Historical — superseded single-arm FrozenLake TP8 carrier

Do not follow this block. The P45/M15 p66-off wrapper described here was
superseded by the matched P45 pair at the top of this file.

```bash
# historical only; do not execute
```

The retained text below describes the old single-arm classifier semantics.

Each JobSet executes the full production rollout/rescore producer unit but
stops at the pre-backward gate. Require exactly one controlled exit and one
classification artifact. `ZERO_TIM_RECOVERED` means A−B/B−C `0/0` for that
workload. `A_B_RED_REPRODUCED` means the single p66-off bisection did not fix
the red. Any B−C drift, depth miss, non-finite record, backward, optimizer
commit, retry, or reused evidence path is fatal. Neither outcome permits a
full train until the forward regression is repaired and re-certified.

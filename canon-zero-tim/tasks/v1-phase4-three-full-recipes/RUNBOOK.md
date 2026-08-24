# V1 high-performance three-full runbook

This renderer prepares exactly three JobSets and never launches them. Use one
approved, pushed 40-character source SHA. Run IDs and the campaign root are
single-use; failed attempts are never reused.

```bash
python3 canon-zero-tim/tasks/v1-phase4-three-full-recipes/scripts/render_three_full_recipes.py \
  --source-commit <approved-40-character-sha> \
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
mean using those receipts. This is not called training-only: evaluation may
change APC cache occupancy, so the view is not a counterfactual no-eval run.
The raw mean remains visible so direct evaluation cost is not hidden.

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
registered P59-local/fixed-head/token/APC/optimizer receipts, but do not stop or
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
parallel/reduction receipts, positive APC hits for P45, and an exact APC-off
resolved profile plus exactly one `[P3_APC_CONFIG] enabled=0` runtime receipt
and no APC-on marker for M15. It also requires one XPlane plus UI trace and
exactly one semantic Perfetto file. Use `xprof-trace-analysis`
after packaging each returned run; operation attribution is a separate claim
from the `[PERF]` timing verdict.

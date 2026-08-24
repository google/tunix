# V1 Phase4 three-full handoff

## Mission and current boundary

Prepare exactly three strict optimized Zero-TIM full-training recipes from one
approved immutable source: GSM8K Qwen3-1.7B DP16xTP4 for 200 updates,
FrozenLake P45 Qwen3-8B DP8xTP8 for 300 updates, and FrozenLake M15-main
Qwen3-8B DP8xTP8 for 300 updates. M15 is a production/scientific recipe, not a
canary. The original three-recipe stack is published in the operator history.
The active worktree is
`/home/yuxuan/code_rl_repro/worktrees/p58_zero_hp_release3_0823`, branch
`local/p58-zero-hp-release3-0823`. The attempt-1 repairs and release contracts
were published through `71d889a32f4668353c758d5c00df88299e6c0d35`.
The latest pulled operator tip is
`238ca28cf6eb642429de66c0da58b68ea659309f`; it adds the GSM8K anti-affinity
repair plus immutable attempt-2 error evidence. The attempt-2 repair stack is
split into P59 q_proj, M15 APC-off, and evidence/ledger CLs. Host gates and the
r5 dependency-complete pinned-image gate are green, but the repaired target has
not run. After authorized publication, render only from the exact 40-character
SHA read back from the operator branch and require a clean worktree.

Do not push, rerun the pinned image, publish an image, apply a JobSet, or occupy
TPU resources without the separate user approval for that boundary. Never
launch through a pipe. Run IDs, campaign roots, and evidence directories are
first-use only; preserve every failed run.

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
tolerance was introduced. The local P59 repair admits the legitimate q_proj
one-layout-shard boundary while retaining invalid-layout and width negatives.
The full classifier now requires exactly one explicit APC-off runtime receipt
for M15 and rejects a missing, duplicate, or opposite-arm receipt.

## Resolved bundle

- All recipes: automatic P47a, continue-decode K=8, fixed-AR gather,
  DP-aware gathered logprobs, logprob step fusion, fixed LM head, resident
  trainer placement, batched report, and P59 rank-parallel backward.
- GSM8K only: batched evidence on; APC off.
- FrozenLake: batched evidence off. APC remains on for P45 only and is forced
  off for M15/main after its attempt-2 target red. B rescore remains an
  independent full recomputation with `reset_prefix_cache=True` in every case.
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
  APC knife is VETOED; P45 remains APC-on but has only one target pre-alignment
  PASS before an unrelated backward carrier stop. Strict A(APC)-B(full-reset)=0
  bytes remains mandatory and was not relaxed.
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
bash canon-zero-tim/tests/v1_phase4/run_exact_image.sh
```

Require the exact terminal marker:
`V1_HP_EXACT_IMAGE_PASS dp16_gathered=1 dp2tp2_parallel=2 p59_tp4_tp8=2 p59_real_shim=4 p57_wandb=1 perfetto_window=1 manifests=3`.
This is an exact-image admission receipt for the pre-attempt-1 tree. The
historical receipt is not a signed raw-log artifact: the
stdout/stderr log was not durably preserved, so no raw-log path or SHA exists.

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
and freeze every YAML hash. Apply the 200-update GSM8K
full train first. Its first real optimizer commit is an early admission
checkpoint, not a shortened run: require zero real alignment FAIL plus the
registered P59-local, fixed-head, and optimizer receipts, then let the same
JobSet continue to its full horizon. Only after that checkpoint may the two
300-update FrozenLake full trains start, P45 first and M15 second, from the same
source SHA. Each remains an uninterrupted full train and must receive its own
complete strict-alignment, P59/APC/fixed-head, timing, XProf, Perfetto, eval,
and horizon postflight. A GSM8K green does not certify APC, TP8 fixed head,
DP8xTP8, FrozenLake evaluation, or M15 workload geometry.

Any real `CANON_ALIGN` or `CANON_ALIGN_PRE verdict=FAIL` kills that recipe.
Missing horizon, receipts, trace, checkpoint, or artifacts is INCONCLUSIVE,
not PASS. Performance judgment comes from `[PERF]`; XProf/Perfetto provides
operation attribution and never overrides the bitwise gate.

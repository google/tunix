# V1 Phase4 three-full handoff

## Mission and current boundary

Prepare exactly three strict optimized Zero-TIM full-training recipes from one
approved immutable source: GSM8K Qwen3-1.7B DP16xTP4 for 200 updates,
FrozenLake P45 Qwen3-8B DP8xTP8 for 300 updates, and FrozenLake M15-main
Qwen3-8B DP8xTP8 for 300 updates. M15 is a production/scientific recipe, not a
canary. The original three-recipe stack is published in the operator history.
The active worktree is
`/home/yuxuan/code_rl_repro/worktrees/p58_is_zero_refine_0824`, branch
`local/p58-is-zero-refine-0824`. The attempt-1 repairs and release contracts
were published through `71d889a32f4668353c758d5c00df88299e6c0d35`.
The latest pulled operator tip is
`7e9b31cb`; it adds the immutable Attempt-4 logs and receipt. Attempt 4 proves
that the published q_proj, RPA, and M15-width repairs all took effect, then all
three recipes stop at the same P59-local gate/up projection layout seam before
an optimizer commit. Runtime CL `5bd90bff` makes the live TP degree,
not engine `config.n_shards=1`, the divisor for gate/up's globally declared
last-axis width; q/k/v retain their one-layout-shard contract. Host admission,
the focused installed-shim image gate, and the complete V1 exact-image gate
are green. The complete gate is durably sealed under
`evidence/v1_hp_attempt4_fix_exact_image_20260824_r1/`. Every repaired target
optimizer commit remains unrun. After authorized publication, render
only from the exact 40-character SHA read back from the operator branch and
require a clean worktree.

After exact-image admission, publication, exact remote readback, rendering,
and separate launch approval, start all three full JobSets in one wave. Do not
gate P45 or M15 launch on GSM8K's first optimizer commit. Every recipe still
owns an independent first-commit admission and strict zero-TIM verdict; a red
freezes and kills only that recipe while the other healthy full runs continue.

Do not push, rerun the pinned image, publish an image, apply a JobSet, or occupy
TPU resources without the separate user approval for that boundary. The user
approved the current exact-image rerun and publication only; no JobSet launch
is implied. Never
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
registered P59-local, fixed-head, token/APC, and optimizer receipts, then let
the same JobSet continue to its full horizon. A red freezes only that recipe;
it does not stop another healthy full run. Each must receive its own complete
strict-alignment, P59/APC/fixed-head, timing, XProf, Perfetto, eval, and horizon
postflight. A GSM8K green does not certify APC, TP8 fixed head, DP8xTP8,
FrozenLake evaluation, or M15 workload geometry.

Any real `CANON_ALIGN` or `CANON_ALIGN_PRE verdict=FAIL` kills that recipe.
Missing horizon, receipts, trace, checkpoint, or artifacts is INCONCLUSIVE,
not PASS. Performance judgment comes from `[PERF]`; XProf/Perfetto provides
operation attribution and never overrides the bitwise gate.

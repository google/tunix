# V1 Phase4 three-full handoff

## START HERE — next P45/M15 full wave must use the P74 system-optimization bundle

This section is the authoritative launch preparation for the two optimized
FrozenLake Zero full recipes. It supersedes older P45/M15 render commands in
this historical handoff, but does not erase their incident evidence.

Status is `M15 A-B WARNING LANE LOCAL IMPLEMENTED / HOST PASS /
PINNED-IMAGE PASS / SOURCE NOT COMMITTED / TARGET NOT RUN`. The implementation
CL must first be reviewed, explicitly
approved for commit/push, published, read back at one exact 40-character SHA,
and checked out clean. Do not render the next wave from the older published
`a8449b3d...` source or from this dirty development worktree.

Run the exact-image admission and then the render-only wrapper from the
physical repository root:

```bash
bash canon-zero-tim/tests/v1_phase4/run_exact_image.sh \
  tunix_frozenlake_image:vllm-tpu0.25.0

bash canon-zero-tim/tasks/v1-phase4-three-full-recipes/scripts/prepare_p67_frozenlake_two_full_wave.sh \
  <approved-40-character-sha> \
  <fresh-output-dir> \
  <fresh-campaign-root> \
  <fresh-p45-run-id> \
  <fresh-m15-run-id>
```

The wrapper never launches. It must emit
`V1_P67_FROZENLAKE_WAVE_READY ... manifests=2 ... launch=not-executed`.
Before a separately approved apply, both resolved environments must contain:

```text
CANON_P59_CHECKED_VMA=1
CANON_P66_P59_CHECK_VMA=1        # derived compatibility alias
CANON_P67_P66_VMA_P59_ONLY=1
CANON_V1_HP_FIRST_UPDATE_GATE=1
CANON_DP_COMPARE_MODE=fingerprint-hybrid
CANON_DP_DISTINCT_SCHEDULE=first-group-warmup
CANON_DP_FINITE_FETCH=batched-commit
CANON_P71_SCAN=fwd
```

`CANON_DP_COLLECTIVE_REDUCE` must remain absent. P74 is source behavior under
the checked-VMA path, not a flag to add by hand. P45 retains strict Zero.
M15/main temporarily sets `CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY=1`, but only
finite `S_decode_vs_S_prefill` and its direct w/wr/clip/TIS consequences are
warnings; B-C, T-current/r, nonfinite, gradient, replica, and optimizer faults
still stop the run. Both remain APC-off, no-eval, no-checkpoint, 300-update
identities. M15 output is `convergence-only / alignment-degraded`, never a
Zero-TIM pass. Native, IS, diagnostic, legacy resident, and evaluation
carriers do not inherit this exact-arm policy.

The M15 warning-lane changes are local and have not yet produced target
evidence. Host admission passed, and pinned image
`sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
exited zero with terminal `V1_HP_EXACT_IMAGE_PASS ... m15_ab_warning=1 ...
manifests=3`; the terminal is an admission receipt, not a durably saved raw
log. Commit, push, and launch still require separate approvals. Offline P74
admission is recorded under
`../v1-system-optimization-workload-rollout/validation.log`. It does not
certify DP8xTP8 performance or convergence. Each target run must independently
return strict alignment, checked-VMA/P67 receipts, the first-update gate,
complete `p32_vag_reverse` timing, 300 committed updates, and its own final
classification.

## START HERE — GSM8K Native/mismatch and Zero full now have matched controls

The optimized GSM8K full renderer remains the single-job P74 wrapper:

```text
scripts/prepare_gsm8k_full_dp16tp4_p74.sh
```

Its matched stock Native/mismatch control is now prepared by:

```text
../v1-gsm8k-native-full-control/prepare_gsm8k_native_full.sh
```

The two arms use the same source, `_gsm8k_command(200)`, driver `SEED=42`,
DP16xTP4 geometry, resident optimizer, and W&B project/group
`zero-tim-gsm8k-dp16-tp4` / `qwen3-1p7b-dp16-tp4`. They use distinct
JobSet-derived run names. The Native arm is the stock P56 vanilla path:
`CANON_GSM8K_VANILLA=1`, no `CANON_P32_WORKLOAD`, no canonical engine overlay,
no alignment observer, ordinary Tunix backward, and stock lm head. All Zero
numerical and optimization selectors remain absent from its raw manifest. See
`../v1-gsm8k-native-full-control/HANDOFF.md` for the two render-only commands,
the exact manifest gate, and the comparison receipt bundle. No target run is
authorized or claimed by this documentation update.

## START HERE — GSM8K Zero-TIM Full (`gfull1`) step 64 rescore incident sealed

This section supersedes every later `START HERE` block.

GSM8K full target run `canon-v1hp-gsm8k-gfull1-799a0bd1` (64 TPU v5p, DP16xTP4, source commit `799a0bd1ed5ecfd7a2f6e42eeaced82886fec76c`) executed 64 continuous full train updates at ~46s/step with 100% Zero-TIM compliance (`alignment_max_differing_bytes=0`). Solve ratio progressed from 34.8% to 77.7% (reward mean 0.792).

At step 64, rollout batch generation completed call 65. Multi-turn trajectory clipping triggered `MAX_CONTEXT_LIMIT_REACHED` on row 255 (total length 1130 tokens). During `get_prefill_rescore_logps` (`tunix/rl/rollout/vllm_rollout.py:526`), vLLM evaluated the request under `prompt_logprobs=0` and returned only 1 prompt logprob element. The strict fail-closed assertion in `vllm_rollout.py` failed:
`RuntimeError: row 255: engine returned 1 prompt logprobs for 1130 tokens; cannot align the re-score`

Raw evidence package sealed in `evidence/v1_hp_gsm8k_gfull1_step64_incident_20260828/`:
- `run.log` (60,072 lines, 6.9 MiB)
- `RAW_ERROR.log`
- `pre_alignment.jsonl`
- `updates.jsonl`
- `env.sh`
- `receipt.json`
- `SHA256SUMS`

Next action: clamp prompt + completion lengths in prefill rescore to avoid exceeding vLLM context bounds, and relaunch alongside M15 and DeepSWE.

## START HERE — optimized P45 and M15 are no-eval/no-checkpoint fast concept runs

This section supersedes every later `START HERE` block.

Latest incident source `19d105377197e9299ae8f93096627a18a130cf33`
(`f45w09`) did not fail Zero-TIM or backward. Step-0 training completed strict
pre-alignment, all 32/32 post-backward alignment records, finite healthy
gradients, AdamW, and the first optimizer commit. It then launched held-out
evaluation because the standard wave wrapper had never forwarded the existing
low-level `--disable-eval` selector. The eval rescore failed with `row 7:
engine returned 1 prompt logprobs for 1025 tokens`. The raw incident head log
is `evidence/incident_20260828_failures/f45w09_head.log`.

The repair was published and exactly read back as runtime source
`a8449b3ddc2187806341b280f9d659028b3936c6`; no launch manifest or TPU target
run exists for it yet. It fixes the launch identity rather than the eval
algorithm: both optimized Zero full recipes—P45 and M15/main—now require
evaluation disabled and checkpoint mode `disabled`. All checkpoint residual
fields (root, tag, interval, max-to-keep, milestone) must be empty. Native/IS,
historical discovery, and isolated evaluation paths retain their existing
evaluation/checkpoint contracts. The Python trainer admits checkpoint-free
execution only for the exact `frozenlake-v1-hp`, Zero, 300-update, P45 or
M15/main, no-eval identity and prints `[P45.CHECKPOINT] DISABLED ...`.

This is deliberately an efficiency-first concept run. It removes held-out
evaluation work and Orbax/GCS checkpoint I/O from the training process. The
tradeoff is explicit: there is no resume point after a failure, no final model
checkpoint, and no in-process held-out curve. W&B training metrics, strict
Zero-TIM evidence, optimizer receipts, timing, XProf/Perfetto, and the full
300-update completion remain required. A failed or interrupted run must start
again from step 0 with a fresh run ID.

From the clean published runtime source above, render fresh P45 and M15
identities with the two-full wrapper in `RUNBOOK.md`.
The two jobs may be launched together by the other operator. Do not reuse any
prior run label or manifest. Before launch, the manifest/resolved-env gate must
show:

- `CANON_P33_ENABLE_EVAL=0`, `CANON_P33_DISABLE_EVAL=1`,
  `CANON_P31_ENABLE_EVAL=0`, `--eval_every_n_steps=0`, and no
  `--num_test_batches`;
- `CANON_FROZENLAKE_CKPT_MODE=disabled` and all five residual checkpoint
  fields empty for both P45 and M15/main;
- the exact v1-hp Zero/P59/P66/P67/full-300 workload identity, APC-off, and
  strict alignment.

### What the relaunch operator must return

- exact published source SHA, fresh run/JobSet IDs, YAML paths and SHA256s,
  retry status, and durable run directories;
- resolved `env.sh` plus the evaluation-disabled and checkpoint-disabled
  runtime markers;
- first-update precommit/commit receipts, complete P59/fixed-head/reduction
  receipts, 300 update rows, and zero real alignment FAIL;
- W&B run links/identities, raw and steady step timing, JAX-cache receipts,
  XProf/Perfetto artifacts, and final classifier JSON plus SHA256.

Do not request evaluation JSON or checkpoint paths from these two optimized
Zero runs; their absence is the signed fast-run contract, not missing evidence.
No target claim exists until fresh P45 and M15 runs pass their full gates.

Local validation on the dirty implementation tree: P57 155/155, Phase4
90/90, the four P45-owned suites 32/32, and flag audit 393/393 pass. The P45
aggregate runner additionally imports two cross-suite modules that cannot load
in this host Python because `datasets` and `metrax` are absent; those are host
dependency errors, not test assertions, and the affected P45-owned tests pass
when invoked directly. The complete exact-image gate against immutable image
`sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
exited zero with terminal `V1_HP_EXACT_IMAGE_PASS ... manifests=3`, including
the P45 overlay and P59 TP4/TP8 real-shim gates. This is an execution-transcript
admission receipt only; no durable raw image log was saved. No TPU target has
run for this repair.

## START HERE — P45 Wave 02 reached AdamW; LR receipt wiring repaired locally

This section supersedes every later `START HERE` block.

P45 Wave 02 ran source `bde8f4c6e055ff077b24af716857786ce967f422`
as `canon-p57-fl-zero-f45w02-bde8f4c6`. Its raw head-container log is
`../p57-frozenlake-tim-causal-study/evidence/f45w02_head_container.log`
(SHA-256 `1f5455b707599ff7fcff6976b980a441434479c4ee27621744808faa19bdff20`).
The target passed strict Step-0 pre-alignment over 45,727 action tokens, then
all 32/32 post-backward alignment records were PASS with all three byte
boundaries zero. The complete accumulator was finite and nonzero with
`stable_norm=0.6722502708435059`, denominator 32, and 399/399 nonzero leaves.
AdamW completed in 73.546 seconds, advanced trainer step 0 to 1, and changed
6,950,316,141 parameter elements with finite deltas.

The first red happened after that trainer-local commit but before outer weight
sync or checkpoint. The first-update receipt reported
`effective_learning_rate=None`, so the fail-closed gate rejected it. This was
not a missing optimizer learning rate: FrozenLake constructed AdamW with the
constant scalar `LEARNING_RATE`, but unlike GSM8K it never registered that
same value with `PeftTrainer.effective_learning_rate()`. Plain scalar Optax
transforms do not retain a readable hyperparameter field, so the observer
returned `None` even though the update used the configured rate.

The local repair leaves the scalar AdamW transform unchanged and registers
`optax.constant_schedule(LEARNING_RATE)` only with the trainer's observation
API. A syntax/AST negative pins both facts: one registration must exist and
AdamW must still receive scalar `LEARNING_RATE`. P57 passes 147/147 and Phase4
passes 89/89. The latter first encountered host `/tmp` ENOSPC and passed
unchanged after directing test temporaries to the work disk; that is an
infrastructure incident, not a product red. Immutable image
`sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
passes the new regression plus the complete P45 gate and ends
`P45_EXACT_IMAGE_CPU_PASS overlay=qwen8b_tp8`. The image output is not stored
as a durable raw artifact.

This CL publishes the repair and admission ledger; its authoritative identity
is the exact remote-read SHA after push. No post-fix TPU run, render, launch,
weight sync, policy step 1, evaluation, or checkpoint exists. Use only that
remote-read exact SHA and fresh P45/M15 identities. The first target gate must report the configured
positive effective learning rate, complete outer weight sync, and enter policy
step 1; strict Zero-TIM and all backward-health gates remain unchanged.

## START HERE — Attempt 10 checkpoint-admission repair; local gates green, target not rerun

This section supersedes every later `START HERE` block.

Attempt 10 launched P45 and M15/main from
`8eb65480d3705d96ab282799ad5a6c1901596248`. Both Step-0 strict pre-alignment
gates are real target greens: P45 has 48,753 action tokens and M15 has 122,162,
with A-B/B-C `0/0` in both runs. Each run then completed the head and all 36
decoder-layer VJPs for reverse group 1/32 and reached the first gradient sink.
Neither completed the remaining 31 groups, gradient accumulation, the
first-update precommit gate, AdamW, weight sync, evaluation, or checkpoint.

The first red is a stale duplicated checkpoint admission in
`tunix/sft/peft_trainer.py`: it still hard-coded the historical ten-update
P45 cadence after `tunix/rl/frozenlake_checkpoint.py` and the exact P57
profiles moved primary P45/M15 runs to final-only interval 300. Contrary to
the older note at the end of this file, `CANON_P32_WORKLOAD` was present and
correct; both raw logs print `P32 admission arithmetic OK: DP8xTP8`, and the
profile could not otherwise have loaded.

The local repair makes the G6 guard consume the existing fail-closed
`frozenlake_checkpoint.from_env()` plus `require_p45()` source of truth. It
retains the historical interval-10 contract, admits interval 300 only for the
exact registered P57 primary identity, and rejects wrong workload, run kind,
horizon, candidate/split, or cadence. It does not change loss, gradient,
optimizer, checkpoint scheduling, forward kernels, or backward kernels.

Local admission on base `3820b168` passes checkpoint contract 15/15, Phase4
89/89, P57 146/146, Python syntax, and diff hygiene. The immutable image
`sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
executes the real trainer positive/negative gate and ends with the complete
`V1_HP_EXACT_IMAGE_PASS ... manifests=3`. The image output is an execution
transcript without a durable raw-log SHA, so it is admission-grade only.

This CL publishes the repair and its admission ledger; its authoritative
identity is the exact remote-read SHA after push. No fresh render, relaunch, or
optimizer target transaction has occurred for this repair. After remote exact
SHA read-back, render fresh P45 and M15 full identities and launch both. The
first required target evidence is one complete 32/32 reverse, finite/nonzero
precommit receipt, valid AdamW receipt, weight sync, and policy step 1; strict
Zero-TIM remains fatal at every step.

## START HERE — P4.11 P67 FrozenLake full promotion; prepare two jobs, not launched

This section supersedes every later `START HERE` block.

Wave 5 resolved the FrozenLake TP8 forward regression. The P45
`serving-scope` arm retained P59 checked VMA with
`CANON_P67_P66_VMA_P59_ONLY=1` and produced strict A-B/B-C `0/0` over 48,594
action tokens to depth 2,472, then exited before backward with zero optimizer
commits. The user accepted this candidate and explicitly waived another M15
scope precheck. The next target wave is exactly two direct 300-update full
trains: P45 and M15/main.

P4.11 makes P67 mandatory only for the exact FrozenLake V1 high-performance
full profile. GSM8K does not receive the flag. P67 restores the historical
ordinary-serving graph while retaining checked-VMA ownership inside the P59
manual DP/TP pullback. It does not change serving mathematics or fixed TP
reduction order. APC remains off for both FrozenLake recipes. Evaluation stays
rollout-only at policy steps `0,50,100,150,200,250,300`; checkpointing stays
final-only at step 300 with latest-1 retention. The base 64-chip YAML,
autoscaling, worker node selector, and `exclusive-topology` annotation are
unchanged.

Publication contract: P4.11 was prepared and verified on base
`e5c596a4e7621e7442606cfc4dbbb39005eba4eb`. Host gates pass Phase4 89/89,
P57 146/146, P59 37/37, P66 16/16, APC 31/31, and flags 385/385. The immutable
image passes the P67-enabled installed DP2xTP4/TP8 ladder and complete
`V1_HP_EXACT_IMAGE_PASS ... manifests=3`; this was observed in the execution
transcript and has no durable raw-log SHA. The authoritative publication
identity is the exact 40-character SHA read back from the remote branch after
the approved push; the development base above is not launchable. No manifest
render, JobSet, or TPU full run has occurred. Do not render or launch until a
clean checkout is at the remote-read SHA. The user waived only the extra M15
precheck—not strict Zero-TIM, backward-health, or full postflight gates.

After publication, use only the first procedure in `RUNBOOK.md`. The wrapper
must produce exactly two manifest hashes and two unpiped apply commands; it
never launches. P45 and M15 may then be applied together from the same SHA
with fresh, never-reused IDs. One run's red does not cancel the other healthy
run, but it is fatal for that recipe.

### What must be returned from each full run

- exact source SHA, run/JobSet ID, rendered YAML path and SHA256, retry or
  replacement-pod status, and durable run directory;
- resolved `env.sh` proving the exact FrozenLake profile,
  `CANON_P59_CHECKED_VMA=1`, `CANON_P66_P59_CHECK_VMA=1`,
  `CANON_P67_P66_VMA_P59_ONLY=1`, P59 parallel backward, APC-off, strict mode,
  full/300-update identity, and the correct P45 or M15/main workload tuple;
- first-update precommit and commit JSON receipts proving the expected 32
  microsteps/denominator, finite nonzero gradient, `0 < stable_norm <= 1e6`,
  valid AdamW transaction, finite parameter delta, and train step `0 -> 1`;
- complete strict alignment counts with zero real FAIL, 300 update rows,
  complete P59/fixed-head/reduction receipts, and no non-finite sentinel;
- held-out evaluation records for policy steps `0,50,100,150,200,250,300`,
  final step-300 checkpoint, JAX-cache restore/save receipts, XProf XPlane and
  trace JSON, Perfetto, and the final full-classification JSON plus SHA256;
- `[PERF]` steady steps 2+ mean excluding the profiled update and direct eval
  enclosing cycles, reported separately from raw wall time.

Claim ceiling before the two full runs finish: P45 serving-forward recovery is
target-verified; M15 serving-forward and both recipes' P59 backward, AdamW,
performance, convergence, full evaluation, and checkpoint contracts are
target-unverified.

## START HERE — P4.10 FrozenLake TP8 Wave 5 dual-arm recovery verified GREEN (0/0 differing bytes)

This section supersedes every later `START HERE` block.

Wave 5 target bisection on dual 64-TPU allocations completed and verified:
1. `p66-off` (`canon-v1fl-ab-p45-off1-5ade89aa`):
   - Completed 256 trajectories, $N_{\text{action}}=42,149$, max depth 2,135 ($\ge 1686$).
   - $A-B = 0$, $B-C = 0$ differing bytes, Max abs = 0.0.
   - Masked hash: `c882114e27f53e1672656bf6d0350b4e3973003e7e36d18d594ec4df52df3e9a` across all 3 terms.
   - Controlled exit 42, `backward=0 optimizer_commits=0`.
   - Classification verdict: `PASS`, outcome: `ZERO_TIM_RECOVERED`.
2. `serving-scope` (`canon-v1fl-ab-p45-scp1-5ade89aa`):
   - Completed 256 trajectories, $N_{\text{action}}=48,594$, max depth 2,472 ($\ge 1686$).
   - $A-B = 0$, $B-C = 0$ differing bytes, Max abs = 0.0.
   - Masked hash: `40589a703447199dd0bd28b4817f120204ad5d261086d8362a15be06b10bf844` across all 3 terms.
   - Controlled exit 42, `backward=0 optimizer_commits=0`.
   - Classification verdict: `PASS`, outcome: `ZERO_TIM_RECOVERED`.

Paired Decision: `CAUSE_FAMILY_CONFIRMED / SCOPE_CANDIDATE_GREEN`.
Both arms recovered strict 0/0 differing bytes, confirming that checked-VMA serving leakage was the root cause of Attempt 9's 1,755 differing bytes drift, and that `serving-scope` (`CANON_P67_P66_VMA_P59_ONLY=1`) cleanly eliminates serving leakage while retaining the P59 backward checked-VMA implementation.

Durable evidence archived under `tasks/v1-phase4-three-full-recipes/evidence/v1_fl_tp8_ab_diagnostic_w05/` (`SHA256SUMS`).

## START HERE — P4.10 FrozenLake TP8 dual-arm recovery publication contract; target not run

This section supersedes every later `START HERE` block.

- Worktree: `/mnt/disks/tunix-data/worktrees/v1_fl_tp8_ab_diag_0826`
- Branch: `local/v1-fl-tp8-ab-diag-0826`
- Development base: fetched operator tip `ff0acaaa2ad6bbd9dcdf0589c343a7c13f242c9a`
- Runtime/source CL: `47219e0729d5bbdbe43bc407e19aa056c80f02c3`
- Current phase: `phases/v1-p4-10-frozenlake-tp8-ab-recovery.md`
- Publication state: the user approved this P4.10 CL for commit and push on
  2026-08-26. The exact published SHA is the remote branch tip containing this
  block and must be read back before rendering. No JobSet launch has occurred.

Attempt 9 stopped before backward with P45 A−B/B−C `1755/0` and M15/main
`93/0`. P4.10 prepares a matched P45 DP8×TP8, full-32-prompt,
256-trajectory, one-round, zero-backward, zero-optimizer pair:

1. `p66-off`: disables checked-VMA and its P66 alias as a cause-family arm;
2. `serving-scope`: keeps checked-VMA for P59 but sets the default-off
   `CANON_P67_P66_VMA_P59_ONLY=1`, restoring the historical ordinary-serving
   graph across fixed-AR embed, Pallas operands/out-shapes, and RPA out-shapes.

Both arms retain fixed-AR gather, continue-decode 8, gathered logprobs, step
fusion, fixed LM head, APC-off, seed 42, strict A/B/C, and exit 42 before
backward. B−C must remain exactly zero. The classifier permits either A−B zero
or a preserved positive red as a diagnostic outcome, but any B−C red,
insufficient deep-prefix coverage, backward marker, or optimizer activity is
fatal.

Host admission is Phase4 82/82, P57 146/146, P59 37/37, P66 16/16, APC
31/31, flags 385/385. Rebuilt qwen8b_tp8 shim manifest is 37/37. With
checked-VMA plus P59-only scoping, the immutable image passes the installed
DP2×TP4/TP8 P59 shim gate. The complete image regression also exits zero
with `V1_HP_EXACT_IMAGE_PASS ... p59_checked_vma_real_shim=4 ...
manifests=3`. That invocation is an execution-transcript receipt; no durable
raw image log was saved. Real DP8×TP8 target is unrun.

After remote read-back and clean checkout of the exact published SHA, use the
first block in `RUNBOOK.md`. The wrapper renders the P45 pair and only prints
apply commands. It never launches. Launch commands must not have a pipe. With
only 64 chips, run `p66-off` first; with 128 chips, the user may launch both
together.

### Exact operator procedure after publication

1. Confirm the checkout is clean and `HEAD` is the exact 40-character SHA
   read back from `origin/yuxzhang/canon-zero-tim`. Never render from this
   dirty preparation tree.
2. Run the first render-only command in `RUNBOOK.md` with one never-used
   output directory and two never-used IDs. Review both YAML hashes and
   `render-receipt.json` files. The renderer must report `backward=0
   optimizer_commits=0` for both arms.
3. Confirm no conflicting P51/P59/P62/P64/Phase4 workload is live. The user,
   not the renderer, applies the printed commands with no pipe. With 128 chips
   the pair may run together; with 64 chips complete `p66-off` first and then
   render `serving-scope` again with a new ID.
4. Preserve each complete run directory and raw log, including failures. Do
   not reuse a run label, retry into the same evidence path, or delete an
   inconclusive result.
5. Classify each arm from its generated
   `v1_fl_tp8_ab.classification.json`. A Kubernetes `Complete` condition or
   runner exit zero is transport evidence only; it is not the numerical
   verdict.

### Required return bundle

Return one record per arm containing all of the following, then one paired
decision:

- exact published source SHA, JobSet/run ID, arm name, rendered YAML SHA256,
  and whether the job had any retry or replacement pod;
- durable `run.log` path and SHA256;
- durable pre-alignment JSONL path and SHA256;
- durable `v1_fl_tp8_ab.classification.json` path and SHA256;
- the exact `V1_FL_TP8_AB_CLASSIFICATION` terminal plus the exact
  `[V1.FL.AB] DIAGNOSTIC_COMPLETE` terminal;
- `verdict`, `outcome`, `N_action`, `max_logical_kv_prefix_length`,
  `A_B_differing_bytes`, `B_C_differing_bytes`, `backward`,
  `optimizer_commits`, and the complete `errors` array from the JSON;
- confirmation that the exact profile marker occurred once, P38 completed
  exactly one round, controlled exit 42 occurred once, and no backward or
  optimizer marker occurred.

The paired decision must use only this matrix:

| `p66-off` | `serving-scope` | Return verdict and next action |
|---|---|---|
| `ZERO_TIM_RECOVERED` | `ZERO_TIM_RECOVERED` | `CAUSE_FAMILY_CONFIRMED / SCOPE_CANDIDATE_GREEN`; next gate is TP8 trainer-forward and backward certification before any full train. |
| `ZERO_TIM_RECOVERED` | `A_B_RED_REPRODUCED` | `SCOPE_INCOMPLETE`; retain both logs and bisect the four registered serving leak sites. |
| `A_B_RED_REPRODUCED` | `A_B_RED_REPRODUCED` | `CHECKED_VMA_EXONERATED`; preserve the frozen contract and next bisect fixed-AR gather or continue-decode. |
| any other pair | any other pair | `INCONCLUSIVE`; do not train or infer a fix. |

Any arm with classifier `verdict=FAIL`, nonzero B−C, non-finite evidence,
insufficient depth, missing/duplicate terminal, retry contamination, backward,
or optimizer activity makes the pair `INCONCLUSIVE/FATAL`. Return the first
failing condition and its raw-log line together with all preserved artifacts.
This carrier intentionally returns no optimizer, convergence, performance, or
XProf claim.

## START HERE — checked-VMA three-full wave published, ready to render, not launched

This section supersedes every later `START HERE` block. The older sections are
an append-only history of Attempt 7 and P64; do not follow their launch matrix.

- Worktree: `/home/yuxuan/code_rl_repro/worktrees/p66_gsm8k_convergence_0825`
- Branch: `local/p66-gsm8k-onehost-convergence`
- Rebase base: fetched operator tip
  `75e97a1db4a4bb328fa174f75869f039defc4b98`.
- Approved launch source: the four-CL runtime/launch stack was pushed and
  exactly read back at
  `ff33ea1a38d1d75c2409ccf480c57e9ff0151075`.
- Tree state: this follow-up is a documentation-only publication ledger. No
  final YAML, JobSet, TPU launch,
  optimizer target result, or performance result has been produced by P4.9.
- Current phase: `phases/v1-p4-9-checked-vma-full-wave.md`.

P66 G1 identified the old P59 TP>1 backward regression as erased
varying-manual-axis/replication ownership under the unchecked nested
`shard_map`. G1.5 compared the repaired, source-frozen pullback with ordinary
JAX at the same model/input/cache/cotangent at six full-Qwen endpoints. All
registered envelopes passed; worst relative-L2 was `0.0052568`. This supersedes
the old padding/RMS hypothesis but remains one-host evidence, not target
certification.

P4.9 promotes that exact repaired core into only the three production full
profiles behind default-off `CANON_P59_CHECKED_VMA=1`. The P66 spelling is an
internal compatibility alias so the adapter/shim bytes tested by the oracle
are unchanged. Each first update now fails before AdamW unless the complete
accumulator has the exact microstep count and denominator, is finite and
nonzero, and has stable-L2 in `(0, 1e6]`. After AdamW, a second receipt must
show a valid `0 -> 1` optimizer step, finite gradient/delta, and parameter
change coherent with the learning rate before outer weight sync or checkpoint.

The prepared concurrent wave is exactly:

1. GSM8K Qwen3-1.7B DP16xTP4, 200 full updates;
2. FrozenLake P45 Qwen3-8B DP8xTP8, 300 full updates; and
3. FrozenLake M15/main Qwen3-8B DP8xTP8, 300 full updates.

All three remain strict Zero-TIM, APC-off, full B-arm rescore, JAX persistent
cache, XProf, Perfetto, evaluation, and checkpoint carriers. They are
independent: a red freezes only that recipe while the two healthy jobs keep
running. Any real `CANON_ALIGN_PRE`/`CANON_ALIGN verdict=FAIL`, first-update
gradient red, or optimizer-admission red is fatal and evidence must be kept.

Admission completed on the final rebased runtime tree:

- V1 74/74, P57 146/146, P59 37/37, P66 16/16, P61 6/6,
  APC 31/31;
- flag audit 383/383 with `FLAG_AUDIT_PASS`;
- syntax and diff hygiene pass; and
- immutable image
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
  exited zero with one terminal containing
  `p59_checked_vma_real_shim=4`, `first_update_gate=4`,
  `apc_m15_carrier=46`, and `manifests=3`.

The image result is an execution-transcript receipt; no durable raw image log
was saved for this invocation. It is construction admission only. DP16xTP4
and DP8xTP8 optimizer correctness, convergence, and performance are unrun.

Next boundary:

1. Create or use a clean worktree whose HEAD is exactly
   `ff33ea1a38d1d75c2409ccf480c57e9ff0151075`.
2. From that exact SHA, run the render-only wrapper at the top of
   `RUNBOOK.md`. It refuses a dirty tree, mismatched HEAD, reused output root,
   or duplicate IDs and never launches.
3. Review `manifest-index.json` and the three printed apply commands. The user
   launches all three together. Never attach a pipe to an apply command.
4. Watch each job's first-update precommit and commit receipts immediately.
   Preserve a FrozenLake red without cancelling another healthy recipe.
5. Performance and XProf comparison begin only after the user supplies the
   launched run IDs and logs.

## Historical — P64 remote 64-TPU diagnostic and old launch decision

This section records the pre-P66 source of truth. It is historical context
unless explicitly linked from the current P4.9 section above.

- Worktree: `/home/yuxuan/code_rl_repro/worktrees/v1_stable_clip_0825`
- Local branch: `local/v1-stable-clip-0825`
- Operator branch: `origin/yuxzhang/canon-zero-tim`
- Source commit: `a909fda14ee3f7e5d2334812a02b1f8ef94b0fbb`
- Evidence directory: `tasks/v1-phase4-three-full-recipes/evidence/v1_hp_p64_remote_64tpu_20260825/`
- P64 Diagnostic Run: `canon-p64-p45-num-p64c11-a909fda1` (64 v5p TPUs, 16 nodes, DP8xTP8, `frozenlake-dp8-tp8`, `optimizer_commits=0`)

### P64 Remote 64-TPU Diagnostic Findings:
1. **Strict Zero-TIM Pre-alignment**: `PASS` across $N_{\text{action}}=46,276$ tokens, differing bytes = 0, masked hash `9215a7c7dca806973d545a278616bbdc7e3d862613f51cd0d128fdb70a8814bc`.
2. **P64 Training Capsule**: Atomic capture complete (`p64_training_capsule.npz`, SHA `af0dc4fc2f8dfb592682b70f752779b970fe9f47713f7fb0e05a5079d982e041`, 22.5 MB, 17 arrays).
3. **Loss Scale & Cotangent Proof**:
   - `loss_scale` = 0.00390625, element-finite.
   - `loss_cotangent` = stable norm 0.43028, max abs 0.01190, 100% element-finite.
   - `group_input_cotangent` = stable norm 0.06981, max abs 0.00571, 100% element-finite (`rank_max_abs` = `[0.0, 0.0, 0.0, 0.00571, 0.00391, 0.0, 0.00068, 0.00317]`).
4. **First Non-Finite Boundary Localized**:
   - Stage: `engine_vjp` on Group 0, leaf 1 (`[1]`), rank 3.
   - Rank pattern: TPU ranks receiving exact 0.0 cotangent inputs (ranks 0, 1, 2, 5) produce finite 0.0 VJP, while TPU ranks receiving non-zero cotangents (ranks 3, 4, 6, 7) produce `NaN`.
   - Mathematical Conclusion: Policy alignment, token sampling, and loss cotangents are 100% sound. The NaN in Qwen3-8B DP8xTP8 occurs specifically inside the Pallas `wrapped_model_fn` reverse VJP pass on TPU.

The published recovery consists of four independently revertible CLs:

1. `4c59ba5d` restores Pathways XProf from an attempt-scoped GCS directory and
   fails closed on missing XPlane/trace artifacts;
2. `f62eb4bf` adds the frozen DP16xTP4 GSM scale/ownership carrier against an
   FP64 oracle, with zero optimizer commits;
3. `3533146d` adds default-off P64 exact-P45 capture plus hash/model-bound
   group-0 diagnostic replay, also with zero optimizer commits;
4. `548db7e9` records the phase, signed evidence, gates, and launch-matrix
   checkpoint.

Release gates are green on the final committed runtime: V1 67/67, P57
144/144, P59 37/37, APC 31/31, flags 378/378, syntax and diff hygiene. The
complete immutable-image gate on
`sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
exited zero with one terminal containing `p64_capsule=3`,
`gsm_scale_replay=1`, and `manifests=3`. This is construction admission, not
target success.

### Target facts that control the next action

- GSM8K DP16xTP4: strict Zero-TIM passed and two real optimizer commits were
  completed. Its gradients were element-finite; the old `norm=inf` was FP32
  sum-of-squares overflow. The later stop was the now-repaired XProf path.
- P45 DP8xTP8: strict Zero-TIM passed, then rank 1 produced 253 non-finite
  staged gradient leaves before the first optimizer commit.
- M15 DP8xTP8: strict Zero-TIM passed, then rank 3 produced 122 non-finite
  staged gradient leaves before the first optimizer commit.
- Both FrozenLake recipes use RLOO. An all-equal reward group gives exact-zero
  advantages without a standard-deviation division. Zero reward remains a
  possible trigger for a zero-cotangent-unsafe VJP, not a proven divide-by-zero
  root cause.

### The only pending decision

The user requested four concurrent jobs: GSM8K full, P64 P45 capture, P45
full, and M15 full. Do not apply anything until the user chooses one of these
matrices explicitly:

| Choice | Apply now | Meaning |
|---|---|---|
| A — recommended | GSM8K full + P64 P45 capture | Obtain useful full-training and first-red evidence; hold the two unchanged known-red FrozenLake full recipes. |
| B — accepted-risk | all four requested jobs | P45/M15 full are expected to reproduce Step-0 backward reds and add little localization evidence. Preserve them as failed target evidence if they do. |

### Executable next action after that decision

1. Fetch and require the operator branch still resolves exactly to
   `548db7e9f014def3cb2b37e66c6f0e62c2041f1d`.
2. Generate fresh, never-reused IDs and output directories. Render only—do not
   apply—using the commands at the top of `RUNBOOK.md`. Both renderers must use
   the exact published SHA above.
3. Require `P64_P45_NUMERIC_RENDER_PASS ... capsule_mode=capture
   optimizer_commits=0`, three `V1_HP_MANIFEST_PASS` records, and one
   `V1_HP_THREE_FULL_RENDER_PASS manifests=3`. Record every YAML SHA; never
   hand-edit a rendered manifest.
4. Check that no conflicting P51/P59/P62/P64 or three-full JobSet is live and
   obtain explicit approval for the exact `kubectl apply` set. Never pipe an
   apply command.
5. On every target, any real `CANON_ALIGN`/`CANON_ALIGN_PRE verdict=FAIL` is a
   hard death. Freeze that run and preserve all evidence. P64 must commit zero
   updates; its capture executes all 32 groups. Replay is optional, uses a new
   label, executes group 0 only, and is permanently `certification=0`.

The active technical plan and classifier branches are in
`phases/v1-p4-8-attempt7-target-recovery.md`. `RUNBOOK.md` owns rendering and
apply commands. `state.md` owns the one-line resumable state.

## 2026-08-25 launch-matrix checkpoint — both FrozenLake full recipes are known red

The user requested one concurrent wave containing GSM8K full, P64 P45 capture,
P45 full, and M15 full. Before publication, remote tip `53876c15` supplied the
previously missing M15 Attempt-7 log. M15 passed strict prealignment for
118,816 actions with zero A/B and B/C bytes, then rank 3 produced 122
non-finite staged leaves before the first commit. P45 already showed the same
family on rank 1 with 253 leaves. Both recipes use RLOO, so an all-equal reward
group produces exact-zero advantages without a standard-deviation division.
The leading hypothesis is a data-dependent TP8 VJP/zero-cotangent failure, not
reward-normalization divide-by-zero.

Do not call unchanged P45/M15 full relaunches expected-green. At this historical
checkpoint, safe progress was to publish the admitted recovery tree and render
P64 P45 capture; publication is now complete as recorded in START HERE.
Applying the two known-red full recipes remains a distinct final matrix choice;
their current profiles do not add first-red instrumentation and would most
likely reproduce the existing Step-0 evidence.

## 2026-08-25 superseding status — P64 capsule replay admitted through pinned image

Attempt-7 P45's real rank-1 non-finite source is still unknown, but repeated
localization iterations no longer need to regenerate the expensive FrozenLake
rollout. Default-off P64 now has two exact modes on the original P45 DP8xTP8
geometry:

- `capture` waits for strict prealignment PASS, atomically stores all 17
  training/observation arrays, binds per-array plus whole-file SHA-256 and a
  bounded live-model sample, uploads the capsule/model sidecar to a unique GCS
  URI, then executes the normal 32-group no-commit diagnostic;
- `replay` verifies both immutable files and the live model, rechecks the
  frozen A/B/C alignment values, bypasses environment/rollout/B rescore, and
  executes reverse group 0 only. It emits `certification=0` and can never
  replace the capture arm as strict target evidence.

Admission is complete through host and immutable image: V1 67/67, P57
144/144, P59 37/37, APC 31/31, flags 378/378, syntax and diff hygiene pass.
The durable pinned-image run is 966 lines with SHA
`c8121b6668e4fbdcceec14214966c7ba8ef55ba30ff4b9b1a52e1baa7c70177c`
and exactly one terminal `V1_HP_EXACT_IMAGE_PASS ... p64_numeric=4
p64_capsule=3 ... manifests=3`; receipt and checksums are in
`evidence/v1_hp_p64_capsule_exact_image_20260825_r1/`.

After the runtime was split into the three local CLs, the complete immutable
image gate was run once more directly on the clean committed tree. It exited
zero with the same unique terminal, including `p64_capsule=3`,
`gsm_scale_replay=1`, and `manifests=3`. That final-tree rerun is an execution
receipt only; the earlier r1 directory remains the durable signed raw log.

Claim ceiling: `IMPLEMENTED / HOST PASS / EXACT_IMAGE PASS / TPU TARGET NOT
RUN`. The admitted runtime was split into three runtime commits:
`4c59ba5d` (GCS XProf restore), `f62eb4bf` (frozen GSM scale oracle), and
`3533146d` (P64 capsule/replay), followed by ledger commit `548db7e9`. The
complete chain is published and read back at
`548db7e9f014def3cb2b37e66c6f0e62c2041f1d`. No JobSet or TPU workload has
been launched from it. Use replay only if a fresh full capture shows that
another earlier observer is needed. Do not relaunch from any other SHA.

## Historical — P63 publication boundary before Attempt-7 recovery

G5b resolved the Attempt-7 ambiguity on the real GSM8K DP16xTP4 target: all
16 backward groups and the full accumulator are element-finite, denominator
16 is exact, and only Optax's naive FP32 sum of squares overflows. The scoped
`CANON_P63_OVERFLOW_SAFE_CLIP` repair is now implemented. It returns the stock
Optax transform byte-for-byte whenever the stock norm is finite, selects
max-scaled L2 only when an independent all-finite predicate proves finite
overflow, and never admits a tree containing NaN/Inf. Max norms remain GSM8K
1 and FrozenLake 100.

The final tested runtime was published as
`98be7b291ddb92391f71d360dd59b09f83edc118` and exactly read back from the
operator branch. Its clean isolated worktree is
`/home/yuxuan/code_rl_repro/worktrees/v1_stable_clip_0825`, branch
`local/v1-stable-clip-0825`; the published commit's parent is admitted base
`22da654ab846b6d3b8a5c0e78e9ded6e04178fd1`. Host gates pass V1 45/45,
P57 144/144, P59 37/37, APC 31/31, flags 372/372, syntax, and diff hygiene.
The complete pinned image
`sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
exits zero with exactly one `V1_HP_EXACT_IMAGE_PASS ... p63_clip=1 ...
manifests=3`. Its raw log SHA is
`31126e623c7ad775614a3ce1ff89d3798d095482d0cbefc84a47ae0d0a2d6c44`;
receipt, checksums, and tested runtime hashes are under
`evidence/v1_hp_p63_exact_image_20260825_r2/`. The sandbox-blocked r1 never
entered Docker and is preserved as infrastructure-inconclusive.

Release verdict: `PUBLISHED / GO FOR MANIFEST RENDER`, not yet `GO FOR
LAUNCH`. No manifest has been rendered. Render three fresh manifests only from
the exact published SHA, verify their hashes, check TPU/storage availability,
then obtain separate explicit launch approval. Apply GSM8K
DP16xTP4 (200), FrozenLake P45 DP8xTP8 (300), and FrozenLake M15 DP8xTP8
(300) in one wave; do not gate one healthy launch on another recipe's first
commit.

Every target remains strict Zero-TIM and APC-off. Each first optimizer
transaction independently requires zero real `CANON_ALIGN verdict=FAIL`, the
registered P59 target-shape receipts, and one valid P63 receipt. GSM8K must
actually observe the finite-overflow fallback; every receipt must show an
all-finite tree, finite selected norm and clip factor, and the exact max norm.
The complete horizon, XProf, Perfetto, JAX-cache, evaluation, checkpoint, and
postflight requirements are unchanged. Current claim ceiling is `PUBLISHED /
HOST PASS / EXACT_IMAGE PASS / TARGET OPTIMIZER COMMIT NOT RUN`.

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

## 2026-08-26 Attempt 8 regression and unpublished local repair

Do not relaunch, render, commit, or push the three-full wave from published
source `c2833eea`. Attempt 8 P45 failed the pre-backward hard gate at step 0:
`S_decode_vs_S_prefill=396` differing bytes while
`S_prefill_vs_T_old=0`. M15 independently failed the same boundary with
`20` differing bytes over `15` elements and `S_prefill_vs_T_old=0` across
`123381` action tokens; its first mismatch is at logical KV prefix `6544`,
turn `14`, and its max-abs delta is `0.007526397705078125`. The retained evidence is
`evidence/v1_hp_three_full_attempt8_20260826/`; the failure is before trainer
backward and therefore does not invalidate the separate P66 gradient repair.
Attempt 7's same P45/TP8 recipe had `0/0` bytes with APC disabled, so this is a
new forward regression, not the experimental prefix-cache path.

Leading causal mechanism (target confirmation pending): the P66 completed-sum `pmean` in
`src/engine_shims/linear_p22xf.py` was guarded only by the process-wide
`CANON_P66_P59_CHECK_VMA` flag. That flag is also present during serving, so
the P59 backward ownership annotation changed the ordinary o_proj/down_proj
serving collective graph. The local repair additionally requires the live
P59 outer manual data/model context before executing the `pmean`; every
serving/global path now returns the historical fixed-order sum unchanged.

The runtime repair CL `41f50d23` and evidence/handoff CL `299fca0c` were
published by ordinary fast-forward under the user's explicit 2026-08-26
approval. Remote readback exactly matched
`299fca0c9f2fff9679076c6d938185696c0f3d2d`. Host gates pass: V1 74/74,
P59 37/37,
P66 16/16, syntax, manifest, and `git diff --check`. The immutable image
`sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
passes the installed-shim TP4/TP8 terminal with `manifests=2x37/37`; the new
negative executes both ring and production gather contract-parallel serving
branches with the P66 flag off/on, requires byte-exact historical output, and
makes any serving `pmean` fatal. The raw image stream was not durably saved,
so this is a reproducible admission receipt rather than a signed raw artifact.

Two real one-host v5p Qwen3-8B DP1xTP4 mechanisms are green. Ring label
`p66scopefix_20260826t0204z` and gather label
`p66scopegather_20260826t0212z` each completed three rounds with actions
`409,565,897`, A-B/B-C `0/0` every round, and zero backward/optimizer commits.
The gather log contains 216 `gather-ordered-sum` PATHTRACEs. Raw/prealignment
SHA256 pairs are `9022cad0.../6ec6cf96...` and
`601d0ffc.../955ae87b...`; full hashes and paths are recorded in the P4.9
phase Result log. These are TP4 mechanism evidence only. The one-host exposes
four JAX TPU devices and cannot certify the failed DP8xTP8 P45/M15 executable.

Claim ceiling: `HOST PASS / PINNED-IMAGE PASS / TP4 MECHANISM PASS / TP8
TARGET ZERO-TIM NOT YET REVERIFIED`. After exact remote readback, run fresh
P45 and M15 full trains as the user requested; each remains independently
fail-closed at the strict pre-backward gate and must require both boundaries
to be zero bytes. A CPU/image/TP4 result may not promote the failed Attempt 8
target.

Launch ownership: the user will start both FrozenLake full trains from the
published repair stack. On first evidence pull, classify each recipe
independently. Any `CANON_ALIGN_PRE` or `CANON_ALIGN` FAIL is fatal and stops
that recipe; require A-B/B-C `0/0` before accepting backward/optimizer
evidence. Do not infer M15 success from P45 or vice versa.

## Historical — 2026-08-26 Attempt 9 single-arm first screen

Attempt 9 proves the published serving-scope repair did not restore TP8
Zero-TIM: P45 is A−B/B−C `1755/0` bytes and M15/main is `93/0`; both are
APC-off and stop before backward. Do not relaunch either full train yet.

The then-active bounded recovery was
`phases/v1-p4-10-frozenlake-tp8-ab-recovery.md`. Its first arm is
`CANON_V1_FL_TP8_AB_ARM=p66-off`: exact production serving geometry with only
checked-VMA/P66 compatibility selection disabled. It runs one full producer
unit, requires deep-prefix coverage and B−C exactness, then exits before
backward/optimizer. `ZERO_TIM_RECOVERED` and `A_B_RED_REPRODUCED` are both
informative diagnostic outcomes; neither is training certification.

This single-arm P45/M15 launch matrix is superseded by the matched P45 pair in
the top `START HERE` section. No local commit/push or target launch may occur
without the user's separate approval.


## 2026-08-26 Attempt 10 FrozenLake Full Runs (P45 & M15 Wave 01)

Attempt 10 (wave 01 of commit `8eb65480d3705d96ab282799ad5a6c1901596248`) executed on dual 64-TPU (DP8xTP8) allocations alongside the active GSM8K full run (`canon-v1hp-gsm8k-g11-c2833eea`):

- **GSM8K Full (`canon-v1hp-gsm8k-g11-c2833eea`)**: Actively training past Update 122/200 (61.0%), solve rate 72.7%-77.3%, 100% Zero-TIM compliance (`logp_diff=0.00000`).
- **FrozenLake P45 Full (`canon-p57-fl-zero-f45w01-8eb65480`)**:
  - Rollout: 256 trajectories completed (Solve rate 58.6%).
  - Step-0 Pre-alignment: `[CANON_ALIGN_PRE] step=0 verdict=PASS N_action=48753 bounds=[('S_decode_vs_S_prefill', 0), ('S_prefill_vs_T_old', 0)]` (`0/0` differing bytes, 100% Zero-TIM pass).
  - Backward pass: Successfully completed all 36 P59 backward layer VJPs.
  - Step-0 Update contract fault: Raised `ValueError: P28 G6 canary requires checkpointing disabled unless the committed P45 checkpoint contract is admitted` at `tunix/sft/peft_trainer.py:945` in `_validate_precomputed_gradient_contract()`. `CANON_FROZENLAKE_CKPT_INTERVAL="300"` differed from the expected interval `"10"`.
- **FrozenLake M15 Full (`canon-p57-fl-zero-m15-mw01-8eb65480`)**:
  - Rollout: 256 trajectories completed, Rescore B completed.
  - Step-0 Pre-alignment: `[CANON_ALIGN_PRE] step=0 verdict=PASS N_action=122162 bounds=[('S_decode_vs_S_prefill', 0), ('S_prefill_vs_T_old', 0)]` (`0/0` differing bytes, 100% Zero-TIM pass).
  - Backward pass: Successfully completed all 36 P59 backward layer VJPs.
  - Step-0 Update contract fault: Raised `ValueError: P28 G6 canary requires checkpointing disabled unless the committed P45 checkpoint contract is admitted` at `tunix/sft/peft_trainer.py:945` in `_validate_precomputed_gradient_contract()`.
- Retained evidence: `evidence/v1_hp_three_full_attempt10_20260826/`.

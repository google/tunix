# P57 300-update FrozenLake runbook

This runbook is the authoritative operator procedure for the two-workload,
three-treatment concept study. It renders JobSets but never applies them.
Every `kubectl apply`, commit, and push requires separate user approval. Never
hand-edit a rendered YAML.

## Active contract

| Wave | Runtime arm | Numerical program | Sampler correction | Launch now |
|---|---|---|---|---|
| `native` | `mismatch` | stock-fast / zero-TIM bundle off | none; old=A | yes |
| `is` | `is` | identical stock-fast program | token TIS; old=C | yes |
| `zero` | `zero` | complete registered zero-TIM bundle | none; old=A | deferred |

Each wave renders two independent 64-chip DP8xTP8 jobs:

- P45 original: generator-backed seed 42/123 maps, side 2–9, five turns,
  prompt/response 4,096/2,048;
- M15: materialized `m15/main`, side 5–12, 15 turns,
  prompt/response 4,096/8,192.

Both jobs run Qwen3-8B for exactly 300 optimizer updates with 32 prompts x
eight generations, temperature 0.7, top-p 1, top-k 0, GSPO-token/RLOO, AdamW
1e-6, and resident optimizer state. Native A-B remains warning-only; B-C,
nonfinite, structural, replica, transaction, optimizer, checkpoint, and
evaluation-completeness failures remain fatal. Zero remains strict.

## Reproducibility and data identity

Every paired command contains exactly one `--seed=42`. The runtime also pins
the vLLM engine global seed to `0`. Startup must emit exactly one:

~~~text
[P57.SEED] CONTRACT_PASS data_shuffle_seed=42 vllm_global_seed=0 per_request_seed=unsupported
~~~

The experiment seed fixes the dataset shuffle and the supported host/JAX random
streams. vLLM's current backend does not expose a stable per-request sampling
seed in this path. Therefore the scientific guarantee is **same signed data,
same seed configuration, and statistically paired recipes**; it is not a claim
that two temperature-0.7 launches produce bitwise-identical token trajectories
or identical curves. One curve per cell is a one-seed concept study. A general
stability claim requires a later preregistered 42/43/44 replication wave.

The primary train/eval dataset identities are frozen and checked row-by-row at
runtime before rollout:

| Workload/split | Rows | Generator namespace | Required SHA-256 |
|---|---:|---|---|
| P45 train | 10,000 | 42 | `ddc96fd9ae4e807d8aa8e800795aa743e423ffe4f936f681596460d28e670487` |
| P45 eval | 100 | 123 | `b10add7f31b2cc9931c65b4cc59780004fd3d52a4fce9d20ed565c87df44b580` |
| M15 `main` train | 10,000 | 57,400,000 | `ff1e659b80a0c9bd640e616972a523132f4a333ef174b1a0b13b202958a30e43` |
| M15 `main` eval | 100 | 57,500,000 | `8edb61cb995b4abe8d3f90b32e961be74b8b74ab46120e0d43513ea26d324089` |

Startup must emit one `[P57.DATASET] MATERIALIZED_PASS ...` line containing the
appropriate pair of hashes. The postflight classifier independently requires
the seed receipt and registered hashes. A row mutation, wrong count, wrong
hash, or `--seed` drift is fatal before a run can be accepted.

## Evaluation contract

Evaluation is enabled inside the training JobSet and is rollout-only. It uses
the held-out 100-row dataset, eight generations per row, and the same
temperature-0.7 sampling recipe as training. It does not feed evaluation
examples to the trainer, does not execute an evaluation backward, and does not
write an evaluation checkpoint.

The required policy steps are exactly:

~~~text
0, 50, 100, 150, 200, 250, 300
~~~

The policy-step labels are the number of updates represented by the weights
currently installed in the rollout engine, not host loop indices. In the
segmented path the trainer can compute the next update before reaching the
shared evaluation block, but those newer weights are not synced to rollout
until after evaluation. The observed policy is therefore still exactly N:

| Policy step | Exact timing |
|---:|---|
| 0 | initial rollout weights; update-1 weights are not yet synced |
| 50 | rollout weights after 50 updates; update-51 weights are not yet synced |
| 100 | rollout weights after 100 updates; update-101 weights are not yet synced |
| 150 | rollout weights after 150 updates; update-151 weights are not yet synced |
| 200 | rollout weights after 200 updates; update-201 weights are not yet synced |
| 250 | rollout weights after 250 updates; update-251 weights are not yet synced |
| 300 | after update 300 commits and its weights are synced to rollout |

Every point reuses the same signed 100-row held-out set and contains exactly
100 x 8 = 800 finite rewards. The postflight classifier rejects a missing,
duplicated, nonfinite, under-covered, wrong-seed, or wrong-dataset point.

W&B metrics are under:

~~~text
frozenlake_eval/eval/reward
frozenlake_eval/eval/solve
frozenlake_eval/eval/n
frozenlake_eval/eval/wall_seconds
frozenlake_eval/eval/policy_step
~~~

The raw log also contains one P42 evaluation JSON record per point
and a unique `[P57.EVAL] FINAL policy_step=300 ...` receipt.

The primary curve does not use `render_eval_schedule.sh`. That script now
renders only optional step-0/final recovery audits and cannot reconstruct an
intermediate point. If the seven-point in-process curve is incomplete, classify
the run `INCONCLUSIVE`; do not fabricate it from training reward.

## Checkpoint contract

Checkpoints write every 10 updates to
`gs://yuxzhang-tunix-models/canon-zero-tim/checkpoints/frozenlake`. The rolling
policy keeps only the latest checkpoint (`max_to_keep=1`). The active campaign
sets `CANON_FROZENLAKE_CKPT_MILESTONE_INTERVAL=0`; no additional 50-step full
model checkpoints are retained. Evaluation therefore no longer creates the
previous multi-terabyte milestone-storage envelope.

## Preflight gates

Run from the exact source worktree:

~~~bash
bash canon-zero-tim/tests/p57_frozenlake_tim/run_cpu.sh
bash canon-zero-tim/tests/p45_frozenlake_dp8_tp8/run_exact_image.sh
git diff --check
~~~

Require at least:

~~~text
P57_FROZENLAKE_TIM_CPU_PASS
P57_INPROCESS_EVAL_CLASSIFIER_PASS steps=7
P57_STOCK_RUNTIME_MATRIX_PASS variants=5 stages=train,eval
P57_TRAJECTORY_PROMPT_PROVENANCE_PASS frozenlake=merge deepswe=environment reset_timeout=preserved missing_prompt=fail_closed
P57_STOCK_POST_BACKWARD_MODULE_C_PASS arms=mismatch,is
P57_STOCK_POST_BACKWARD_MODULE_C_NEGATIVE_PASS arm=unknown
P57_STOCK_OBSERVER_EXACT_IMAGE_PASS targets=absolute values=processed
P45_EXACT_IMAGE_CPU_PASS overlay=qwen8b_tp8
~~~

Missing any required marker forbids launch. Local gates prove construction,
not target TPU behavior.

## Render the immediate four jobs

First package/stop every earlier P57 JobSet. Never reuse an old run ID,
campaign root, output directory, or checkpoint tag. Replace the example IDs
below if they have already been consumed.

~~~bash
cd /home/yuxuan/code_rl_repro/worktrees/p57_frozenlake_tim_0820
SOURCE=<approved-pushed-full-40-character-sha>
OUT_NATIVE=/tmp/p57-primary-native-300-c
OUT_IS=/tmp/p57-primary-is-300-c

bash canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/render_three_arm_wave.sh \
  native "$SOURCE" "$OUT_NATIVE" n45e n15e p57-native-300-c
bash canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/render_three_arm_wave.sh \
  is "$SOURCE" "$OUT_IS" i45e i15e p57-is-300-c
~~~

Each command must emit two `P57_THREE_ARM_MANIFEST_PASS` lines plus:

~~~text
P57_THREE_ARM_WAVE_PASS wave=<native|is> manifests=2
P57_THREE_ARM_RENDER_PASS wave=<native|is> ...
~~~

The four manifests are:

~~~text
$OUT_NATIVE/p45/jobset-p57-frozenlake-mismatch-300.yaml
$OUT_NATIVE/m15/jobset-p57-frozenlake-mismatch-m15-main-300.yaml
$OUT_IS/p45/jobset-p57-frozenlake-is-300.yaml
$OUT_IS/m15/jobset-p57-frozenlake-is-m15-main-300.yaml
~~~

Inspect and retain every rendered YAML SHA-256. Confirm each manifest contains:

~~~text
CANON_P57_EXPECTED_UPDATES=300
CANON_P57_STOP_AFTER_STEP=300
CANON_P33_ENABLE_EVAL=1
CANON_P33_DISABLE_EVAL=0
CANON_P31_ENABLE_EVAL=1
CANON_FROZENLAKE_CKPT_MILESTONE_INTERVAL=0
--seed=42
--num_test_batches=4
--eval_every_n_steps=50
~~~

## Launch

Only after separate user approval:

~~~bash
kubectl apply -f "$OUT_NATIVE/p45/jobset-p57-frozenlake-mismatch-300.yaml"
kubectl apply -f "$OUT_NATIVE/m15/jobset-p57-frozenlake-mismatch-m15-main-300.yaml"
kubectl apply -f "$OUT_IS/p45/jobset-p57-frozenlake-is-300.yaml"
kubectl apply -f "$OUT_IS/m15/jobset-p57-frozenlake-is-m15-main-300.yaml"
~~~

The jobs are independent. Do not cancel a healthy arm because another fails,
and do not relaunch automatically under a changed tag.

## Live and terminal gates

All four logs require exactly one enabled receipt with `cadence=50`, seven
P42 JSON records at the registered policy steps, and the final step-300
receipt. They also require exactly one `[P57.SEED] CONTRACT_PASS` and one
`[P57.DATASET] MATERIALIZED_PASS` whose hashes match the registered table.
Native/no-IS additionally requires:

~~~text
[P57.TIM_PURITY] PASS sampler_is=none old_logps=rollout tis_weights=absent trainer_rescore=observer-only
~~~

Native/token-IS requires:

~~~text
[P57.TIM_PURITY] PASS sampler_is=token old_logps=trainer tis_weights=present trainer_rescore=training-input
~~~

Postflight must emit:

~~~text
P57_INPROCESS_EVAL_PASS steps=0,50,100,150,200,250,300 rewards_per_step=800 ...
[P57.EVAL] EVIDENCE classification=... classification_sha256=...
[P57.TRAIN] EVIDENCE classification=... classification_sha256=...
~~~

Finite A-B is the native treatment, not failure. Missing A-B is
`NO_TREATMENT`. B-C mismatch, nonfinite data, wrong sampler receipt, canonical
marker leakage into native, missing eval point, restart without explicit resume
decision, or an incomplete log makes that run `INCONCLUSIVE` or invalid.

## Package and return

Run the existing `scripts/package_run.sh` for every success or failure. Return
one block per JobSet containing:

- source/image/jobset/run/attempt identity and YAML SHA;
- full raw log from byte zero and resolved environment;
- training and in-process-eval classifier JSON plus SHA-256;
- checkpoint tag, latest durable step/path, and checkpoint contract;
- exact purity, stock-route, segment-preflight/completion, and eval receipts;
- seven evaluation JSON records and the W&B run URL/name;
- A-B dose, B-C/nonfinite/structural verdicts;
- training solve/reward, eval solve/reward, step timing, sampled tokens/s,
  gradient/update norms, and IS mean/max/clip fraction where applicable;
- every infrastructure event or restart.

Do not summarize away failure. If a large artifact cannot be committed, run
the checked-in classifier beside it and return the complete classifier JSON,
input inventory, paths, sizes, and SHA ledger.

## Deferred Zero-TIM pair

Do not launch `zero` in the immediate four-job assignment. After the four
native-program runs are packaged, the user may separately authorize:

~~~bash
bash canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/render_three_arm_wave.sh \
  zero "$SOURCE" /tmp/p57-deferred-zero-300-a z45a z15a p57-zero-300-a
~~~

The same 300-update/evaluation/checkpoint contract applies. Compare treatments
only within P45 or only within M15.

## Rollback

P57 is isolated behind its renderer/profile and explicit arm fields. Leaving
P57 fields unset restores existing workloads. Do not reset shared history or
edit historical P45 files to recover from a P57 failure.

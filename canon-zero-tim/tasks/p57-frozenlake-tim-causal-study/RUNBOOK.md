# P57 FrozenLake M15 stock-curve runbook

This runbook executes only the untreated stock/mismatch arm selected by
`p57cal6`. Use one immutable 40-character source SHA. Never hand-edit rendered
YAML. Every TPU launch requires explicit user approval.

## Frozen recipe

| field | value |
|---|---:|
| model/topology | Qwen3-8B, DP8xTP8, 64 TPU chips |
| maps | M15, balanced 5x5–12x12, deterministic `selection` split |
| max turns | 15 |
| physical prompt / response | 4,096 / 8,192 tokens |
| signed horizon | 200 optimizer updates |
| process stop | update 200 only; one uninterrupted JobSet |
| held-out eval | none before/during discovery training; optional eval-200 afterward |
| rollout group | 32 prompts x 8 generations = 256 trajectories/update |
| trajectory mini / micro | 32 / 8 |
| objective | GSPO-token, RLOO, beta 0, epsilon 0.003/0.005 |
| optimizer | AdamW, lr 1e-6, b1 0.9, b2 0.95, wd 0, resident on TPU |
| sampling | temperature 0.7, top-p 1, top-k 0 |
| checkpoint | every 10 updates, GCS LatestN(1) |

Checkpoint root:
`gs://yuxzhang-tunix-models/canon-zero-tim/checkpoints/frozenlake`.
Frozen tag: `p57-m15-selection-mismatch`. The run keeps
`--expected-updates 200`; omitting `--stop-after-step` resolves the stop to the
full horizon. Checkpoints at 10-step intervals are recovery points, not planned
pause/evaluation barriers. Changing horizon or source SHA makes resume
provenance fail.

## Original-arm contract

`stock-fast` removes the rollout/trainer numerical zero-TIM treatment, not only
fixed lm-head. Twelve presence-sensitive switches are absent; canonical
attention, fixed-shape logprob, trainer, reducer, and VJP gates are zero; the
excess-precision XLA pin is absent. The entrypoint first verifies all six engine
files equal the pinned-image bytes. For mismatch training only, it then applies
a signed two-file observer delta to `runner/tpu_runner.py` plus one helper.
Calibration/evaluation leave all six files unchanged. Training retains
non-treatment services plus the finite warning-only observer. That observer sets
`CANON_PROMPT_PROCESSED_LOGPROBS=1` so its temperature-0.7 prefill rescore has
the same semantic transform as decode and gathers each target from absolute
request history instead of rolling a DP-packed buffer. Sampling does not
request prompt logprobs, and the learner continues to use `S_decode`—not this
observer `S_prefill`—as `old_per_token_logps`. Calibration/evaluation keep the
switch at zero because their alignment observer is off.

## Local gates

~~~bash
cd /home/yuxuan/code_rl_repro/worktrees/p57_frozenlake_tim_0820
bash canon-zero-tim/tests/p57_frozenlake_tim/run_cpu.sh
bash canon-zero-tim/tests/p45_frozenlake_dp8_tp8/run_exact_image.sh
git diff --check
~~~

Require `P57_FROZENLAKE_TIM_CPU_PASS` and
`P57_STOCK_OBSERVER_EXACT_IMAGE_PASS targets=absolute values=processed` plus
`P45_EXACT_IMAGE_CPU_PASS overlay=qwen8b_tp8`. These do not authorize launch.

## Common immutable variables

~~~bash
SOURCE=<approved-full-40-character-source-sha>
CAMPAIGN=p57-m15-selection
BASE_ARGS="--source-commit $SOURCE --campaign-tag $CAMPAIGN --expected-updates 200 --workload-candidate m15 --data-split selection --stock-only"
test "$(printf %s "$SOURCE" | wc -c)" -eq 40
git cat-file -e "$SOURCE^{commit}"
~~~

## Direct stock full training: 0→200

~~~bash
OUT=/tmp/p57-m15-stock-full200
python3 canon-zero-tim/cluster/render_p57_frozenlake_tim.py \
  $BASE_ARGS \
  --run-id p57m15att3 \
  --output-dir "$OUT" \
  --checkpoint-mode new \
  --run-kind train
find "$OUT" -maxdepth 1 -name 'jobset-*.yaml' -print
sha256sum "$OUT"/jobset-*.yaml
~~~

Require one manifest, `--max_steps=200`, prompt/response `4096/8192`, and
`CANON_P57_STOP_AFTER_STEP=200`. The command must not contain
`--evaluation_only`. The resolved preflight must include
`observer=train processed_b=on`; a train environment with processed-B zero is
invalid. `p57m15att3` is the registered next fresh attempt id; it is an
11-character lowercase DNS label component and therefore satisfies the
renderer limit of 1–16 characters. Startup must also emit
`[P57.STOCK_OBSERVER] OVERLAY_PASS files=2 stock_runner_verified=1 treatment=observer-only`.
The first B call must emit exactly one
`[P57.STOCK_OBSERVER] PROCESSED_PROMPT_LOGPROBS_PASS ... targets=absolute-request-history treatment=observer-only`.
After explicit launch approval:

~~~bash
kubectl apply -f "$OUT/jobset-p57-frozenlake-mismatch-m15-selection-200.yaml"
~~~

Pod success alone is insufficient. Require
`SEGMENT_PREFLIGHT restored=0 stop_after=200 horizon=200`,
`SEGMENT_COMPLETE step=200 durable_checkpoint=200 horizon=200`, and the stock
postflight. New training intentionally has no resume-sync marker. Do not launch
eval-0 or intentionally stop at 50/100/150. Never run two jobs against the same
checkpoint tag concurrently.

If infrastructure interrupts the run, preserve the failed run and request a
resume decision. Do not change the source SHA, final horizon, recipe, or tag.
An optional eval-200 can be rendered from checkpoint 200 after the training
curve is classified; it is not part of the launch critical path.

## Monitoring and acceptance

Finite training A-B differences are warnings and must be retained as treatment
dose. Structural errors, nonfinite values, B-C/non-treatment failures,
gradient/update failures, and checkpoint failures remain fatal.

Preserve complete log/YAML hashes, source/image/profile identity, stock
zero-bundle and runtime-route markers, checkpoint markers/URI, W&B training
curve, alignment/update reports, and all infrastructure events.

## Decision after update 200

Automatic freeze requires the preregistered trailing update-200 on-policy solve
statistic to be 60–70%, with valid mismatch-dose, structural, and trajectory
health receipts. Because eval-0 was intentionally skipped, P57.1 makes no
same-split held-out improvement claim. A final 55–60% or 70–75% stops for user
review. Outside 55–75% is floor/ceiling. Do not launch or inspect zero-TIM until
this decision and the unseen `main` split are frozen in the ledger.

## Rollback

P57 paths are env-gated. Revert the eventual P57 concern or leave P57 fields
unset to restore prior P45 behavior. Never repair a failed segment by deleting
or overwriting its GCS checkpoint; stop and request a decision.

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
| segment stops | 50, 100, 150, 200 |
| held-out eval | updates 0, 50, 100, 150, 200; 100 maps, 8 deterministic generations |
| rollout group | 32 prompts x 8 generations = 256 trajectories/update |
| trajectory mini / micro | 32 / 8 |
| objective | GSPO-token, RLOO, beta 0, epsilon 0.003/0.005 |
| optimizer | AdamW, lr 1e-6, b1 0.9, b2 0.95, wd 0, resident on TPU |
| sampling | temperature 0.7, top-p 1, top-k 0 |
| checkpoint | every 10 updates, GCS LatestN(1) |

Checkpoint root:
`gs://yuxzhang-tunix-models/canon-zero-tim/checkpoints/frozenlake`.
Frozen tag: `p57-m15-selection-mismatch`. Every segment keeps
`--expected-updates 200`; `--stop-after-step` only chooses where that process
pauses. Changing horizon or source SHA makes resume provenance fail.

## Original-arm contract

`stock-fast` removes the full numerical zero-TIM bundle, not only fixed
lm-head. Twelve presence-sensitive switches are absent; the canonical
attention/logprob/trainer/reducer/VJP gates are zero; the excess-precision XLA
pin is absent; and the entrypoint leaves engine files equal to pinned-image
bytes. Training retains only non-treatment services plus the finite
warning-only observer. Evaluation uses the same stock engine with no training
or alignment admission.

## Local gates

~~~bash
cd /home/yuxuan/code_rl_repro/worktrees/p57_frozenlake_tim_0820
bash canon-zero-tim/tests/p57_frozenlake_tim/run_cpu.sh
bash canon-zero-tim/tests/p45_frozenlake_dp8_tp8/run_exact_image.sh
git diff --check
~~~

Require `P57_FROZENLAKE_TIM_CPU_PASS` and
`P45_EXACT_IMAGE_CPU_PASS overlay=qwen8b_tp8`. These do not authorize launch.

## Common immutable variables

~~~bash
SOURCE=<approved-full-40-character-source-sha>
CAMPAIGN=p57-m15-selection
BASE_ARGS="--source-commit $SOURCE --campaign-tag $CAMPAIGN --expected-updates 200 --workload-candidate m15 --data-split selection --stock-only"
test "$(printf %s "$SOURCE" | wc -c)" -eq 40
git cat-file -e "$SOURCE^{commit}"
~~~

## Step-0 isolated stock evaluation

Run this before training so LatestN(1) cannot erase the baseline state:

~~~bash
OUT=/tmp/p57-m15-stock-eval0
python3 canon-zero-tim/cluster/render_p57_frozenlake_tim.py \
  $BASE_ARGS \
  --run-id p57-m15-stock-eval0 \
  --output-dir "$OUT" \
  --checkpoint-mode new \
  --run-kind eval \
  --checkpoint-step 0
find "$OUT" -maxdepth 1 -name 'jobset-*.yaml' -print
sha256sum "$OUT"/jobset-*.yaml
~~~

Require one manifest and renderer `INTENT_PASS run_kind=eval`. After approval:

~~~bash
kubectl apply -f "$OUT/jobset-p57-frozenlake-mismatch-m15-selection-eval-0.yaml"
~~~

Accept only an eval receipt/classification at step 0 with mutation counters 0.
The evaluator intentionally keeps trainer-side rescore enabled. Its global
trajectory row count is therefore 8, shard-local row count is 1 on DP8, and
all 800 rewards must be present. Do not reduce `--num_generations`: attempt 2
proved that a global row count of 2 cannot enter the DP8 Splash Attention
program. The classifier requires all eight greedy rewards for each map to be
identical and computes capability from the resulting 100 map-level values.
Attempt 3 proved that renderer/profile checks alone are insufficient: the real
workload entrypoint must consume the same registered generation count. Before
launch, the host gate must report at least 90 tests and include
`test_generation_contract_is_shared_with_real_workload_entrypoint`.

## First stock training segment: 0→50

~~~bash
OUT=/tmp/p57-m15-stock-train50
python3 canon-zero-tim/cluster/render_p57_frozenlake_tim.py \
  $BASE_ARGS \
  --run-id p57-m15-stock-train50 \
  --output-dir "$OUT" \
  --checkpoint-mode new \
  --run-kind train \
  --stop-after-step 50
find "$OUT" -maxdepth 1 -name 'jobset-*.yaml' -print
sha256sum "$OUT"/jobset-*.yaml
~~~

Require one manifest, `--max_steps=200`, prompt/response `4096/8192`, and
`CANON_P57_STOP_AFTER_STEP=50`. After approval:

~~~bash
kubectl apply -f "$OUT/jobset-p57-frozenlake-mismatch-m15-selection-200.yaml"
~~~

Pod success alone is insufficient. Require
`SEGMENT_COMPLETE step=50 durable_checkpoint=50 horizon=200` and the stock
postflight. New training intentionally has no resume-sync marker.

## Evaluate and resume later boundaries

Evaluate the just-written boundary first. Step-50 example:

~~~bash
OUT=/tmp/p57-m15-stock-eval50
python3 canon-zero-tim/cluster/render_p57_frozenlake_tim.py \
  $BASE_ARGS \
  --run-id p57-m15-stock-eval50 \
  --output-dir "$OUT" \
  --checkpoint-mode resume \
  --run-kind eval \
  --checkpoint-step 50
~~~

Then resume to the next absolute stop:

~~~bash
OUT=/tmp/p57-m15-stock-train100
python3 canon-zero-tim/cluster/render_p57_frozenlake_tim.py \
  $BASE_ARGS \
  --run-id p57-m15-stock-train100 \
  --output-dir "$OUT" \
  --checkpoint-mode resume \
  --run-kind train \
  --stop-after-step 100
~~~

Repeat at 100, 150, and 200. Each resume restores exactly the previous retained
checkpoint and emits one stock sync marker. Never run two jobs against the
same tag concurrently.

## Monitoring and acceptance

Finite training A-B differences are warnings and must be retained as treatment
dose. Structural errors, nonfinite values, B-C/non-treatment failures,
gradient/update failures, and checkpoint failures remain fatal.

Preserve complete log/YAML hashes, source/image/profile identity, stock
zero-bundle and runtime-route markers, checkpoint markers/URI, W&B run for
training, evaluation JSON/classification, alignment/update reports, and all
infrastructure events.

## Decision after update 200

Automatic freeze requires held-out solve 60–70% and at least +15 points over
eval-0. A final 55–60% or 70–75% stops for user review. Outside 55–75% is
floor/ceiling. Do not launch or inspect zero-TIM until this decision and the
unseen `main` split are frozen in the ledger.

## Rollback

P57 paths are env-gated. Revert the eventual P57 concern or leave P57 fields
unset to restore prior P45 behavior. Never repair a failed segment by deleting
or overwriting its GCS checkpoint; stop and request a decision.

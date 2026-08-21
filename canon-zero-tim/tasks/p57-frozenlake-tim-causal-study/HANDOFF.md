# P57 M15 stock-curve execution handoff

## Mission

Run the untreated Qwen3-8B FrozenLake M15 `selection` curve before observing
any zero-TIM learning outcome. The registered horizon is 200 updates, but each
JobSet stops on one durable boundary: 50, 100, 150, then 200. Run isolated
stock evaluations at 0, 50, 100, 150, and 200.

Do not render or launch the `zero` arm. Do not hand-edit YAML. Do not commit or
push unless the user separately authorizes it. Every `kubectl apply` requires
explicit user approval.

## Read first

Read `state.md`, `plan.md`, `phases/p57-1-stock-discovery.md`, then
`RUNBOOK.md`. The runbook contains the exact commands and is authoritative.

## Scientific state already established

`p57cal6` completed 2,400 stock-fast trajectories. The original receipt
preserved all measured outcomes and exact group/pair identity, but four map
provenance fields were sentinels because the recorder read a post-construction
trajectory object instead of the source dataset row. The source receipt was
not overwritten. A deterministic rematerialization plus exact `group_id` join
derived those fields for all 2,400 records, with a separate SHA proof. The
unchanged classifier returns:

~~~text
verdict=PASS selection=FREEZE_M15 selected_recipe=m15
~~~

M15 starts at 24.625% solve, has 56% mixed groups, reaches 7,403 context tokens
and 6,223 completion tokens, and had no physical/logical cap hit. M15 is frozen
for the stock curve. M10 and M20 are no longer candidates.

## Treatment identity

The stock arm is not merely `CANON_P38_FIXED_LM_HEAD=0`. It requires:

- `CANON_P57_TIM_ARM=mismatch` and `CANON_P57_INFERENCE_REGIME=stock-fast`;
- all 12 presence-sensitive canonical switches absent;
- the complete numerical trainer/serving bundle literal zero;
- canonical install/overlay/verify skipped, leaving the engine equal to the
  pinned-image stock bytes;
- training admission, checkpointing, W&B, host telemetry, and the finite
  warning-only alignment observer enabled;
- evaluation on the same untreated engine with training/alignment gates off.

The profile, entrypoint, workload validator, and postflight independently
enforce this contract.

## Execute exactly this workflow

1. Check out the approved immutable 40-character source SHA. Keep this exact
   SHA and campaign tag for every segment and evaluation.
2. Run both local gates in `RUNBOOK.md`; stop at the first red gate.
3. Run stock eval-0 with checkpoint mode `new`.
4. Run stock train segment 0→50 with checkpoint mode `new`.
5. Accept it only after the log proves step 50 and durable checkpoint 50.
6. Run stock eval-50 with checkpoint mode `resume`.
7. Repeat train/eval at 100, 150, and 200. Later train segments use `resume`;
   the signed horizon remains 200.
8. Return all receipts/logs/hashes and stop before any zero-arm render.

## Required first-segment markers

For new 0→50 training, require exactly one each:

~~~text
[P57.STOCK_FAST] ZERO_TIM_OFF_PASS mode=train absent=12 observer=train
[entrypoint] P57_STOCK_FAST_PATH run_kind=train regime=stock-fast ... canonical_overlay=skipped
[P57.STOCK] TRAIN_RUNTIME_PASS regime=stock-fast arm=mismatch canonical_bundle=off observer=warning-only
[P57.STOCK] SEGMENT_PREFLIGHT restored=0 stop_after=50 horizon=200 checkpoint_interval=10 max_to_keep=1
[P57.STOCK] SEGMENT_COMPLETE step=50 durable_checkpoint=50 horizon=200 next_action=isolated-eval
[P57.STOCK_FAST] RUNTIME_PATH_PASS canonical_markers=0 overlay=skipped
~~~

New training has zero `ROLLOUT_SYNC_PASS` resume markers. A resumed segment
has exactly one marker at the restored boundary.

For stock eval-0 and later stock evaluations, require:

~~~text
[P57.STOCK_FAST] ZERO_TIM_OFF_PASS mode=eval absent=12 observer=off
[entrypoint] P57_STOCK_FAST_PATH run_kind=eval regime=stock-fast ... canonical_overlay=skipped
[P57.STOCK_FAST] ROLLOUT_SYNC_PASS step=<boundary> transport=update_params exact_weight_attestation=unavailable-by-design
[CANON_P57_EVAL] COMPLETE arm=mismatch step=<boundary> ... backward=0 optimizer_commits=0 checkpoint_writes=0
[P57.STOCK_FAST] RUNTIME_PATH_PASS canonical_markers=0 overlay=skipped
~~~

## Stop conditions

Stop on source/image/profile drift, any canonical runtime marker, missing stock
route, wrong dataset SHA, cap hit, nonfinite value, structural/B-C/gradient or
checkpoint failure, OOM, restart, IFRT disconnect, wrong restored step, or
missing segment completion. Finite A-B drift is warning-only evidence.

Do not automatically rerun or change context, batch, seed, learning rate,
horizon, or checkpoint tag. Preserve partial evidence and report the first
failing marker.

## Return template

~~~text
P57 M15 stock segment/eval complete or inconclusive
source_sha: <same 40 hex for the whole curve>
campaign_tag: p57-m15-selection
run_kind/boundary/checkpoint_mode: <train|eval>/<step>/<new|resume>
jobset/run/attempt/exit: <...>
yaml_path/sha256: <...>
raw_log_path/sha256: <complete log from byte zero>
zero_tim_off_marker: <exact line>
runtime_route_marker: <exact line>
segment_preflight/complete: <exact lines or n/a>
rollout_sync_marker: <exact line, or absent for new training>
checkpoint_latest/uri: <step and gs:// path>
evaluation_json/classification/sha256: <paths or n/a>
wandb_run: <url/id or n/a>
alignment_summary: <A-B warning dose; B-C/structural/nonfinite verdict>
infra_events: <none or exact list>
~~~

# P57 M15 stock-curve execution handoff

## Mission

Run the untreated Qwen3-8B FrozenLake M15 `selection` curve before observing
any zero-TIM learning outcome. The registered horizon is one uninterrupted
200-update JobSet. Do not run eval-0 and do not pause at 50/100/150. Durable
checkpoints remain enabled every 10 updates with LatestN(1) for infrastructure
recovery; they are not evaluation barriers.

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

`p57_eval0_att1` is not an evaluation result. Source `200b244c...` passed the
resolved stock contract but stopped before model load because Steps 37 and 38
still had calibration-only leaf guards. Commit `86112838...` repaired and
certified those guards.

`p57_eval0_att2` is also not an evaluation result. It proved the repaired stock
runtime, loaded the model, synchronized rollout weights, and completed real
rollouts, then failed in trainer-side EVAL rescore. The global input had shape
`[2, ...]` while Splash Attention mapped its row axis over DP8, so JAX rejected
the non-divisible shape before producing a receipt. The repair uses eight
deterministic generations: caller-global M=8, shard-local M=1, semantic rows=8,
and the same DP8xTP8 trainer program. Do not resume attempts 1 or 2. The plan
at that point was a fresh eval-0; the later direct-run decision below supersedes
that next action.

`p57_eval0_att3` is likewise not an evaluation result. Source `8acfe784...`
rendered eight generations correctly, but the real workload entrypoint retained
an older `eval ? 2 : 8` P32 geometry assertion and rejected the rendered value
before model load. The repair moves the P57 count to one
`GENERATIONS_PER_PROMPT=8` registry value consumed by both renderer and
entrypoint. Do not resume attempt 3; after publication, use a new run id and
checkpoint mode `new`.

User decision on 2026-08-21 supersedes the eval-first segmented discovery
flow: P57.1 now launches the stock 0→200 training run directly. Attempts 1–3
remain preserved as `INCONCLUSIVE` evidence, but no fourth eval-0 attempt is
required. Isolated evaluation code remains available for an optional final
step-200 measurement and for the later paired causal campaign.

`p57_stock_full_att1` is the first direct-training attempt and is also
`INCONCLUSIVE`. Source `7e608682...` passed stock routing, data materialization,
checkpoint preflight, model/engine startup, one complete 256-trajectory rollout,
and the training runtime admission. Before backward or update 0, the
warning-only observer requested processed `S_prefill` while the stock profile
still forced `CANON_PROMPT_PROCESSED_LOGPROBS=0`; the rollout API correctly
refused to label raw prompt logprobs as processed. No checkpoint was written.
Do not resume this attempt. The first repair made processed-B an observer-only
training exception and retained zero for calibration/evaluation.

`p57_stock_full_att2` is also `INCONCLUSIVE`. Source `c5cc71b5...` proved the
first repair reached a real B call after 256 trajectories, then failed before
backward/update 0 because sequence target 304 was missing from the returned
prompt-logprob dictionary and key 5795 appeared instead. The pinned stock
runner uses `roll(input_ids, -1)` across DP-packed rows and does not process
prompt logits at temperature 0.7. The current repair first verifies all stock
bytes, then installs a signed observer-only runner/helper delta for mismatch
training B. It does not enable fixed-M or any canonical model/trainer kernel.

## Treatment identity

The stock arm is not merely `CANON_P38_FIXED_LM_HEAD=0`. It requires:

- `CANON_P57_TIM_ARM=mismatch` and `CANON_P57_INFERENCE_REGIME=stock-fast`;
- all 12 presence-sensitive canonical switches absent;
- the rollout, trainer, backward, reducer, and optimizer numerical treatment
  bundle literal zero;
- canonical install/overlay/verify skipped; calibration/evaluation leave the
  engine equal to pinned-image stock bytes, while mismatch training verifies
  those bytes and then installs only the two-file B-observer delta;
- training admission, checkpointing, W&B, host telemetry, and the finite
  warning-only alignment observer enabled. Its processed-B interface is on
  only for post-rollout rescore; sampling and gradients do not request prompt
  logprobs;
- both arms pin `--sampler_is=none`; rollout A is the old-policy denominator,
  sampler-TIS weights are absent, and processed B/trainer C remain read-only.
  The ordinary GSPO epsilon clip is shared base-algorithm behavior and remains
  enabled;
- evaluation on the same untreated engine with training/alignment gates off.

The profile, entrypoint, workload validator, and postflight independently
enforce this contract.

## Execute exactly this workflow

1. Check out the approved immutable 40-character source SHA. Keep this exact
   SHA and campaign tag for every segment and evaluation.
2. Run both local gates in `RUNBOOK.md`; stop at the first red gate.
3. Render one stock train run with run id `p57m15att3`, checkpoint mode `new`,
   horizon 200, and no explicit `--stop-after-step`; the renderer must resolve
   the stop to 200. Do not use the obsolete overlength id
   `p57-m15-stock-full200`, which the 1–16 character DNS gate rejects.
4. Launch that one JobSet and let it run continuously through update 200.
5. Accept it only after the log proves step 200 and durable checkpoint 200.
6. Return the complete training logs, checkpoint identity, W&B curve, and
   alignment/update evidence; stop before any zero-arm render.

## Required first-segment markers

For the direct new 0→200 training run, require exactly one each:

~~~text
[P57.STOCK_FAST] ZERO_TIM_OFF_PASS mode=train absent=12 observer=train processed_b=on
[P57.STOCK_OBSERVER] OVERLAY_PASS files=2 stock_runner_verified=1 treatment=observer-only
[entrypoint] P57_STOCK_FAST_PATH run_kind=train regime=stock-fast ... canonical_overlay=skipped observer_overlay=installed
[P57.STOCK] TRAIN_RUNTIME_PASS regime=stock-fast arm=mismatch canonical_bundle=off observer=warning-only processed_b=observer-only
[P57.STOCK_OBSERVER] PROCESSED_PROMPT_LOGPROBS_PASS ... targets=absolute-request-history treatment=observer-only
[P57.TIM_PURITY] PASS sampler_is=none old_logps=rollout tis_weights=absent trainer_rescore=observer-only
[P57.STOCK] SEGMENT_PREFLIGHT restored=0 stop_after=200 horizon=200 checkpoint_interval=10 max_to_keep=1
[P57.STOCK] SEGMENT_COMPLETE step=200 durable_checkpoint=200 horizon=200 next_action=complete
[P57.STOCK_FAST] RUNTIME_PATH_PASS canonical_markers=0 overlay=skipped
~~~

New training has zero `ROLLOUT_SYNC_PASS` resume markers. Do not intentionally
split or resume a healthy run; resume is reserved for infrastructure recovery.

If the user later requests an optional stock eval-200, require:

~~~text
[P57.STOCK_FAST] ZERO_TIM_OFF_PASS mode=eval absent=12 observer=off
[entrypoint] P57_STOCK_FAST_PATH run_kind=eval regime=stock-fast ... canonical_overlay=skipped observer_overlay=absent
[P57.STOCK_FAST] ROLLOUT_SYNC_PASS step=200 transport=update_params exact_weight_attestation=unavailable-by-design
[CANON_P57_EVAL] COMPLETE arm=mismatch step=200 ... backward=0 optimizer_commits=0 checkpoint_writes=0
[P57.STOCK_FAST] RUNTIME_PATH_PASS canonical_markers=0 overlay=skipped
~~~

The COMPLETE marker must report `prompts=100 generations=8 rewards=800`.
Local admission is complete: P57 host tests pass `102/102`, and the pinned-image
gate executes the eight-generation evaluator lifecycle plus calibration/train/
eval stock modes, installs the exact two-file observer delta, proves absolute
targets and processed values end to end, and passes all registered negatives
before ending
`P45_EXACT_IMAGE_CPU_PASS overlay=qwen8b_tp8`. This is not target evidence.

## Stop conditions

Stop on source/image/profile drift, any canonical runtime marker, missing stock
route, wrong dataset SHA, cap hit, nonfinite value, structural/B-C/gradient or
checkpoint failure, OOM, restart, IFRT disconnect, wrong restored step, or
missing segment completion. Finite A-B drift is warning-only evidence.
Any token sampler-IS path, TIS weight, old-logprob source other than rollout A,
or missing/duplicate P57 purity receipt is a hard causal-contract failure.

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

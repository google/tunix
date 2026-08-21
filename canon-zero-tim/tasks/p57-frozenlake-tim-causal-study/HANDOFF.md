# P57 stock-fast calibration execution handoff

## Mission

Run one 64-chip temperature-0.7 Qwen3-8B rollout-only JobSet that evaluates
M10, M15, and M20 sequentially under the untreated `stock-fast` inference
regime. Return the complete evidence bundle. Do not train or launch zero-TIM.

## Read first

Read, in order:

1. `state.md`;
2. `plan.md`;
3. `phases/p57-1-stock-discovery.md`;
4. `RUNBOOK.md`.

The runbook is authoritative and contains exact commands. Ask the user before
`kubectl apply`. Do not commit or push unless separately authorized.

## Honest validation status

The dependency-light host suite and pinned-image CPU/overlay suite are green.
No real DP8xTP8 / 64-chip target run has yet proved distributed startup,
vLLM/Pathways initialization, or live HBM/KV capacity for the 16,384-token
physical envelope. Do not report “target tested” from the local gates and do
not substitute a DP1 one-host run for this missing evidence. The first approved
calibration launch closes that boundary if it reaches real rollout progress
under the unchanged signed manifest.

## Execute exactly this workflow

1. Check out the approved immutable 40-character source SHA.
2. Run both local gates in the runbook; stop at the first red gate.
3. Render once with `cluster/render_p57_calibration.py`.
4. Run `scripts/verify_calibration_manifest.py` on the rendered YAML.
5. Confirm exactly one manifest exists.
6. Obtain explicit launch approval, then apply that manifest once.
7. Capture the complete chief log from byte zero and immediately check the
   runbook's first-target-launch startup list; preserve and stop on any red.
8. Extract the JSON v2 receipt and run `classify_stock_discovery.py`.
9. Return every item in the runbook's evidence contract and stop.

## Nonnegotiable intent

- `CANON_P57_TIM_ARM=mismatch`;
- `CANON_P57_INFERENCE_REGIME=stock-fast`;
- exact resolved marker `ZERO_TIM_OFF_PASS absent=12 zero=25`;
- exact startup route `P57_STOCK_FAST_PATH ... canonical_overlay=skipped`;
- pre-model stock proof `[P57.STOCK_FAST] PREFLIGHT_PASS files=6 import=pass overlay=absent`;
- exact postflight `RUNTIME_PATH_PASS canonical_markers=0 overlay=skipped`;
- fixed AR, pinned RPA, canonical Pallas trunk/VJP, and fixed logprob M absent;
- RPA VJP2, processed logprobs, log-softmax, module C, KV unified, fixed
  lm-head, segmented training/update, and all alignment/training admissions 0;
- no canonical excess-precision XLA pin;
- M10/M15/M20 only, in that order; no L0 or greedy arm;
- 100 maps x 8 generations per recipe at temperature 0.7;
- `--evaluation_only`; base weights remain immutable;
- no trainer rescore, backward, optimizer commit, checkpoint write, or
  in-process train evaluation;
- prefix cache off; no YAML/threshold hand edits;
- no zero-arm result or W&B run may be launched or inspected.

`CANON_P38_FIXED_LM_HEAD=0` by itself does not satisfy this contract.

## Required success evidence

- renderer `VERDICT PASS count=1`;
- manifest preflight `PASS regime=stock-fast`;
- resolved-container `ZERO_TIM_OFF_PASS absent=12 zero=25`;
- stock engine route and zero-canonical-runtime-marker postflight;
- 3 dataset attestations, starts, and completes;
- one JSON v2 receipt with complete zero-TIM-off attestation;
- one terminal complete record with all mutation counters zero;
- offline classifier `verdict=PASS`.

The classifier may select `FREEZE_*` or `REVIEW_NO_ELIGIBLE_RECIPE`; either is
valid evidence. Do not launch training after classification.

## Stop conditions

Stop on source/image drift, local gate failure, manifest preflight failure,
more/fewer than one manifest, missing zero-TIM-off marker, any canonical switch
enabled, missing recipe, context-limit/timeout/failed trajectory, OOM, restart,
missing terminal marker, or classifier `FAIL`. Preserve partial evidence; do
not patch the cluster or rerun automatically.

## Return template

~~~text
P57 stock-fast stochastic calibration complete/inconclusive
source_sha: <40 hex>
image_digest: <digest>
jobset/run/attempt/exit: <...>
yaml_path/sha256: <...>
renderer_stdout: <path>
manifest_preflight_stdout: <path>
zero_tim_off_marker: <exact line>
raw_log: <complete path>
receipt_v2_json/sha256: <...>
dataset_attestations: <three lines or path>
classifier_json/sha256/exit: <...>
infra_events: <none or exact list>
~~~

Attach the files. Do not paste only selected solve rates.

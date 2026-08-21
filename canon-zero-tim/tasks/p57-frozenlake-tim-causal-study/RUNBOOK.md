# P57 FrozenLake stock-fast calibration runbook

This runbook covers P57.1 workload calibration only. It is executable only
from an immutable 40-character source SHA. Every TPU launch requires explicit
user approval. Never hand-edit rendered YAML.

## Goal and scientific boundary

Measure base-policy solve rate and training-signal density for three long-form
FrozenLake recipes without training and without exposing a zero-TIM result.
This calibration selects a workload; it does not estimate the causal effect of
trainer-inference mismatch.

| recipe | grid sides | max turns | logical context cap |
|---|---|---:|---:|
| M10 | 5x5–10x10 | 10 | 8,192 |
| M15 | 5x5–12x12 | 15 | 12,288 |
| M20 | 5x5–15x15 | 20 | 16,384 |

One DP8xTP8 / 64-chip JobSet initializes the Qwen3-8B carrier once and evaluates
M10, M15, and M20 sequentially. Each recipe contains 100 immutable maps and
uses eight temperature-0.7 generations: 2,400 trajectories total. The shared
physical prompt/response envelope is 16,384 tokens; the offline classifier
applies each recipe's smaller logical cap to observed lengths.

## What `stock-fast` means

`CANON_P57_INFERENCE_REGIME=stock-fast` is a mechanical contract, not a label.
The topology, pinned image, Qwen3-8B weights/TP8 geometry, vLLM server,
resident HBM placement, and rollout sampling configuration remain shared. All numerical
zero-TIM paths inherited from `_canonical_engine.env` are disabled before the
engine starts:

- absent, because presence itself admits their shim: fixed AR/embed, pinned RPA
  D/P/M, fixed logprob M, canonical Pallas projections/RMSNorm/SwiGLU/padding,
  and canonical VJP;
- explicitly `0`: RPA VJP2, processed-logprob carrier, Pallas log-softmax,
  engine-module C, KV unified, fixed lm-head, segmented trainer/update paths,
  all alignment gates, and all P32 training/reduction/launch admissions;
- `XLA_FLAGS` does not contain `--xla_allow_excess_precision=false`.

The entrypoint recognizes this exact profile/run-kind/regime tuple and skips
canonical install, overlay, and overlay verification. The six engine modules
therefore remain the pinned image's stock bytes. A stock-only preparation step
installs the missing nonnumerical workload dependencies (`gymnasium`,
`sentencepiece`, and `tiktoken`) without building or exposing a canonical
overlay. The independent R2E step remains in the sequence but is a no-op for
this workload. Merely installing the overlay and unsetting flags is invalid:
some overlay shims enforce canonical dependencies at import time.
Every P57 command uses the package-safe entrypoint
`python3 -u -m examples.frozenlake.train_frozenlake_qwen3`; file-path execution
is invalid because it does not place the repository root on Python's import
path in the stock environment.
`return_logprobs=True` still requests sampled-token rollout logprobs; it does
not run trainer rescore. Prefix caching remains off. No trainer logprob
recomputation, backward, optimizer commit, checkpoint write, or in-process
train evaluation is allowed.

The rollout engine still has to receive the immutable actor weights before
sampling. Stock-fast therefore calls the same `update_params` transport used
by canonical resume/evaluation, but the untreated engine intentionally lacks
the canonical adapter needed for a live-leaf equality proof. Its honest
receipt is `completed=true`, `transport=update_params`, and
`exact_weight_attestation=unavailable-by-design`. Canonical resume/evaluation
continues to require exact adapter-backed equality and fails closed on a
mismatch. Never claim exact stock weight equality from transport completion.

The container preflight must emit exactly:

~~~text
[P57.STOCK_FAST] ZERO_TIM_OFF_PASS absent=12 zero=25
~~~

The full pod log must also contain exactly one of each of these startup/runtime
markers:

~~~text
[entrypoint] P57_STOCK_FAST_PATH run_kind=calibration regime=stock-fast ... canonical_overlay=skipped
[P57.STOCK_FAST] RUNTIME_DEPS_PASS packages=6
[P57.STOCK_FAST] WORKLOAD_IMPORT_PASS entrypoint=module
[P57.STOCK_FAST] PREFLIGHT_PASS files=6 import=pass overlay=absent
[P57.STOCK_FAST] ROLLOUT_SYNC_PASS step=0 transport=update_params exact_weight_attestation=unavailable-by-design
[P57.STOCK_FAST] RUNTIME_PATH_PASS canonical_markers=0 overlay=skipped
~~~

Absence of any marker is a hard stop. `CANON_P38_FIXED_LM_HEAD=0` alone is not
sufficient and must never be described as stock-fast.

## Local preflight

From the worktree containing the intended source:

~~~bash
cd /home/yuxuan/code_rl_repro/worktrees/p57_frozenlake_tim_0820
bash canon-zero-tim/tests/p57_frozenlake_tim/run_cpu.sh
bash canon-zero-tim/tests/p45_frozenlake_dp8_tp8/run_exact_image.sh
git diff --check
~~~

Required terminal markers:

- `P57_FROZENLAKE_TIM_CPU_PASS`;
- `P45_EXACT_IMAGE_CPU_PASS overlay=qwen8b_tp8`;
- `P57_FILE_ENTRYPOINT_NEGATIVE_PASS script_mode=rejected` and
  `P57_MODULE_ENTRYPOINT_PASS workload_import=complete` occur in the
  exact-image output;
- no traceback, syntax error, manifest mismatch, or flag-census mismatch.

Do not substitute a dirty-tree SHA. Commit and push are separate user-approved
operations and are not authorized by this runbook.

## Validation boundary before the first target launch

The host and exact-image gates prove configuration resolution, manifest
contracts, Python control flow, classifier behavior, overlay importability, and
the rollout-only no-update boundary. They do **not** prove that this exact
DP8xTP8 Pathways/vLLM process reaches its first rollout on 64 real TPU chips.
In particular, the first target launch still has to validate distributed
initialization and the 16,384 prompt/response physical envelope against live
HBM/KV-cache capacity. A one-host DP1 run is not equivalent evidence.

Treat the first approved calibration JobSet as both the scientific run and the
target-startup certification. Before waiting for all 2,400 trajectories, check
the complete log from byte zero for all of the following:

- source and pinned-image identity match the rendered manifest;
- the resolved topology is DP8xTP8 and all workers join once;
- `[P57.STOCK_FAST] ZERO_TIM_OFF_PASS absent=12 zero=25` appears once;
- `P57_STOCK_FAST_PATH ... canonical_overlay=skipped` appears once;
- `[P57.STOCK_FAST] RUNTIME_DEPS_PASS packages=6` appears once;
- `[P57.STOCK_FAST] WORKLOAD_IMPORT_PASS entrypoint=module` appears once;
- `[P57.STOCK_FAST] PREFLIGHT_PASS files=6 import=pass overlay=absent` appears once before model load;
- vLLM/Pathways initialization completes without OOM, KV-block-capacity error,
  restart, or IFRT disconnect;
- the stock rollout-sync marker appears exactly once after `update_params` and
  before the first `RECIPE_START`;
- the first `RECIPE_START` is followed by actual rollout progress;
- no trainer-rescore, backward, optimizer-commit, checkpoint-write, or
  alignment-admission marker appears.

Failure before those checks is infrastructure/startup `INCONCLUSIVE`, not a
scientific outcome. Preserve the complete attempt and do not silently shrink
the envelope, topology, map inventory, or concurrency to make it start.

## Render and mechanically verify the one JobSet

Use a unique empty directory and campaign tag:

~~~bash
SOURCE=<full-40-character-approved-source-sha>
OUT=/tmp/p57-calibration-0821

python3 canon-zero-tim/cluster/render_p57_calibration.py \
  --source-commit "$SOURCE" \
  --run-id p57cal0821 \
  --campaign-tag p57-calibration-0821 \
  --output-dir "$OUT"

python3 \
  canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/verify_calibration_manifest.py \
  "$OUT/jobset-p57-frozenlake-calibration-stochastic.yaml"

sha256sum "$OUT/jobset-p57-frozenlake-calibration-stochastic.yaml"
~~~

Both terminal markers are required:

~~~text
[P57.CALIBRATION.JOBSET] VERDICT PASS count=1 recipes=m10,m15,m20
[P57.CALIBRATION.PREFLIGHT] PASS ... regime=stock-fast recipes=m10,m15,m20
~~~

The manifest verifier proves the signed intent: the exact module entrypoint,
one mismatch arm, stock-fast
regime, fixed head off, train/reduction/launch/alignment admissions all zero,
stochastic M10/M15/M20 inventory, and the exact physical envelope. It does not
replace the resolved-container marker, which proves the sourced profile really
removed the inherited canonical switches.

Any zero arm, canonical regime, fixed-head value 1, warning-only alignment,
nonzero training admission, greedy mode, L0 recipe, resume mode, missing recipe,
or handwritten YAML change is a hard stop.

## Launch only after explicit approval

~~~bash
kubectl apply -f "$OUT/jobset-p57-frozenlake-calibration-stochastic.yaml"
~~~

Use the cluster's normal full-log collector. Save the complete chief log from
byte zero as:

~~~text
p57-calibration-stochastic.raw.log
~~~

A valid log contains:

- one `[P57.STOCK_FAST] ZERO_TIM_OFF_PASS absent=12 zero=25`;
- one runtime-dependency marker, one module-workload-import marker, one
  stock-path routing marker, one transport-only rollout-sync marker, and one
  zero-canonical-marker postflight;
- three dataset attestation records;
- three `RECIPE_START` and three `RECIPE_COMPLETE` records;
- one `CANON_P57_CALIBRATION_JSON` v2 record whose
  `inference_regime=stock-fast` and `zero_tim_off_attestation` lists all 37
  switches (12 absent plus 25 zero), while `rollout_weight_sync` exactly records
  `update_params` completion and `unavailable-by-design` exact attestation;
- one terminal record:

~~~text
[CANON_P57_CALIBRATION] COMPLETE mode=stochastic inference_regime=stock-fast recipes=3 ... backward=0 optimizer_commits=0 checkpoint_writes=0
~~~

`MAX_STEPS_REACHED` is a valid task outcome. Context-limit hits, timeouts,
`FAILED`, OOM, restart, missing recipes, or a missing terminal marker invalidate
the JobSet. Preserve partial evidence and do not auto-rerun.

## Offline classification

The classifier accepts either the emitted JSON file or the complete raw log:

~~~bash
python3 \
  canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/classify_stock_discovery.py \
  --stochastic /path/to/p57-calibration-stochastic.raw.log \
  --output /tmp/p57-calibration-classification.json

sha256sum /tmp/p57-calibration-classification.json
~~~

The classifier rejects a missing or altered stock-fast attestation. Eligibility
is frozen before launch:

- stochastic solve 15–35%;
- mixed-group ratio at least 25%;
- inferred nonzero-advantage ratio at least 25%;
- no invalid terminal status;
- no observed context above the recipe cap;
- no prompt/response length reaching the physical 16,384-token cap.

Among eligible recipes, choose stochastic solve closest to 20%. Exact ties
prefer M15, then M10, then M20.

- `FREEZE_M10`, `FREEZE_M15`, or `FREEZE_M20`: freeze that recipe definition;
- `REVIEW_NO_ELIGIBLE_RECIPE`: valid evidence, but stop for user review;
- classifier `FAIL`: invalid evidence; infer nothing about task difficulty.

## Exact evidence return contract

Return all of the following to the analysis agent:

1. full source SHA, image digest, run id, JobSet, Attempt-0 identity, exit;
2. renderer and manifest-preflight stdout plus YAML SHA-256;
3. complete raw chief log from byte zero through terminal exit;
4. extracted calibration JSON v2 receipt;
5. exact stock rollout-sync marker;
6. all three dataset attestation records;
7. classifier JSON, stdout, exit code, and SHA-256;
8. exact infrastructure events, retries, OOMs, or truncation markers.

Do not return only W&B metrics, screenshots, `tail`, or selected solve rates.
Missing items make the campaign `INCONCLUSIVE`.

## Stop after classification

Do not launch a zero arm or the existing paired-training renderer. Under the
new scientific definition the later treatment is the complete numerical
zero-TIM bundle, not fixed lm-head alone. P57.2 must first register and certify
the stock-fast training arm and its full zero-TIM counterpart with all
nonnumerical recipe/performance settings held equal. Until then, the current
paired renderer is staging code, not an admitted experiment.

## Rollback

P57 is additive/default-off. Leaving `CANON_P57_*` unset restores the ordinary
P45 path. Rendered manifests are disposable and must never be edited into a new
experiment.

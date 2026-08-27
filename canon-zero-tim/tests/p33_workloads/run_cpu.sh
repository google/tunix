#!/usr/bin/env bash
# Run the complete CPU-only gate for the two default-off P33 workloads.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKTREE="$(cd "$ROOT/.." && pwd)"
cd "$WORKTREE"

python3 -c "import ast,pathlib; files=('tunix/rl/dp_workloads.py','tunix/rl/agentic/agentic_grpo_learner.py','tunix/rl/canonical_qwen3_adapter.py','tunix/rl/envelope_probe.py','tunix/rl/p38_frozenlake_replay.py','canon-zero-tim/src/engine_shims/p38_kv_fingerprint.py','canon-zero-tim/src/engine_shims/p38_tail_capture.py','canon-zero-tim/src/engine_shims/models/qwen1p7b_tp1/p22xf_contract.py','canon-zero-tim/src/engine_shims/models/qwen1p7b_tp2/p22xf_contract.py','canon-zero-tim/cluster/render_p33_jobsets.py','canon-zero-tim/cluster/render_p59_backward_ab.py','canon-zero-tim/cluster/render_p35_jobset.py','canon-zero-tim/cluster/render_p38_aval_jobset.py','canon-zero-tim/cluster/render_p38_serving_jobsets.py','canon-zero-tim/tests/t1_tpu/probe_logprob_aval.py','canon-zero-tim/tests/p33_workloads/validate_workload.py','canon-zero-tim/tests/p33_workloads/classify_run.py','canon-zero-tim/tests/p33_workloads/test_dp_workloads.py','canon-zero-tim/tests/p33_workloads/test_decode_logprob_chunking.py','canon-zero-tim/tests/p33_workloads/test_render_p33_jobsets.py','canon-zero-tim/tests/p33_workloads/test_classify_run.py','canon-zero-tim/tests/p33_workloads/test_sampler_is_contract.py','canon-zero-tim/tests/p59_backward/classify_and_analyze.py','canon-zero-tim/tests/p59_backward/probe_dp4_tp1_overlay.py','canon-zero-tim/tests/p59_backward/test_classify_and_analyze.py','canon-zero-tim/tests/p59_backward/test_dp4_carrier.py','canon-zero-tim/tests/p59_backward/test_render_p59_backward_ab.py','canon-zero-tim/tests/p59_backward/test_p59_persistence.py','canon-zero-tim/tests/p35_envelope/probe_memory_space_attestation.py','canon-zero-tim/tests/p35_envelope/test_render_p35_jobset.py','canon-zero-tim/tests/p38_aval/test_render_p38_aval_jobset.py','canon-zero-tim/tests/p38_serving/make_fixture.py','canon-zero-tim/tests/p38_serving/test_kv_fingerprint.py','canon-zero-tim/tests/p38_serving/test_render_p38_serving_jobsets.py','canon-zero-tim/tests/t1_tpu/test_probe_logprob_aval.py','tests/rl/canonical_qwen3_adapter_test.py','tests/rl/envelope_probe_test.py','tests/rl/p38_frozenlake_replay_test.py','examples/math_gsm8k/qwen3_grpo_demo.py','examples/frozenlake/train_frozenlake_qwen3.py'); [ast.parse(pathlib.Path(p).read_text(), filename=p) for p in files]"
if grep -Fq "P35 first target admits only one local-M chunk" \
  tunix/rl/agentic/agentic_grpo_learner.py; then
  echo "[P35.ENVELOPE] learner retained the rejected single-chunk gate" >&2
  exit 1
fi
python3 -c "import ast,pathlib; files=('tunix/rl/alignment.py','tests/rl/alignment_test.py'); [ast.parse(pathlib.Path(p).read_text(), filename=p) for p in files]"
python3 -c "import ast,pathlib; files=('canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/extract_p38_capsule.py','canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/test_extract_p38_capsule.py','canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/extract_p38_serving_archive.py','canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/test_extract_p38_serving_archive.py','canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/prepare_p38_frozenlake_replay.py','canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/classify_p38_frozenlake_replay.py','canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/test_classify_p38_frozenlake_replay.py','canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/classify_p38_serving_capture.py','canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/test_classify_p38_serving_capture.py','canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/classify_p38_kv_observer.py','canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/classify_p38_seam.py','canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/classify_p38_terminal_discriminator.py','canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/classify_p38_fixed_lm_head_receipts.py','canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/probe_p38_lm_head.py','canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/probe_p38_fixed_lm_head.py','canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/probe_p38_fixed_lm_head_vjp.py','canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/stage_p38_round.py','canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/check_p38_intent_diff.py','canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/test_check_p38_intent_diff.py','canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/run_p38_kv_fingerprint_onehost.py','canon-zero-tim/cluster/render_p38_backward_jobset.py','canon-zero-tim/tests/p38_serving/test_render_p38_backward_jobset.py','canon-zero-tim/tests/p38_serving/test_fixed_lm_head_receipts.py','canon-zero-tim/src/engine_shims/p38_fixed_lm_head.py'); [ast.parse(pathlib.Path(p).read_text(), filename=p) for p in files]"
python3 -c "import ast,pathlib; files=('canon-zero-tim/tests/p35_envelope/classify_envelope.py','canon-zero-tim/tests/p35_envelope/test_classify_envelope.py','canon-zero-tim/tests/p35_envelope/classify_exact_replay.py','canon-zero-tim/tests/p35_envelope/test_classify_exact_replay.py','canon-zero-tim/tests/p35_envelope/classify_stage_probe.py','canon-zero-tim/tests/p35_envelope/test_classify_stage_probe.py'); [ast.parse(pathlib.Path(p).read_text(), filename=p) for p in files]"
bash -n \
  canon-zero-tim/cluster/entrypoint.sh \
  canon-zero-tim/cluster/steps/00_env.sh \
  canon-zero-tim/cluster/steps/86_validate_workload.sh \
  canon-zero-tim/cluster/steps/90_run.sh \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/run_p38_aval_onehost.sh \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/run_p38_frozenlake_replay.sh \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/run_p38_incident_onehost.sh \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/run_p38_lm_head_onehost.sh \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/run_p38_fixed_lm_head_onehost.sh \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/run_p38_fixed_lm_head_vjp_onehost.sh \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/launch_p38h_backward.sh \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/collect_p38h_backward_return.sh \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/run_gsm8k_fixed_lm_head_onehost.sh \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/launch_p38y_gsm8k_full.sh \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/p38_live_snapshot_worker.sh \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/persist_p38_gcs.sh \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/seal_p38_evidence.sh \
  canon-zero-tim/tasks/p59-dp16-parallel-backward/scripts/run_and_persist.sh \
  canon-zero-tim/tasks/p59-dp16-parallel-backward/scripts/run_dp4_inner.sh \
  canon-zero-tim/tasks/p59-dp16-parallel-backward/scripts/run_onehost_dp4.sh \
  canon-zero-tim/tests/p59_backward/run_dp4_exact_image.sh \
  canon-zero-tim/tests/p38_serving/fake_gcloud.sh \
  canon-zero-tim/tests/p38_serving/test_evidence_seal.sh \
  canon-zero-tim/tests/p38_serving/test_gcs_persistence.sh \
  canon-zero-tim/tests/p38_serving/test_p38h_backward_operator_scripts.sh \
  canon-zero-tim/tests/p38_serving/test_postflight.sh \
  canon-zero-tim/cluster/profiles/qwen3-8b-dp16-tp4-frozenlake.env \
  canon-zero-tim/cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k.env \
  canon-zero-tim/cluster/profiles/qwen3-1p7b-dp4-tp1-gsm8k-p59.env \
  canon-zero-tim/tests/p33_workloads/run_exact_image.sh \
  canon-zero-tim/tests/p33_workloads/negative_control.sh

grep -Fq "sentencepiece==0.2.2" \
  canon-zero-tim/cluster/steps/30_install_canon.sh
grep -Fq "tiktoken==0.13.0" \
  canon-zero-tim/cluster/steps/30_install_canon.sh
grep -Fq "gymnasium==1.3.0" \
  canon-zero-tim/cluster/steps/30_install_canon.sh
grep -Fq -- "--no-deps" canon-zero-tim/cluster/steps/30_install_canon.sh
if grep -Fq -- "--target" canon-zero-tim/cluster/steps/30_install_canon.sh; then
  echo "[P33.WORKLOAD] canonical overlay accepted runtime dependencies" >&2
  exit 1
fi
grep -Fq "import gymnasium, numba, numpy, sentencepiece, tiktoken" \
  canon-zero-tim/cluster/steps/30_install_canon.sh

JAX_PLATFORMS=cpu python3 -m unittest \
  canon-zero-tim/tests/p33_workloads/test_dp_workloads.py \
  canon-zero-tim/tests/p33_workloads/test_render_p33_jobsets.py \
  canon-zero-tim/tests/p33_workloads/test_classify_run.py \
  canon-zero-tim/tests/p33_workloads/test_sampler_is_contract.py
python3 -m unittest discover \
  -s canon-zero-tim/tests/p59_backward \
  -p 'test_*.py'
JAX_PLATFORMS=cpu python3 -m unittest discover \
  -s tests/rl \
  -p alignment_test.py
JAX_PLATFORMS=cpu python3 -m unittest discover \
  -s tests/rl/agentic \
  -p agentic_rl_learner_test.py
JAX_PLATFORMS=cpu python3 -m unittest discover \
  -s tests/rl/rollout \
  -p vllm_rollout_canonical_test.py
JAX_PLATFORMS=cpu python3 -m unittest discover \
  -s tests/rl \
  -p envelope_probe_test.py
JAX_PLATFORMS=cpu python3 -m unittest discover \
  -s tests/rl \
  -p p38_frozenlake_replay_test.py
python3 -m unittest discover \
  -s canon-zero-tim/tests/p35_envelope \
  -p 'test_*.py'
python3 -m unittest discover \
  -s canon-zero-tim/tests/p38_aval \
  -p 'test_*.py'
JAX_PLATFORMS=cpu python3 -m unittest discover \
  -s canon-zero-tim/tests/p38_serving \
  -p 'test_*.py'
bash canon-zero-tim/tests/p38_serving/test_postflight.sh
bash canon-zero-tim/tests/p38_serving/test_p38h_backward_operator_scripts.sh
bash canon-zero-tim/tests/p38_serving/test_gcs_persistence.sh
python3 canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/test_extract_p38_capsule.py
python3 canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/test_extract_p38_serving_archive.py
python3 canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/test_classify_p38_frozenlake_replay.py
python3 canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/test_classify_p38_serving_capture.py
python3 canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/test_check_p38_intent_diff.py
bash canon-zero-tim/tests/p38_serving/test_evidence_seal.sh
PYTHONPATH=canon-zero-tim/tests/t1_tpu \
python3 -m unittest discover \
  -s canon-zero-tim/tests/t1_tpu \
  -p 'test_probe_logprob_aval.py'
PYTHONPATH=canon-zero-tim/tests/t1_tpu \
python3 -m unittest discover \
  -s canon-zero-tim/tests/t1_tpu \
  -p 'test_unified_runner.py'
JAX_PLATFORMS=cpu python3 -m unittest discover \
  -s tests/rl \
  -p canonical_qwen3_adapter_test.py \
  -k tied_embedding
JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
python3 -m unittest discover \
  -s tests/rl \
  -p canonical_qwen3_adapter_test.py \
  -k p38_frozenlake
JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
PYTHONPATH="$WORKTREE" \
python3 tests/rl/canonical_qwen3_adapter_test.py \
  CanonicalQwen3AdapterTest.test_p59_dp4_logprob_pipeline_pads_request64_per_rank_to_m256
JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
python3 -m unittest discover \
  -s tests/rl \
  -p dp_training_test.py
canon-zero-tim/tests/p33_workloads/negative_control.sh

validate_profile() (
  set -euo pipefail
  local workload="$1" profile="$2" state
  state="$(mktemp -d)"
  trap 'rm -r "$state"' EXIT
  export CANON_PKG="$ROOT"
  export CANON_STATE="$state"
  export CANON_MODE=workload-contract-only
  export CANON_PROFILE_FILE="$profile"
  bash "$ROOT/cluster/steps/00_env.sh"
  bash "$ROOT/cluster/steps/86_validate_workload.sh"
  python3 -c "import json; p='$state/p33_${workload}_contract.classification.json'; r=json.load(open(p)); assert r['verdict']=='PASS'; assert r['scope']=='contract-only'; assert r['dp_reduction_admitted'] is False"
  if [ "$workload" = frozenlake ]; then
    python3 -c "import json; p='$state/p33_frozenlake_contract.classification.json'; r=json.load(open(p)); assert r['periodic_evaluation'] is False"
  fi
  echo "[P33.WORKLOAD] PROFILE_PASS workload=$workload launch=refused"
)

validate_profile \
  gsm8k cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k.env
validate_profile \
  frozenlake cluster/profiles/qwen3-8b-dp16-tp4-frozenlake.env

validate_admitted_preflight() (
  set -euo pipefail
  local state
  state="$(mktemp -d)"
  trap 'rm -r "$state"' EXIT
  export CANON_PKG="$ROOT"
  export CANON_STATE="$state"
  export CANON_MODE=run
  export CANON_PROFILE_FILE=cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k.env
  export CANON_P32_TRAIN_ADMITTED=1
  export CANON_P32_DP_REDUCTION_ADMITTED=1
  export CANON_P33_WORKLOAD_LAUNCH_ADMITTED=1
  export CANON_P33_SHARED_MESH=16,4
  export CANON_P59_RANK_PARALLEL_BACKWARD=1
  export CANON_RUN_CMD="printf admitted-preflight-only"
  export CANON_RUN_LOG="$state/run.log"
  export CANON_PRE_ALIGN_REPORT="$state/pre_alignment.jsonl"
  export CANON_ALIGN_REPORT="$state/alignment.jsonl"
  export CANON_UPDATE_REPORT="$state/updates.jsonl"
  export INJECTED_WANDB_API_KEY=test-key-not-a-credential
  bash "$ROOT/cluster/steps/00_env.sh"
  test -s "$state/env.sh"
  grep -q 'export CANON_WANDB_ONLINE_REQUIRED=1' "$state/env.sh"
  grep -q 'export CANON_P31_MONOTONIC_METRICS=1' "$state/env.sh"
  grep -q 'export CANON_P59_RANK_PARALLEL_BACKWARD=1' "$state/env.sh"
  if grep -q 'WANDB_API_KEY' "$state/env.sh"; then
    echo "[P33.WORKLOAD] admitted preflight persisted a secret" >&2
    exit 1
  fi
  echo "[P33.WORKLOAD] ADMITTED_PREFLIGHT_PASS wandb=online secret_persisted=0"
)

validate_admitted_preflight

validate_gsm8k_alignment_warning_policy_preflight() (
  set -euo pipefail
  local state
  state="$(mktemp -d)"
  trap 'rm -r "$state"' EXIT
  export CANON_PKG="$ROOT"
  export CANON_STATE="$state"
  export CANON_MODE=run
  export CANON_PROFILE_FILE=cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k.env
  export CANON_P32_TRAIN_ADMITTED=1
  export CANON_P32_DP_REDUCTION_ADMITTED=1
  export CANON_P33_WORKLOAD_LAUNCH_ADMITTED=1
  export CANON_P33_SHARED_MESH=16,4
  export CANON_P33_RUN_STAGE=full
  export CANON_P33_NO_COMMIT=0
  export CANON_GSM8K_AB_REPORT_ONLY=0
  export CANON_GSM8K_ALIGNMENT_WARN_ONLY=1
  export CANON_P38_FIXED_LM_HEAD=1
  export CANON_RUN_CMD="printf alignment-warning-policy-preflight-only"
  export CANON_RUN_LOG="$state/run.log"
  export CANON_PRE_ALIGN_REPORT="$state/pre_alignment.jsonl"
  export CANON_ALIGN_REPORT="$state/alignment.jsonl"
  export CANON_UPDATE_REPORT="$state/updates.jsonl"
  export INJECTED_WANDB_API_KEY=test-key-not-a-credential
  bash "$ROOT/cluster/steps/00_env.sh" >/dev/null
  grep -q 'export CANON_GSM8K_ALIGNMENT_WARN_ONLY=1' "$state/env.sh"
  grep -q 'export CANON_P38_FIXED_LM_HEAD=1' "$state/env.sh"

  export CANON_PROFILE_FILE=cluster/profiles/qwen3-8b-dp16-tp4-frozenlake.env
  if bash "$ROOT/cluster/steps/00_env.sh" >/dev/null 2>&1; then
    echo "[P33.WORKLOAD] FrozenLake accepted the GSM8K alignment warning policy" >&2
    exit 1
  fi
  export CANON_PROFILE_FILE=cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k.env
  export CANON_GSM8K_AB_REPORT_ONLY=1
  if bash "$ROOT/cluster/steps/00_env.sh" >/dev/null 2>&1; then
    echo "[P33.WORKLOAD] mutually exclusive GSM8K policies were accepted" >&2
    exit 1
  fi
  echo "[P33.WORKLOAD] ALIGNMENT_WARNING_POLICY_PREFLIGHT_PASS gsm8k=accepted frozenlake_and_dual_policy=rejected"
)

validate_gsm8k_alignment_warning_policy_preflight

validate_p35_preflight() (
  set -euo pipefail
  local state
  state="$(mktemp -d)"
  trap 'rm -r "$state"' EXIT
  export CANON_PKG="$ROOT"
  export CANON_STATE="$state"
  export CANON_MODE=run
  export CANON_PROFILE_FILE=cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k.env
  export CANON_P32_TRAIN_ADMITTED=1
  export CANON_P32_DP_REDUCTION_ADMITTED=1
  export CANON_P33_WORKLOAD_LAUNCH_ADMITTED=1
  export CANON_P33_SHARED_MESH=16,4
  export CANON_P33_RUN_STAGE=envelope-short
  export CANON_P33_NO_COMMIT=1
  export CANON_P35_ENVELOPE=1
  export CANON_P35_ENVELOPE_REPORT="$state/p35.json"
  export CANON_P35_METADATA_DIR="$state/p35_metadata"
  export CANON_P35_CLASSIFICATION="$state/p35.classification.json"
  export CANON_RUN_CMD="python3 probe.py --max_response_length=256"
  export CANON_RUN_LOG="$state/run.log"
  export CANON_PRE_ALIGN_REPORT="$state/pre_alignment.jsonl"
  export CANON_ALIGN_REPORT="$state/alignment.jsonl"
  export CANON_UPDATE_REPORT="$state/updates.jsonl"
  export INJECTED_WANDB_API_KEY=test-key-not-a-credential
  bash "$ROOT/cluster/steps/00_env.sh" >/dev/null
  grep -q 'export CANON_P35_ENVELOPE=1' "$state/env.sh"

  for rejected_response in 64 65; do
    export CANON_RUN_CMD="python3 probe.py --max_response_length=$rejected_response"
    if bash "$ROOT/cluster/steps/00_env.sh" >/dev/null 2>&1; then
      echo "[P35.ENVELOPE] preflight accepted response-length drift: $rejected_response" >&2
      exit 1
    fi
  done
  echo "[P35.ENVELOPE] PREFLIGHT_PASS response256=accepted response64_65=rejected"
)

validate_p35_preflight

validate_p38_aval_preflight() (
  set -euo pipefail
  local state
  state="$(mktemp -d)"
  trap 'rm -r "$state"' EXIT
  export CANON_PKG="$ROOT"
  export CANON_STATE="$state"
  export CANON_MODE=gate-only
  export CANON_PROFILE_FILE=cluster/profiles/qwen3-8b-dp16-tp4-admission.env
  export CANON_RUN_P38_AVAL=1
  export CANON_P38_AVAL_REPORT="$state/p38_aval.result.json"
  bash "$ROOT/cluster/steps/00_env.sh" >/dev/null
  grep -q 'export CANON_RUN_P38_AVAL=1' "$state/env.sh"
  grep -q 'export CANON_P38_AVAL_REPORT=' "$state/env.sh"

  export CANON_RUN_P38_AVAL=invalid
  if bash "$ROOT/cluster/steps/00_env.sh" >/dev/null 2>&1; then
    echo "[P38.AVAL] preflight accepted an invalid switch" >&2
    exit 1
  fi
  echo "[P38.AVAL] PREFLIGHT_PASS enabled=accepted invalid=rejected"
)

validate_p38_aval_preflight

validate_p38_serving_preflight() (
  set -euo pipefail
  local state
  state="$(mktemp -d)"
  trap 'rm -r "$state"' EXIT
  export CANON_PKG="$ROOT"
  export CANON_STATE="$state"
  export CANON_MODE=run
  export CANON_PROFILE_FILE=cluster/profiles/qwen3-8b-dp16-tp4-frozenlake.env
  export CANON_P32_TRAIN_ADMITTED=1
  export CANON_P32_DP_REDUCTION_ADMITTED=1
  export CANON_P33_WORKLOAD_LAUNCH_ADMITTED=1
  export CANON_P33_SHARED_MESH=16,4
  export CANON_P33_RUN_STAGE=backward-no-commit
  export CANON_P33_NO_COMMIT=1
  export CANON_RUN_CMD="printf p38-serving-preflight-only --max_concurrency=256"
  export CANON_RUN_LOG="$state/run.log"
  export CANON_PRE_ALIGN_REPORT="$state/pre_alignment.jsonl"
  export CANON_ALIGN_REPORT="$state/alignment.jsonl"
  export CANON_UPDATE_REPORT="$state/updates.jsonl"
  export CANON_P38_MISMATCH_CAPSULE="$state/mismatch.npz"
  export CANON_P38_MISMATCH_CAPSULE_MAX_ROWS=256
  export CANON_P38_SERVING_CAPTURE_DIR="$state/serving"
  export CANON_P38_REQUEST_JOURNAL="$state/serving/p38_request_journal.jsonl"
  export CANON_P38_INCIDENT_LEDGER="$state/serving/p38_incident_ledger.jsonl"
  export CANON_P38_DIAGNOSTIC_ROUND_FILE="$state/p38_diagnostic_round"
  export CANON_P38_ROUND_SEAL_REQUEST_DIR="$state/p38_round_seal_requests"
  export CANON_P38_ROUND_SEAL_ACK_DIR="$state/p38_round_seal_acks"
  export CANON_P38_INCIDENT_MIN_PREFIX=1400
  export CANON_P38_INCIDENT_MAX_PREFIX=3072
  export CANON_P38_INCIDENT_MAX_BYTES=134217728
  export CANON_P38_DURABILITY_PROFILE=full-v1
  export CANON_P38_LIVE_SNAPSHOT_INTERVAL_SECONDS=30
  export CANON_P38_LIVE_SNAPSHOT_STOP_FILE="$state/p38_live.stop"
  export CANON_P38_LIVE_SNAPSHOT_WORKER_LOG="$state/p38_live_worker.log"
  export CANON_P38_LIVE_COLLECT_REQUEST_FILE="$state/p38_collect.request"
  export CANON_P38_LIVE_COLLECT_ACK_FILE="$state/p38_collect.ack"
  export CANON_P38_LIVE_COMPLETE_REQUEST_FILE="$state/p38_complete.request"
  export CANON_P38_LIVE_COMPLETE_ACK_FILE="$state/p38_complete.ack"
  export CANON_P38_SERVING_CAPTURE_MAX_CALLS=4
  export CANON_P38_SERVING_CAPTURE_MIN_PREFIX=1536
  export CANON_P38_SERVING_CAPTURE_PREFIX_BOUNDS=1536,1664,1792,1920,2048
  export CANON_P38_SERVING_CAPTURE_FREE_SPACE_MULTIPLIER=5
  export CANON_P38_SERVING_CAPTURE_EXPECTED_PATH=standard
  export CANON_P38_SERVING_CAPTURE_EXPECTED_RECORDS=4
  export CANON_P38_SERVING_CAPTURE_CLASSIFICATION="$state/serving.json"
  export CANON_P38_SERVING_CAPTURE_ARCHIVE="$state/serving.tar"
  export CANON_P38_GCS_PREFIX="gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/$(basename "$state")/attempt-0"
  export CANON_P38_PRECHECK_ONLY=1
  export CANON_P38_CONTROLLED_EXIT=1
  export CANON_P38_DIAGNOSTIC_ROUNDS=3
  export CANON_P38_MIN_ACTION_KV=1686
  export CANON_KV_UNIFIED=1
  export INJECTED_WANDB_API_KEY=test-key-not-a-credential
  bash "$ROOT/cluster/steps/00_env.sh" >/dev/null
  grep -q 'export CANON_KV_UNIFIED=1' "$state/env.sh"
  grep -q 'export CANON_P38_SERVING_CAPTURE_MAX_CALLS=4' "$state/env.sh"
  grep -Fq 'export CANON_P38_SERVING_CAPTURE_PREFIX_BOUNDS=1536\,1664\,1792\,1920\,2048' "$state/env.sh"
  grep -q 'export CANON_P38_SERVING_CAPTURE_FREE_SPACE_MULTIPLIER=5' "$state/env.sh"
  grep -q 'export CANON_P38_SERVING_CAPTURE_EXPECTED_PATH=standard' "$state/env.sh"
  grep -q 'export CANON_P38_CONTROLLED_EXIT=1' "$state/env.sh"
  grep -q 'export CANON_P38_DIAGNOSTIC_ROUNDS=3' "$state/env.sh"
  grep -Fq "export CANON_P38_ROUND_SEAL_REQUEST_DIR=$state/p38_round_seal_requests" "$state/env.sh"
  grep -Fq "export CANON_P38_ROUND_SEAL_ACK_DIR=$state/p38_round_seal_acks" "$state/env.sh"
  grep -q 'export CANON_P38_MIN_ACTION_KV=1686' "$state/env.sh"
  grep -q 'export CANON_P38_DURABILITY_PROFILE=full-v1' "$state/env.sh"
  grep -Fq "export CANON_P38_GCS_PREFIX=$CANON_P38_GCS_PREFIX" "$state/env.sh"

  valid_p38_gcs_prefix="$CANON_P38_GCS_PREFIX"
  export CANON_P38_GCS_PREFIX=gs://wrong-bucket/p38
  if bash "$ROOT/cluster/steps/00_env.sh" >/dev/null 2>&1; then
    echo "[P38.SERVING] preflight accepted a drifted GCS evidence prefix" >&2
    exit 1
  fi
  export CANON_P38_GCS_PREFIX="$valid_p38_gcs_prefix"
  grep -Fq "export CANON_P38_REQUEST_JOURNAL=$state/serving/p38_request_journal.jsonl" "$state/env.sh"
  grep -Fq "export CANON_P38_INCIDENT_LEDGER=$state/serving/p38_incident_ledger.jsonl" "$state/env.sh"
  grep -Fq "export CANON_P38_LIVE_SNAPSHOT_STOP_FILE=$state/p38_live.stop" "$state/env.sh"
  grep -Fq "export CANON_P38_LIVE_SNAPSHOT_WORKER_LOG=$state/p38_live_worker.log" "$state/env.sh"
  grep -Fq "export CANON_P38_LIVE_COLLECT_REQUEST_FILE=$state/p38_collect.request" "$state/env.sh"
  grep -Fq "export CANON_P38_LIVE_COLLECT_ACK_FILE=$state/p38_collect.ack" "$state/env.sh"
  grep -Fq "export CANON_P38_LIVE_COMPLETE_REQUEST_FILE=$state/p38_complete.request" "$state/env.sh"
  grep -Fq "export CANON_P38_LIVE_COMPLETE_ACK_FILE=$state/p38_complete.ack" "$state/env.sh"

  export CANON_KV_UNIFIED=0
  export CANON_P38_KV_OBSERVER_DIR="$state/serving"
  export CANON_P38_KV_OBSERVER_MAX_CANDIDATES=3
  export CANON_P38_KV_OBSERVER_MAX_PAGES=16
  export CANON_P38_KV_OBSERVER_MAX_BYTES=134217728
  export CANON_P38_KV_OBSERVER_MAX_READ_BYTES=671088640
  export CANON_P38_KV_OBSERVER_CLASSIFICATION="$state/p38_kv_observer.classification.json"
  bash "$ROOT/cluster/steps/00_env.sh" >/dev/null
  grep -q 'export CANON_P38_KV_OBSERVER_MAX_CANDIDATES=3' "$state/env.sh"
  grep -q 'export CANON_P38_KV_OBSERVER_MAX_PAGES=16' "$state/env.sh"
  export CANON_P38_KV_OBSERVER_MAX_PAGES=8
  if bash "$ROOT/cluster/steps/00_env.sh" >/dev/null 2>&1; then
    echo "[P38.SERVING] preflight accepted drifted KV observer bounds" >&2
    exit 1
  fi
  export CANON_P38_KV_OBSERVER_MAX_PAGES=16

  unset CANON_P38_KV_OBSERVER_DIR \
        CANON_P38_KV_OBSERVER_MAX_CANDIDATES \
        CANON_P38_KV_OBSERVER_MAX_PAGES \
        CANON_P38_KV_OBSERVER_MAX_BYTES \
        CANON_P38_KV_OBSERVER_MAX_READ_BYTES \
        CANON_P38_KV_OBSERVER_CLASSIFICATION
  export CANON_P38_DURABILITY_PROFILE=round-alignment-v1
  export CANON_P38_FIXED_LM_HEAD=1
  bash "$ROOT/cluster/steps/00_env.sh" >/dev/null
  grep -q 'export CANON_P38_FIXED_LM_HEAD=1' "$state/env.sh"
  grep -q 'export CANON_P38_DURABILITY_PROFILE=round-alignment-v1' "$state/env.sh"
  export CANON_P38_KV_OBSERVER_DIR="$state/serving"
  if bash "$ROOT/cluster/steps/00_env.sh" >/dev/null 2>&1; then
    echo "[P38.SERVING] preflight accepted fixed lm-head with KV observer" >&2
    exit 1
  fi
  unset CANON_P38_KV_OBSERVER_DIR
  export CANON_P38_DURABILITY_PROFILE=full-v1
  if bash "$ROOT/cluster/steps/00_env.sh" >/dev/null 2>&1; then
    echo "[P38.SERVING] preflight accepted fixed lm-head with full durability" >&2
    exit 1
  fi
  export CANON_P38_DURABILITY_PROFILE=round-alignment-v1
  export CANON_KV_UNIFIED=1
  if bash "$ROOT/cluster/steps/00_env.sh" >/dev/null 2>&1; then
    echo "[P38.SERVING] preflight accepted fixed lm-head on unified KV" >&2
    exit 1
  fi
  export CANON_KV_UNIFIED=0
  export CANON_MM_ALGO=1
  if bash "$ROOT/cluster/steps/00_env.sh" >/dev/null 2>&1; then
    echo "[P38.SERVING] preflight accepted fixed lm-head with MM_ALGO" >&2
    exit 1
  fi
  unset CANON_MM_ALGO
  export CANON_P38_FIXED_LM_HEAD=invalid
  if bash "$ROOT/cluster/steps/00_env.sh" >/dev/null 2>&1; then
    echo "[P38.SERVING] preflight accepted invalid fixed lm-head value" >&2
    exit 1
  fi
  unset CANON_P38_FIXED_LM_HEAD
  export CANON_P38_DURABILITY_PROFILE=full-v1

  export CANON_P38_SEAM_OBSERVER=layer
  export CANON_P38_SEAM_OBSERVER_DIR="$state/serving"
  export CANON_P38_SEAM_MIN_POSITION=1400
  export CANON_P38_SEAM_MAX_POSITION=3072
  export CANON_P38_SEAM_MAX_BYTES=4294967296
  export CANON_P38_SEAM_CLASSIFICATION="$state/p38_seam.classification.json"
  bash "$ROOT/cluster/steps/00_env.sh" >/dev/null
  grep -q 'export CANON_P38_SEAM_OBSERVER=layer' "$state/env.sh"
  grep -q 'export CANON_P38_SEAM_MAX_BYTES=4294967296' "$state/env.sh"
  export CANON_P38_TAIL_OBSERVER=1
  export CANON_P38_TAIL_MAX_BYTES=268435456
  bash "$ROOT/cluster/steps/00_env.sh" >/dev/null
  grep -q 'export CANON_P38_TAIL_OBSERVER=1' "$state/env.sh"
  export CANON_P38_TAIL_MAX_BYTES=1
  if bash "$ROOT/cluster/steps/00_env.sh" >/dev/null 2>&1; then
    echo "[P38.SERVING] preflight accepted drifted terminal-tail bounds" >&2
    exit 1
  fi
  export CANON_P38_TAIL_MAX_BYTES=268435456
  unset CANON_P38_TAIL_OBSERVER CANON_P38_TAIL_MAX_BYTES
  export CANON_P38_SEAM_OBSERVER=full
  if bash "$ROOT/cluster/steps/00_env.sh" >/dev/null 2>&1; then
    echo "[P38.SERVING] preflight accepted full seam mode without a layer" >&2
    exit 1
  fi
  export CANON_P38_SEAM_LAYER=17
  bash "$ROOT/cluster/steps/00_env.sh" >/dev/null
  export CANON_P38_SEAM_LAYER=36
  if bash "$ROOT/cluster/steps/00_env.sh" >/dev/null 2>&1; then
    echo "[P38.SERVING] preflight accepted an out-of-range seam layer" >&2
    exit 1
  fi
  export CANON_P38_SEAM_LAYER=17
  unset CANON_P38_SEAM_OBSERVER CANON_P38_SEAM_OBSERVER_DIR \
        CANON_P38_SEAM_MIN_POSITION CANON_P38_SEAM_MAX_POSITION \
        CANON_P38_SEAM_MAX_BYTES CANON_P38_SEAM_LAYER \
        CANON_P38_SEAM_CLASSIFICATION CANON_P38_TAIL_OBSERVER \
        CANON_P38_TAIL_MAX_BYTES

  export CANON_P38_SERVING_CAPTURE_EXPECTED_PATH=continue_decode
  if bash "$ROOT/cluster/steps/00_env.sh" >/dev/null 2>&1; then
    echo "[P38.SERVING] preflight accepted the unreachable continue-decode path" >&2
    exit 1
  fi
  export CANON_P38_SERVING_CAPTURE_EXPECTED_PATH=standard

  export CANON_P38_SERVING_CAPTURE_PREFIX_BOUNDS=1536,1792,2048
  if bash "$ROOT/cluster/steps/00_env.sh" >/dev/null 2>&1; then
    echo "[P38.SERVING] preflight accepted drifted prefix strata" >&2
    exit 1
  fi
  export CANON_P38_SERVING_CAPTURE_PREFIX_BOUNDS=1536,1664,1792,1920,2048

  unset CANON_P38_SERVING_CAPTURE_DIR
  if bash "$ROOT/cluster/steps/00_env.sh" >/dev/null 2>&1; then
    echo "[P38.SERVING] preflight accepted KV-unified without capture" >&2
    exit 1
  fi
  if bash "$ROOT/cluster/steps/00_env.sh" >/dev/null 2>&1; then
    echo "[P38.SERVING] preflight accepted partial capture configuration" >&2
    exit 1
  fi
  echo "[P38.SERVING] PREFLIGHT_PASS bounded=accepted partial_and_unbounded=rejected"
)

validate_p38_serving_preflight

validate_frozenlake_eval_postflight() (
  set -euo pipefail
  local state
  state="$(mktemp -d)"
  trap 'rm -r "$state"' EXIT
  export CANON_STATE="$state"
  export CANON_PKG="$ROOT"
  export CANON_RUN_CWD="$WORKTREE"
  cat > "$state/env.sh" <<'EOF'
export CANON_P32_TRAIN_ADMITTED=1
export CANON_P32_WORKLOAD=frozenlake
EOF
  export CANON_RUN_LOG="$state/missing-eval.log"
  export CANON_RUN_CMD="printf '%s\n' 'CANON_FIXED_AR=1 fixed-order tree' 'CANON_FIXED_AR_EMBED=1 fixed-order embed gather' '[CANON_P33_WANDB] ONLINE_RUN_PASS test'"
  if bash "$ROOT/cluster/steps/90_run.sh" >/dev/null 2>&1; then
    echo "[P33.WORKLOAD] eval postflight accepted a missing attestation" >&2
    exit 1
  fi
  export CANON_RUN_LOG="$state/present-eval.log"
  export CANON_RUN_CMD="printf '%s\n' 'CANON_FIXED_AR=1 fixed-order tree' 'CANON_FIXED_AR_EMBED=1 fixed-order embed gather' '[CANON_P33_WANDB] ONLINE_RUN_PASS test' '[CANON_P33_EVAL] DISABLED workload=frozenlake'"
  bash "$ROOT/cluster/steps/90_run.sh" >/dev/null
  echo "[P33.WORKLOAD] EVAL_POSTFLIGHT_PASS missing=rejected present=accepted"
)

validate_frozenlake_eval_postflight

validate_stale_evidence_rejected() (
  set -euo pipefail
  local state
  state="$(mktemp -d)"
  trap 'rm -r "$state"' EXIT
  export CANON_STATE="$state"
  export CANON_PKG="$ROOT"
  export CANON_RUN_CWD="$WORKTREE"
  cat > "$state/env.sh" <<EOF
export CANON_P32_TRAIN_ADMITTED=1
export CANON_P33_WORKLOAD_LAUNCH_ADMITTED=1
export CANON_P32_WORKLOAD=gsm8k
export CANON_RUN_LOG=$state/run.log
export CANON_PRE_ALIGN_REPORT=$state/pre_alignment.jsonl
export CANON_ALIGN_REPORT=$state/alignment.jsonl
export CANON_UPDATE_REPORT=$state/updates.jsonl
EOF
  touch "$state/updates.jsonl"
  export CANON_RUN_CMD="printf '%s\n' 'CANON_FIXED_AR=1 fixed-order tree' 'CANON_FIXED_AR_EMBED=1 fixed-order embed gather'"
  if bash "$ROOT/cluster/steps/90_run.sh" >/dev/null 2>&1; then
    echo "[P33.WORKLOAD] stale evidence path was accepted" >&2
    exit 1
  fi
  test ! -e "$state/run.log"
  echo "[P33.WORKLOAD] STALE_EVIDENCE_REJECTED"
)

validate_stale_evidence_rejected

validate_failed_pre_alignment_stdout() (
  set -euo pipefail
  local state output
  state="$(mktemp -d)"
  trap 'rm -r "$state"' EXIT
  output="$state/runner.stdout"
  cat > "$state/env.sh" <<EOF
export CANON_P33_WORKLOAD_LAUNCH_ADMITTED=1
export CANON_P32_TRAIN_ADMITTED=0
export CANON_RUN_LOG=$state/run.log
export CANON_PRE_ALIGN_REPORT=$state/pre_alignment.jsonl
export CANON_ALIGN_REPORT=$state/alignment.jsonl
export CANON_UPDATE_REPORT=$state/updates.jsonl
export CANON_RUN_CMD="printf '%s\\n' '{\"verdict\":\"FAIL\",\"reds\":[\"S_decode_vs_S_prefill\"]}' > '$state/pre_alignment.jsonl'; printf '%s\\n' 'CANON_FIXED_AR=1 fixed-order tree' 'CANON_FIXED_AR_EMBED=1 fixed-order embed gather'; exit 17"
EOF
  export CANON_STATE="$state"
  export CANON_PKG="$ROOT"
  export CANON_RUN_CWD="$WORKTREE"
  if bash "$ROOT/cluster/steps/90_run.sh" >"$output" 2>&1; then
    echo "[P38.EVIDENCE] failed workload was accepted" >&2
    exit 1
  fi
  grep -Fq '[CANON_PRE_ALIGN_ARTIFACT] path=' "$output"
  grep -Fq 'rows=1 sha256=' "$output"
  grep -Fq '[CANON_PRE_ALIGN_ARTIFACT_JSON] {"verdict":"FAIL","reds":["S_decode_vs_S_prefill"]}' "$output"
  echo "[P38.EVIDENCE] FAILED_REPORT_STDOUT_PASS"
)

validate_failed_pre_alignment_stdout

validate_p35_postflight() (
  set -euo pipefail
  local state report_source report_output classification
  state="$(mktemp -d)"
  trap 'rm -r "$state"' EXIT
  report_source="$state/source.json"
  report_output="$state/report.json"
  classification="$state/classification.json"
  python3 - "$report_source" <<'PY'
import importlib.util
import json
import pathlib
import sys

target = pathlib.Path("canon-zero-tim/tests/p35_envelope/test_classify_envelope.py")
spec = importlib.util.spec_from_file_location("p35_fixture", target)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
pathlib.Path(sys.argv[1]).write_text(
    json.dumps(module._report(False, True)) + "\n", encoding="utf-8"
)
PY
  export CANON_STATE="$state"
  export CANON_PKG="$ROOT"
  export CANON_RUN_CWD="$WORKTREE"
  export CANON_P35_ENVELOPE=1
  export CANON_P35_ENVELOPE_REPORT="$report_output"
  export CANON_P35_METADATA_DIR="$state/metadata"
  export CANON_P35_CLASSIFICATION="$classification"
  export CANON_RUN_LOG="$state/run.log"
  : > "$state/env.sh"

  export CANON_RUN_CMD="cp '$report_source' '$report_output'; printf '%s\n' 'CANON_FIXED_AR=1 fixed-order tree' 'CANON_FIXED_AR_EMBED=1 fixed-order embed gather'"
  if bash "$ROOT/cluster/steps/90_run.sh" >/dev/null 2>&1; then
    echo "[P35.ENVELOPE] postflight accepted report without stop marker" >&2
    exit 1
  fi
  rm -f "$report_output" "$classification" "$CANON_RUN_LOG"

  export CANON_RUN_CMD="printf '%s\n' 'CANON_FIXED_AR=1 fixed-order tree' 'CANON_FIXED_AR_EMBED=1 fixed-order embed gather' '[CANON_P35] REPORT_COMPLETE path=missing STOP_BEFORE_BACKWARD'"
  if bash "$ROOT/cluster/steps/90_run.sh" >/dev/null 2>&1; then
    echo "[P35.ENVELOPE] postflight accepted stop marker without report" >&2
    exit 1
  fi
  rm -f "$classification" "$CANON_RUN_LOG"

  export CANON_RUN_CMD="cp '$report_source' '$report_output'; printf '%s\n' 'CANON_FIXED_AR=1 fixed-order tree' 'CANON_FIXED_AR_EMBED=1 fixed-order embed gather' '[CANON_P35] REPORT_COMPLETE path=$report_output STOP_BEFORE_BACKWARD'; exit 17"
  if bash "$ROOT/cluster/steps/90_run.sh" >/dev/null 2>&1; then
    echo "[P35.ENVELOPE] postflight accepted an unexpected diagnostic exit" >&2
    exit 1
  fi
  rm -f "$report_output" "$classification" "$CANON_RUN_LOG"

  export CANON_RUN_CMD="cp '$report_source' '$report_output'; printf '%s\n' 'CANON_FIXED_AR=1 fixed-order tree' 'CANON_FIXED_AR_EMBED=1 fixed-order embed gather' '[CANON_P35] REPORT_COMPLETE path=$report_output STOP_BEFORE_BACKWARD'; exit 1"
  bash "$ROOT/cluster/steps/90_run.sh" >/dev/null
  python3 -c "import json; r=json.load(open('$classification')); assert r['measurement_verdict']=='COMPLETE'; assert r['classification']=='packing_metadata_carrier'"
  echo "[P35.ENVELOPE] POSTFLIGHT_PASS expected_exit_accepted=1 fail_closed_controls=3"
)

validate_p35_postflight

validate_p35_exact_postflight() (
  set -euo pipefail
  local state base_source exact_source base_pre base_final exact_output driver
  state="$(mktemp -d)"
  trap 'rm -r "$state"' EXIT
  base_source="$state/base_source.json"
  exact_source="$state/exact_source.json"
  base_pre="$state/p35_envelope.pre_replay.json"
  base_final="$state/p35_envelope.json"
  exact_output="$state/p35_exact_replay.json"
  driver="$state/driver.log"
  python3 - "$base_source" "$exact_source" <<'PY'
import importlib.util
import json
import pathlib
import sys

def load(name, path):
  spec = importlib.util.spec_from_file_location(name, pathlib.Path(path))
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module

base = load("p35_base_fixture", "canon-zero-tim/tests/p35_envelope/test_classify_envelope.py")
exact = load("p35_exact_fixture", "canon-zero-tim/tests/p35_envelope/test_classify_exact_replay.py")
pathlib.Path(sys.argv[1]).write_text(json.dumps(base._report(False, True)) + "\n")
pathlib.Path(sys.argv[2]).write_text(json.dumps(exact._report()) + "\n")
PY
  export CANON_STATE="$state"
  export CANON_PKG="$ROOT"
  export CANON_RUN_CWD="$WORKTREE"
  export CANON_P35_ENVELOPE=1
  export CANON_P35_ENVELOPE_REPORT="$base_final"
  export CANON_P35_PRE_REPLAY_REPORT="$base_pre"
  export CANON_P35_METADATA_DIR="$state/metadata"
  export CANON_P35_CLASSIFICATION="$state/p35_envelope.classification.json"
  export CANON_P35_EXACT_REPLAY=1
  export CANON_P35_EXACT_REPLAY_REPORT="$exact_output"
  export CANON_P35_EXACT_REPLAY_CLASSIFICATION="$state/p35_exact_replay.classification.json"
  export CANON_RUN_LOG="$state/run.log"
  : > "$state/env.sh"

  export CANON_RUN_CMD="cp '$base_source' '$base_pre'; printf '%s\n' 'CANON_FIXED_AR=1 fixed-order tree' 'CANON_FIXED_AR_EMBED=1 fixed-order embed gather' '[CANON_P35] BASE_REPORT_COMPLETE path=$base_pre REPLAY_PENDING'; exit 1"
  if bash "$ROOT/cluster/steps/90_run.sh" >"$driver" 2>&1; then
    echo "[P35.ENVELOPE] exact postflight accepted a missing replay" >&2
    exit 1
  fi
  grep -q '\[CANON_P35.3\] PRE_REPLAY_EVIDENCE .*sha256=' "$driver"
  rm -f "$base_pre" "$CANON_RUN_LOG" "$driver"

  export CANON_RUN_CMD="cp '$base_source' '$base_pre'; cp '$base_source' '$base_final'; cp '$exact_source' '$exact_output'; printf '%s\n' 'CANON_FIXED_AR=1 fixed-order tree' 'CANON_FIXED_AR_EMBED=1 fixed-order embed gather' '[CANON_P35] BASE_REPORT_COMPLETE path=$base_pre REPLAY_PENDING' '[CANON_P35.3] REPLAY_COMPLETE path=$exact_output' '[CANON_P35] REPORT_COMPLETE path=$base_final STOP_BEFORE_BACKWARD'; exit 1"
  bash "$ROOT/cluster/steps/90_run.sh" >"$driver"
  python3 -c "import json; b=json.load(open('$CANON_P35_CLASSIFICATION')); e=json.load(open('$CANON_P35_EXACT_REPLAY_CLASSIFICATION')); assert b['measurement_verdict']=='COMPLETE'; assert e['measurement_verdict']=='COMPLETE'"
  grep -q '\[CANON_P35.3\] PRE_REPLAY_EVIDENCE .*sha256=' "$driver"
  echo "[P35.ENVELOPE] EXACT_POSTFLIGHT_PASS preliminary_failure_preserved=1 complete_accepted=1"
)

validate_p35_exact_postflight

validate_p35_stage_probe_postflight() (
  set -euo pipefail
  local state base_source stage_source base_pre stage_output driver
  state="$(mktemp -d)"
  trap 'rm -r "$state"' EXIT
  base_source="$state/base_source.json"
  stage_source="$state/stage_source.jsonl"
  base_pre="$state/p35_envelope.pre_replay.json"
  stage_output="$state/p35_replay_stages.jsonl"
  driver="$state/driver.log"
  python3 - "$base_source" "$stage_source" <<'PY'
import importlib.util
import json
import pathlib
import sys

def load(name, path):
  spec = importlib.util.spec_from_file_location(name, pathlib.Path(path))
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module

base = load("p35_stage_base", "canon-zero-tim/tests/p35_envelope/test_classify_envelope.py")
stage = load("p35_stage_fixture", "canon-zero-tim/tests/p35_envelope/test_classify_stage_probe.py")
pathlib.Path(sys.argv[1]).write_text(json.dumps(base._report(False, True)) + "\n")
pathlib.Path(sys.argv[2]).write_text(
    "".join(json.dumps(event) + "\n" for event in stage._events())
)
PY
  export CANON_STATE="$state"
  export CANON_PKG="$ROOT"
  export CANON_RUN_CWD="$WORKTREE"
  export CANON_P35_ENVELOPE=1
  export CANON_P35_ENVELOPE_REPORT="$state/p35_envelope.json"
  export CANON_P35_PRE_REPLAY_REPORT="$base_pre"
  export CANON_P35_METADATA_DIR="$state/metadata"
  export CANON_P35_CLASSIFICATION="$state/p35_envelope.classification.json"
  export CANON_P35_EXACT_REPLAY=1
  export CANON_P35_EXACT_REPLAY_REPORT="$state/p35_exact_replay.json"
  export CANON_P35_EXACT_REPLAY_CLASSIFICATION="$state/p35_exact_replay.classification.json"
  export CANON_P35_REPLAY_STAGE_PROBE=1
  export CANON_P35_REPLAY_STAGE_REPORT="$stage_output"
  export CANON_P35_REPLAY_STAGE_CLASSIFICATION="$state/p35_replay_stages.classification.json"
  export CANON_RUN_LOG="$state/run.log"
  : > "$state/env.sh"

  export CANON_RUN_CMD="cp '$base_source' '$base_pre'; head -n 5 '$stage_source' > '$stage_output'; printf '%s\n' 'CANON_FIXED_AR=1 fixed-order tree' 'CANON_FIXED_AR_EMBED=1 fixed-order embed gather' '[CANON_P35] BASE_REPORT_COMPLETE path=$base_pre REPLAY_PENDING' '[CANON_P35.3C] STAGE_BEGIN stage=model' '[CANON_P35.3C] STAGE_READY stage=model' '[CANON_P35.3C] STAGE_BEGIN stage=logits' '[CANON_P35.3C] STAGE_READY stage=logits' '[CANON_P35.3C] STAGE_BEGIN stage=sample' '[CANON_P35.3C] STAGE_READY stage=sample' '[CANON_P35.3C] STAGE_BEGIN stage=logprobs' '[CANON_P35.3C] STAGE_READY stage=logprobs' '[CANON_P35.3C] STAGE_BEGIN stage=target_gathers' '[CANON_P35.3C] STAGE_READY stage=target_gathers' '[CANON_P35.3C] STAGE_PROBE_COMPLETE NO_NUMERICAL_VERDICT'; exit 1"
  if bash "$ROOT/cluster/steps/90_run.sh" >"$driver" 2>&1; then
    echo "[P35.ENVELOPE] stage postflight accepted a missing stage" >&2
    exit 1
  fi
  python3 -c "import json; s=json.load(open('$CANON_P35_REPLAY_STAGE_CLASSIFICATION')); assert s['measurement_verdict']=='INCONCLUSIVE'; assert s['last_ready_stage']=='target_gathers'; assert s['first_missing_stage']=='record_outputs'"
  rm -f "$base_pre" "$stage_output" "$CANON_P35_CLASSIFICATION" \
    "$CANON_P35_REPLAY_STAGE_CLASSIFICATION" "$CANON_RUN_LOG" "$driver"

  export CANON_RUN_CMD="cp '$base_source' '$base_pre'; cp '$stage_source' '$stage_output'; printf '%s\n' 'CANON_FIXED_AR=1 fixed-order tree' 'CANON_FIXED_AR_EMBED=1 fixed-order embed gather' '[CANON_P35] BASE_REPORT_COMPLETE path=$base_pre REPLAY_PENDING' '[CANON_P35.3C] STAGE_BEGIN stage=model' '[CANON_P35.3C] STAGE_READY stage=model' '[CANON_P35.3C] STAGE_BEGIN stage=logits' '[CANON_P35.3C] STAGE_READY stage=logits' '[CANON_P35.3C] STAGE_BEGIN stage=sample' '[CANON_P35.3C] STAGE_READY stage=sample' '[CANON_P35.3C] STAGE_BEGIN stage=logprobs' '[CANON_P35.3C] STAGE_READY stage=logprobs' '[CANON_P35.3C] STAGE_BEGIN stage=target_gathers' '[CANON_P35.3C] STAGE_READY stage=target_gathers' '[CANON_P35.3C] STAGE_BEGIN stage=record_outputs' '[CANON_P35.3C] STAGE_READY stage=record_outputs' '[CANON_P35.3C] STAGE_PROBE_COMPLETE NO_NUMERICAL_VERDICT'; exit 1"
  bash "$ROOT/cluster/steps/90_run.sh" >"$driver"
  python3 -c "import json; s=json.load(open('$CANON_P35_REPLAY_STAGE_CLASSIFICATION')); assert s['measurement_verdict']=='COMPLETE'; assert s['numerical_verdict'] is False"
  grep -q '\[run\] P35.3c first-record stage probe accepted; NO_NUMERICAL_VERDICT' "$driver"
  echo "[P35.ENVELOPE] STAGE_POSTFLIGHT_PASS missing_stage_rejected=1 numerical_verdict=0"
)

validate_p35_stage_probe_postflight

echo "[P33.WORKLOAD] CPU_GATE PASS workloads=2 p35_postflight=1 p35_stage_probe=1"

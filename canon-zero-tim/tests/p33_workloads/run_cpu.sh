#!/usr/bin/env bash
# Run the complete CPU-only gate for the two default-off P33 workloads.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKTREE="$(cd "$ROOT/.." && pwd)"
cd "$WORKTREE"

python3 -c "import ast,pathlib; files=('tunix/rl/dp_workloads.py','tunix/rl/agentic/agentic_grpo_learner.py','tunix/rl/canonical_qwen3_adapter.py','tunix/rl/envelope_probe.py','canon-zero-tim/cluster/render_p33_jobsets.py','canon-zero-tim/cluster/render_p35_jobset.py','canon-zero-tim/tests/p33_workloads/validate_workload.py','canon-zero-tim/tests/p33_workloads/classify_run.py','canon-zero-tim/tests/p33_workloads/test_dp_workloads.py','canon-zero-tim/tests/p33_workloads/test_decode_logprob_chunking.py','canon-zero-tim/tests/p33_workloads/test_render_p33_jobsets.py','canon-zero-tim/tests/p33_workloads/test_classify_run.py','canon-zero-tim/tests/p33_workloads/test_sampler_is_contract.py','canon-zero-tim/tests/p35_envelope/probe_memory_space_attestation.py','canon-zero-tim/tests/p35_envelope/test_render_p35_jobset.py','tests/rl/canonical_qwen3_adapter_test.py','tests/rl/envelope_probe_test.py','examples/math_gsm8k/qwen3_grpo_demo.py','examples/frozenlake/train_frozenlake_qwen3.py'); [ast.parse(pathlib.Path(p).read_text(), filename=p) for p in files]"
if grep -Fq "P35 first target admits only one local-M chunk" \
  tunix/rl/agentic/agentic_grpo_learner.py; then
  echo "[P35.ENVELOPE] learner retained the rejected single-chunk gate" >&2
  exit 1
fi
python3 -c "import ast,pathlib; files=('tunix/rl/alignment.py','tests/rl/alignment_test.py'); [ast.parse(pathlib.Path(p).read_text(), filename=p) for p in files]"
python3 -c "import ast,pathlib; files=('canon-zero-tim/tests/p35_envelope/classify_envelope.py','canon-zero-tim/tests/p35_envelope/test_classify_envelope.py','canon-zero-tim/tests/p35_envelope/classify_exact_replay.py','canon-zero-tim/tests/p35_envelope/test_classify_exact_replay.py'); [ast.parse(pathlib.Path(p).read_text(), filename=p) for p in files]"
bash -n \
  canon-zero-tim/cluster/entrypoint.sh \
  canon-zero-tim/cluster/steps/00_env.sh \
  canon-zero-tim/cluster/steps/86_validate_workload.sh \
  canon-zero-tim/cluster/steps/90_run.sh \
  canon-zero-tim/cluster/profiles/qwen3-8b-dp16-tp4-frozenlake.env \
  canon-zero-tim/cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k.env \
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
JAX_PLATFORMS=cpu python3 -m unittest discover \
  -s tests/rl \
  -p alignment_test.py
JAX_PLATFORMS=cpu python3 -m unittest discover \
  -s tests/rl/rollout \
  -p vllm_rollout_canonical_test.py
JAX_PLATFORMS=cpu python3 -m unittest discover \
  -s tests/rl \
  -p envelope_probe_test.py
python3 -m unittest discover \
  -s canon-zero-tim/tests/p35_envelope \
  -p 'test_*.py'
JAX_PLATFORMS=cpu python3 -m unittest discover \
  -s tests/rl \
  -p canonical_qwen3_adapter_test.py \
  -k tied_embedding
JAX_PLATFORMS=cpu \
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
python3 -m unittest discover \
  -s tests/rl \
  -p dp_training_test.py \
  -k explicit_data_axis
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
  if grep -q 'WANDB_API_KEY' "$state/env.sh"; then
    echo "[P33.WORKLOAD] admitted preflight persisted a secret" >&2
    exit 1
  fi
  echo "[P33.WORKLOAD] ADMITTED_PREFLIGHT_PASS wandb=online secret_persisted=0"
)

validate_admitted_preflight

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

echo "[P33.WORKLOAD] CPU_GATE PASS workloads=2 p35_postflight=1"

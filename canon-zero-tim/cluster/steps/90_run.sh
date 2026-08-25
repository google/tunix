#!/usr/bin/env bash
# The actual workload.
#
# CANON_RUN_CMD carries the command so the same entry point serves gates, canaries and
# training without editing YAML.  The exit code is propagated, and the PATHTRACE tally is
# checked afterwards: an intervention that never fired produces a perfectly green run.
set -uo pipefail
source "$CANON_STATE/env.sh"
# shellcheck disable=SC1091
source "$CANON_PKG/cluster/steps/p57_runtime_contract.sh"
# shellcheck disable=SC1091
source "$CANON_PKG/cluster/steps/jax_cache_sync_lib.sh"
# shellcheck disable=SC1091
source "$CANON_PKG/cluster/steps/xprof_gcs_sync_lib.sh"

# A full GSM8K JobSet may be recreated after an eviction or node loss. Preserve
# fail-closed no-overwrite semantics within each attempt without letting an
# Attempt-0 run.log make all later infrastructure retries fail immediately.
# Diagnostics and other workloads retain their historical paths unchanged.
if [ "${CANON_P32_WORKLOAD:-}" = "gsm8k" ] && \
   [ "${CANON_P33_RUN_STAGE:-}" = "full" ] && \
   [ "${CANON_P33_NO_COMMIT:-0}" = "0" ] && \
   [ -n "${JOBSET_RESTART_ATTEMPT:-}" ]; then
  if [[ ! "$JOBSET_RESTART_ATTEMPT" =~ ^[0-9]+$ ]]; then
    echo "[run] FATAL: GSM8K full restart attempt must be a non-negative integer" >&2
    exit 1
  fi
  attempt_evidence_dir="${CANON_STATE%/}/attempt-$JOBSET_RESTART_ATTEMPT"
  export CANON_RUN_LOG="$attempt_evidence_dir/run.log"
  export CANON_PRE_ALIGN_REPORT="$attempt_evidence_dir/pre_alignment.jsonl"
  export CANON_ALIGN_REPORT="$attempt_evidence_dir/alignment.jsonl"
  export CANON_UPDATE_REPORT="$attempt_evidence_dir/updates.jsonl"
  echo "[run] GSM8K_FULL_ATTEMPT_EVIDENCE attempt=$JOBSET_RESTART_ATTEMPT dir=$attempt_evidence_dir"
fi
for k in HF_TOKEN WANDB_API_KEY; do
  inj="INJECTED_$k"
  if [ -n "${!inj:-}" ]; then
    v="$(printf '%s' "${!inj}" | tr -d '[:space:]')"
    export "$k=$v"
  fi
done
export JAX_PLATFORMS="proxy,cpu"
export JAX_BACKEND_TARGET="grpc://localhost:29000"
export PATHWAYS_HEAD="localhost"

p38_stop_live_worker() {
  if [ -z "${p38_live_pid:-}" ]; then
    return 0
  fi
  touch "$CANON_P38_LIVE_SNAPSHOT_STOP_FILE"
  p38_live_rc=0
  wait "$p38_live_pid" || p38_live_rc=$?
  cat "$CANON_P38_LIVE_SNAPSHOT_WORKER_LOG"
  echo "[P38.GCS] LIVE_WORKER_JOINED rc=$p38_live_rc"
  p38_live_pid=""
  return 0
}

p38_request_live_action() {
  local action="$1" request_file ack_file partial unused
  case "$action" in
    collect)
      request_file="$CANON_P38_LIVE_COLLECT_REQUEST_FILE"
      ack_file="$CANON_P38_LIVE_COLLECT_ACK_FILE"
      ;;
    complete)
      request_file="$CANON_P38_LIVE_COMPLETE_REQUEST_FILE"
      ack_file="$CANON_P38_LIVE_COMPLETE_ACK_FILE"
      ;;
    *)
      echo "[run] FATAL: unknown P38 live-worker action: $action" >&2
      return 2
      ;;
  esac
  if [ -e "$request_file" ] || [ -e "$ack_file" ]; then
    echo "[run] FATAL: repeated P38 live-worker action: $action" >&2
    return 2
  fi
  partial="${request_file}.partial"
  (umask 077; printf 'action=%s\n' "$action" > "$partial")
  mv -- "$partial" "$request_file"
  echo "[P38.GCS] LIVE_ACTION_REQUESTED action=$action request=$request_file"
  for unused in $(seq 1 900); do
    if [ -s "$ack_file" ]; then
      if [ "$(cat "$ack_file")" != "action=$action status=PASS" ]; then
        echo "[run] FATAL: malformed P38 live-worker acknowledgement: $ack_file" >&2
        return 2
      fi
      echo "[P38.GCS] LIVE_ACTION_ACKNOWLEDGED action=$action ack=$ack_file"
      return 0
    fi
    if ! kill -0 "$p38_live_pid" 2>/dev/null; then
      echo "[run] FATAL: P38 live worker exited before $action acknowledgement" >&2
      cat "$CANON_P38_LIVE_SNAPSHOT_WORKER_LOG" >&2 || true
      return 2
    fi
    sleep 1
  done
  echo "[run] FATAL: timed out waiting for P38 live-worker action: $action" >&2
  return 2
}
if [ "${CANON_P32_DP_ADMISSION:-0}" = "1" ] && \
   [ "${CANON_P32_TRAIN_ADMITTED:-0}" != "1" ]; then
  if ! p57_is_nontraining_runtime; then
    echo "[run] REFUSING: P32 profile is admission-only." >&2
    echo "[run] A real (dp,tp) replicated-parameter segmented VJP has not passed remote gates;" >&2
    echo "[run] set all three P33 admissions only after the rank-local reducer gate passes." >&2
    exit 2
  fi
fi
: "${CANON_RUN_CMD:?CANON_RUN_CMD unset -- nothing to run}"
LOG="${CANON_RUN_LOG:-$CANON_STATE/run.log}"
if [ "${CANON_P33_WORKLOAD_LAUNCH_ADMITTED:-0}" = "1" ]; then
  report_keys=(CANON_RUN_LOG CANON_PRE_ALIGN_REPORT CANON_ALIGN_REPORT CANON_UPDATE_REPORT)
  if [ "${CANON_P34_DEEPSWE:-0}" = "1" ]; then
    report_keys+=(CANON_P34_WEIGHT_REPORT)
    if [ "${CANON_P44_DEEPSWE_PARITY:-0}" = "1" ]; then
      report_keys+=(CANON_P44_DEBUG_DIR)
    elif [ "${CANON_P58_DEEPSWE_TIM:-0}" = "1" ]; then
      report_keys+=(CANON_P58_DEBUG_DIR)
    elif [ "${CANON_P43_DEEPSWE_DEBUG:-0}" = "1" ]; then
      report_keys+=(CANON_P43_DEBUG_DIR)
    elif [ "${CANON_P34_TRAJECTORY_CAPTURE:-0}" = "1" ]; then
      report_keys+=(CANON_P34_DEBUG_DIR)
    fi
  fi
  if [ -n "${CANON_P38_MISMATCH_CAPSULE:-}" ]; then
    report_keys+=(CANON_P38_MISMATCH_CAPSULE)
  fi
  if [ -n "${CANON_P38_SERVING_CAPTURE_DIR:-}" ]; then
    report_keys+=(CANON_P38_SERVING_CAPTURE_CLASSIFICATION
                  CANON_P38_SERVING_CAPTURE_ARCHIVE)
    if [ -n "${CANON_P38_KV_OBSERVER_DIR:-}" ]; then
      report_keys+=(CANON_P38_KV_OBSERVER_CLASSIFICATION)
    elif [ -n "${CANON_P38_SEAM_OBSERVER:-}" ]; then
      report_keys+=(CANON_P38_SEAM_CLASSIFICATION)
    fi
    if [ -e "$CANON_P38_SERVING_CAPTURE_DIR" ]; then
      echo "[run] FATAL: P38 serving-capture directory already exists: $CANON_P38_SERVING_CAPTURE_DIR" >&2
      exit 1
    fi
    mkdir -p "$(dirname "$CANON_P38_SERVING_CAPTURE_DIR")"
  fi
  for report_key in "${report_keys[@]}"; do
    report_path="${!report_key:-}"
    if [ -z "$report_path" ]; then
      echo "[run] FATAL: admitted P33 workload lacks $report_key" >&2
      exit 1
    fi
    if [ -e "$report_path" ]; then
      echo "[run] FATAL: admitted P33 evidence path already exists: $report_key=$report_path" >&2
      exit 1
    fi
    mkdir -p "$(dirname "$report_path")"
  done
  if [ "${CANON_P35_ENVELOPE:-0}" = "1" ]; then
    for report_key in CANON_P35_ENVELOPE_REPORT CANON_P35_METADATA_DIR \
                      CANON_P35_CLASSIFICATION; do
      report_path="${!report_key:-}"
      if [ -z "$report_path" ]; then
        echo "[run] FATAL: P35 lacks $report_key" >&2
        exit 1
      fi
      if [ -e "$report_path" ]; then
        echo "[run] FATAL: P35 evidence path already exists: $report_key=$report_path" >&2
        exit 1
      fi
      mkdir -p "$(dirname "$report_path")"
    done
    if [ "${CANON_P35_EXACT_REPLAY:-0}" = "1" ]; then
      for report_key in CANON_P35_PRE_REPLAY_REPORT \
                        CANON_P35_EXACT_REPLAY_REPORT \
                        CANON_P35_EXACT_REPLAY_CLASSIFICATION; do
        report_path="${!report_key:-}"
        if [ -z "$report_path" ]; then
          echo "[run] FATAL: P35.3 lacks $report_key" >&2
          exit 1
        fi
        if [ -e "$report_path" ]; then
          echo "[run] FATAL: P35.3 evidence path already exists: $report_key=$report_path" >&2
          exit 1
        fi
        mkdir -p "$(dirname "$report_path")"
      done
      if [ "${CANON_P35_REPLAY_STAGE_PROBE:-0}" = "1" ]; then
        for report_key in CANON_P35_REPLAY_STAGE_REPORT \
                          CANON_P35_REPLAY_STAGE_CLASSIFICATION; do
          report_path="${!report_key:-}"
          if [ -z "$report_path" ]; then
            echo "[run] FATAL: P35.3c lacks $report_key" >&2
            exit 1
          fi
          if [ -e "$report_path" ]; then
            echo "[run] FATAL: P35.3c evidence path already exists: $report_key=$report_path" >&2
            exit 1
          fi
          mkdir -p "$(dirname "$report_path")"
        done
      fi
    fi
  fi
fi
if [ -n "${CANON_P38_SERVING_CAPTURE_DIR:-}" ]; then
  bash "$CANON_PKG/tasks/p38-pathways-decode-prefill-carrier/scripts/persist_p38_gcs.sh" \
    probe || {
      echo "[run] FATAL: P38 GCS write/read preflight failed" >&2
      exit 1
    }
  : "${CANON_P38_LIVE_SNAPSHOT_STOP_FILE:?}"
  : "${CANON_P38_LIVE_SNAPSHOT_WORKER_LOG:?}"
  : "${CANON_P38_LIVE_COLLECT_REQUEST_FILE:?}"
  : "${CANON_P38_LIVE_COLLECT_ACK_FILE:?}"
  : "${CANON_P38_LIVE_COMPLETE_REQUEST_FILE:?}"
  : "${CANON_P38_LIVE_COMPLETE_ACK_FILE:?}"
  : "${CANON_P38_DIAGNOSTIC_ROUND_FILE:?}"
  : "${CANON_P38_ROUND_SEAL_REQUEST_DIR:?}"
  : "${CANON_P38_ROUND_SEAL_ACK_DIR:?}"
  if [ -e "$CANON_P38_LIVE_SNAPSHOT_STOP_FILE" ] || \
     [ -e "$CANON_P38_LIVE_SNAPSHOT_WORKER_LOG" ] || \
     [ -e "$CANON_P38_LIVE_COLLECT_REQUEST_FILE" ] || \
     [ -e "$CANON_P38_LIVE_COLLECT_ACK_FILE" ] || \
     [ -e "$CANON_P38_LIVE_COMPLETE_REQUEST_FILE" ] || \
     [ -e "$CANON_P38_LIVE_COMPLETE_ACK_FILE" ] || \
     [ -e "$CANON_P38_DIAGNOSTIC_ROUND_FILE" ] || \
     [ -e "$CANON_P38_ROUND_SEAL_REQUEST_DIR" ] || \
     [ -e "$CANON_P38_ROUND_SEAL_ACK_DIR" ]; then
    echo "[run] FATAL: P38 live snapshot state already exists" >&2
    exit 1
  fi
  (umask 077; printf '0\n' > "$CANON_P38_DIAGNOSTIC_ROUND_FILE")
  mkdir -m 700 "$CANON_P38_ROUND_SEAL_REQUEST_DIR" \
    "$CANON_P38_ROUND_SEAL_ACK_DIR"
  bash "$CANON_PKG/tasks/p38-pathways-decode-prefill-carrier/scripts/p38_live_snapshot_worker.sh" \
    > "$CANON_P38_LIVE_SNAPSHOT_WORKER_LOG" 2>&1 &
  p38_live_pid=$!
  trap 'p38_stop_live_worker' EXIT
  echo "[P38.GCS] LIVE_WORKER_LAUNCHED pid=$p38_live_pid"
fi
LOG_BASE="$LOG"
mkdir -p "$(dirname "$LOG_BASE")"
if [ "${CANON_P46_EVALUATION:-0}" = "1" ] && \
   [ "${CANON_P46_FULL_CAMPAIGN:-0}" = "1" ]; then
  # A resume must retain the prior failure/timeout log, while postflight must
  # inspect only this launch (an appended old timeout would poison success).
  log_stem="$(basename "${LOG_BASE%.log}")"
  LOG="$(mktemp "$(dirname "$LOG_BASE")/${log_stem}.attempt-$(date -u +%Y%m%dT%H%M%SZ).XXXXXX.log")"
  echo "[P46.RESUME] ATTEMPT_LOG path=$LOG base=$LOG_BASE resume_tag=${CANON_P46_RESUME_TAG:-missing} launch_id=${CANON_RUN_ID:-missing}"
fi
echo "[run] cmd: $CANON_RUN_CMD"
echo "[run] log: $LOG"
cd "${CANON_RUN_CWD:-$CANON_PKG/..}"
set +e
set -o pipefail
run_tee_args=("$LOG")
if [ "${CANON_P62_BACKWARD_NUMERIC_DEBUG:-0}" = "1" ]; then
  if [ "${CANON_PROFILE_FILE:-}" != \
       "cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k-p62-debug.env" ]; then
    echo "[run] FATAL: P62 full-log seed requires its exact profile" >&2
    exit 1
  fi
  p62_profile_receipt="[P62.NUMERIC] profile_resolved workload=gsm8k dp=16 tp=4 stage=backward-no-commit optimizer_commits=0"
  printf '%s\n' "$p62_profile_receipt" > "$LOG"
  echo "$p62_profile_receipt"
  run_tee_args=(-a "$LOG")
fi
bash -c "$CANON_RUN_CMD" 2>&1 | tee "${run_tee_args[@]}"
pipeline_status=("${PIPESTATUS[@]}")
rc="${pipeline_status[0]}"
tee_rc="${pipeline_status[1]:-1}"
# Keep postflight explicitly non-errexit. P38 exit 42 and classifier failures
# are data that must be persisted before the final fail-closed verdict.
set +e
echo "[run] exit=$rc"
echo "[run] transport_rc=$tee_rc"
# Persist compiled executables before any fail-closed postflight branch can
# exit. The receipt is informational: cache transport never relaxes or replaces
# a numerical/alignment verdict.
jax_cache_saved_early=0
if [ "${CANON_V1_HP_FULL:-0}" = "1" ]; then
  canon_jax_cache_sync save
  jax_cache_saved_early=1
fi
xprof_restore_rc=0
xprof_local_dir=""
xprof_restore_receipt=""
if [ "${CANON_V1_HP_FULL:-0}" = "1" ]; then
  xprof_attempt="${JOBSET_RESTART_ATTEMPT:-direct}"
  if [[ ! "$xprof_attempt" =~ ^(direct|[0-9]+)$ ]]; then
    echo "[run] FATAL: V1 XProf attempt is not direct or a non-negative integer" >&2
    exit 1
  fi
  if [ "$xprof_attempt" = direct ]; then
    xprof_local_dir="${CANON_STATE%/}/xprof-update"
  else
    xprof_local_dir="${CANON_STATE%/}/attempt-$xprof_attempt/xprof-update"
  fi
  xprof_restore_receipt="$(dirname -- "$xprof_local_dir")/xprof_gcs_restore.receipt"
  canon_xprof_gcs_restore "$xprof_local_dir" "$xprof_restore_receipt" || \
    xprof_restore_rc=$?
  if [ "$rc" -eq 0 ] && [ "$xprof_restore_rc" -ne 0 ]; then
    echo "[run] FATAL: completed V1 workload lacks a durable GCS XProf restore: rc=$xprof_restore_rc" >&2
    exit "$xprof_restore_rc"
  fi
fi
if [ "${CANON_P46_EVALUATION:-0}" = "1" ]; then
  n_eval_subshard=$(grep -ac '^P46_EVAL_SUBSHARD_PASS ' "$LOG" || true)
  n_eval_report=$(grep -ac '^P46_EVAL_LOGICAL_REPORT_PASS ' "$LOG" || true)
  n_eval_campaign=$(grep -ac '^P46_EVAL_CAMPAIGN_PASS tasks=1851 n_sample=16 valid_trajectories=29616 logical_shards=58 ' "$LOG" || true)
  n_eval_campaign_logical=$(grep -ac '^P46_EVAL_CAMPAIGN_LOGICAL_PASS ' "$LOG" || true)
  n_eval_census=$(grep -ac '^P46_EVAL_CENSUS_PASS tasks=1851 scheduled_identities=29616 attempted_identities=29616 ' "$LOG" || true)
  n_eval_census_logical=$(grep -ac '^P46_EVAL_CENSUS_LOGICAL_COMPLETE ' "$LOG" || true)
  n_eval_census_incomplete=$(grep -ac '^P46_EVAL_CENSUS_INCOMPLETE ' "$LOG" || true)
  n_eval_timeout=$(grep -aEc 'P46_EVAL_(SHARD|CAMPAIGN_WAVE)_TIMEOUT' "$LOG" || true)
  echo "[P46.EVAL.POSTFLIGHT] rc=$rc transport_rc=$tee_rc subshard=$n_eval_subshard report=$n_eval_report campaign=$n_eval_campaign campaign_logical=$n_eval_campaign_logical census=$n_eval_census census_logical=$n_eval_census_logical census_incomplete=$n_eval_census_incomplete timeout=$n_eval_timeout log=$LOG"
  if [ "$rc" -ne 0 ]; then
    exit "$rc"
  fi
  if [ "$tee_rc" -ne 0 ]; then
    echo "[run] FATAL: P46 evaluation log transport failed: rc=$tee_rc" >&2
    exit 1
  fi
  if [ "${CANON_P46_FULL_CAMPAIGN:-0}" = "1" ]; then
    if [ "${CANON_P46_CENSUS_FIRST_PASS:-0}" = "1" ]; then
      if [ "$n_eval_census" -ne 1 ] || \
         [ "$n_eval_census_logical" -ne 58 ] || \
         [ "$n_eval_census_incomplete" -ne 0 ] || \
         [ "$n_eval_campaign" -ne 0 ] || \
         [ "$n_eval_campaign_logical" -ne 0 ] || \
         [ "$n_eval_subshard" -ne 0 ] || [ "$n_eval_report" -ne 0 ]; then
        echo "[run] FATAL: P46 census completion marker contract failed" >&2
        exit 1
      fi
    elif [ "$n_eval_campaign" -ne 1 ] || \
         [ "$n_eval_campaign_logical" -ne 58 ] || \
         [ "$n_eval_census" -ne 0 ] || \
         [ "$n_eval_census_logical" -ne 0 ] || \
         [ "$n_eval_census_incomplete" -ne 0 ] || \
         [ "$n_eval_subshard" -ne 0 ] || [ "$n_eval_report" -ne 0 ] || \
         [ "$n_eval_timeout" -ne 0 ]; then
      echo "[run] FATAL: P46 full-campaign completion marker contract failed" >&2
      exit 1
    fi
  elif [ "$((n_eval_subshard + n_eval_report))" -ne 1 ] || \
       [ "$n_eval_campaign" -ne 0 ] || [ "$n_eval_census" -ne 0 ] || \
       [ "$n_eval_census_logical" -ne 0 ] || \
       [ "$n_eval_census_incomplete" -ne 0 ] || \
       [ "$n_eval_timeout" -ne 0 ]; then
    echo "[run] FATAL: P46 evaluation completion marker contract failed" >&2
    exit 1
  fi
  if [ "${CANON_P46_CENSUS_FIRST_PASS:-0}" = "1" ]; then
    echo "[P46.EVAL.POSTFLIGHT] PASS mode=census"
  else
    echo "[P46.EVAL.POSTFLIGHT] PASS mode=strict"
  fi
  exit 0
fi
# A fail-closed numerical gate exits before the normal P33 classifier runs.  Preserve the
# complete pre-backward record in the pod log, which is the only artifact guaranteed to survive
# a deleted pod.  The report contains hashes and numerical diagnostics, never credentials.
if [ "$rc" -ne 0 ] && [ -s "${CANON_PRE_ALIGN_REPORT:-}" ]; then
  pre_align_sha="$(sha256sum "$CANON_PRE_ALIGN_REPORT" | awk '{print $1}')"
  pre_align_rows="$(wc -l < "$CANON_PRE_ALIGN_REPORT" | tr -d '[:space:]')"
  echo "[CANON_PRE_ALIGN_ARTIFACT] path=$CANON_PRE_ALIGN_REPORT rows=$pre_align_rows sha256=$pre_align_sha"
  sed 's/^/[CANON_PRE_ALIGN_ARTIFACT_JSON] /' "$CANON_PRE_ALIGN_REPORT"
fi
if [ "$rc" -ne 0 ] && [ -s "${CANON_P34_WEIGHT_REPORT:-}" ]; then
  weight_sha="$(sha256sum "$CANON_P34_WEIGHT_REPORT" | awk '{print $1}')"
  weight_rows="$(wc -l < "$CANON_P34_WEIGHT_REPORT" | tr -d '[:space:]')"
  echo "[P34.WEIGHT_ARTIFACT] path=$CANON_P34_WEIGHT_REPORT rows=$weight_rows sha256=$weight_sha"
  sed 's/^/[P34.WEIGHT_ARTIFACT_JSON] /' "$CANON_P34_WEIGHT_REPORT"
fi
if [ "$rc" -ne 0 ] && [ -s "${CANON_P38_MISMATCH_CAPSULE:-}" ]; then
  capsule_sha="$(sha256sum "$CANON_P38_MISMATCH_CAPSULE" | awk '{print $1}')"
  capsule_bytes="$(wc -c < "$CANON_P38_MISMATCH_CAPSULE" | tr -d '[:space:]')"
  p38_capsule_artifact_marker='CANON_''P38_CAPSULE_ARTIFACT'
  p38_capsule_b64_marker='CANON_''P38_CAPSULE_B64'
  if [ -n "${CANON_APC_M15_TARGET_DEBUG:-}" ]; then
    echo "[$p38_capsule_artifact_marker] path=$CANON_P38_MISMATCH_CAPSULE bytes=$capsule_bytes sha256=$capsule_sha encoding=gcs-only"
  else
    echo "[$p38_capsule_artifact_marker] path=$CANON_P38_MISMATCH_CAPSULE bytes=$capsule_bytes sha256=$capsule_sha encoding=base64"
    base64 "$CANON_P38_MISMATCH_CAPSULE" | sed "s/^/[$p38_capsule_b64_marker] /"
  fi
fi
if [ -n "${CANON_P38_SERVING_CAPTURE_DIR:-}" ]; then
  p38_join_args=()
  if [ -n "${CANON_APC_M15_TARGET_DEBUG:-}" ] && \
     [ -s "${CANON_P38_MISMATCH_CAPSULE:-}" ]; then
    p38_join_args+=(--require-mismatch-join)
  elif [ -z "${CANON_APC_M15_TARGET_DEBUG:-}" ] && \
       [ "${CANON_KV_UNIFIED:-0}" = "0" ] && \
       [ "${CANON_P38_FIXED_LM_HEAD:-0}" != "1" ]; then
    p38_join_args+=(--require-mismatch-join)
  fi
  JAX_PLATFORMS=cpu PYTHONPATH="$CANON_PKG/..:${PYTHONPATH:-}" \
    python3 "$CANON_PKG/tasks/p38-pathways-decode-prefill-carrier/scripts/classify_p38_serving_capture.py" \
      --directory "$CANON_P38_SERVING_CAPTURE_DIR" \
      --expected-records "$CANON_P38_SERVING_CAPTURE_EXPECTED_RECORDS" \
      --expected-program-path "$CANON_P38_SERVING_CAPTURE_EXPECTED_PATH" \
      --prefix-bounds "$CANON_P38_SERVING_CAPTURE_PREFIX_BOUNDS" \
      --incident-min-prefix "$CANON_P38_INCIDENT_MIN_PREFIX" \
      --incident-max-prefix "$CANON_P38_INCIDENT_MAX_PREFIX" \
      --mismatch-capsule "$CANON_P38_MISMATCH_CAPSULE" \
      "${p38_join_args[@]}" \
      --output "$CANON_P38_SERVING_CAPTURE_CLASSIFICATION"
  p38_capture_rc=$?
  if [ -s "$CANON_P38_SERVING_CAPTURE_CLASSIFICATION" ]; then
    p38_class_sha="$(sha256sum "$CANON_P38_SERVING_CAPTURE_CLASSIFICATION" | awk '{print $1}')"
    echo "[CANON_P38_SERVING_CLASSIFICATION] path=$CANON_P38_SERVING_CAPTURE_CLASSIFICATION sha256=$p38_class_sha"
    sed 's/^/[CANON_P38_SERVING_CLASSIFICATION_JSON] /' \
      "$CANON_P38_SERVING_CAPTURE_CLASSIFICATION"
  fi
  if [ -n "${CANON_APC_M15_TARGET_DEBUG:-}" ]; then
    m15_apc_classification="$CANON_P38_SERVING_CAPTURE_DIR/m15_apc_target.classification.json"
    m15_replay_bundle_dir="$CANON_P38_SERVING_CAPTURE_DIR/m15_first_red_replay"
    m15_full_replay_dir="$CANON_P38_SERVING_CAPTURE_DIR/m15_full_replay_carrier"
    m15_replay_bundle_rc=0
    m15_full_replay_rc=0
    m15_apc_capsule_args=()
    if [ -s "${CANON_P38_MISMATCH_CAPSULE:-}" ]; then
      m15_apc_capsule_args+=(--mismatch-capsule "$CANON_P38_MISMATCH_CAPSULE")
    fi
    JAX_PLATFORMS=cpu PYTHONPATH="$CANON_PKG/..:${PYTHONPATH:-}" \
      python3 "$CANON_PKG/tasks/v1-apc-m15-target-debug/scripts/classify_m15_apc_target_run.py" \
        --raw "$LOG" \
        --report "$CANON_PRE_ALIGN_REPORT" \
        --capture-classification "$CANON_P38_SERVING_CAPTURE_CLASSIFICATION" \
        --arm "$CANON_APC_M15_TARGET_DEBUG" \
        --expected-source-commit "$CANON_EXPECT_COMMIT" \
        "${m15_apc_capsule_args[@]}" \
        --output "$m15_apc_classification"
    m15_apc_rc=$?
    if [ -s "$m15_apc_classification" ]; then
      m15_apc_sha="$(sha256sum "$m15_apc_classification" | awk '{print $1}')"
      echo "[CAN""ON_APC_M15_CLASSIFICATION] path=$m15_apc_classification sha256=$m15_apc_sha"
      sed 's/^/[CAN''ON_APC_M15_CLASSIFICATION_JSON] /' "$m15_apc_classification"
    fi
    if [ "${m15_apc_rc:-1}" -ne 0 ] && [ "$rc" -eq 0 ]; then
      rc=2
    fi
    if [ "${m15_apc_rc:-1}" -eq 0 ] && \
       [ -s "${CANON_P38_MISMATCH_CAPSULE:-}" ]; then
      JAX_PLATFORMS=cpu PYTHONPATH="$CANON_PKG/..:${PYTHONPATH:-}" \
        python3 "$CANON_PKG/tasks/v1-apc-m15-target-debug/scripts/package_first_red_replay.py" \
          --capsule "$CANON_P38_MISMATCH_CAPSULE" \
          --capture-classification "$CANON_P38_SERVING_CAPTURE_CLASSIFICATION" \
          --m15-classification "$m15_apc_classification" \
          --output-dir "$m15_replay_bundle_dir" || \
        m15_replay_bundle_rc=$?
      if [ -s "$m15_replay_bundle_dir/first_red_contract.json" ]; then
        m15_replay_bundle_sha="$(sha256sum "$m15_replay_bundle_dir/first_red_contract.json" | awk '{print $1}')"
        echo "[CAN""ON_APC_M15_REPLAY_BUNDLE] path=$m15_replay_bundle_dir sha256=$m15_replay_bundle_sha"
      fi
      if [ "${m15_replay_bundle_rc:-1}" -eq 0 ]; then
        JAX_PLATFORMS=cpu PYTHONPATH="$CANON_PKG/..:${PYTHONPATH:-}" \
          python3 "$CANON_PKG/tasks/v1-apc-m15-target-debug/scripts/package_full_replay_carrier.py" \
            --producer-unit "$CANON_P38_SERVING_CAPTURE_DIR/m15_producer_unit.npz" \
            --serving-envelope "$CANON_APC_M15_REPLAY_LEDGER" \
            --first-red-dir "$m15_replay_bundle_dir" \
            --capture-classification "$CANON_P38_SERVING_CAPTURE_CLASSIFICATION" \
            --m15-classification "$m15_apc_classification" \
            --output-dir "$m15_full_replay_dir" || \
          m15_full_replay_rc=$?
        if [ -s "$m15_full_replay_dir/replay_contract.json" ]; then
          m15_full_replay_sha="$(sha256sum "$m15_full_replay_dir/replay_contract.json" | awk '{print $1}')"
          echo "[CAN""ON_APC_M15_FULL_REPLAY_CARRIER] path=$m15_full_replay_dir sha256=$m15_full_replay_sha"
        fi
      fi
    fi
  fi
  p38_kv_observer_rc=0
  if [ -n "${CANON_P38_KV_OBSERVER_DIR:-}" ]; then
    shopt -s nullglob
    p38_kv_round_capsules=(
      "${CANON_P38_MISMATCH_CAPSULE%.npz}".round-*.npz
    )
    shopt -u nullglob
    if [ "${#p38_kv_round_capsules[@]}" -gt 0 ]; then
      # The stable capsule aliases the latest immutable round.  Passing both
      # would classify the same diagnostic round twice and must not be hidden
      # as two independent inputs.
      p38_kv_capsules=("${p38_kv_round_capsules[@]}")
      p38_kv_capsule_source=immutable-rounds
    else
      p38_kv_capsules=("$CANON_P38_MISMATCH_CAPSULE")
      p38_kv_capsule_source=stable-fallback
    fi
    p38_kv_args=()
    for p38_kv_capsule in "${p38_kv_capsules[@]}"; do
      [ -s "$p38_kv_capsule" ] || continue
      p38_kv_args+=(--capsule "$p38_kv_capsule")
    done
    echo "[CANON_P38_KV_OBSERVER_INPUTS] source=$p38_kv_capsule_source capsules=$((${#p38_kv_args[@]} / 2))"
    JAX_PLATFORMS=cpu PYTHONPATH="$CANON_PKG/..:${PYTHONPATH:-}" \
      python3 "$CANON_PKG/tasks/p38-pathways-decode-prefill-carrier/scripts/classify_p38_kv_observer.py" \
        --directory "$CANON_P38_KV_OBSERVER_DIR" \
        "${p38_kv_args[@]}" \
        --require-red-join \
        --output "$CANON_P38_KV_OBSERVER_CLASSIFICATION" || \
      p38_kv_observer_rc=$?
    if [ -s "$CANON_P38_KV_OBSERVER_CLASSIFICATION" ]; then
      p38_kv_sha="$(sha256sum "$CANON_P38_KV_OBSERVER_CLASSIFICATION" | awk '{print $1}')"
      echo "[CANON_P38_KV_OBSERVER_CLASSIFICATION] path=$CANON_P38_KV_OBSERVER_CLASSIFICATION sha256=$p38_kv_sha"
      sed 's/^/[CANON_P38_KV_OBSERVER_CLASSIFICATION_JSON] /' \
        "$CANON_P38_KV_OBSERVER_CLASSIFICATION"
    fi
  fi
  p38_seam_rc=0
  if [ -n "${CANON_P38_SEAM_OBSERVER:-}" ]; then
    shopt -s nullglob
    p38_seam_round_capsules=(
      "${CANON_P38_MISMATCH_CAPSULE%.npz}".round-*.npz
    )
    shopt -u nullglob
    if [ "${#p38_seam_round_capsules[@]}" -gt 0 ]; then
      p38_seam_capsules=("${p38_seam_round_capsules[@]}")
      p38_seam_capsule_source=immutable-rounds
    else
      p38_seam_capsules=("$CANON_P38_MISMATCH_CAPSULE")
      p38_seam_capsule_source=stable-fallback
    fi
    p38_seam_args=()
    for p38_seam_capsule in "${p38_seam_capsules[@]}"; do
      [ -s "$p38_seam_capsule" ] || continue
      p38_seam_args+=(--capsule "$p38_seam_capsule")
    done
    p38_seam_tail_args=()
    if [ "${CANON_P38_TAIL_OBSERVER:-0}" = "1" ]; then
      p38_seam_tail_args+=(--require-tail)
    fi
    echo "[CANON_P38_SEAM_INPUTS] source=$p38_seam_capsule_source capsules=$((${#p38_seam_args[@]} / 2)) mode=$CANON_P38_SEAM_OBSERVER"
    JAX_PLATFORMS=cpu PYTHONPATH="$CANON_PKG/..:${PYTHONPATH:-}" \
      python3 "$CANON_PKG/tasks/p38-pathways-decode-prefill-carrier/scripts/classify_p38_seam.py" \
        --directory "$CANON_P38_SEAM_OBSERVER_DIR" \
        "${p38_seam_args[@]}" \
        --mode "$CANON_P38_SEAM_OBSERVER" \
        "${p38_seam_tail_args[@]}" \
        --output "$CANON_P38_SEAM_CLASSIFICATION" || \
      p38_seam_rc=$?
    if [ -s "$CANON_P38_SEAM_CLASSIFICATION" ]; then
      p38_seam_sha="$(sha256sum "$CANON_P38_SEAM_CLASSIFICATION" | awk '{print $1}')"
      echo "[CANON_P38_SEAM_CLASSIFICATION] path=$CANON_P38_SEAM_CLASSIFICATION sha256=$p38_seam_sha"
      sed 's/^/[CANON_P38_SEAM_CLASSIFICATION_JSON] /' \
        "$CANON_P38_SEAM_CLASSIFICATION"
    fi
  fi
  p38_terminal_rc=0
  if [ "${CANON_P38_TERMINAL_DISCRIMINATOR:-0}" = "1" ]; then
    JAX_PLATFORMS=cpu PYTHONPATH="$CANON_PKG/..:${PYTHONPATH:-}" \
      python3 "$CANON_PKG/tasks/p38-pathways-decode-prefill-carrier/scripts/classify_p38_terminal_discriminator.py" \
        --input "$CANON_P38_SEAM_OBSERVER_DIR" \
        "${p38_seam_args[@]}" \
        --require-red-join \
        --output "$CANON_P38_TERMINAL_CLASSIFICATION" || \
      p38_terminal_rc=$?
    if [ -s "$CANON_P38_TERMINAL_CLASSIFICATION" ]; then
      p38_terminal_sha="$(sha256sum "$CANON_P38_TERMINAL_CLASSIFICATION" | awk '{print $1}')"
      echo "[CANON_P38_TERMINAL_CLASSIFICATION] path=$CANON_P38_TERMINAL_CLASSIFICATION sha256=$p38_terminal_sha"
      sed 's/^/[CANON_P38_TERMINAL_CLASSIFICATION_JSON] /' \
        "$CANON_P38_TERMINAL_CLASSIFICATION"
    fi
  fi
  if [ -d "$CANON_P38_SERVING_CAPTURE_DIR" ]; then
    tar --sort=name --mtime=@0 --owner=0 --group=0 \
      -C "$CANON_P38_SERVING_CAPTURE_DIR" \
      -cf "$CANON_P38_SERVING_CAPTURE_ARCHIVE" .
    p38_archive_sha="$(sha256sum "$CANON_P38_SERVING_CAPTURE_ARCHIVE" | awk '{print $1}')"
    p38_archive_bytes="$(wc -c < "$CANON_P38_SERVING_CAPTURE_ARCHIVE" | tr -d '[:space:]')"
    p38_persist_rc=0
    p38_request_live_action collect || p38_persist_rc=$?
    p38_serving_archive_marker='CANON_''P38_SERVING_ARCHIVE'
    p38_serving_archive_b64_marker='CANON_''P38_SERVING_ARCHIVE_B64'
    if [ -n "${CANON_APC_M15_TARGET_DEBUG:-}" ]; then
      echo "[$p38_serving_archive_marker] path=$CANON_P38_SERVING_CAPTURE_ARCHIVE bytes=$p38_archive_bytes sha256=$p38_archive_sha encoding=gcs-only"
    else
      echo "[$p38_serving_archive_marker] path=$CANON_P38_SERVING_CAPTURE_ARCHIVE bytes=$p38_archive_bytes sha256=$p38_archive_sha encoding=base64"
      base64 "$CANON_P38_SERVING_CAPTURE_ARCHIVE" | \
        sed "s/^/[$p38_serving_archive_b64_marker] /"
    fi
  fi
  if [ "$p38_capture_rc" -ne 0 ] && [ "$rc" -eq 0 ]; then
    rc=2
  fi
  if [ "${p38_kv_observer_rc:-0}" -ne 0 ] && [ "$rc" -eq 0 ]; then
    rc=2
  fi
  if [ "${p38_seam_rc:-0}" -ne 0 ] && [ "$rc" -eq 0 ]; then
    rc=2
  fi
  if [ "${p38_terminal_rc:-0}" -ne 0 ] && [ "$rc" -eq 0 ]; then
    rc=2
  fi
fi
# grep -a: progress-bar control characters make grep treat the log as binary and drop every
# match silently, which reads exactly like "the intervention never fired".
n_ar=$(grep -ac 'CANON_FIXED_AR=1 fixed-order tree' "$LOG" || true)
n_emb=$(grep -ac 'CANON_FIXED_AR_EMBED=1 fixed-order embed gather' "$LOG" || true)
n_lp=$(grep -ac 'CANON_LOGPROB_M on' "$LOG" || true)
n_wandb=$(grep -ac '\[CANON_P33_WANDB\] ONLINE_RUN_PASS' "$LOG" || true)
n_wandb_p34=$(grep -ac '\[CANON_P34_WANDB\] ONLINE_RUN_PASS' "$LOG" || true)
n_eval_off=$(grep -ac '\[CANON_P33_EVAL\] DISABLED workload=frozenlake' "$LOG" || true)
n_eval_on=$(grep -aEc '\[CANON_''P33_EVAL\] ENABLED workload=frozenlake cadence=[0-9]+ held_out_rows=100 generations=8' "$LOG" || true)
n_p35_stop=$(grep -ac '^\[CANON_P35\] REPORT_COMPLETE .*STOP_BEFORE_BACKWARD' "$LOG" || true)
n_p35_base=$(grep -ac '^\[CANON_P35\] BASE_REPORT_COMPLETE .*REPLAY_PENDING' "$LOG" || true)
n_p35_replay=$(grep -ac '^\[CANON_P35.3\] REPLAY_COMPLETE' "$LOG" || true)
n_p35_stage_begin=$(grep -ac '^\[CANON_P35.3C\] STAGE_BEGIN' "$LOG" || true)
n_p35_stage_ready=$(grep -ac '^\[CANON_P35.3C\] STAGE_READY' "$LOG" || true)
n_p35_stage_complete=$(grep -ac '^\[CANON_P35.3C\] STAGE_PROBE_COMPLETE .*NO_NUMERICAL_VERDICT' "$LOG" || true)
n_p38_precheck=$(grep -ac '^\[CANON_P38\] PRECHECK_COMPLETE STOP_BEFORE_BACKWARD' "$LOG" || true)
n_p38_rounds=$(grep -ac '^\[CANON_P38\] PRECHECK_ROUND_COMPLETE ' "$LOG" || true)
n_p38_controlled_exit=$(grep -ac '^\[CANON_P38\] CONTROLLED_EXIT code=42 backward=0 optimizer_commits=0' "$LOG" || true)
n_p38_fixed_primal=$(grep -ac '^\[PATHTRACE\] CANON_P38_FIXED_LM_HEAD=1 ' "$LOG" || true)
n_p38_fixed_vjp=$(grep -ac '^\[PATHTRACE\] CANON_P38_FIXED_LM_HEAD_VJP=1 semantic_M=4096 fixed_M=256 chunks=16 accumulation=lax.scan order=ascending' "$LOG" || true)
n_p38_kv_unified=$(grep -ac 'KV_UNIFIED_two_pass' "$LOG" || true)
n_p38_capture_init=$(grep -ac '^\[CANON_P38_SERVING_CAPTURE_INIT\]' "$LOG" || true)
n_p38_capture_observe=$(grep -ac '^\[CANON_P38_SERVING_CAPTURE_OBSERVE\]' "$LOG" || true)
n_p38_capture_error=$(grep -ac '^\[CANON_P38_SERVING_CAPTURE_ERROR\]' "$LOG" || true)
n_p38_request_journal=$(grep -ac '^\[CANON_P38_REQUEST_JOURNAL\]' "$LOG" || true)
n_p38_incident_ledger=$(grep -ac '^\[CANON_P38_INCIDENT_LEDGER\]' "$LOG" || true)
n_m15_replay_ledger=$(grep -ac '^\[CAN''ON_APC_M15_REPLAY_LEDGER\]' "$LOG" || true)
n_m15_producer_carrier=$(grep -ac '^\[CAN''ON_APC_M15_PRODUCER_CARRIER\]' "$LOG" || true)
n_p38_kv_observer_init=$(grep -ac '^\[CANON_P38_KV_OBSERVER_INIT\]' "$LOG" || true)
n_p38_kv_observer_candidate=$(grep -ac '^\[CANON_P38_KV_OBSERVER_CANDIDATE\]' "$LOG" || true)
n_p38_kv_observer_a=$(grep -ac '^\[CANON_P38_KV_OBSERVER_RECORD\] arm=A ' "$LOG" || true)
n_p38_kv_observer_b=$(grep -ac '^\[CANON_P38_KV_OBSERVER_RECORD\] arm=B ' "$LOG" || true)
n_p38_seam_init=$(grep -ac '^\[CANON_P38_SEAM_OBSERVER_INIT\] ' "$LOG" || true)
n_p38_seam_records=$(grep -ac '^\[CANON_P38_SEAM_OBSERVER_RECORD\] ' "$LOG" || true)
n_p38_tail_init=$(grep -ac '^\[CANON_P38_TAIL_OBSERVER_INIT\] ' "$LOG" || true)
n_p38_tail_a=$(grep -ac '^\[CANON_P38_TAIL_OBSERVER_RECORD\] .* arm=A ' "$LOG" || true)
n_p38_tail_b=$(grep -ac '^\[CANON_P38_TAIL_OBSERVER_RECORD\] .* arm=B ' "$LOG" || true)
n_p38_terminal_init=$(grep -ac '^\[CANON_P38_TERMINAL_DISCRIMINATOR_INIT\] ' "$LOG" || true)
n_p38_terminal_a=$(grep -ac '^\[CANON_P38_TERMINAL_DISCRIMINATOR_RECORD\] .* arm=A ' "$LOG" || true)
n_p38_terminal_b=$(grep -ac '^\[CANON_P38_TERMINAL_DISCRIMINATOR_RECORD\] .* arm=B ' "$LOG" || true)
if [ -n "${CANON_APC_M15_TARGET_DEBUG:-}" ]; then
  n_p38_coverage=$(grep -ac '^\[CANON_P38\] DIAGNOSTIC_COVERAGE_CONTRACT .*prompt_groups=32 .*unit_prompts=32 .*units=1 .*trajectories=256 .*partial_tail=reject verdict=PASS' "$LOG" || true)
else
  n_p38_coverage=$(grep -ac '^\[CANON_P38\] DIAGNOSTIC_COVERAGE_CONTRACT .*prompt_groups=32 .*unit_prompts=4 .*units=8 .*trajectories=256 .*partial_tail=reject verdict=PASS' "$LOG" || true)
fi
n_p57_stock_sync=$(grep -aEc '^\[P57.STOCK_FAST\] ROLLOUT_SYNC_PASS step=[0-9]+ transport=update_params exact_weight_attestation=unavailable-by-design$' "$LOG" || true)
n_p57_stock_train_runtime=$(grep -aEc '^\[P57.STOCK\] TRAIN_RUNTIME_PASS regime=stock-fast arm=(mismatch|is) canonical_bundle=off observer=warning-only processed_b=observer-only$' "$LOG" || true)
n_p57_stock_observer=$(grep -ac '^\[P57.STOCK_OBSERVER\] PROCESSED_PROMPT_LOGPROBS_PASS .*targets=absolute-request-history treatment=observer-only$' "$LOG" || true)
n_p58_stock_observer=$(grep -ac '^\[P58.STOCK_OBSERVER\] PROCESSED_PROMPT_LOGPROBS_PASS .*targets=absolute-request-history treatment=observer-only$' "$LOG" || true)
n_p58_seed=$(grep -ac '^\[P58.SEED\] PASS dataset_seed=42 rollout_seed=42 scope=config-level async_completion_order=not-claimed$' "$LOG" || true)
n_p58_recipe_raw=$(grep -ac '^\[P58.TIM_RECIPE\] PASS recipe=native-raw sampler_is=none old_logps=rollout tis_weights=absent threshold=inactive group_filter=none$' "$LOG" || true)
n_p58_recipe_is=$(grep -ac '^\[P58.TIM_RECIPE\] PASS recipe=native-is sampler_is=token old_logps=trainer tis_weights=present threshold=2.0 group_filter=none$' "$LOG" || true)
n_p57_stock_segment_preflight=$(grep -ac '^\[P57.STOCK\] SEGMENT_PREFLIGHT ' "$LOG" || true)
n_p57_stock_segment_complete=$(grep -ac '^\[P57.STOCK\] SEGMENT_COMPLETE ' "$LOG" || true)
n_p57_tim_purity_none=$(grep -ac '^\[P57.TIM_PURITY\] PASS sampler_is=none old_logps=rollout tis_weights=absent trainer_rescore=observer-only$' "$LOG" || true)
n_p57_tim_purity_is=$(grep -ac '^\[P57.TIM_PURITY\] PASS sampler_is=token old_logps=trainer tis_weights=present trainer_rescore=training-input$' "$LOG" || true)
n_p38_standard_init=$(grep -ac '^\[CANON_P38_SERVING_CAPTURE_INIT\].*expected_path=standard' "$LOG" || true)
n_p38_standard_observe=$(grep -aEc '^\[CANON_P38_SERVING_CAPTURE_OBSERVE\].*"program_path"[[:space:]]*:[[:space:]]*"standard"' "$LOG" || true)
echo "[run] PATHTRACE fixed_ar=$n_ar embed=$n_emb logprob_m=$n_lp wandb_online=$n_wandb p34_wandb_online=$n_wandb_p34 eval_off=$n_eval_off eval_on=$n_eval_on p35_base=$n_p35_base p35_stop=$n_p35_stop p35_replay=$n_p35_replay p35_stage_begin=$n_p35_stage_begin p35_stage_ready=$n_p35_stage_ready p35_stage_complete=$n_p35_stage_complete p38_precheck=$n_p38_precheck p38_rounds=$n_p38_rounds p38_controlled_exit=$n_p38_controlled_exit p38_fixed_primal=$n_p38_fixed_primal p38_fixed_vjp=$n_p38_fixed_vjp p38_kv_unified=$n_p38_kv_unified p38_capture_init=$n_p38_capture_init p38_capture_observe=$n_p38_capture_observe p38_capture_error=$n_p38_capture_error p38_request_journal=$n_p38_request_journal p38_incident_ledger=$n_p38_incident_ledger p38_kv_observer_init=$n_p38_kv_observer_init p38_kv_observer_candidate=$n_p38_kv_observer_candidate p38_kv_observer_a=$n_p38_kv_observer_a p38_kv_observer_b=$n_p38_kv_observer_b p38_seam_init=$n_p38_seam_init p38_seam_records=$n_p38_seam_records p38_tail_init=$n_p38_tail_init p38_tail_a=$n_p38_tail_a p38_tail_b=$n_p38_tail_b p38_terminal_init=$n_p38_terminal_init p38_terminal_a=$n_p38_terminal_a p38_terminal_b=$n_p38_terminal_b p38_coverage=$n_p38_coverage p57_stock_sync=$n_p57_stock_sync p57_stock_train_runtime=$n_p57_stock_train_runtime p57_stock_observer=$n_p57_stock_observer p57_tim_purity_none=$n_p57_tim_purity_none p57_tim_purity_is=$n_p57_tim_purity_is p58_stock_observer=$n_p58_stock_observer p58_seed=$n_p58_seed p58_recipe_raw=$n_p58_recipe_raw p58_recipe_is=$n_p58_recipe_is p57_stock_segment_preflight=$n_p57_stock_segment_preflight p57_stock_segment_complete=$n_p57_stock_segment_complete"

if [ "${CANON_P57_RUN_KIND:-}" = "train" ]; then
  case "${CANON_P57_TIM_ARM:-}" in
    mismatch|zero)
      if [ "$n_p57_tim_purity_none" -ne 1 ] || \
         [ "$n_p57_tim_purity_is" -ne 0 ]; then
        echo "[run] FATAL: P57 no-IS purity marker contract failed: none=$n_p57_tim_purity_none/1 is=$n_p57_tim_purity_is/0" >&2
        exit 1
      fi
      ;;
    is)
      if [ "$n_p57_tim_purity_none" -ne 0 ] || \
         [ "$n_p57_tim_purity_is" -ne 1 ]; then
        echo "[run] FATAL: P57 IS purity marker contract failed: none=$n_p57_tim_purity_none/0 is=$n_p57_tim_purity_is/1" >&2
        exit 1
      fi
      ;;
  esac
fi
if [ -n "${CANON_P38_SERVING_CAPTURE_DIR:-}" ]; then
  if [ "$n_p38_capture_init" -ne 1 ] || [ "$n_p38_capture_observe" -le 0 ]; then
    echo "[run] FATAL: P38 serving capture hook was not observed: init=$n_p38_capture_init observe=$n_p38_capture_observe" >&2
    exit 1
  fi
  if [ "$n_p38_standard_init" -ne 1 ] || [ "$n_p38_standard_observe" -le 0 ]; then
    echo "[run] FATAL: P38 capture did not execute the standard runner path: init=$n_p38_standard_init observe=$n_p38_standard_observe" >&2
    exit 1
  fi
  if [ "$n_p38_capture_error" -ne 0 ]; then
    echo "[run] FATAL: P38 serving capture reported internal errors: $n_p38_capture_error" >&2
    exit 1
  fi
  if [ -n "${CANON_APC_M15_TARGET_DEBUG:-}" ] && \
     { [ "${m15_apc_rc:-1}" -ne 0 ] || \
       [ ! -s "${m15_apc_classification:-}" ]; }; then
    echo "[run] FATAL: M15 APC target classification failed: rc=${m15_apc_rc:-unset} artifact=${m15_apc_classification:-unset}" >&2
    exit 1
  fi
  if [ -n "${CANON_APC_M15_TARGET_DEBUG:-}" ] && \
     [ -s "${CANON_P38_MISMATCH_CAPSULE:-}" ] && \
     { [ "${m15_replay_bundle_rc:-1}" -ne 0 ] || \
       [ ! -s "${m15_replay_bundle_dir:-}/first_red_capsule.npz" ] || \
       [ ! -s "${m15_replay_bundle_dir:-}/first_red_contract.json" ] || \
       [ ! -s "${m15_replay_bundle_dir:-}/SHA256SUMS" ]; }; then
    echo "[run] FATAL: M15 first-red replay bundle failed: rc=${m15_replay_bundle_rc:-unset} directory=${m15_replay_bundle_dir:-unset}" >&2
    exit 1
  fi
  if [ -n "${CANON_APC_M15_TARGET_DEBUG:-}" ] && \
     { [ "$n_m15_replay_ledger" -le 0 ] || \
       [ "$n_m15_producer_carrier" -ne 1 ] || \
       [ ! -s "${CANON_APC_M15_REPLAY_LEDGER:-}" ] || \
       [ ! -s "$CANON_P38_SERVING_CAPTURE_DIR/m15_producer_unit.npz" ]; }; then
    echo "[run] FATAL: M15 full replay inputs are incomplete: ledger_markers=$n_m15_replay_ledger producer_markers=$n_m15_producer_carrier ledger=${CANON_APC_M15_REPLAY_LEDGER:-unset}" >&2
    exit 1
  fi
  if [ -n "${CANON_APC_M15_TARGET_DEBUG:-}" ] && \
     [ -s "${CANON_P38_MISMATCH_CAPSULE:-}" ] && \
     { [ "${m15_full_replay_rc:-1}" -ne 0 ] || \
       [ ! -s "${m15_full_replay_dir:-}/replay_contract.json" ] || \
       [ ! -s "${m15_full_replay_dir:-}/request_row_joins.jsonl" ] || \
       [ ! -s "${m15_full_replay_dir:-}/SHA256SUMS" ]; }; then
    echo "[run] FATAL: M15 full replay carrier failed: rc=${m15_full_replay_rc:-unset} directory=${m15_full_replay_dir:-unset}" >&2
    exit 1
  fi
  if [ -n "${CANON_P38_KV_OBSERVER_DIR:-}" ] && \
     { [ "$n_p38_kv_observer_init" -ne 1 ] || \
       [ "$n_p38_kv_observer_candidate" -ne 3 ] || \
       [ "$n_p38_kv_observer_a" -ne 3 ] || \
       [ "$n_p38_kv_observer_b" -ne 3 ] || \
       [ "${p38_kv_observer_rc:-1}" -ne 0 ] || \
       [ ! -s "${CANON_P38_KV_OBSERVER_CLASSIFICATION:-}" ]; }; then
    echo "[run] FATAL: P38 KV observer contract failed: init=$n_p38_kv_observer_init candidates=$n_p38_kv_observer_candidate A=$n_p38_kv_observer_a B=$n_p38_kv_observer_b classifier=${p38_kv_observer_rc:-unset}" >&2
    exit 1
  fi
  if [ -n "${CANON_P38_SEAM_OBSERVER:-}" ] && \
     { [ "$n_p38_seam_init" -ne 1 ] || \
       [ "$n_p38_seam_records" -le 0 ] || \
       [ "${p38_seam_rc:-1}" -ne 0 ] || \
       [ ! -s "${CANON_P38_SEAM_CLASSIFICATION:-}" ]; }; then
    echo "[run] FATAL: P38 seam observer contract failed: init=$n_p38_seam_init records=$n_p38_seam_records classifier=${p38_seam_rc:-unset}" >&2
    exit 1
  fi
  if [ "${CANON_P38_TAIL_OBSERVER:-0}" = "1" ] && \
     { [ "$n_p38_tail_init" -ne 1 ] || [ "$n_p38_tail_a" -le 0 ] || \
       [ "$n_p38_tail_b" -le 0 ]; }; then
    echo "[run] FATAL: P38 terminal-tail observer contract failed: init=$n_p38_tail_init A=$n_p38_tail_a B=$n_p38_tail_b" >&2
    exit 1
  fi
  if [ "${CANON_P38_TERMINAL_DISCRIMINATOR:-0}" = "1" ] && \
     { [ "$n_p38_terminal_init" -ne 1 ] || \
       [ "$n_p38_terminal_a" -le 0 ] || [ "$n_p38_terminal_b" -le 0 ] || \
       [ "${p38_terminal_rc:-1}" -ne 0 ] || \
       [ ! -s "${CANON_P38_TERMINAL_CLASSIFICATION:-}" ]; }; then
    echo "[run] FATAL: P38 terminal discriminator contract failed: init=$n_p38_terminal_init A=$n_p38_terminal_a B=$n_p38_terminal_b classifier=${p38_terminal_rc:-unset}" >&2
    exit 1
  fi
  if [ "${p38_persist_rc:-1}" -ne 0 ]; then
    echo "[run] FATAL: P38 GCS evidence collection failed: rc=${p38_persist_rc:-unset}" >&2
    exit 1
  fi
  if ! kill -0 "${p38_live_pid:?}" 2>/dev/null; then
    echo "[run] FATAL: P38 live snapshot worker is not alive after collection" >&2
    exit 1
  fi
  if [ "${tee_rc:-1}" -ne 0 ]; then
    echo "[run] FATAL: P38 workload log transport failed: rc=${tee_rc:-unset}" >&2
    exit 1
  fi
  if [ "$n_p38_request_journal" -le 0 ] || \
     [ ! -s "${CANON_P38_REQUEST_JOURNAL:-}" ]; then
    echo "[run] FATAL: P38 request journal is absent: markers=$n_p38_request_journal" >&2
    exit 1
  fi
  if [ "$n_p38_incident_ledger" -le 0 ] || \
     [ ! -s "${CANON_P38_INCIDENT_LEDGER:-}" ]; then
    echo "[run] FATAL: P38 incident ledger is absent: markers=$n_p38_incident_ledger" >&2
    exit 1
  fi
  if [ "$n_p38_coverage" -ne 1 ]; then
    echo "[run] FATAL: P38 diagnostic did not attest full 32-prompt coverage: $n_p38_coverage" >&2
    exit 1
  fi
  if [ "$n_p38_precheck" -gt 0 ] && \
     [ "$n_p38_rounds" -ne "${CANON_P38_DIAGNOSTIC_ROUNDS:-1}" ]; then
    echo "[run] FATAL: P38 frozen-weight round contract failed: observed=$n_p38_rounds expected=${CANON_P38_DIAGNOSTIC_ROUNDS:-1}" >&2
    exit 1
  fi
  expected_p38_round=$((CANON_P38_DIAGNOSTIC_ROUNDS - 1))
  actual_p38_round="$(tr -d '[:space:]' < "$CANON_P38_DIAGNOSTIC_ROUND_FILE")"
  if [ "$n_p38_precheck" -gt 0 ] && \
     [ "$actual_p38_round" != "$expected_p38_round" ]; then
    echo "[run] FATAL: P38 diagnostic round publication drifted: observed=$actual_p38_round expected=$expected_p38_round" >&2
    exit 1
  fi
  if [ "$n_p38_precheck" -gt 0 ] && \
     [ "${CANON_P38_CONTROLLED_EXIT:-0}" = "1" ] && \
     [ "$n_p38_controlled_exit" -ne 1 ]; then
    echo "[run] FATAL: P38 controlled-exit marker contract failed: $n_p38_controlled_exit" >&2
    exit 1
  fi
  if [ "${CANON_KV_UNIFIED:-0}" = "1" ] && [ "$n_p38_kv_unified" -le 0 ]; then
    echo "[run] FATAL: P38 U arm did not execute KV_UNIFIED_two_pass" >&2
    exit 1
  fi
  if [ "${CANON_KV_UNIFIED:-0}" = "0" ] && [ "$n_p38_kv_unified" -ne 0 ]; then
    echo "[run] FATAL: P38 stock arm executed KV_UNIFIED_two_pass" >&2
    exit 1
  fi
fi
if [ "${CANON_P38_FIXED_LM_HEAD:-0}" = "1" ]; then
  case "${CANON_PROFILE_FILE:-}" in
    cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k.env|\
    cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k-v1-hp.env|\
    cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k-p62-debug.env)
      p38_fixed_endpoint=tied_embed
      p38_fixed_hidden=2048
      p38_fixed_tp=4
      ;;
    cluster/profiles/qwen3-8b.env)
      p38_fixed_endpoint=untied_lm_head
      p38_fixed_hidden=4096
      p38_fixed_tp=4
      ;;
    cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-tim.env|\
    cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-v1-hp.env|\
    cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-apc-debug.env)
      p38_fixed_endpoint=untied_lm_head
      p38_fixed_hidden=4096
      p38_fixed_tp=8
      ;;
    cluster/profiles/qwen3-4b-dp-parity-deepswe-debug.env)
      p38_fixed_endpoint=tied_embed
      p38_fixed_hidden=2560
      p38_fixed_tp=8
      ;;
    cluster/profiles/qwen3-4b-dp8-tp8-deepswe-v1-hp.env)
      p38_fixed_endpoint=tied_embed
      p38_fixed_hidden=2560
      p38_fixed_tp=8
      ;;
    cluster/profiles/qwen3-32b-dp16-tp8-deepswe.env|\
    cluster/profiles/qwen3-32b-dp-parity-deepswe-full.env)
      p38_fixed_endpoint=untied_lm_head
      p38_fixed_hidden=5120
      p38_fixed_tp=8
      ;;
    *)
      echo "[run] FATAL: fixed lm-head receipt classifier has no admitted workload/stage/profile contract" >&2
      exit 1
      ;;
  esac
  p38_fixed_receipt_args=(
    --log "$LOG"
    --endpoint "$p38_fixed_endpoint"
    --hidden "$p38_fixed_hidden"
    --tp-size "$p38_fixed_tp"
    --output "$CANON_STATE/p38_fixed_lm_head_receipts.json"
  )
  case "${CANON_PROFILE_FILE:-}" in
    cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-tim.env)
      if [ "${CANON_P57_RUN_KIND:-}" = "eval" ]; then
        p38_fixed_receipt_args+=(--request-only)
      else
        p38_fixed_receipt_args+=(--learner-m 2048)
        if [ "${CANON_P59_RANK_PARALLEL_BACKWARD:-0}" = "1" ]; then
          p38_fixed_receipt_args+=(--p59-local-dp-size 8)
        fi
      fi
      ;;
    cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-v1-hp.env)
      p38_fixed_receipt_args+=(--learner-m 2048)
      if [ "${CANON_P59_RANK_PARALLEL_BACKWARD:-0}" = "1" ]; then
        p38_fixed_receipt_args+=(--p59-local-dp-size 8)
      fi
      ;;
    cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-apc-debug.env)
      p38_fixed_receipt_args+=(--learner-m 2048)
      ;;
    cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k-v1-hp.env|\
    cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k-p62-debug.env)
      if [ "${CANON_P59_RANK_PARALLEL_BACKWARD:-0}" = "1" ]; then
        p38_fixed_receipt_args+=(--p59-local-dp-size 16)
      fi
      ;;
  esac
  if [ -z "${CANON_P38_SERVING_CAPTURE_DIR:-}" ] && \
     ! { [ "${CANON_PROFILE_FILE:-}" = \
             "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-tim.env" ] && \
         [ "${CANON_P57_RUN_KIND:-}" = "eval" ]; }; then
    p38_fixed_receipt_args+=(--require-vjp)
  fi
  if ! JAX_PLATFORMS=cpu PYTHONPATH="$CANON_PKG/..:${PYTHONPATH:-}" \
      python3 "$CANON_PKG/tasks/p38-pathways-decode-prefill-carrier/scripts/classify_p38_fixed_lm_head_receipts.py" \
        "${p38_fixed_receipt_args[@]}"; then
    echo "[run] FATAL: fixed lm-head executable receipt contract failed" >&2
    exit 1
  fi
  p38_fixed_receipt_report="$CANON_STATE/p38_fixed_lm_head_receipts.json"
  p38_fixed_receipt_sha="$(sha256sum "$p38_fixed_receipt_report" | awk '{print $1}')"
  echo "[P38.FIXED_LM_HEAD] RECEIPT_ARTIFACT path=$p38_fixed_receipt_report sha256=$p38_fixed_receipt_sha"
  sed 's/^/[P38.FIXED_LM_HEAD_RECEIPTS_JSON] /' "$p38_fixed_receipt_report"
  if [ -n "${attempt_evidence_dir:-}" ]; then
    cp -- "$p38_fixed_receipt_report" \
      "$attempt_evidence_dir/p38_fixed_lm_head_receipts.json"
  fi
fi
if [ "${CANON_P35_EXACT_REPLAY:-0}" = "1" ] && \
   [ -s "${CANON_P35_PRE_REPLAY_REPORT:-}" ]; then
  p35_base_sha="$(sha256sum "$CANON_P35_PRE_REPLAY_REPORT" | awk '{print $1}')"
  echo "[CANON_P35.3] PRE_REPLAY_EVIDENCE path=$CANON_P35_PRE_REPLAY_REPORT sha256=$p35_base_sha"
fi
if [ "${CANON_P58_DEEPSWE_TIM:-0}" = "1" ] && \
   [ "$n_p58_seed" -ne 1 ]; then
  echo "[run] FATAL: P58 fixed-seed marker contract failed: $n_p58_seed" >&2
  exit 1
fi
if [ "${CANON_P58_DEEPSWE_TIM:-0}" = "1" ] && \
   [ "${CANON_P58_TIM_ARM:-}" = "native" ]; then
  for marker_count in "$n_ar" "$n_emb" "$n_lp" "$n_p38_fixed_primal" \
                      "$n_p38_fixed_vjp" "$n_p38_kv_unified"; do
    if [ "$marker_count" -ne 0 ]; then
      echo "[run] FATAL: canonical runtime marker leaked into P58 native" >&2
      exit 1
    fi
  done
  if [ "$n_p58_stock_observer" -ne 1 ]; then
    echo "[run] FATAL: P58 native stock observer marker contract failed: $n_p58_stock_observer" >&2
    exit 1
  fi
  case "${CANON_P34_DISABLE_SAMPLER_IS:-}:${CANON_P34_DISABLE_TIS:-}" in
    1:1)
      [ "$n_p58_recipe_raw" -eq 1 ] && [ "$n_p58_recipe_is" -eq 0 ] || {
        echo "[run] FATAL: P58 native-raw recipe marker contract failed: raw=$n_p58_recipe_raw/1 is=$n_p58_recipe_is/0" >&2
        exit 1
      }
      ;;
    0:0)
      [ "$n_p58_recipe_raw" -eq 0 ] && [ "$n_p58_recipe_is" -eq 1 ] || {
        echo "[run] FATAL: P58 native-is recipe marker contract failed: raw=$n_p58_recipe_raw/0 is=$n_p58_recipe_is/1" >&2
        exit 1
      }
      ;;
    *)
      echo "[run] FATAL: P58 native sampler environment tuple drifted" >&2
      exit 1
      ;;
  esac
  [ "$n_wandb_p34" -eq 1 ] || {
    echo "[run] FATAL: P58 native did not attest exactly one W&B run" >&2
    exit 1
  }
  echo "[P58.NATIVE] RUNTIME_PATH_PASS canonical_markers=0 canonical_overlay=skipped stock_observer=observer-only"
elif p57_is_stock_fast_runtime; then
  p57_validate_stock_fast_runtime_markers \
    "$n_ar" "$n_emb" "$n_lp" "$n_p38_fixed_primal" \
    "$n_p38_fixed_vjp" "$n_p38_kv_unified" || exit 1
  if p57_is_stock_fast_calibration || p57_is_stock_fast_evaluation; then
    if [ "$n_p57_stock_sync" -ne 1 ]; then
      echo "[run] FATAL: P57 calibration rollout weight sync marker contract failed: $n_p57_stock_sync" >&2
      exit 1
    fi
  else
    p57_expected_sync=0
    [ "${CANON_FROZENLAKE_CKPT_MODE:-}" != "resume" ] || p57_expected_sync=1
    if [ "$n_p57_stock_sync" -ne "$p57_expected_sync" ] || \
       [ "$n_p57_stock_train_runtime" -ne 1 ] || \
       [ "$n_p57_stock_observer" -ne 1 ] || \
       [ "$n_p57_stock_segment_preflight" -ne 1 ] || \
       [ "$n_p57_stock_segment_complete" -ne 1 ]; then
      echo "[run] FATAL: P57 stock training segment contract failed: sync=$n_p57_stock_sync/$p57_expected_sync runtime=$n_p57_stock_train_runtime observer=$n_p57_stock_observer preflight=$n_p57_stock_segment_preflight complete=$n_p57_stock_segment_complete" >&2
      exit 1
    fi
  fi
elif [ "$n_ar" -eq 0 ] || [ "$n_emb" -eq 0 ]; then
  echo "[run] FATAL: no PATHTRACE for the fixed-order reductions -- the intervention did not" >&2
  echo "[run]        execute.  Any result from this run is void regardless of its exit code." >&2
  exit 1
fi
if [ "${CANON_P32_TRAIN_ADMITTED:-0}" = "1" ] && [ "$n_wandb" -ne 1 ]; then
  if [ "${CANON_P34_DEEPSWE:-0}" != "1" ] || [ "$n_wandb_p34" -ne 1 ]; then
    echo "[run] FATAL: admitted workload did not attest exactly one online W&B run" >&2
    exit 1
  fi
fi
if [ "${CANON_P32_TRAIN_ADMITTED:-0}" = "1" ]; then
  case "${CANON_P32_WORKLOAD:-}" in
    frozenlake|frozenlake-dp8-tp8) is_frozenlake=1 ;;
    *) is_frozenlake=0 ;;
  esac
fi
if [ "${is_frozenlake:-0}" = "1" ]; then
  if [ "${CANON_P33_ENABLE_EVAL:-0}" = "1" ]; then
    if [ "$n_eval_on" -ne 1 ] || [ "$n_eval_off" -ne 0 ]; then
      echo "[run] FATAL: admitted FrozenLake evaluation marker contract failed" >&2
      exit 1
    fi
  elif [ "$n_eval_off" -ne 1 ] || [ "$n_eval_on" -ne 0 ]; then
    echo "[run] FATAL: admitted P33 FrozenLake did not attest evaluation disabled exactly once" >&2
    exit 1
  fi
fi
if [ "${CANON_P62_BACKWARD_NUMERIC_DEBUG:-0}" = "1" ]; then
  p62_classification="$CANON_STATE/p62_backward_numeric.classification.json"
  p62_classifier_rc=0
  JAX_PLATFORMS=cpu PYTHONPATH="$CANON_PKG/..:${PYTHONPATH:-}" \
    python3 "$CANON_PKG/tasks/v1-phase4-three-full-recipes/scripts/classify_attempt7_numeric_debug.py" \
      "$LOG" --output "$p62_classification" || p62_classifier_rc=$?
  if [ ! -s "$p62_classification" ]; then
    echo "[run] FATAL: P62 classifier did not persist its result" >&2
    exit 1
  fi
  p62_log_sha="$(sha256sum "$LOG" | awk '{print $1}')"
  p62_log_bytes="$(wc -c < "$LOG" | tr -d '[:space:]')"
  p62_log_lines="$(wc -l < "$LOG" | tr -d '[:space:]')"
  p62_class_sha="$(sha256sum "$p62_classification" | awk '{print $1}')"
  p62_verdict="$(JAX_PLATFORMS=cpu python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["verdict"])' \
    "$p62_classification")" || {
      echo "[run] FATAL: P62 classification verdict is unreadable" >&2
      exit 1
    }
  echo "[P62.NUMERIC.POSTFLIGHT] verdict=$p62_verdict workload_rc=$rc transport_rc=$tee_rc run_log=$LOG run_log_sha256=$p62_log_sha run_log_bytes=$p62_log_bytes run_log_lines=$p62_log_lines classification=$p62_classification classification_sha256=$p62_class_sha"
  sed 's/^/[P62.NUMERIC.CLASSIFICATION_JSON] /' "$p62_classification"
  if [ "$tee_rc" -ne 0 ]; then
    echo "[run] FATAL: P62 full-log transport failed: rc=$tee_rc" >&2
    exit 1
  fi
  if [ "$p62_classifier_rc" -ne 0 ]; then
    echo "[run] FATAL: P62 full-log classification failed: rc=$p62_classifier_rc verdict=$p62_verdict" >&2
    exit "$p62_classifier_rc"
  fi
  if [ "$p62_verdict" = "ROOT_LOCALIZED_NONFINITE" ]; then
    if [ "$rc" -eq 0 ]; then
      echo "[run] FATAL: P62 localized non-finite without fail-closed workload exit" >&2
      exit 1
    fi
    echo "[P62.NUMERIC.POSTFLIGHT] ROOT_LOCALIZED workload_exit_preserved=$rc"
    exit "$rc"
  fi
  if [ "$rc" -ne 0 ]; then
    echo "[run] FATAL: P62 workload failed after successful finite classification: rc=$rc verdict=$p62_verdict" >&2
    exit "$rc"
  fi
  echo "[P62.NUMERIC.POSTFLIGHT] PASS complete_log=1 optimizer_commits=0"
  exit 0
elif [ "${CANON_P35_ENVELOPE:-0}" = "1" ]; then
  if [ "${CANON_P35_REPLAY_STAGE_PROBE:-0}" = "1" ]; then
    if [ "$rc" -ne 1 ]; then
      echo "[run] FATAL: P35.3c must terminate with diagnostic exit=1; got $rc" >&2
      exit 1
    fi
    if [ "$n_p35_base" -ne 1 ] || [ "$n_p35_stop" -ne 0 ] || [ "$n_p35_replay" -ne 0 ]; then
      echo "[run] FATAL: P35.3c marker contract drifted: base=$n_p35_base stop=$n_p35_stop replay=$n_p35_replay" >&2
      exit 1
    fi
    if [ ! -s "$CANON_P35_PRE_REPLAY_REPORT" ]; then
      echo "[run] FATAL: P35.3c missing preliminary evidence: $CANON_P35_PRE_REPLAY_REPORT" >&2
      exit 1
    fi
    JAX_PLATFORMS=cpu PYTHONPATH="$CANON_PKG/..:${PYTHONPATH:-}" \
      python3 "$CANON_PKG/tests/p35_envelope/classify_envelope.py" \
        --report "$CANON_P35_PRE_REPLAY_REPORT" \
        --output "$CANON_P35_CLASSIFICATION" || exit 1
    stage_class_rc=1
    if [ -e "$CANON_P35_REPLAY_STAGE_REPORT" ]; then
      if JAX_PLATFORMS=cpu PYTHONPATH="$CANON_PKG/..:${PYTHONPATH:-}" \
        python3 "$CANON_PKG/tests/p35_envelope/classify_stage_probe.py" \
          --report "$CANON_P35_REPLAY_STAGE_REPORT" \
          --output "$CANON_P35_REPLAY_STAGE_CLASSIFICATION"; then
        stage_class_rc=0
      else
        stage_class_rc=$?
      fi
    fi
    for evidence_path in \
      "$CANON_P35_PRE_REPLAY_REPORT" \
      "$CANON_P35_CLASSIFICATION" \
      "$CANON_P35_REPLAY_STAGE_REPORT" \
      "$CANON_P35_REPLAY_STAGE_CLASSIFICATION"; do
      if [ -e "$evidence_path" ]; then
        evidence_sha="$(sha256sum "$evidence_path" | awk '{print $1}')"
        echo "[CANON_P35.3C] EVIDENCE path=$evidence_path sha256=$evidence_sha"
      fi
    done
    if [ "$n_p35_stage_begin" -ne 6 ] || \
       [ "$n_p35_stage_ready" -ne 6 ] || \
       [ "$n_p35_stage_complete" -ne 1 ] || \
       [ "$stage_class_rc" -ne 0 ]; then
      echo "[run] FATAL: P35.3c incomplete stages: begin=$n_p35_stage_begin ready=$n_p35_stage_ready complete=$n_p35_stage_complete classifier_rc=$stage_class_rc" >&2
      exit 1
    fi
    echo "[run] P35.3c first-record stage probe accepted; NO_NUMERICAL_VERDICT"
    rc=0
  else
    if [ "$rc" -ne 1 ]; then
      echo "[run] FATAL: P35 producer must terminate with its expected diagnostic exit=1; got $rc" >&2
      exit 1
    fi
    if [ "$n_p35_stop" -ne 1 ]; then
      echo "[run] FATAL: P35 did not stop exactly once before backward" >&2
      exit 1
    fi
    if [ ! -s "$CANON_P35_ENVELOPE_REPORT" ]; then
      echo "[run] FATAL: P35 stop marker exists without a report" >&2
      exit 1
    fi
    JAX_PLATFORMS=cpu PYTHONPATH="$CANON_PKG/..:${PYTHONPATH:-}" \
      python3 "$CANON_PKG/tests/p35_envelope/classify_envelope.py" \
        --report "$CANON_P35_ENVELOPE_REPORT" \
        --output "$CANON_P35_CLASSIFICATION" || exit 1
    if [ "${CANON_P35_EXACT_REPLAY:-0}" = "1" ]; then
      if [ "$n_p35_base" -ne 1 ]; then
        echo "[run] FATAL: P35.3 did not emit exactly one pre-replay marker" >&2
        exit 1
      fi
      if [ ! -s "$CANON_P35_PRE_REPLAY_REPORT" ]; then
        echo "[run] FATAL: P35.3 pre-replay marker exists without a report" >&2
        exit 1
      fi
      if [ "$n_p35_replay" -ne 1 ]; then
        echo "[run] FATAL: P35.3 did not emit exactly one replay marker" >&2
        exit 1
      fi
      if [ ! -s "$CANON_P35_EXACT_REPLAY_REPORT" ]; then
        echo "[run] FATAL: P35.3 marker exists without a replay report" >&2
        exit 1
      fi
      JAX_PLATFORMS=cpu PYTHONPATH="$CANON_PKG/..:${PYTHONPATH:-}" \
        python3 "$CANON_PKG/tests/p35_envelope/classify_exact_replay.py" \
          --report "$CANON_P35_EXACT_REPLAY_REPORT" \
          --output "$CANON_P35_EXACT_REPLAY_CLASSIFICATION" || exit 1
      for evidence_path in \
        "$CANON_P35_PRE_REPLAY_REPORT" \
        "$CANON_P35_ENVELOPE_REPORT" \
        "$CANON_P35_CLASSIFICATION" \
        "$CANON_P35_EXACT_REPLAY_REPORT" \
        "$CANON_P35_EXACT_REPLAY_CLASSIFICATION"; do
        evidence_sha="$(sha256sum "$evidence_path" | awk '{print $1}')"
        echo "[CANON_P35.3] EVIDENCE path=$evidence_path sha256=$evidence_sha"
      done
    fi
    echo "[run] P35 expected diagnostic exit=1 accepted after COMPLETE classification"
    rc=0
  fi
elif [ -n "${CANON_P38_SERVING_CAPTURE_DIR:-}" ] && \
     [ "$n_p38_precheck" -gt 0 ]; then
  p38_expected_rc=1
  if [ "${CANON_P38_CONTROLLED_EXIT:-0}" = "1" ]; then
    p38_expected_rc=42
  fi
  p38_depth_rc=0
  p38_depth_observed=""
  p38_depth_observed="$(python3 - "$CANON_PRE_ALIGN_REPORT" <<'PY'
import json
import pathlib
import sys

lines = [
    line for line in pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if not lines:
  raise SystemExit("empty P38 pre-alignment report")
record = json.loads(lines[-1])
geometry = record.get("action_geometry", {})
if geometry.get("valid") is not True:
  raise SystemExit(f"invalid P38 action geometry: {geometry}")
print(int(geometry["max_logical_kv_prefix_length"]))
PY
)" || p38_depth_rc=$?
  if [ "$p38_depth_rc" -eq 0 ] && \
     [ "$p38_depth_observed" -ge "${CANON_P38_MIN_ACTION_KV:?}" ]; then
    echo "[CANON_P38] DEPTH_SUFFICIENCY min=$CANON_P38_MIN_ACTION_KV observed=$p38_depth_observed verdict=PASS"
  else
    echo "[run] FATAL: P38 depth sufficiency failed: min=${CANON_P38_MIN_ACTION_KV:-unset} observed=${p38_depth_observed:-invalid}" >&2
    exit 1
  fi
  if [ "$rc" -ne "$p38_expected_rc" ] || [ "$n_p38_precheck" -ne 1 ] || \
     [ "${p38_capture_rc:-1}" -ne 0 ] || \
     [ "${p38_kv_observer_rc:-0}" -ne 0 ] || \
     [ "${p38_seam_rc:-0}" -ne 0 ] || \
     [ "${p38_terminal_rc:-0}" -ne 0 ]; then
    echo "[run] FATAL: P38 serving precheck is incomplete: rc=$rc expected_rc=$p38_expected_rc markers=$n_p38_precheck capture_rc=${p38_capture_rc:-unset} kv_observer_rc=${p38_kv_observer_rc:-unset} seam_rc=${p38_seam_rc:-unset} terminal_rc=${p38_terminal_rc:-unset}" >&2
    exit 1
  fi
  p38_request_live_action complete || {
      echo "[run] FATAL: P38 GCS completion marker failed" >&2
      exit 1
    }
  p38_stop_live_worker
  trap - EXIT
  if [ "${p38_live_rc:-1}" -ne 0 ]; then
    echo "[run] FATAL: P38 live snapshot worker failed: rc=${p38_live_rc:-unset}" >&2
    exit 1
  fi
  echo "[run] P38 serving controlled precheck accepted exit=$p38_expected_rc; backward=0 optimizer_commits=0"
  rc=0
elif [ "$rc" -eq 0 ] && [ "${CANON_P34_DEEPSWE:-0}" = "1" ]; then
  if [ "${CANON_P58_DEEPSWE_TIM:-0}" = "1" ]; then
    classification="$CANON_STATE/p58_deepswe_${CANON_P58_TIM_ARM}_${CANON_P34_RUN_STAGE}.classification.json"
    JAX_PLATFORMS=cpu PYTHONPATH="$CANON_PKG/..:${PYTHONPATH:-}" \
      python3 "$CANON_PKG/tests/p58_deepswe_native_zero/classify_run.py" \
        --arm "$CANON_P58_TIM_ARM" \
        --stage "$CANON_P34_RUN_STAGE" \
        --run-log "$LOG" \
        --debug-dir "$CANON_P58_DEBUG_DIR" \
        --weight-report "$CANON_P34_WEIGHT_REPORT" \
        --pre-alignment-report "$CANON_PRE_ALIGN_REPORT" \
        --update-report "$CANON_UPDATE_REPORT" \
        --alignment-report "$CANON_ALIGN_REPORT" \
        --output "$classification" || exit 1
    if [ "${CANON_V1_HP_FULL:-0}" = "1" ]; then
      p58_hp_classification="$CANON_STATE/p58_zero_hp_full.classification.json"
      JAX_PLATFORMS=cpu PYTHONPATH="$CANON_PKG/..:${PYTHONPATH:-}" \
        python3 "$CANON_PKG/tasks/p58-deepswe-native-zero-comparison/scripts/classify_zero_hp_full.py" \
          --state "$CANON_STATE" \
          --run-log "$LOG" \
          --update-report "$CANON_UPDATE_REPORT" \
          --base-classification "$classification" \
          --output "$p58_hp_classification" || exit 1
    fi
  elif [ "${CANON_P46_DEEPSWE_TRAIN:-0}" = "1" ]; then
    classification="$CANON_STATE/p46_deepswe_q32_${CANON_P46_TOPOLOGY}_full.classification.json"
    JAX_PLATFORMS=cpu PYTHONPATH="$CANON_PKG/..:${PYTHONPATH:-}" \
      python3 "$CANON_PKG/tests/p34_deepswe/classify_run.py" \
        --p46-profile \
        --topology "$CANON_P46_TOPOLOGY" \
        --stage full \
        --run-log "$LOG" \
        --debug-dir "$CANON_P34_DEBUG_DIR" \
        --weight-report "$CANON_P34_WEIGHT_REPORT" \
        --pre-alignment-report "$CANON_PRE_ALIGN_REPORT" \
        --update-report "$CANON_UPDATE_REPORT" \
        --alignment-report "$CANON_ALIGN_REPORT" \
        --output "$classification" || exit 1
  elif [ "${CANON_P44_DEEPSWE_PARITY:-0}" = "1" ]; then
    classification="$CANON_STATE/p44_deepswe_${CANON_P44_TOPOLOGY}_${CANON_P34_RUN_STAGE}.classification.json"
    JAX_PLATFORMS=cpu PYTHONPATH="$CANON_PKG/..:${PYTHONPATH:-}" \
      python3 "$CANON_PKG/tests/p44_deepswe_qwen4b_parity/classify_run.py" \
        --topology "$CANON_P44_TOPOLOGY" \
        --stage "$CANON_P34_RUN_STAGE" \
        --run-log "$LOG" \
        --debug-dir "$CANON_P44_DEBUG_DIR" \
        --weight-report "$CANON_P34_WEIGHT_REPORT" \
        --pre-alignment-report "$CANON_PRE_ALIGN_REPORT" \
        --update-report "$CANON_UPDATE_REPORT" \
        --alignment-report "$CANON_ALIGN_REPORT" \
        --output "$classification" || exit 1
  elif [ "${CANON_P43_DEEPSWE_DEBUG:-0}" = "1" ]; then
    classification="$CANON_STATE/p43_deepswe_${CANON_P34_RUN_STAGE}.classification.json"
    JAX_PLATFORMS=cpu PYTHONPATH="$CANON_PKG/..:${PYTHONPATH:-}" \
      python3 "$CANON_PKG/tests/p43_deepswe_debug/classify_run.py" \
        --stage "$CANON_P34_RUN_STAGE" \
        --run-log "$LOG" \
        --debug-dir "$CANON_P43_DEBUG_DIR" \
        --weight-report "$CANON_P34_WEIGHT_REPORT" \
        --pre-alignment-report "$CANON_PRE_ALIGN_REPORT" \
        --update-report "$CANON_UPDATE_REPORT" \
        --alignment-report "$CANON_ALIGN_REPORT" \
        --output "$classification" || exit 1
  elif [ "${CANON_P39_64CHIP_PILOT:-0}" = "1" ]; then
    classification="$CANON_STATE/p39_deepswe_${CANON_P34_RUN_STAGE}.classification.json"
    JAX_PLATFORMS=cpu PYTHONPATH="$CANON_PKG/..:${PYTHONPATH:-}" \
      python3 "$CANON_PKG/tests/p39_deepswe_pilot/classify_run.py" \
        --stage "$CANON_P34_RUN_STAGE" \
        --run-log "$LOG" \
        --weight-report "$CANON_P34_WEIGHT_REPORT" \
        --pre-alignment-report "$CANON_PRE_ALIGN_REPORT" \
        --update-report "$CANON_UPDATE_REPORT" \
        --alignment-report "$CANON_ALIGN_REPORT" \
        --output "$classification" || exit 1
  else
    classification="$CANON_STATE/p34_deepswe_${CANON_P34_RUN_STAGE}.classification.json"
    JAX_PLATFORMS=cpu PYTHONPATH="$CANON_PKG/..:${PYTHONPATH:-}" \
      python3 "$CANON_PKG/tests/p34_deepswe/classify_run.py" \
        --stage "$CANON_P34_RUN_STAGE" \
        --run-log "$LOG" \
        --debug-dir "$CANON_P34_DEBUG_DIR" \
        --weight-report "$CANON_P34_WEIGHT_REPORT" \
        --pre-alignment-report "$CANON_PRE_ALIGN_REPORT" \
        --update-report "$CANON_UPDATE_REPORT" \
        --alignment-report "$CANON_ALIGN_REPORT" \
        --output "$classification" || exit 1
  fi
elif [ "$rc" -eq 0 ] && [ "${CANON_P33_WORKLOAD_LAUNCH_ADMITTED:-0}" = "1" ]; then
  if [ "${CANON_PROFILE_FILE:-}" = \
       "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-tim.env" ] && \
     [ "${CANON_P57_RUN_KIND:-}" = "eval" ]; then
    classification="$CANON_STATE/p57_${CANON_P57_TIM_ARM}_eval_${CANON_P57_EVAL_CHECKPOINT_STEP}.classification.json"
    JAX_PLATFORMS=cpu PYTHONPATH="$CANON_PKG/..:${PYTHONPATH:-}" \
      python3 "$CANON_PKG/tasks/p57-frozenlake-tim-causal-study/scripts/classify_checkpoint_eval.py" \
        --evaluation "$CANON_P57_EVAL_OUTPUT" \
        --run-log "$LOG" \
        --arm "$CANON_P57_TIM_ARM" \
        --source-commit "$CANON_EXPECT_COMMIT" \
        --checkpoint-tag "$CANON_FROZENLAKE_CKPT_TAG" \
        --checkpoint-step "$CANON_P57_EVAL_CHECKPOINT_STEP" \
        --expected-updates "$CANON_P57_EXPECTED_UPDATES" \
        --workload-candidate "${CANON_P57_WORKLOAD_CANDIDATE:-}" \
        --data-split "${CANON_P57_DATA_SPLIT:-}" \
        --output "$classification" || exit 1
    eval_sha="$(sha256sum "$CANON_P57_EVAL_OUTPUT" | awk '{print $1}')"
    class_sha="$(sha256sum "$classification" | awk '{print $1}')"
    echo "[P57.EVAL] EVIDENCE evaluation=$CANON_P57_EVAL_OUTPUT evaluation_sha256=$eval_sha classification=$classification classification_sha256=$class_sha"
    sed 's/^/[P57.EVAL.CLASSIFICATION] /' "$classification"
    JAX_PLATFORMS=cpu python3 -c \
      'import json,sys; print("[P57.EVAL.CLASSIFICATION_JSON] "+json.dumps(json.load(open(sys.argv[1], encoding="utf-8")), sort_keys=True, separators=(",", ":")))' \
      "$classification"
  else
    classification="$CANON_STATE/p33_${CANON_P32_WORKLOAD}_${CANON_P33_RUN_STAGE}.classification.json"
    p57_classifier_args=()
    if { [ "${CANON_PROFILE_FILE:-}" = \
           "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-tim.env" ] || \
         [ "${CANON_PROFILE_FILE:-}" = \
           "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-v1-hp.env" ]; } && \
       [ "${CANON_P33_ENABLE_EVAL:-0}" = "1" ]; then
      p57_eval_classification="$CANON_STATE/p57_inprocess_eval.classification.json"
      JAX_PLATFORMS=cpu PYTHONPATH="$CANON_PKG/..:${PYTHONPATH:-}" \
        python3 "$CANON_PKG/tasks/p57-frozenlake-tim-causal-study/scripts/classify_inprocess_eval.py" \
          --run-log "$LOG" \
          --expected-updates "$CANON_P57_EXPECTED_UPDATES" \
          --interval 50 \
          --held-out-rows 100 \
          --generations 8 \
          --workload-candidate "${CANON_P57_WORKLOAD_CANDIDATE:-}" \
          --data-split "${CANON_P57_DATA_SPLIT:-}" \
          --output "$p57_eval_classification" || exit 1
      p57_eval_class_sha="$(sha256sum "$p57_eval_classification" | awk '{print $1}')"
      echo "[P57.EVAL] EVIDENCE classification=$p57_eval_classification classification_sha256=$p57_eval_class_sha"
    fi
    if [ "${CANON_V1_HP_FULL:-0}" = "1" ]; then
      p57_classifier_args+=(--alignment-warning-only 0)
      if [ "${CANON_P32_WORKLOAD:-}" = "frozenlake-dp8-tp8" ]; then
        p57_classifier_args+=(--expected-updates "$CANON_P57_EXPECTED_UPDATES")
      fi
    elif [ "${CANON_PROFILE_FILE:-}" = \
         "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-tim.env" ]; then
      p57_classifier_args+=(
        --expected-updates "$CANON_P57_EXPECTED_UPDATES"
        --alignment-warning-only \
          "$([ "${CANON_P57_TIM_ARM:-}" != zero ] && printf 1 || printf 0)"
        --p57-ab-only \
          "$([ "${CANON_P57_TIM_ARM:-}" != zero ] && printf 1 || printf 0)"
      )
    fi
    JAX_PLATFORMS=cpu PYTHONPATH="$CANON_PKG/..:${PYTHONPATH:-}" \
      python3 "$CANON_PKG/tests/p33_workloads/classify_run.py" \
        --workload "$CANON_P32_WORKLOAD" \
        --dp-size "$CANON_DP_SIZE" \
        --tp-size "$CANON_TP_SIZE" \
        --stage "$CANON_P33_RUN_STAGE" \
        --run-log "$LOG" \
        --pre-alignment-report "$CANON_PRE_ALIGN_REPORT" \
        --update-report "$CANON_UPDATE_REPORT" \
        --alignment-report "$CANON_ALIGN_REPORT" \
        "${p57_classifier_args[@]}" \
        --output "$classification" || exit 1
    if [ "${CANON_V1_HP_FULL:-0}" = "1" ]; then
      case "${CANON_P32_WORKLOAD:-}:${CANON_P57_WORKLOAD_CANDIDATE:-}:${CANON_P57_DATA_SPLIT:-}" in
        gsm8k::) v1_recipe=gsm8k ;;
        frozenlake-dp8-tp8::) v1_recipe=p45 ;;
        frozenlake-dp8-tp8:m15:main) v1_recipe=m15 ;;
        *)
          echo "[run] FATAL: unknown V1 high-performance recipe identity" >&2
          exit 1
          ;;
      esac
      v1_classification="$CANON_STATE/v1_hp_${v1_recipe}_full.classification.json"
      JAX_PLATFORMS=cpu PYTHONPATH="$CANON_PKG/..:${PYTHONPATH:-}" \
        python3 "$CANON_PKG/tasks/v1-phase4-three-full-recipes/scripts/classify_full_recipe.py" \
          --recipe "$v1_recipe" \
          --state "$CANON_STATE" \
          --run-log "$LOG" \
          --update-report "$CANON_UPDATE_REPORT" \
          --base-classification "$classification" \
          --xprof-dir "$xprof_local_dir" \
          --xprof-receipt "$xprof_restore_receipt" \
          --output "$v1_classification" || exit 1
      unset v1_recipe
      unset v1_classification
    fi
    if [ "${CANON_PROFILE_FILE:-}" = \
         "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-tim.env" ] || \
       [ "${CANON_PROFILE_FILE:-}" = \
         "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-v1-hp.env" ]; then
      class_sha="$(sha256sum "$classification" | awk '{print $1}')"
      echo "[P57.TRAIN] EVIDENCE classification=$classification classification_sha256=$class_sha"
      JAX_PLATFORMS=cpu python3 -c \
        'import json,sys; print("[P57.TRAIN.CLASSIFICATION_JSON] "+json.dumps(json.load(open(sys.argv[1], encoding="utf-8")), sort_keys=True, separators=(",", ":")))' \
        "$classification"
    fi
  fi
fi
if [ "${CANON_P38_FIXED_LM_HEAD:-0}" = "1" ] && \
   [ -z "${CANON_P38_SERVING_CAPTURE_DIR:-}" ] && \
   [ "${CANON_P32_WORKLOAD:-}" = "frozenlake" ] && \
   [ "${CANON_P33_RUN_STAGE:-}" = "backward-no-commit" ]; then
  for p38h_artifact in \
      "pre-alignment:$CANON_PRE_ALIGN_REPORT" \
      "alignment:$CANON_ALIGN_REPORT" \
      "update:$CANON_UPDATE_REPORT"; do
    p38h_name="${p38h_artifact%%:*}"
    p38h_path="${p38h_artifact#*:}"
    if [ ! -s "$p38h_path" ]; then
      echo "[run] FATAL: P38.2h evidence absent: $p38h_name=$p38h_path" >&2
      exit 1
    fi
    p38h_sha="$(sha256sum "$p38h_path" | awk '{print $1}')"
    p38h_data="$(base64 -w 0 "$p38h_path")"
    echo "[CANON_P38H_ARTIFACT] name=$p38h_name sha256=$p38h_sha encoding=base64 data=$p38h_data"
  done
fi
if [ "$jax_cache_saved_early" -ne 1 ]; then
  canon_jax_cache_sync save
fi
exit "$rc"

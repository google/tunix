#!/usr/bin/env bash
# The actual workload.
#
# CANON_RUN_CMD carries the command so the same entry point serves gates, canaries and
# training without editing YAML.  The exit code is propagated, and the PATHTRACE tally is
# checked afterwards: an intervention that never fired produces a perfectly green run.
set -uo pipefail
source "$CANON_STATE/env.sh"
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
if [ "${CANON_P32_DP_ADMISSION:-0}" = "1" ] && \
   [ "${CANON_P32_TRAIN_ADMITTED:-0}" != "1" ]; then
  echo "[run] REFUSING: P32 profile is admission-only." >&2
  echo "[run] A real (dp,tp) replicated-parameter segmented VJP has not passed remote gates;" >&2
  echo "[run] set all three P33 admissions only after the rank-local reducer gate passes." >&2
  exit 2
fi
: "${CANON_RUN_CMD:?CANON_RUN_CMD unset -- nothing to run}"
LOG="${CANON_RUN_LOG:-$CANON_STATE/run.log}"
if [ "${CANON_P33_WORKLOAD_LAUNCH_ADMITTED:-0}" = "1" ]; then
  report_keys=(CANON_RUN_LOG CANON_ALIGN_REPORT CANON_UPDATE_REPORT)
  if [ "${CANON_P34_DEEPSWE:-0}" != "1" ]; then
    report_keys+=(CANON_PRE_ALIGN_REPORT)
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
echo "[run] cmd: $CANON_RUN_CMD"
echo "[run] log: $LOG"
cd "${CANON_RUN_CWD:-$CANON_PKG/..}"
set -o pipefail
bash -c "$CANON_RUN_CMD" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
echo "[run] exit=$rc"
# grep -a: progress-bar control characters make grep treat the log as binary and drop every
# match silently, which reads exactly like "the intervention never fired".
n_ar=$(grep -ac 'CANON_FIXED_AR=1 fixed-order tree' "$LOG" || true)
n_emb=$(grep -ac 'CANON_FIXED_AR_EMBED=1 fixed-order embed gather' "$LOG" || true)
n_lp=$(grep -ac 'CANON_LOGPROB_M on' "$LOG" || true)
n_wandb=$(grep -ac '\[CANON_P33_WANDB\] ONLINE_RUN_PASS' "$LOG" || true)
n_wandb_p34=$(grep -ac '\[CANON_P34_WANDB\] ONLINE_RUN_PASS' "$LOG" || true)
n_eval_off=$(grep -ac '\[CANON_P33_EVAL\] DISABLED workload=frozenlake' "$LOG" || true)
n_p35_stop=$(grep -ac '^\[CANON_P35\] REPORT_COMPLETE .*STOP_BEFORE_BACKWARD' "$LOG" || true)
n_p35_base=$(grep -ac '^\[CANON_P35\] BASE_REPORT_COMPLETE .*REPLAY_PENDING' "$LOG" || true)
n_p35_replay=$(grep -ac '^\[CANON_P35.3\] REPLAY_COMPLETE' "$LOG" || true)
n_p35_stage_begin=$(grep -ac '^\[CANON_P35.3C\] STAGE_BEGIN' "$LOG" || true)
n_p35_stage_ready=$(grep -ac '^\[CANON_P35.3C\] STAGE_READY' "$LOG" || true)
n_p35_stage_complete=$(grep -ac '^\[CANON_P35.3C\] STAGE_PROBE_COMPLETE .*NO_NUMERICAL_VERDICT' "$LOG" || true)
echo "[run] PATHTRACE fixed_ar=$n_ar embed=$n_emb logprob_m=$n_lp wandb_online=$n_wandb p34_wandb_online=$n_wandb_p34 eval_off=$n_eval_off p35_base=$n_p35_base p35_stop=$n_p35_stop p35_replay=$n_p35_replay p35_stage_begin=$n_p35_stage_begin p35_stage_ready=$n_p35_stage_ready p35_stage_complete=$n_p35_stage_complete"
if [ "${CANON_P35_EXACT_REPLAY:-0}" = "1" ] && \
   [ -s "${CANON_P35_PRE_REPLAY_REPORT:-}" ]; then
  p35_base_sha="$(sha256sum "$CANON_P35_PRE_REPLAY_REPORT" | awk '{print $1}')"
  echo "[CANON_P35.3] PRE_REPLAY_EVIDENCE path=$CANON_P35_PRE_REPLAY_REPORT sha256=$p35_base_sha"
fi
if [ "$n_ar" -eq 0 ] || [ "$n_emb" -eq 0 ]; then
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
if [ "${CANON_P32_TRAIN_ADMITTED:-0}" = "1" ] && \
   [ "${CANON_P32_WORKLOAD:-}" = "frozenlake" ] && \
   [ "$n_eval_off" -ne 1 ]; then
  echo "[run] FATAL: admitted P33 FrozenLake did not attest evaluation disabled exactly once" >&2
  exit 1
fi
if [ "${CANON_P35_ENVELOPE:-0}" = "1" ]; then
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
elif [ "$rc" -eq 0 ] && [ "${CANON_P34_DEEPSWE:-0}" = "1" ]; then
  classification="$CANON_STATE/p34_deepswe_${CANON_P34_RUN_STAGE}.classification.json"
  JAX_PLATFORMS=cpu PYTHONPATH="$CANON_PKG/..:${PYTHONPATH:-}" \
    python3 "$CANON_PKG/tests/p34_deepswe/classify_run.py" \
      --stage "$CANON_P34_RUN_STAGE" \
      --run-log "$LOG" \
      --update-report "$CANON_UPDATE_REPORT" \
      --alignment-report "$CANON_ALIGN_REPORT" \
      --output "$classification" || exit 1
elif [ "$rc" -eq 0 ] && [ "${CANON_P33_WORKLOAD_LAUNCH_ADMITTED:-0}" = "1" ]; then
  classification="$CANON_STATE/p33_${CANON_P32_WORKLOAD}_${CANON_P33_RUN_STAGE}.classification.json"
  JAX_PLATFORMS=cpu PYTHONPATH="$CANON_PKG/..:${PYTHONPATH:-}" \
    python3 "$CANON_PKG/tests/p33_workloads/classify_run.py" \
      --workload "$CANON_P32_WORKLOAD" \
      --stage "$CANON_P33_RUN_STAGE" \
      --run-log "$LOG" \
      --pre-alignment-report "$CANON_PRE_ALIGN_REPORT" \
      --update-report "$CANON_UPDATE_REPORT" \
      --alignment-report "$CANON_ALIGN_REPORT" \
      --output "$classification" || exit 1
fi
if [ -n "${CANON_GCS_CACHE_BUCKET:-}" ] && [ -d "${JAX_COMPILATION_CACHE_DIR:-}" ]; then
  PROFILE_NAME="$(basename "${CANON_PROFILE_FILE:-default}" .env)"
  GCS_PATH="${CANON_GCS_CACHE_BUCKET}/${PROFILE_NAME}"
  echo "[cache] Syncing persistent compilation cache back to $GCS_PATH..."
  if command -v gcloud >/dev/null 2>&1; then
    gcloud storage rsync -r "$JAX_COMPILATION_CACHE_DIR" "$GCS_PATH" 2>/dev/null || true
  elif command -v gsutil >/dev/null 2>&1; then
    gsutil -m rsync -r "$JAX_COMPILATION_CACHE_DIR" "$GCS_PATH" 2>/dev/null || true
  fi
fi
exit "$rc"

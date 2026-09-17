#!/bin/bash
# =============================================================================
# raiden_smoke_test/run.sh -- launch / watch / tear down the smoke test
# =============================================================================
#   bash raiden_smoke_test/run.sh dryrun   # render + assert manifests, no cluster
#   bash raiden_smoke_test/run.sh start    # launch and start log streamers
#   bash raiden_smoke_test/run.sh status   # pods + gate progress at a glance
#   bash raiden_smoke_test/run.sh watch    # follow the orchestrator, de-noised
#   bash raiden_smoke_test/run.sh stop     # tear down (guarded, see below)
#
# Wall clock: ~9 min TPU provisioning, ~5 min startup+first sync, ~9 min for
# 2 steps, ~4 min final checkpoint finalize.
# =============================================================================
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
c_require_kubectl

MODE="${1:-start}"

# --- Run identity -----------------------------------------------------------
# On start we mint a RUN_ID and persist it. Every other mode READS it. The old
# script regenerated the ID (and rewrote the saved env) on every invocation, so
# a later `stop` silently overwrote the provenance record of the run that had
# just succeeded.
if [[ "$MODE" == "start" || "$MODE" == "dryrun" ]]; then
  RUN_ID="${RUN_ID:-smoke-$(date +%m%d-%H%M)}"
else
  if [[ -z "${RUN_ID:-}" ]]; then
    [[ -f "$LAST_RUN_POINTER" ]] || c_die "no previous run found; pass RUN_ID=<id> explicitly."
    RUN_ID="$(cat "$LAST_RUN_POINTER")"
  fi
fi
LOGDIR="${LOG_ROOT}/${RUN_ID}"
PIDFILE="/tmp/raiden_smoke_streamers_${USER}.pids"
export MAXTEXT_OUTPUT_DIR="${GCS_OUTPUT_ROOT}/${RUN_ID}"

CKPT_URI="${MAXTEXT_OUTPUT_DIR}/${USER}-train/checkpoints"

# =============================================================================
# stop
# =============================================================================
if [[ "$MODE" == "stop" ]]; then
  echo "Tearing down ${RUN_ID}"
  # Bare HEAD has orchestrator _STOP_TIMEOUT_S = 60 s while the final save takes
  # ~236 s. The orchestrator gives up waiting but the trainer keeps finalizing,
  # so the checkpoint normally survives -- unless teardown wins the race.
  last_step="${MAX_STEPS}"
  if ! gsutil -q stat "${CKPT_URI}/${last_step}/commit_success.txt" 2>/dev/null; then
    c_warn "checkpoints/${last_step}/commit_success.txt does NOT exist yet."
    c_warn "Tearing down now can destroy the final checkpoint (save takes ~236 s"
    c_warn "and the orchestrator only waits 60 s). Set FORCE_STOP=1 to override."
    [[ "${FORCE_STOP:-0}" == "1" ]] || c_die "refusing to stop. Re-run with FORCE_STOP=1 if intentional."
  else
    c_info "checkpoints/${last_step}/commit_success.txt present -- safe to stop"
  fi

  bash "$LAUNCHER" stop
  if [[ -f "$PIDFILE" ]]; then
    while read -r spid; do
      [[ "$spid" =~ ^[0-9]+$ ]] && { kill -TERM -"$spid" 2>/dev/null || kill -TERM "$spid" 2>/dev/null; }
    done < "$PIDFILE"
    rm -f "$PIDFILE"
  fi
  # Bracket trick: a bare pattern matches this pkill's own cmdline and SIGTERMs
  # the calling shell.
  pkill -f "[w]hile true.*kubectl logs -f" 2>/dev/null || true
  sleep 1
  pkill -f "[k]ubectl logs -f .*${K8S_NAMESPACE}" 2>/dev/null || true
  echo "torn down. Logs kept at ${LOGDIR}"
  exit 0
fi

# =============================================================================
# status / watch -- read-only, safe at any time
# =============================================================================
if [[ "$MODE" == "status" ]]; then
  echo "=== run ${RUN_ID} ==="
  kubectl get pods -n "${K8S_NAMESPACE}" 2>/dev/null | sed -n "1p;/${USER}/p"
  echo
  echo "=== progress ==="
  [[ -f "${LOGDIR}/orch.log" ]] && sed -n \
    '/Weight synchronization complete/p;/Train step/p;/Step . starting/p;/GRPO Training Finished/p;/EXIT_CODE/p;/Traceback/p;/Error/p' \
    "${LOGDIR}/orch.log" | sort -u | tail -20
  echo
  echo "=== checkpoints ==="
  gsutil ls "${CKPT_URI}/" 2>/dev/null || echo "  (none yet)"
  echo
  echo "=== log sizes (all must be non-zero once running) ==="
  ls -lh "${LOGDIR}"/*.log 2>/dev/null || echo "  (no logs)"
  exit 0
fi

if [[ "$MODE" == "watch" ]]; then
  [[ -f "${LOGDIR}/orch.log" ]] || c_die "no orch.log at ${LOGDIR}"
  tail -f "${LOGDIR}/orch.log" | sed -u \
    -e '/HTTP Request/d' -e '/receive_response/d' -e '/send_request/d' \
    -e '/response_closed/d' -e '/cache_key/d' -e '/COMPILATION CACHE/d' \
    -e '/_cygrpc/d' -e '/XLA compilation/d' -e '/get_compile_options/d' \
    -e '/Finished tracing/d' -e '/jaxpr to MLIR/d' -e '/get_executable/d'
  exit 0
fi

# =============================================================================
# dryrun / start
# =============================================================================
mkdir -p "$LOGDIR"
DRY_RUN=true bash "$LAUNCHER" start > "${LOGDIR}/manifests.yaml" 2>&1

fail=0
check() {
  local want="$1" needle="$2" got
  got=$(sed -n "\#${needle}#p" "${LOGDIR}/manifests.yaml" | sed '/^[[:space:]]*#/d' | wc -l)
  if [[ "$got" -ne "$want" ]]; then
    printf '  FAIL  expected %sx, got %sx: %s\n' "$want" "$got" "$needle"; fail=1
  else
    printf '  ok    %sx  %s\n' "$got" "$needle"
  fi
}

echo "asserting rendered manifest:"
check 1 "memory: ${PATHWAYS_PROXY_MEMORY_LIMIT}"
check 1 "memory: ${USER_CONTAINER_MEMORY}"
check 1 "memory: ${USER_CONTAINER_MEMORY_LIMIT}"
check 1 "memory: ${PATHWAYS_WORKER_MEMORY}"
check 1 "sampler_type=${SAMPLER}"
check 1 "sampler=${SAMPLER}"
check 3 "max_response_length=${MAX_RESPONSE_LENGTH}"
# 1x, not 2x: upstream removed --checkpoint_save_interval_steps from the
# orchestrator (run_gsm8k_dist_grpo has no add_argument for it). It now lives
# solely on the trainer, where Orbax consumes it.
check 1 "checkpoint_save_interval_steps=${CHECKPOINT_SAVE_INTERVAL_STEPS}"
check 1 "ENABLE_PATHWAYS_PERSISTENCE=${ENABLE_PATHWAYS_PERSISTENCE}"
check 1 "RAIDEN_USE_FFI=1"                      # trainer (Pathways) only
check 1 "RAIDEN_USE_FFI=0"                      # rollout (mcJAX) only
check 1 "mesh_fsdp=${ROLLOUT_MESH_FSDP}"        # rollout dp
check 1 "prefuse_moe_weights=${PREFUSE_MOE_WEIGHTS}"  # rollout only
# Guards the two upstream breaking changes this config works around.
# 2x is correct: --mini_batch_size goes to BOTH the orchestrator
# (run_gsm8k_dist_grpo.main, which enforces batch_size % mini_batch_size == 0)
# and the trainer (run_trainer_node.main, which derives
# update_trajectories = mini_batch_size * num_generations).
check 2 "mini_batch_size=${MINI_BATCH_SIZE}"
if [[ "${USE_ROLLOUT_LOGPS}" == "false" ]]; then
  check 1 "no-use_rollout_logps"
fi
# NOT asserted: `restartPolicy: Always`. Bare HEAD renders OnFailure; the
# auto-restart patch was never upstreamed.

[[ "$fail" -eq 0 ]] || c_die "manifest assertion failed -- not launching."

if [[ "$MODE" == "dryrun" ]]; then
  echo "dryrun ok."
  exit 0
fi

# --- launch -----------------------------------------------------------------
echo
echo "Launching ${RUN_ID}"
echo "  image  ${TUNIX_IMAGE}"
echo "  gcs    ${MAXTEXT_OUTPUT_DIR}"
echo "  logs   ${LOGDIR}"

# Persist provenance BEFORE launching, and never rewrite it afterwards.
{ echo "# ${RUN_ID}  launched $(date -u +%FT%TZ)"
  echo "# image: ${TUNIX_IMAGE}"
  for v in TUNIX_IMAGE MAXTEXT_OUTPUT_DIR PROJECT CLUSTER LOCATION_NAME K8S_NAMESPACE \
           PATHWAYS_SERVER_IMAGE PATHWAYS_PROXY_IMAGE TRAINER_BACKEND MAXTEXT_MODEL_NAME \
           MODEL_ID MAXTEXT_CKPT TRAINER_TPU_SLICE TRAINER_MESH_FSDP ROLLOUT_TPU_SLICE \
           ROLLOUT_MESH_FSDP ROLLOUT_MESH_TP SAMPLER WEIGHT_SYNC_MODE PREFUSE_MOE_WEIGHTS \
           ENABLE_PATHWAYS_PERSISTENCE MAX_STEPS CHECKPOINT_SAVE_INTERVAL_STEPS BATCH_SIZE \
           NUM_GENERATIONS TRAIN_MICRO_BATCH_SIZE MINI_BATCH_SIZE USE_ROLLOUT_LOGPS \
           VERIFY_WEIGHTS DEBUG; do
    printf 'export %s=%q\n' "$v" "${!v}"
  done
} > "${LOGDIR}/run_env.sh"
echo "$RUN_ID" > "$LAST_RUN_POINTER"

bash "$LAUNCHER" start || c_die "launcher failed"

# --- log streamers ----------------------------------------------------------
# Two bugs the previous script had:
#   1. `kubectl logs job/<trainer>` attaches to the pathways-proxy container, so
#      every [TrainerNode] line -- checksums, devices_per_host, all checkpoint
#      activity -- was missing. The trainer needs `-c main`.
#   2. Streamers could silently fail to start, leaving a 0-byte log nobody
#      noticed until the run was over. We now verify and retry.
rm -f "$PIDFILE"
_stream() {  # role container
  local role="$1" container="$2"
  setsid bash -c "
    echo \$\$ >> '${PIDFILE}'
    while true; do
      kubectl logs -f job/${USER}-${role}-proc-0 -n '${K8S_NAMESPACE}' \
        ${container:+-c ${container}} >> '${LOGDIR}/${role}.log' 2>/dev/null
      sleep 3
    done
  " </dev/null >/dev/null 2>&1 &
}
_stream orch  ""
_stream train "main"
_stream roll  ""

echo -n "waiting for log streamers"
for _ in $(seq 1 40); do
  sleep 15; echo -n "."
  missing=()
  for role in orch train roll; do
    [[ -s "${LOGDIR}/${role}.log" ]] || missing+=("$role")
  done
  [[ "${#missing[@]}" -eq 0 ]] && break
done
echo
if [[ "${#missing[@]}" -ne 0 ]]; then
  c_warn "restarting streamers with no output yet: ${missing[*]}"
  for role in "${missing[@]}"; do
    [[ "$role" == "train" ]] && _stream train "main" || _stream "$role" ""
  done
fi
for role in orch train roll; do
  printf '  %-6s %s\n' "$role" "$( [[ -s "${LOGDIR}/${role}.log" ]] && echo OK || echo 'STILL EMPTY (pods may still be Pending)' )"
done

cat <<EOF

Launched ${RUN_ID}.
  bash raiden_smoke_test/run.sh status    # progress
  bash raiden_smoke_test/run.sh watch     # follow orchestrator

Do NOT stop until checkpoints/${MAX_STEPS}/commit_success.txt exists.
EOF

#!/bin/bash
# ==============================================================================
# GKE Cluster & Sandbox Monitor for Tunix / Trellis Workloads
# ==============================================================================
# Usage:
#   ./scripts/monitor_sandboxes.sh          # Runs continuous monitor every 60s
#   ./scripts/monitor_sandboxes.sh --once   # Runs once and exits
#   INTERVAL=30 ./scripts/monitor_sandboxes.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REGION="${REGION:-us-east1}"
export CLUSTER="${CLUSTER:-bodaborg-tpu7x-gsc-elm}"
export NAMESPACE="${NAMESPACE:-priority-dev-scheduled}"
export JOB_FILTER="${JOB_FILTER:-tr-}"
export INTERVAL="${INTERVAL:-60}"
export PYTHONUNBUFFERED=1

python3 "${SCRIPT_DIR}/monitor_sandboxes.py" "$@"

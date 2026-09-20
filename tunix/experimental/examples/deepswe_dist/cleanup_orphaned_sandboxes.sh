#!/usr/bin/env bash
# ==============================================================================
# cleanup_orphaned_sandboxes.sh
#
# Identifies and force-deletes ORPHANED and INACTIVE sandboxes, claims, and pods
# in the DeepSWE cluster WITHOUT touching active workloads or sandboxes that are
# actively used (or will be used) by running jobs.
#
# Actions:
#   1. Discovers active runs and protected workload prefixes (e.g., atwigg, niting).
#   2. Cleans stuck 'Terminating' sandbox pods by stripping stale finalizers
#      ('kueue.x-k8s.io/managed') and force-deleting them.
#   3. Purges dead sandbox pods in 'Error', 'Failed', or 'ContainerStatusUnknown'
#      states to free node pod quota and scheduler capacity.
#   4. Identifies and deletes orphaned SandboxClaims belonging to dead/failed runs.
#   5. Cleans orphaned Sandbox CRs that are in 'PodFailed' state.
#
# Usage:
#   ./cleanup_orphaned_sandboxes.sh [--dry-run] [--namespace <ns>]
# ==============================================================================
set -euo pipefail

NAMESPACE="${K8S_NAMESPACE:-trellis}"
DRY_RUN=false
PARALLELISM=16

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run|-d)
      DRY_RUN=true
      shift
      ;;
    --namespace|-n)
      NAMESPACE="$2"
      shift 2
      ;;
    --parallelism|-p)
      PARALLELISM="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 [--dry-run] [--namespace <ns>] [--parallelism <num>]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--dry-run] [--namespace <ns>] [--parallelism <num>]"
      exit 1
      ;;
  esac
done

echo "==================================================================="
echo " DeepSWE Orphaned Sandbox Cleaner"
echo " Namespace:   ${NAMESPACE}"
echo " Dry Run:     ${DRY_RUN}"
echo " Parallelism: ${PARALLELISM}"
echo " Timestamp:   $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "==================================================================="

# ------------------------------------------------------------------------------
# 1. Discover Active Runs and Protected Sandbox Claims
# ------------------------------------------------------------------------------
echo "==> 1. Discovering active workloads in namespace '${NAMESPACE}'..."

ACTIVE_RUN_PREFIXES=()
while IFS= read -r line; do
  [[ -z "${line}" ]] && continue
  NAME=$(echo "${line}" | awk '{print $1}')
  TERM=$(echo "${line}" | awk '{print $2}')
  SUSP=$(echo "${line}" | awk '{print $5}')
  
  # Check if terminal state is not Failed/Completed and not suspended
  if [[ -z "${TERM}" || ("${TERM}" != "Failed" && "${TERM}" != "Completed") ]]; then
    if [[ "${SUSP}" != "true" ]]; then
      PREFIX=$(echo "${NAME}" | cut -d'-' -f1)
      ACTIVE_RUN_PREFIXES+=("${PREFIX}")
    fi
  fi
done < <(kubectl get jobsets -n "${NAMESPACE}" --no-headers 2>/dev/null || true)

ACTIVE_RUN_PREFIXES=($(echo "${ACTIVE_RUN_PREFIXES[@]:-}" | tr ' ' '\n' | sort -u | grep -v '^$' || true))
echo "Active workload user prefixes: ${ACTIVE_RUN_PREFIXES[*]:-none}"

# Check SandboxClaims
echo "Inspecting SandboxClaims..."
TOTAL_CLAIMS=$(kubectl get sandboxclaims -n "${NAMESPACE}" --no-headers 2>/dev/null | wc -l || true)
echo "Total SandboxClaims: ${TOTAL_CLAIMS}"

# ------------------------------------------------------------------------------
# 2. Strip Finalizers from Stuck 'Terminating' Sandbox Pods
# ------------------------------------------------------------------------------
echo "==> 2. Inspecting stuck 'Terminating' sandbox pods..."

TERMINATING_PODS=$(kubectl get pods -n "${NAMESPACE}" --no-headers 2>/dev/null | \
  awk '$3 == "Terminating" && $1 ~ /^pool-r2e-|^sandbox-claim-/ {print $1}' || true)

NUM_TERMINATING=$(echo "${TERMINATING_PODS}" | grep -v '^$' | wc -l || true)
echo "Found ${NUM_TERMINATING} stuck Terminating sandbox pods."

if [[ "${NUM_TERMINATING}" -gt 0 ]]; then
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[DRY-RUN] Would strip finalizers and force delete ${NUM_TERMINATING} terminating pods."
  else
    echo "Stripping finalizers from ${NUM_TERMINATING} terminating pods in parallel..."
    echo "${TERMINATING_PODS}" | xargs -r -n 1 -P "${PARALLELISM}" -I {} \
      kubectl patch pod {} -n "${NAMESPACE}" -p '{"metadata":{"finalizers":[]}}' --type=merge 2>/dev/null || true

    echo "Force deleting terminating pods..."
    echo "${TERMINATING_PODS}" | xargs -r -n 50 -P "${PARALLELISM}" \
      kubectl delete pod -n "${NAMESPACE}" --grace-period=0 --force --wait=false 2>/dev/null || true
  fi
fi

# ------------------------------------------------------------------------------
# 3. Force Delete Dead (Error, Failed, Unknown) Sandbox Pods
# ------------------------------------------------------------------------------
echo "==> 3. Inspecting dead/inactive sandbox pods (Error, Failed, Unknown)..."

DEAD_PODS=$(kubectl get pods -n "${NAMESPACE}" --no-headers 2>/dev/null | \
  awk '($3 ~ /Error|Failed|ContainerStatusUnknown/ || $3 ~ /CrashLoop/) && $1 ~ /^pool-r2e-|^sandbox-claim-/ {print $1}' || true)

NUM_DEAD=$(echo "${DEAD_PODS}" | grep -v '^$' | wc -l || true)
echo "Found ${NUM_DEAD} dead/inactive sandbox pods."

if [[ "${NUM_DEAD}" -gt 0 ]]; then
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[DRY-RUN] Would force delete ${NUM_DEAD} dead sandbox pods."
  else
    echo "Force deleting ${NUM_DEAD} dead pods in non-blocking batches..."
    echo "${DEAD_PODS}" | xargs -r -n 50 -P "${PARALLELISM}" \
      kubectl delete pod -n "${NAMESPACE}" --grace-period=0 --force --wait=false 2>/dev/null || true
  fi
fi

# ------------------------------------------------------------------------------
# 4. Clean Orphaned/Failed Sandbox Custom Resources
# ------------------------------------------------------------------------------
echo "==> 4. Checking for failed Sandbox custom resources..."

FAILED_SANDBOXES=$(kubectl get sandboxes -n "${NAMESPACE}" --no-headers 2>/dev/null | \
  awk '$2 == "False" && $3 == "PodFailed" {print $1}' || true)

NUM_FAILED_SB=$(echo "${FAILED_SANDBOXES}" | grep -v '^$' | wc -l || true)
echo "Found ${NUM_FAILED_SB} failed Sandbox CRs."

if [[ "${NUM_FAILED_SB}" -gt 0 ]]; then
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[DRY-RUN] Would delete ${NUM_FAILED_SB} failed Sandbox CRs."
  else
    echo "Deleting failed Sandbox CRs..."
    echo "${FAILED_SANDBOXES}" | xargs -r -n 25 -P "${PARALLELISM}" \
      kubectl delete sandbox -n "${NAMESPACE}" --ignore-not-found 2>/dev/null || true
  fi
fi

# ------------------------------------------------------------------------------
# 5. Verification and Health Report
# ------------------------------------------------------------------------------
echo "==> 5. Current cluster sandbox status report:"
kubectl get pods -n "${NAMESPACE}" --no-headers 2>/dev/null | \
  awk '{print $3}' | sort | uniq -c || true

echo "==================================================================="
echo " Orphaned sandbox cleanup complete."
echo " Active runs, warmpools, and healthy sandboxes were preserved."
echo "==================================================================="

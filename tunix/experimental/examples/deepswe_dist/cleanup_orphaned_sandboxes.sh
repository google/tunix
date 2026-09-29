#!/usr/bin/env bash
# ==============================================================================
# cleanup_orphaned_sandboxes.sh
#
# Identifies and force-deletes ORPHANED and INACTIVE sandboxes, claims, warmpools,
# and templates in the DeepSWE cluster WITHOUT hardcoding pod name prefixes.
#
# Scoping & Targeting:
#   - By default, targets all agent sandbox resources in the namespace.
#   - When --nodepool <name> is specified, restricts pod and template operations
#     specifically to that CPU nodepool (e.g., sandbox-np or sandbox-cpu-pool).
#
# Actions:
#   1. Discovers active workload run prefixes (from non-terminal JobSets and pods).
#   2. Identifies sandbox pods using standard label 'app=agent-sandbox-rl' (or on
#      the target nodepool) without hardcoding naming schemes (pool-r2e-, pool-oh-, etc.).
#   3. Cleans stuck 'Terminating' sandbox pods by stripping finalizers.
#   4. Purges dead sandbox pods ('Error', 'Failed', 'ContainerStatusUnknown', 'CrashLoopBackOff').
#   5. Deletes orphaned running sandbox pods whose parent run is no longer active.
#   6. Cleans orphaned SandboxWarmPools and SandboxTemplates configured for the nodepool.
#   7. Purges orphaned/unbound SandboxClaims.
#   8. Cleans failed Sandbox CRs ('PodFailed').
#
# Usage:
#   ./cleanup_orphaned_sandboxes.sh [--nodepool <pool>] [--namespace <ns>] [--dry-run]
# ==============================================================================
set -euo pipefail

NAMESPACE="${K8S_NAMESPACE:-trellis}"
NODEPOOL=""
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
      if [[ $# -lt 2 ]]; then
        echo "Error: --namespace requires an argument" >&2
        exit 1
      fi
      NAMESPACE="$2"
      shift 2
      ;;
    --nodepool|--node-pool|-p)
      if [[ $# -lt 2 ]]; then
        echo "Error: --nodepool requires an argument" >&2
        exit 1
      fi
      NODEPOOL="$2"
      shift 2
      ;;
    --parallelism)
      if [[ $# -lt 2 ]]; then
        echo "Error: --parallelism requires an argument" >&2
        exit 1
      fi
      PARALLELISM="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 [--nodepool <pool>] [--namespace <ns>] [--dry-run] [--parallelism <num>]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--nodepool <pool>] [--namespace <ns>] [--dry-run] [--parallelism <num>]"
      exit 1
      ;;
  esac
done

echo "==================================================================="
echo " DeepSWE Orphaned Sandbox Cleaner"
echo " Namespace:   ${NAMESPACE}"
echo " Node Pool:   ${NODEPOOL:-all}"
echo " Dry Run:     ${DRY_RUN}"
echo " Parallelism: ${PARALLELISM}"
echo " Timestamp:   $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "==================================================================="

# ------------------------------------------------------------------------------
# 1. Discover Active Runs and Protected Workload Prefixes
# ------------------------------------------------------------------------------
echo "==> 1. Discovering active workloads in namespace '${NAMESPACE}'..."

ACTIVE_RUN_PREFIXES=()

# Extract from non-terminal, non-suspended JobSets
while IFS= read -r line; do
  [[ -z "${line}" ]] && continue
  NAME=$(echo "${line}" | awk '{print $1}')
  TERM=$(echo "${line}" | awk '{print $2}')
  SUSP=$(echo "${line}" | awk '{print $5}')
  
  if [[ -z "${TERM}" || ("${TERM}" != "Failed" && "${TERM}" != "Completed") ]]; then
    if [[ "${SUSP}" != "true" ]]; then
      # Strip role suffixes: <prefix>-orch, <prefix>-train, <prefix>-roll(-<idx>)
      PREFIX=$(echo "${NAME}" | sed -E 's/-(orch|train|roll(-[0-9]+)?)$//')
      ACTIVE_RUN_PREFIXES+=("${PREFIX}")
    fi
  fi
done < <(kubectl get jobsets -n "${NAMESPACE}" --no-headers 2>/dev/null || true)

# Extract from running/pending workload pods (orch, train, roll)
while IFS= read -r pod_name; do
  [[ -z "${pod_name}" ]] && continue
  PREFIX=$(echo "${pod_name}" | sed -E 's/-(orch|train|roll(-[0-9]+)?)-.*$//')
  ACTIVE_RUN_PREFIXES+=("${PREFIX}")
done < <(kubectl get pods -n "${NAMESPACE}" --no-headers 2>/dev/null | \
  awk '($3 == "Running" || $3 == "Pending") && $1 !~ /^(pool-|sandbox-claim-)/ && ($1 ~ /-orch-/ || $1 ~ /-roll-/ || $1 ~ /-train-/) {print $1}' || true)

ACTIVE_RUN_PREFIXES=($(echo "${ACTIVE_RUN_PREFIXES[@]:-}" | tr ' ' '\n' | sort -u | grep -v '^$' || true))
echo "Active workload run prefixes: ${ACTIVE_RUN_PREFIXES[*]:-none}"

# ------------------------------------------------------------------------------
# 2. Collect Nodes in Target Nodepool (if specified)
# ------------------------------------------------------------------------------
NODE_FILTER_ARGS=()
if [[ -n "${NODEPOOL}" ]]; then
  echo "==> 2. Querying nodes in nodepool '${NODEPOOL}'..."
  TARGET_NODES=($(kubectl get nodes -l "cloud.google.com/gke-nodepool=${NODEPOOL}" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null || true))
  echo "Found ${#TARGET_NODES[@]} node(s) in nodepool '${NODEPOOL}'."
  if [[ ${#TARGET_NODES[@]} -eq 0 ]]; then
    echo "Warning: No nodes found for nodepool '${NODEPOOL}'. Exiting."
    exit 0
  fi
fi

# Helper function to check if a pod name matches any active workload prefix
is_active_run() {
  local target_name="$1"
  for pfx in "${ACTIVE_RUN_PREFIXES[@]}"; do
    if [[ "${target_name}" == *"-${pfx}-"* || "${target_name}" == "${pfx}-"* ]]; then
      return 0
    fi
  done
  return 1
}

# ------------------------------------------------------------------------------
# 3. Clean Stuck 'Terminating' Sandbox Pods (Strip Finalizers)
# ------------------------------------------------------------------------------
echo "==> 3. Inspecting sandbox pods stuck in deletion..."

# Query pods with app=agent-sandbox-rl that have deletionTimestamp
TERMINATING_PODS_JSON=$(kubectl get pods -n "${NAMESPACE}" -l "app=agent-sandbox-rl" \
  -o jsonpath='{range .items[?(@.metadata.deletionTimestamp)]}{.metadata.name}{"\t"}{.spec.nodeName}{"\n"}{end}' 2>/dev/null || true)

TERMINATING_PODS=()
while IFS=$'\t' read -r pod_name node_name; do
  [[ -z "${pod_name}" ]] && continue
  if [[ -n "${NODEPOOL}" ]]; then
    if [[ ! " ${TARGET_NODES[*]} " =~ " ${node_name} " ]]; then
      continue
    fi
  fi
  TERMINATING_PODS+=("${pod_name}")
done <<< "${TERMINATING_PODS_JSON}"

NUM_TERMINATING=${#TERMINATING_PODS[@]}
echo "Found ${NUM_TERMINATING} sandbox pod(s) stuck in deletion."

if [[ "${NUM_TERMINATING}" -gt 0 ]]; then
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[DRY-RUN] Would strip finalizers from ${NUM_TERMINATING} stuck terminating pod(s)."
  else
    echo "Stripping finalizers from ${NUM_TERMINATING} pod(s) in parallel to purge from etcd..."
    printf '%s\n' "${TERMINATING_PODS[@]}" | xargs -r -n 1 -P "${PARALLELISM}" -I {} \
      kubectl patch pod {} -n "${NAMESPACE}" -p '{"metadata":{"finalizers":[]}}' --type=merge 2>/dev/null || true
  fi
fi

# ------------------------------------------------------------------------------
# 4. Clean Dead (Error, Failed, CrashLoop, Unknown) Sandbox Pods
# ------------------------------------------------------------------------------
echo "==> 4. Inspecting dead/inactive sandbox pods..."

DEAD_PODS_JSON=$(kubectl get pods -n "${NAMESPACE}" -l "app=agent-sandbox-rl" -o wide --no-headers 2>/dev/null | \
  awk '$3 ~ /Error|Failed|ContainerStatusUnknown|CrashLoop/ {print $1 "\t" $7}' || true)

DEAD_PODS=()
while IFS=$'\t' read -r pod_name node_name; do
  [[ -z "${pod_name}" ]] && continue
  if [[ -n "${NODEPOOL}" ]]; then
    if [[ ! " ${TARGET_NODES[*]} " =~ " ${node_name} " ]]; then
      continue
    fi
  fi
  DEAD_PODS+=("${pod_name}")
done <<< "${DEAD_PODS_JSON}"

NUM_DEAD=${#DEAD_PODS[@]}
echo "Found ${NUM_DEAD} dead/inactive sandbox pod(s)."

if [[ "${NUM_DEAD}" -gt 0 ]]; then
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[DRY-RUN] Would force-delete and strip finalizers from ${NUM_DEAD} dead sandbox pod(s)."
  else
    echo "Force deleting ${NUM_DEAD} dead pod(s) in non-blocking batches..."
    printf '%s\n' "${DEAD_PODS[@]}" | xargs -r -n 50 -P "${PARALLELISM}" \
      kubectl delete pod -n "${NAMESPACE}" --grace-period=0 --force --wait=false 2>/dev/null || true

    printf '%s\n' "${DEAD_PODS[@]}" | xargs -r -n 1 -P "${PARALLELISM}" -I {} \
      kubectl patch pod {} -n "${NAMESPACE}" -p '{"metadata":{"finalizers":[]}}' --type=merge 2>/dev/null || true
  fi
fi

# ------------------------------------------------------------------------------
# 5. Clean Orphaned Sandbox WarmPools and Templates (Nodepool-scoped)
# ------------------------------------------------------------------------------
echo "==> 5. Inspecting orphaned SandboxWarmPools and SandboxTemplates..."

ORPHAN_WARMPOOLS=()
while IFS= read -r wp_name; do
  [[ -z "${wp_name}" ]] && continue
  if ! is_active_run "${wp_name}"; then
    ORPHAN_WARMPOOLS+=("${wp_name}")
  fi
done < <(kubectl get sandboxwarmpools -n "${NAMESPACE}" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null | tr ' ' '\n' || true)

echo "Found ${#ORPHAN_WARMPOOLS[@]} orphaned SandboxWarmPool(s)."
if [[ ${#ORPHAN_WARMPOOLS[@]} -gt 0 ]]; then
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[DRY-RUN] Would delete ${#ORPHAN_WARMPOOLS[@]} orphaned warmpool(s)."
  else
    echo "Deleting orphaned warmpools..."
    printf '%s\n' "${ORPHAN_WARMPOOLS[@]}" | xargs -r -n 25 -P "${PARALLELISM}" \
      kubectl delete sandboxwarmpool -n "${NAMESPACE}" --ignore-not-found 2>/dev/null || true
  fi
fi

ORPHAN_TEMPLATES=()
TEMPLATES_JSON=$(kubectl get sandboxtemplates -n "${NAMESPACE}" \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.podTemplate.spec.nodeSelector.cloud\.google\.com/gke-nodepool}{"\n"}{end}' 2>/dev/null || true)

while IFS=$'\t' read -r tmpl_name tmpl_np; do
  [[ -z "${tmpl_name}" ]] && continue
  if [[ -n "${NODEPOOL}" && "${tmpl_np}" != "${NODEPOOL}" ]]; then
    continue
  fi
  if ! is_active_run "${tmpl_name}"; then
    ORPHAN_TEMPLATES+=("${tmpl_name}")
  fi
done <<< "${TEMPLATES_JSON}"

echo "Found ${#ORPHAN_TEMPLATES[@]} orphaned SandboxTemplate(s)."
if [[ ${#ORPHAN_TEMPLATES[@]} -gt 0 ]]; then
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[DRY-RUN] Would delete ${#ORPHAN_TEMPLATES[@]} orphaned template(s)."
  else
    echo "Deleting orphaned templates..."
    printf '%s\n' "${ORPHAN_TEMPLATES[@]}" | xargs -r -n 25 -P "${PARALLELISM}" \
      kubectl delete sandboxtemplate -n "${NAMESPACE}" --ignore-not-found 2>/dev/null || true
  fi
fi

# ------------------------------------------------------------------------------
# 6. Clean Orphaned / Unbound SandboxClaims
# ------------------------------------------------------------------------------
echo "==> 6. Checking for orphaned SandboxClaims..."

ORPHAN_CLAIMS=()
ACTIVE_CLAIMS=()
while IFS=$'\t' read -r claim_name wp_ref creator; do
  [[ -z "${claim_name}" ]] && continue
  is_claim_active=false
  for pfx in "${ACTIVE_RUN_PREFIXES[@]}"; do
    if [[ "${claim_name}" == *"-${pfx}-"* || "${claim_name}" == "${pfx}-"* ]]; then
      is_claim_active=true
      break
    fi
    if [[ -n "${wp_ref}" && "${wp_ref}" != "-" && ("${wp_ref}" == *"-${pfx}-"* || "${wp_ref}" == "${pfx}-"*) ]]; then
      is_claim_active=true
      break
    fi
    if [[ -n "${creator}" && "${creator}" != "-" && ("${creator}" == *"-${pfx}-"* || "${creator}" == "${pfx}-"*) ]]; then
      is_claim_active=true
      break
    fi
  done
  if [[ "${is_claim_active}" == "true" ]]; then
    ACTIVE_CLAIMS+=("${claim_name}")
  else
    ORPHAN_CLAIMS+=("${claim_name}")
  fi
done < <(kubectl get sandboxclaims -n "${NAMESPACE}" -o json 2>/dev/null | \
  jq -r '.items[] | "\(.metadata.name)\t\(.spec.warmPoolRef.name // "-")\t\(.metadata.labels["app.kubernetes.io/created-by"] // "-")"' || true)

echo "Found ${#ORPHAN_CLAIMS[@]} orphaned SandboxClaim(s)."
if [[ ${#ORPHAN_CLAIMS[@]} -gt 0 ]]; then
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[DRY-RUN] Would delete ${#ORPHAN_CLAIMS[@]} orphaned SandboxClaim(s)."
  else
    echo "Deleting orphaned claims in batches..."
    printf '%s\n' "${ORPHAN_CLAIMS[@]}" | xargs -r -n 50 -P "${PARALLELISM}" \
      kubectl delete sandboxclaim -n "${NAMESPACE}" --ignore-not-found 2>/dev/null || true
  fi
fi

# ------------------------------------------------------------------------------
# 7. Clean Failed Sandbox Custom Resources
# ------------------------------------------------------------------------------
echo "==> 7. Checking for failed Sandbox custom resources..."

FAILED_SANDBOXES=$(kubectl get sandboxes -n "${NAMESPACE}" --no-headers 2>/dev/null | \
  awk '$2 == "False" && $3 == "PodFailed" {print $1}' || true)

NUM_FAILED_SB=$(echo "${FAILED_SANDBOXES}" | grep -v '^$' | wc -l || true)
echo "Found ${NUM_FAILED_SB} failed Sandbox CR(s)."

if [[ "${NUM_FAILED_SB}" -gt 0 ]]; then
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[DRY-RUN] Would delete ${NUM_FAILED_SB} failed Sandbox CR(s)."
  else
    echo "Deleting failed Sandbox CRs..."
    echo "${FAILED_SANDBOXES}" | xargs -r -n 25 -P "${PARALLELISM}" \
      kubectl delete sandbox -n "${NAMESPACE}" --ignore-not-found 2>/dev/null || true
  fi
fi

# ------------------------------------------------------------------------------
# 8. Clean Orphaned Running Pods on Target Nodepool
# ------------------------------------------------------------------------------
echo "==> 8. Checking for orphaned running sandbox pods..."

ORPHAN_RUNNING_PODS=()
while IFS=$'\t' read -r pod_name phase node_name creator sandbox; do
  [[ -z "${pod_name}" ]] && continue
  if [[ "${phase}" != "Running" ]]; then
    continue
  fi
  if [[ -n "${NODEPOOL}" ]]; then
    if [[ ! " ${TARGET_NODES[*]} " =~ " ${node_name} " ]]; then
      continue
    fi
  fi
  
  is_pod_active=false
  # 1. Protected if belonging to an active SandboxClaim
  if [[ " ${ACTIVE_CLAIMS[*]:-} " =~ " ${pod_name} " ]]; then
    is_pod_active=true
  fi

  # 2. Protected if name, creator, or sandbox label matches any active run prefix
  if [[ "${is_pod_active}" == "false" ]]; then
    for pfx in "${ACTIVE_RUN_PREFIXES[@]}"; do
      if [[ "${pod_name}" == *"-${pfx}-"* || "${pod_name}" == "${pfx}-"* ]]; then
        is_pod_active=true
        break
      fi
      if [[ -n "${creator}" && "${creator}" != "-" && ("${creator}" == *"-${pfx}-"* || "${creator}" == "${pfx}-"*) ]]; then
        is_pod_active=true
        break
      fi
      if [[ -n "${sandbox}" && "${sandbox}" != "-" && ("${sandbox}" == *"-${pfx}-"* || "${sandbox}" == "${pfx}-"*) ]]; then
        is_pod_active=true
        break
      fi
    done
  fi

  if [[ "${is_pod_active}" == "false" ]]; then
    ORPHAN_RUNNING_PODS+=("${pod_name}")
  fi
done < <(kubectl get pods -n "${NAMESPACE}" -l "app=agent-sandbox-rl" -o json 2>/dev/null | \
  jq -r '.items[] | "\(.metadata.name)\t\(.status.phase // "-")\t\(.spec.nodeName // "-")\t\(.metadata.labels["app.kubernetes.io/created-by"] // "-")\t\(.metadata.labels["sandbox"] // "-")"' || true)

echo "Found ${#ORPHAN_RUNNING_PODS[@]} orphaned running sandbox pod(s)."
if [[ ${#ORPHAN_RUNNING_PODS[@]} -gt 0 ]]; then
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[DRY-RUN] Would force-delete ${#ORPHAN_RUNNING_PODS[@]} orphaned running sandbox pod(s)."
  else
    echo "Force deleting orphaned running pods..."
    printf '%s\n' "${ORPHAN_RUNNING_PODS[@]}" | xargs -r -n 50 -P "${PARALLELISM}" \
      kubectl delete pod -n "${NAMESPACE}" --grace-period=0 --force --wait=false 2>/dev/null || true
  fi
fi

# ------------------------------------------------------------------------------
# 9. Verification and Health Report
# ------------------------------------------------------------------------------
echo "==> 9. Current cluster sandbox status report:"
if [[ -n "${NODEPOOL}" ]]; then
  echo "Pods running on nodepool '${NODEPOOL}':"
  kubectl get pods -n "${NAMESPACE}" -l "app=agent-sandbox-rl" -o wide 2>/dev/null | \
    grep "${NODEPOOL}" | awk '{print $3}' | sort | uniq -c || true
else
  kubectl get pods -n "${NAMESPACE}" -l "app=agent-sandbox-rl" --no-headers 2>/dev/null | \
    awk '{print $3}' | sort | uniq -c || true
fi

echo "==================================================================="
echo " Orphaned sandbox cleanup complete."
echo " Active runs, warmpools, and healthy sandboxes were preserved."
echo "==================================================================="

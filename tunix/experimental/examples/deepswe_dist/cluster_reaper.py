#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cluster Reaper Daemon.

Continuously monitors the cluster namespace for:
1. Orphaned and stuck sandboxes:
   - Pods stuck in Terminating with finalizers (e.g. kueue.x-k8s.io/managed) -> strip finalizers.
   - Dead sandbox pods (Error, Failed, ContainerStatusUnknown, CrashLoop) -> delete and strip finalizers.
   - Orphaned SandboxClaims whose parent run has ended -> delete claims and associated resources.
   - Orphaned SandboxWarmPools and SandboxTemplates whose parent run has ended -> delete CRs.
   - Failed Sandbox CRs (status.phase == 'PodFailed') -> delete CRs.
2. Failed, crashed, or hung workloads using cluster resources:
   - JobSets with terminalState == 'Failed' -> delete JobSet and sibling workers matching run prefix.
   - Workload pods (orch, train, roll) that are marked Running by K8s but have crashed -> delete JobSet group.
   - Workload pods in CrashLoopBackOff or terminated with non-zero exit code -> delete JobSet group.
   - Failed batch/v1 Jobs -> delete Jobs.
3. Safety invariants:
   - NEVER kill healthy workloads quietly computing rollouts (no silence timeouts).
   - Only teardown on verified failure signals (fatal tracebacks, exit codes, or terminalState).
   - Never cross-kill sibling runs (uses exact run prefix extraction, not split("-")[0]).
   - Never inspect sandbox pods for tracebacks (sandboxes run user test suites with pytest tracebacks).
"""

import datetime
import logging
import os
import re
import sys
import time
try:
  from kubernetes import client, config
except ImportError:
  client = None
  config = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [cluster-reaper] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cluster-reaper")

NAMESPACE = os.environ.get("NAMESPACE", "trellis")
LOOP_INTERVAL_SECONDS = int(os.environ.get("INTERVAL_SECONDS", "30"))
SANDBOX_POD_PREFIXES = ("pool-", "sandbox-claim-")

FATAL_EXCEPTION_PATTERNS = (
    "Error:",
    "Exception:",
    "SIGSEGV",
    "Segmentation fault",
    "OutOfMemoryError",
    "ResourceExhaustedError",
    "Fatal Python error:",
    "Aborted (core dumped)",
)


def init_k8s():
  try:
    config.load_incluster_config()
  except Exception:
    config.load_kube_config()
  return client.CoreV1Api(), client.BatchV1Api(), client.CustomObjectsApi()


def extract_run_prefix(js_name: str) -> str:
  """Extracts the run prefix (JOB_PREFIX) from a JobSet name.

  DeepSWE/Tunix JobSets follow the naming conventions:
    <prefix>-orch
    <prefix>-train
    <prefix>-roll
    <prefix>-roll-<index>

  For standalone JobSets that do not follow this convention, the entire
  JobSet name is returned so it only matches itself.
  """
  match = re.match(r"^(.*?)-(orch|train|roll(-\d+)?)$", js_name)
  if match:
    return match.group(1)
  return js_name


def reap_stuck_terminating_pods(core_api):
  """Strips finalizers from any sandbox pod stuck in deletion."""
  count = 0
  try:
    pods = core_api.list_namespaced_pod(
        namespace=NAMESPACE, label_selector="app=agent-sandbox-rl"
    )
    for pod in pods.items:
      name = pod.metadata.name
      if not name.startswith(SANDBOX_POD_PREFIXES):
        continue
      if pod.metadata.deletion_timestamp is not None:
        finalizers = pod.metadata.finalizers or []
        if finalizers:
          logger.info(
              "Pod %s has deletionTimestamp with finalizers %s. Stripping finalizers...",
              name, finalizers,
          )
          try:
            core_api.patch_namespaced_pod(
                name=name,
                namespace=NAMESPACE,
                body={"metadata": {"finalizers": []}},
            )
            count += 1
          except Exception as e:
            logger.warning("Failed to strip finalizers on pod %s: %s", name, e)
  except Exception as e:
    logger.error("Error checking terminating pods: %s", e)
  return count


def reap_dead_sandbox_pods(core_api):
  """Finds dead/failed sandbox pods and force deletes them with finalizers stripped."""
  count = 0
  try:
    pods = core_api.list_namespaced_pod(
        namespace=NAMESPACE, label_selector="app=agent-sandbox-rl"
    )
    now = datetime.datetime.now(datetime.timezone.utc)
    for pod in pods.items:
      name = pod.metadata.name
      if not name.startswith(SANDBOX_POD_PREFIXES):
        continue
      if pod.metadata.deletion_timestamp is not None:
        continue

      phase = pod.status.phase
      is_dead = phase in ("Failed", "Error")

      if not is_dead and pod.status.container_statuses:
        for cs in pod.status.container_statuses:
          if cs.state.waiting and cs.state.waiting.reason in (
              "CrashLoopBackOff",
              "ImagePullBackOff",
              "ErrImagePull",
          ):
            age = (now - pod.metadata.creation_timestamp).total_seconds()
            if age > 300:
              is_dead = True
              break
          if cs.state.terminated and cs.state.terminated.exit_code != 0:
            is_dead = True
            break

      if is_dead:
        logger.info("Found dead sandbox pod %s (phase: %s). Deleting...", name, phase)
        try:
          core_api.patch_namespaced_pod(
              name=name,
              namespace=NAMESPACE,
              body={"metadata": {"finalizers": []}},
          )
          core_api.delete_namespaced_pod(
              name=name,
              namespace=NAMESPACE,
              grace_period_seconds=0,
          )
          count += 1
        except Exception as e:
          logger.warning("Failed to delete dead pod %s: %s", name, e)
  except Exception as e:
    logger.error("Error checking dead sandbox pods: %s", e)
  return count


def reap_orphaned_claims(custom_api, core_api):
  """Deletes SandboxClaims whose parent run is no longer active on the cluster.

  A SandboxClaim is considered orphaned if:
  1. No workload pods (*-roll-*, *-orch-*, *-train-*) are currently Running or Pending,
     and the claim is older than 5 minutes.
  2. Workload pods ARE active, but the claim was created BEFORE the oldest active
     workload pod started (with a 5-minute safety buffer), meaning it was left behind
     by an earlier completed or crashed workload run.
  """
  count = 0
  try:
    running_pods = core_api.list_namespaced_pod(namespace=NAMESPACE)
    workload_start_times = []
    for pod in running_pods.items:
      name = pod.metadata.name
      if name.startswith(SANDBOX_POD_PREFIXES) or name.startswith("cluster-reaper"):
        continue
      if (pod.metadata.labels or {}).get("app") == "agent-sandbox-rl":
        continue
      if pod.status.phase in ("Running", "Pending"):
        if pod.metadata.creation_timestamp:
          workload_start_times.append(pod.metadata.creation_timestamp)

    now = datetime.datetime.now(datetime.timezone.utc)
    claims = custom_api.list_namespaced_custom_object(
        group="extensions.agents.x-k8s.io",
        version="v1beta1",
        namespace=NAMESPACE,
        plural="sandboxclaims",
    )

    earliest_workload_time = min(workload_start_times) if workload_start_times else None

    for claim in claims.get("items", []):
      cname = claim["metadata"]["name"]
      created_str = claim["metadata"]["creationTimestamp"]
      created_dt = datetime.datetime.fromisoformat(created_str.replace("Z", "+00:00"))
      age_s = (now - created_dt).total_seconds()

      # Never reap claims created within the last 5 minutes (safety buffer)
      if age_s < 300:
        continue

      is_orphan = False
      reason = ""
      if earliest_workload_time is None:
        # No workloads running at all
        is_orphan = True
        reason = f"no active workload pods in namespace {NAMESPACE}"
      elif created_dt < earliest_workload_time - datetime.timedelta(minutes=5):
        # Claim was created well before current active workloads started
        is_orphan = True
        reason = (
            f"created at {created_str} before earliest running workload"
            f" ({earliest_workload_time.isoformat()})"
        )

      if is_orphan:
        logger.info(
            "Found orphaned SandboxClaim %s (age: %.1f min, reason: %s). Deleting...",
            cname,
            age_s / 60,
            reason,
        )
        try:
          custom_api.patch_namespaced_custom_object(
              group="extensions.agents.x-k8s.io",
              version="v1beta1",
              namespace=NAMESPACE,
              plural="sandboxclaims",
              name=cname,
              body={"metadata": {"finalizers": []}},
          )
          custom_api.delete_namespaced_custom_object(
              group="extensions.agents.x-k8s.io",
              version="v1beta1",
              namespace=NAMESPACE,
              plural="sandboxclaims",
              name=cname,
          )
          count += 1
        except Exception as e:
          logger.warning("Failed to delete orphaned claim %s: %s", cname, e)
  except Exception as e:
    logger.error("Error checking orphaned claims: %s", e)
  return count


def reap_orphaned_warmpools_and_templates(custom_api, core_api):
  """Cleans up SandboxWarmPool and SandboxTemplate resources whose run is gone."""
  count = 0
  try:
    # 1. Discover all active run prefixes from running or pending workload pods & jobsets
    active_prefixes = set()
    jobsets = custom_api.list_namespaced_custom_object(
        group="jobset.x-k8s.io",
        version="v1alpha2",
        namespace=NAMESPACE,
        plural="jobsets",
    )
    for js in jobsets.get("items", []):
      active_prefixes.add(extract_run_prefix(js["metadata"]["name"]))

    pods = core_api.list_namespaced_pod(namespace=NAMESPACE)
    for pod in pods.items:
      pname = pod.metadata.name
      if pname.startswith(SANDBOX_POD_PREFIXES) or pname.startswith("cluster-reaper"):
        continue
      if (pod.metadata.labels or {}).get("app") == "agent-sandbox-rl":
        continue
      js_name = (pod.metadata.labels or {}).get("jobset.sigs.k8s.io/jobset-name")
      if js_name:
        active_prefixes.add(extract_run_prefix(js_name))

    now = datetime.datetime.now(datetime.timezone.utc)

    # 2. Check SandboxWarmPools
    warmpools = custom_api.list_namespaced_custom_object(
        group="extensions.agents.x-k8s.io",
        version="v1beta1",
        namespace=NAMESPACE,
        plural="sandboxwarmpools",
    )
    for wp in warmpools.get("items", []):
      wp_name = wp["metadata"]["name"]
      created_str = wp["metadata"]["creationTimestamp"]
      created_dt = datetime.datetime.fromisoformat(created_str.replace("Z", "+00:00"))
      age_m = (now - created_dt).total_seconds() / 60

      # Safety: only reap warmpools older than 10 minutes
      if age_m < 10:
        continue

      is_active = any(f"-{prefix}-" in wp_name for prefix in active_prefixes)
      if not is_active:
        logger.info(
            "Found orphaned SandboxWarmPool %s (age: %.1fm, no active workloads for run). Deleting...",
            wp_name, age_m,
        )
        try:
          custom_api.delete_namespaced_custom_object(
              group="extensions.agents.x-k8s.io",
              version="v1beta1",
              namespace=NAMESPACE,
              plural="sandboxwarmpools",
              name=wp_name,
          )
          count += 1
        except Exception as e:
          logger.warning("Failed to delete orphaned warmpool %s: %s", wp_name, e)

    # 3. Check SandboxTemplates
    templates = custom_api.list_namespaced_custom_object(
        group="extensions.agents.x-k8s.io",
        version="v1beta1",
        namespace=NAMESPACE,
        plural="sandboxtemplates",
    )
    for tmpl in templates.get("items", []):
      tmpl_name = tmpl["metadata"]["name"]
      created_str = tmpl["metadata"]["creationTimestamp"]
      created_dt = datetime.datetime.fromisoformat(created_str.replace("Z", "+00:00"))
      age_m = (now - created_dt).total_seconds() / 60

      if age_m < 10:
        continue

      is_active = any(
          f"-{prefix}-" in tmpl_name or tmpl_name.startswith(f"oh-{prefix}-") or tmpl_name.startswith(f"{prefix}-")
          for prefix in active_prefixes
      )
      if not is_active:
        logger.info(
            "Found orphaned SandboxTemplate %s (age: %.1fm). Deleting...",
            tmpl_name, age_m,
        )
        try:
          custom_api.delete_namespaced_custom_object(
              group="extensions.agents.x-k8s.io",
              version="v1beta1",
              namespace=NAMESPACE,
              plural="sandboxtemplates",
              name=tmpl_name,
          )
          count += 1
        except Exception as e:
          logger.warning("Failed to delete orphaned template %s: %s", tmpl_name, e)

  except Exception as e:
    logger.error("Error checking orphaned warmpools and templates: %s", e)
  return count


def reap_failed_sandbox_crs(custom_api):
  """Deletes Sandbox CRs that are in PodFailed state."""
  count = 0
  try:
    sandboxes = custom_api.list_namespaced_custom_object(
        group="agents.x-k8s.io",
        version="v1beta1",
        namespace=NAMESPACE,
        plural="sandboxes",
    )
    for sb in sandboxes.get("items", []):
      sname = sb["metadata"]["name"]
      status = sb.get("status", {})
      is_failed = False
      for cond in status.get("conditions", []):
        if cond.get("reason") == "PodFailed" or (
            cond.get("type") == "Ready"
            and cond.get("status") == "False"
            and cond.get("reason") == "PodFailed"
        ):
          is_failed = True
          break
      if is_failed:
        logger.info("Found failed Sandbox CR %s. Deleting...", sname)
        try:
          custom_api.patch_namespaced_custom_object(
              group="agents.x-k8s.io",
              version="v1beta1",
              namespace=NAMESPACE,
              plural="sandboxes",
              name=sname,
              body={"metadata": {"finalizers": []}},
          )
          custom_api.delete_namespaced_custom_object(
              group="agents.x-k8s.io",
              version="v1beta1",
              namespace=NAMESPACE,
              plural="sandboxes",
              name=sname,
          )
          count += 1
        except Exception as e:
          logger.warning("Failed to delete failed sandbox CR %s: %s", sname, e)
  except Exception as e:
    logger.error("Error checking failed sandbox CRs: %s", e)
  return count


def check_pod_fatal_traceback(core_api, pod_name, container_name="main"):
  """Inspects log tail of a workload pod to detect fatal unhandled Python tracebacks.

  Returns the error string if a fatal traceback is detected, or None.
  """
  try:
    log_text = core_api.read_namespaced_pod_log(
        name=pod_name,
        namespace=NAMESPACE,
        container=container_name,
        tail_lines=35,
    )
    if not log_text or "Traceback (most recent call last):" not in log_text:
      return None

    lines = [l.strip() for l in log_text.strip().splitlines() if l.strip()]
    if not lines:
      return None

    # Check the last 10 non-empty lines for fatal exception patterns
    tail_window = lines[-10:]
    for line in reversed(tail_window):
      for pat in FATAL_EXCEPTION_PATTERNS:
        if pat in line:
          return line
  except Exception:
    pass
  return None


def reap_failed_crashed_hung_jobs(custom_api, batch_api, core_api):
  """Checks for failed, crashed (with traceback), or hung JobSets and standard Jobs."""
  count = 0
  failed_run_prefixes = set()
  deleted_jobsets = set()

  # 1. Check JobSets (terminalState == Failed)
  try:
    jobsets = custom_api.list_namespaced_custom_object(
        group="jobset.x-k8s.io",
        version="v1alpha2",
        namespace=NAMESPACE,
        plural="jobsets",
    )
    now = datetime.datetime.now(datetime.timezone.utc)
    for js in jobsets.get("items", []):
      js_name = js["metadata"]["name"]
      status = js.get("status", {})
      terminal_state = status.get("terminalState")
      restarts = status.get("restarts", 0)

      created_str = js["metadata"]["creationTimestamp"]
      created_dt = datetime.datetime.fromisoformat(created_str.replace("Z", "+00:00"))
      age_m = (now - created_dt).total_seconds() / 60

      if terminal_state == "Failed":
        prefix = extract_run_prefix(js_name)
        failed_run_prefixes.add(prefix)
        logger.info(
            "Found FAILED JobSet %s (terminalState: Failed, restarts: %s, age: %.1fm). Deleting...",
            js_name, restarts, age_m
        )
        try:
          custom_api.delete_namespaced_custom_object(
              group="jobset.x-k8s.io",
              version="v1alpha2",
              namespace=NAMESPACE,
              plural="jobsets",
              name=js_name,
          )
          deleted_jobsets.add(js_name)
          count += 1
        except Exception as e:
          logger.warning("Failed to delete JobSet %s: %s", js_name, e)

    # 2. Check Workload Pods for Unhandled Exceptions / Crashes (The Zombie Detector)
    # Only inspect non-sandbox workload pods (orchestrator, trainer, rollout)
    pods = core_api.list_namespaced_pod(namespace=NAMESPACE)
    for pod in pods.items:
      pname = pod.metadata.name
      # Skip sandboxes (they run user test code which legitimately contains tracebacks)
      if pname.startswith(SANDBOX_POD_PREFIXES):
        continue
      if (pod.metadata.labels or {}).get("app") == "agent-sandbox-rl":
        continue
      # Skip cluster-reaper itself
      if pname.startswith("cluster-reaper"):
        continue
      # Skip pods that are already terminating or succeeded
      if pod.metadata.deletion_timestamp is not None:
        continue
      if pod.status.phase == "Succeeded":
        continue

      # Must be older than 5 minutes to allow healthy startup/compilation
      age = (now - pod.metadata.creation_timestamp).total_seconds()
      if age < 300:
        continue

      # Check container status
      is_crashed = False
      crash_reason = None

      if pod.status.container_statuses:
        for cs in pod.status.container_statuses:
          if cs.state.waiting and cs.state.waiting.reason == "CrashLoopBackOff":
            is_crashed = True
            crash_reason = f"Container {cs.name} in CrashLoopBackOff"
            break
          if cs.state.terminated and cs.state.terminated.exit_code != 0:
            is_crashed = True
            crash_reason = f"Container {cs.name} terminated with exit code {cs.state.terminated.exit_code}"
            break

      if is_crashed:
        js_name = (pod.metadata.labels or {}).get("jobset.sigs.k8s.io/jobset-name")
        if js_name:
          prefix = extract_run_prefix(js_name)
          failed_run_prefixes.add(prefix)
          logger.warning(
              "DETECTED CRASHED WORKLOAD: Pod %s in JobSet %s failed: %s. Flagging run '%s' for teardown...",
              pname, js_name, crash_reason, prefix
          )

    # 3. Clean all JobSets belonging to failed/crashed run prefixes
    if failed_run_prefixes:
      for js in jobsets.get("items", []):
        js_name = js["metadata"]["name"]
        if js_name in deleted_jobsets:
          continue
        prefix = extract_run_prefix(js_name)
        if prefix in failed_run_prefixes:
          logger.info(
              "Turning down JobSet %s belonging to crashed/failed run '%s'...",
              js_name, prefix
          )
          try:
            custom_api.delete_namespaced_custom_object(
                group="jobset.x-k8s.io",
                version="v1alpha2",
                namespace=NAMESPACE,
                plural="jobsets",
                name=js_name,
            )
            deleted_jobsets.add(js_name)
            count += 1
          except Exception as e:
            logger.warning("Failed to delete JobSet %s: %s", js_name, e)

  except Exception as e:
    logger.error("Error checking JobSets: %s", e)

  # 4. Standard batch/v1 Jobs
  try:
    jobs = batch_api.list_namespaced_job(namespace=NAMESPACE)
    for job in jobs.items:
      jname = job.metadata.name
      if jname.startswith("cluster-reaper"):
        continue
      owners = job.metadata.owner_references or []
      if any(o.kind == "JobSet" for o in owners):
        continue

      jstatus = job.status
      is_failed = False
      if jstatus.failed and jstatus.failed > 0 and (not jstatus.active or jstatus.active == 0):
        is_failed = True
      if jstatus.conditions:
        for cond in jstatus.conditions:
          if cond.type == "Failed" and cond.status == "True":
            is_failed = True
            break

      if is_failed:
        logger.info("Found FAILED Job %s. Deleting to free resources...", jname)
        try:
          batch_api.delete_namespaced_job(
              name=jname, namespace=NAMESPACE, propagation_policy="Background"
          )
          count += 1
        except Exception as e:
          logger.warning("Failed to delete Job %s: %s", jname, e)
  except Exception as e:
    logger.error("Error checking standard Jobs: %s", e)

  return count


def main():
  logger.info("Starting Cluster Reaper daemon in namespace '%s'...", NAMESPACE)
  core_api, batch_api, custom_api = init_k8s()

  cycle = 0
  while True:
    cycle += 1
    t0 = time.time()
    try:
      stuck_terminating = reap_stuck_terminating_pods(core_api)
      dead_sandboxes = reap_dead_sandbox_pods(core_api)
      orphaned_claims = reap_orphaned_claims(custom_api, core_api)
      failed_crs = reap_failed_sandbox_crs(custom_api)
      failed_jobs = reap_failed_crashed_hung_jobs(custom_api, batch_api, core_api)
      orphaned_pools = reap_orphaned_warmpools_and_templates(custom_api, core_api)

      pods = core_api.list_namespaced_pod(namespace=NAMESPACE)
      running_count = sum(1 for p in pods.items if p.status.phase == "Running")
      pending_count = sum(1 for p in pods.items if p.status.phase == "Pending")
      term_count = sum(1 for p in pods.items if p.metadata.deletion_timestamp is not None)

      dt = time.time() - t0
      logger.info(
          "[Cycle %d] (%.1fs) Pods: %d running, %d pending, %d terminating. Actions: "
          "stuck_terminating=%d, dead_sandboxes=%d, orphaned_claims=%d, failed_crs=%d, failed_jobs=%d, orphaned_pools=%d",
          cycle, dt, running_count, pending_count, term_count,
          stuck_terminating, dead_sandboxes, orphaned_claims, failed_crs, failed_jobs, orphaned_pools,
      )

    except Exception as e:
      logger.error("Unexpected error during reaper cycle: %s", e, exc_info=True)

    time.sleep(LOOP_INTERVAL_SECONDS)


if __name__ == "__main__":
  main()

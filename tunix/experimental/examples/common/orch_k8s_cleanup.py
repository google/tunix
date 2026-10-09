# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Orchestrator-side teardown of the worker JobSets of a DeepSWE run.

Trainer and rollout workers are long-lived gRPC servers: they never exit on
their own, and under ``FAIL_FAST`` the PID-1 wrapper even rewrites a
post-registration exit 0 to exit 1 so that a worker that *does* exit fails its
JobSet. The only supported way to end a worker is therefore to delete its
JobSet. ``ClusterOrchestrator.shutdown()`` only sends a gRPC ``stop`` (which
drains the worker and leaves it serving), so without this module the
``<prefix>-train`` / ``<prefix>-roll*`` JobSets outlive the orchestrator and
keep their TPU slices allocated.

The teardown runs in the orchestrator pod's ``finally`` block, in this order:

1. ``cluster.shutdown()`` - graceful gRPC ``stop`` so the trainer flushes
   in-flight checkpoint saves (``TrainerWorker.stop`` -> ``trainer.close()``).
2. :func:`wait_until_workers_stopped` - poll ``heartbeat`` until every remote
   worker reports ``STOPPED`` or is unreachable. ``shutdown()`` gives up after
   a fixed timeout that a large final checkpoint can exceed; deleting the
   JobSet before the drain finishes would also delete the Pathways TPU-worker
   Job underneath it.
3. :func:`delete_sibling_jobsets` - delete the run's worker JobSets through
   the in-cluster Kubernetes API (the same credentials the orchestrator already
   uses to reap agent sandboxes).

Only the orchestrator's own namespace and the exact names derived from its
``ORCHESTRATOR_ID`` (``<prefix>-orch``) are touched; the orchestrator JobSet
itself is left alone and completes normally when the process exits.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from concurrent import futures
import logging
import os
import re
import threading
import time
from typing import Any

ORCHESTRATOR_SUFFIX = "-orch"
JOBSET_GROUP = "jobset.x-k8s.io"
JOBSET_VERSION = "v1alpha2"
JOBSET_PLURAL = "jobsets"
_SA_NAMESPACE_FILE = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
# ``tunix.experimental.common.datatypes.WorkerState.STOPPED.value``. Compared
# by value so this module does not import datatypes (and with it flax/jax).
STOPPED_STATE = "STOPPED"

# How long to wait for workers to finish draining after shutdown() before
# deleting their JobSets anyway (the pod terminationGracePeriodSeconds still
# applies after that). Overridable for very large final checkpoints.
DEFAULT_WORKER_STOP_DEADLINE_S = float(
    os.environ.get("ORCH_WORKER_STOP_DEADLINE_S", "900")
)
DEFAULT_WORKER_STOP_POLL_S = 10.0
# Bound on a single heartbeat RPC while polling; a hung worker must not pin
# the orchestrator for the handle's (possibly hours-long) RPC deadline.
_HEARTBEAT_TIMEOUT_S = 15.0


def run_prefix(orchestrator_id: str | None) -> str | None:
  """Returns ``<prefix>`` for an ``<prefix>-orch`` id, else ``None``."""
  if not orchestrator_id or not orchestrator_id.endswith(ORCHESTRATOR_SUFFIX):
    return None
  prefix = orchestrator_id[: -len(ORCHESTRATOR_SUFFIX)]
  return prefix or None


def sibling_jobset_names(prefix: str, candidates: Iterable[str]) -> list[str]:
  """Filters ``candidates`` down to this run's worker JobSets.

  Matches exactly ``<prefix>-train``, ``<prefix>-roll`` and
  ``<prefix>-roll-<n>`` (the launcher's multi-replica rollout naming). The
  orchestrator JobSet and any other run sharing a textual prefix (e.g.
  ``<prefix>-v2-train``) are never matched.

  Args:
    prefix: Run prefix (``JOB_PREFIX`` of the launcher).
    candidates: JobSet names present in the namespace.

  Returns:
    Matching names, sorted.
  """
  pattern = re.compile(rf"^{re.escape(prefix)}-(train|roll(-\d+)?)$")
  return sorted(name for name in candidates if pattern.match(name))


def pod_namespace(default: str | None = None) -> str | None:
  """Namespace of the current pod from the service-account downward file.

  Falls back to ``default`` (callers typically pass ``NAMESPACE`` from the
  environment, which the launcher sets to the sandbox namespace that for the
  MLPerf recipes coincides with the JobSet namespace).
  """
  try:
    with open(_SA_NAMESPACE_FILE, encoding="utf-8") as f:
      value = f.read().strip()
    return value or default
  except OSError:
    return default


def _state_value(state: Any) -> str:
  """Normalizes a ``WorkerState`` enum member or plain string to its value."""
  return str(getattr(state, "value", state))


def _heartbeat_state(handle: Any, timeout_s: float) -> str:
  """Returns the worker's reported state value, bounding the RPC by ``timeout_s``.

  Raises whatever the RPC raises (channel closed, deadline exceeded, ...), and
  ``TimeoutError`` if the RPC did not return within ``timeout_s``. The RPC runs
  on a daemon thread so an abandoned call can never block interpreter exit.
  """
  outcome: futures.Future[Any] = futures.Future()

  def _run() -> None:
    try:
      outcome.set_result(handle.submit("heartbeat"))
    except BaseException as err:  # pylint: disable=broad-except
      outcome.set_exception(err)

  threading.Thread(target=_run, name="orch-heartbeat-poll", daemon=True).start()
  report = outcome.result(timeout=timeout_s)
  return _state_value(getattr(report, "state", report))


def wait_until_workers_stopped(
    handles_by_id: Mapping[str, Any],
    *,
    deadline_s: float = DEFAULT_WORKER_STOP_DEADLINE_S,
    poll_s: float = DEFAULT_WORKER_STOP_POLL_S,
    heartbeat_timeout_s: float = _HEARTBEAT_TIMEOUT_S,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> list[str]:
  """Blocks until every remote worker reports ``STOPPED`` or is unreachable.

  A worker whose heartbeat RPC fails is treated as stopped (its process or
  channel is gone). A worker whose heartbeat merely times out is treated as
  still pending.

  Args:
    handles_by_id: ``worker_id -> ActorHandle`` of the remote workers.
    deadline_s: Give up after this long; the caller then deletes JobSets
      anyway and relies on the pod termination grace period.
    poll_s: Pause between polls.
    heartbeat_timeout_s: Per-RPC bound.
    monotonic: Clock, injectable for tests.
    sleep: Sleeper, injectable for tests.

  Returns:
    Worker ids that still had not reported ``STOPPED`` at the deadline.
  """
  pending = dict(handles_by_id)
  if not pending:
    return []
  end = monotonic() + deadline_s
  logging.info(
      "Waiting up to %.0fs for %d worker(s) to finish draining: %s",
      deadline_s,
      len(pending),
      sorted(pending),
  )
  while True:
    for worker_id, handle in list(pending.items()):
      try:
        state = _heartbeat_state(handle, heartbeat_timeout_s)
      except futures.TimeoutError:
        logging.info(
            "Worker %s heartbeat timed out; still draining.", worker_id
        )
        continue
      except Exception as err:  # pylint: disable=broad-except
        logging.info(
            "Worker %s unreachable (%r); treating as stopped.", worker_id, err
        )
        pending.pop(worker_id)
        continue
      if state == STOPPED_STATE:
        logging.info("Worker %s reports STOPPED.", worker_id)
        pending.pop(worker_id)
    if not pending:
      return []
    if monotonic() >= end:
      logging.warning(
          "Worker(s) still not STOPPED after %.0fs; proceeding with JobSet"
          " deletion: %s",
          deadline_s,
          sorted(pending),
      )
      return sorted(pending)
    sleep(poll_s)


def _in_cluster_jobset_api() -> Any:
  """Builds a CustomObjectsApi from the pod's service-account credentials."""
  # pylint: disable=g-import-not-at-top,import-outside-toplevel
  from kubernetes import client
  from kubernetes import config

  config.load_incluster_config()
  return client.CustomObjectsApi()


def delete_sibling_jobsets(
    orchestrator_id: str | None,
    *,
    namespace: str | None = None,
    api: Any = None,
) -> list[str]:
  """Deletes the ``-train`` / ``-roll*`` JobSets of the orchestrator's run.

  Best effort: every failure is logged and swallowed so the orchestrator's
  own exit status still reflects the training outcome.

  Args:
    orchestrator_id: ``ORCHESTRATOR_ID`` of this pod (``<prefix>-orch``).
      Anything else (e.g. the local launcher's ``orchestrator``) is a no-op.
    namespace: JobSet namespace; defaults to the pod's own namespace.
    api: ``kubernetes.client.CustomObjectsApi``-like object, injectable for
      tests. Defaults to an in-cluster client.

  Returns:
    Names of the JobSets for which a delete call was issued.
  """
  prefix = run_prefix(orchestrator_id)
  if prefix is None:
    logging.info(
        "ORCHESTRATOR_ID %r is not a <prefix>-orch id; not deleting worker"
        " JobSets.",
        orchestrator_id,
    )
    return []
  namespace = namespace or pod_namespace(default=os.environ.get("NAMESPACE"))
  if not namespace:
    logging.warning(
        "Could not determine the JobSet namespace; not deleting worker"
        " JobSets of run %r.",
        prefix,
    )
    return []
  try:
    if api is None:
      api = _in_cluster_jobset_api()
    listing = api.list_namespaced_custom_object(
        group=JOBSET_GROUP,
        version=JOBSET_VERSION,
        namespace=namespace,
        plural=JOBSET_PLURAL,
    )
  except Exception as err:  # pylint: disable=broad-except
    logging.warning(
        "Could not list JobSets in %s to clean up run %r: %r",
        namespace,
        prefix,
        err,
    )
    return []
  names = [item["metadata"]["name"] for item in listing["items"]]
  targets = sibling_jobset_names(prefix, names)
  if not targets:
    logging.info("No worker JobSets of run %r left in %s.", prefix, namespace)
    return []
  deleted = []
  for name in targets:
    try:
      logging.info("Deleting worker JobSet %s/%s", namespace, name)
      api.delete_namespaced_custom_object(
          group=JOBSET_GROUP,
          version=JOBSET_VERSION,
          namespace=namespace,
          plural=JOBSET_PLURAL,
          name=name,
      )
      deleted.append(name)
    except Exception as err:  # pylint: disable=broad-except
      logging.warning("Failed to delete JobSet %s/%s: %r", namespace, name, err)
  return deleted


def shutdown_cluster_and_workers(
    cluster: Any,
    *,
    orchestrator_id: str | None,
    delete_jobsets: bool = True,
    stop_deadline_s: float = DEFAULT_WORKER_STOP_DEADLINE_S,
) -> None:
  """``cluster.shutdown()`` followed by a confirmed drain and JobSet deletion.

  The three stages are chained with ``try/finally`` so that an exception in
  ``shutdown()`` (e.g. a stop RPC timing out) still leads to the JobSets being
  deleted: a worker that cannot be stopped gracefully is exactly the one that
  must not be left holding its TPUs.

  Args:
    cluster: ``ClusterOrchestrator``.
    orchestrator_id: ``ORCHESTRATOR_ID`` of this pod.
    delete_jobsets: Set ``False`` to only drain (local runs, debugging).
    stop_deadline_s: Passed to :func:`wait_until_workers_stopped`.
  """
  # Snapshot before shutdown(): it may unregister handles while stopping.
  handles = dict(cluster.remote_worker_handles())
  try:
    logging.info("Shutting down cluster workers...")
    cluster.shutdown()
  finally:
    if delete_jobsets:
      try:
        wait_until_workers_stopped(handles, deadline_s=stop_deadline_s)
      except Exception as err:  # pylint: disable=broad-except
        logging.warning("Error while waiting for workers to stop: %r", err)
      delete_sibling_jobsets(orchestrator_id)

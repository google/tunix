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

"""Pre-pulls static R2E-Gym task images onto sandbox nodes before run_start.

Stages all static task container images (and the OpenHands runtime image when
applicable) into the sandbox nodepool's local containerd image store using
``imagePullPolicy: IfNotPresent``, runs ``sync && sysctl -w vm.drop_caches=3``
across all sandbox nodes, and deletes all pre-pull DaemonSets/Pods before
returning so zero containers remain running prior to ``run_start``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent import futures
import hashlib
import json
import logging
import os
import sys
import time
from typing import Any

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)

# pylint: disable=g-import-not-at-top
from examples.deepswe import sandbox_utils
from examples.deepswe import template
# pylint: enable=g-import-not-at-top

DEFAULT_DATASET_NAME = "R2E-Gym/R2E-Gym-Subset"
PREPULL_APP_LABEL = "deepswe-sandbox-prepull"
DEFAULT_IMAGES_PER_POD = 25
DEFAULT_MAX_PARALLEL_DAEMONSETS = 4
DEFAULT_TIMEOUT_S = 3600.0
DEFAULT_POLL_INTERVAL_S = 5.0


def _as_text(value: Any) -> str:
  """Converts dataset scalar/array/bytes wrappers into plain strings."""
  if hasattr(value, "reshape") and hasattr(value, "tolist"):
    flat = value.reshape(-1).tolist()
    if len(flat) == 1:
      return _as_text(flat[0])
    return str(flat)
  if isinstance(value, (bytes, bytearray)):
    return value.decode("utf-8")
  if hasattr(value, "tobytes"):
    return value.tobytes().decode("utf-8")
  return value if isinstance(value, str) else str(value)


def _extract_raw_image(task: Mapping[str, Any]) -> str | None:
  """Extracts the raw task container image name from a dataset entry."""
  for key in ("docker_image", "image_name", "image"):
    val = task.get(key)
    if val is not None:
      text = _as_text(val).strip()
      if text:
        return text
  return None


def collect_static_sandbox_images(
    tasks: Iterable[Mapping[str, Any]],
    *,
    scaffold: str = "r2egym",
    openhands_server_image: str | None = None,
) -> list[str]:
  """Returns the lexicographically sorted unique list of rewritten image URIs.

  Sorting the deduplicated set guarantees zero dependence on dataset ordering
  or shuffle seed.

  Args:
    tasks: Iterable of dataset rows containing ``docker_image`` /
      ``image_name``.
    scaffold: Agent scaffold name (``"r2egym"`` or ``"openhands"``).
    openhands_server_image: Optional override for ``OPENHANDS_SERVER_IMAGE``.

  Returns:
    Sorted list of unique container image URIs to stage on sandbox nodes.
  """
  rewrite_fn = sandbox_utils.get_image_rewrite_fn()
  images: set[str] = set()
  for i in range(len(tasks)) if hasattr(tasks, "__len__") else []:
    entry = tasks[i]
    if isinstance(entry, Mapping):
      raw = _extract_raw_image(entry)
      if raw:
        images.add(rewrite_fn(raw) if rewrite_fn is not None else raw)

  if not images:
    for entry in tasks:
      if isinstance(entry, Mapping):
        raw = _extract_raw_image(entry)
        if raw:
          images.add(rewrite_fn(raw) if rewrite_fn is not None else raw)

  if scaffold.strip().lower() == "openhands":
    oh_image = (
        openhands_server_image
        or os.environ.get("OPENHANDS_SERVER_IMAGE")
        or os.environ.get("SANDBOX_RUNTIME_CONTAINER_IMAGE")
        or os.environ.get("AGENT_SERVER_IMAGE")
        or template.DEFAULT_OPENHANDS_SERVER_IMAGE
    ).strip()
    if oh_image:
      images.add(oh_image)

  return sorted(images)


def load_images_from_manifest(
    manifest_path: str,
    *,
    scaffold: str = "r2egym",
    openhands_server_image: str | None = None,
) -> list[str]:
  """Loads raw image URIs from a manifest file and applies image rewriting.

  Args:
    manifest_path: Local file path or gs:// URI to a manifest file containing
      one image URI per line.
    scaffold: Agent scaffold name ("r2egym" or "openhands").
    openhands_server_image: Optional override for OPENHANDS_SERVER_IMAGE.

  Returns:
    Sorted list of unique container image URIs to stage on sandbox nodes.
  """
  return sandbox_utils.load_images_from_manifest(
      manifest_path,
      scaffold=scaffold,
      include_runtime_image=True,
      openhands_server_image=openhands_server_image,
  )


def _resolve_node_selector(
    node_selector_key: str | None = None,
    node_selector_val: str | None = None,
) -> dict[str, str]:
  key = (
      node_selector_key
      if node_selector_key is not None
      else os.environ.get("NODE_SELECTOR_KEY")
      or os.environ.get("SANDBOX_NODE_SELECTOR_KEY", "workload")
  ).strip()
  val = (
      node_selector_val
      if node_selector_val is not None
      else os.environ.get("NODE_SELECTOR_VAL")
      or os.environ.get("SANDBOX_NODE_SELECTOR_VAL", "sandbox")
  ).strip()
  if key and val:
    return {key: val}
  return {}


def _resolve_tolerations(
    tolerations_json: str | None = None,
) -> list[dict[str, Any]]:
  raw = (
      tolerations_json
      if tolerations_json is not None
      else os.environ.get("SANDBOX_TOLERATIONS", "")
  ).strip()
  if raw:
    try:
      parsed = json.loads(raw)
      if isinstance(parsed, list):
        return parsed
    except json.JSONDecodeError as e:
      logging.warning("Failed to parse SANDBOX_TOLERATIONS=%r: %s", raw, e)
  return [
      {
          "key": "workload",
          "operator": "Equal",
          "value": "sandbox",
          "effect": "NoSchedule",
      }
  ]


def build_prepull_daemonset_manifest(
    *,
    name: str,
    namespace: str,
    images: Sequence[str],
    run_id: str,
    node_selector: Mapping[str, str] | None = None,
    tolerations: Sequence[Mapping[str, Any]] | None = None,
    image_pull_policy: str = "IfNotPresent",
) -> dict[str, Any]:
  """Builds an apps/v1 DaemonSet manifest that pulls a batch of images."""
  if not images:
    raise ValueError("images must be non-empty.")

  labels = {
      "app": PREPULL_APP_LABEL,
      "app.kubernetes.io/managed-by": "tunix-prepull",
      "tunix.google.com/prepull-run": run_id,
      "tunix.google.com/prepull-ds": name,
  }
  init_containers = []
  for idx, img in enumerate(images):
    init_containers.append({
        "name": f"pull-{idx}",
        "image": img,
        "imagePullPolicy": image_pull_policy,
        "command": ["/bin/sh", "-c", "true"],
    })

  pod_spec: dict[str, Any] = {
      "terminationGracePeriodSeconds": 0,
      "initContainers": init_containers,
      "containers": [{
          "name": "pause",
          "image": images[0],
          "imagePullPolicy": image_pull_policy,
          "command": ["/bin/sh", "-c", "sleep 3600"],
      }],
  }
  if node_selector:
    pod_spec["nodeSelector"] = dict(node_selector)
  if tolerations:
    pod_spec["tolerations"] = [dict(t) for t in tolerations]

  return {
      "apiVersion": "apps/v1",
      "kind": "DaemonSet",
      "metadata": {
          "name": name,
          "namespace": namespace,
          "labels": labels,
      },
      "spec": {
          "selector": {"matchLabels": labels},
          "template": {
              "metadata": {"labels": labels},
              "spec": pod_spec,
          },
      },
  }


def build_drop_caches_daemonset_manifest(
    *,
    name: str,
    namespace: str,
    image: str,
    run_id: str,
    node_selector: Mapping[str, str] | None = None,
    tolerations: Sequence[Mapping[str, Any]] | None = None,
    image_pull_policy: str = "IfNotPresent",
) -> dict[str, Any]:
  """Builds a privileged DaemonSet manifest to sync and drop OS page caches."""
  labels = {
      "app": PREPULL_APP_LABEL,
      "app.kubernetes.io/managed-by": "tunix-prepull",
      "tunix.google.com/prepull-run": run_id,
      "tunix.google.com/prepull-ds": name,
  }
  drop_cmd = (
      "sync && (sysctl -w vm.drop_caches=3 || echo 3 > /proc/sys/vm/drop_caches)"
  )
  pod_spec: dict[str, Any] = {
      "terminationGracePeriodSeconds": 0,
      "initContainers": [{
          "name": "drop-caches",
          "image": image,
          "imagePullPolicy": image_pull_policy,
          "command": ["/bin/sh", "-c", drop_cmd],
          "securityContext": {"privileged": True},
      }],
      "containers": [{
          "name": "pause",
          "image": image,
          "imagePullPolicy": image_pull_policy,
          "command": ["/bin/sh", "-c", "sleep 3600"],
      }],
  }
  if node_selector:
    pod_spec["nodeSelector"] = dict(node_selector)
  if tolerations:
    pod_spec["tolerations"] = [dict(t) for t in tolerations]

  return {
      "apiVersion": "apps/v1",
      "kind": "DaemonSet",
      "metadata": {
          "name": name,
          "namespace": namespace,
          "labels": labels,
      },
      "spec": {
          "selector": {"matchLabels": labels},
          "template": {
              "metadata": {"labels": labels},
              "spec": pod_spec,
          },
      },
  }


def _load_k8s_apis() -> tuple[Any, Any]:
  """Loads in-cluster or local kubeconfig and returns (AppsV1Api, CoreV1Api)."""
  from kubernetes import client  # pyrefly: ignore[missing-import]
  from kubernetes import config  # pyrefly: ignore[missing-import]

  try:
    config.load_incluster_config()
  except config.ConfigException:
    config.load_kube_config()
  cfg = client.Configuration.get_default_copy()
  cfg.connection_pool_maxsize = 64
  api_client = client.ApiClient(configuration=cfg)
  return client.AppsV1Api(api_client), client.CoreV1Api(api_client)


def _daemonset_ready(ds_obj: Any) -> tuple[bool, int, int]:
  """Returns (is_ready, number_ready, desired_number_scheduled)."""
  status = ds_obj.status
  if status is None:
    return False, 0, 0
  desired = int(status.desired_number_scheduled or 0)
  ready = int(status.number_ready or 0)
  return (desired > 0 and ready >= desired), ready, desired


def _unblock_kueue_gated_pods(
    core_api: Any,
    *,
    namespace: str,
    run_id: str,
) -> int:
  """Strips Kueue schedulingGates from prepull DaemonSet pods when injected.

  On clusters where Kueue's pod webhook intercepts all pods in the namespace
  (``manageJobsWithoutQueueName: true``) while ``workload=sandbox`` nodes are
  not part of the Kueue ClusterQueue flavors, DaemonSet pods receive
  ``kueue.x-k8s.io/admission`` schedulingGates. Clearing ``kueue.x-k8s.io/managed``,
  ``finalizers``, and ``schedulingGates`` hands the DaemonSet pods directly to
  ``default-scheduler`` (matching the cluster's ``sandbox-pods-bypass-kueue``
  policy for Sandbox pods).

  Args:
    core_api: Kubernetes CoreV1Api client.
    namespace: Kubernetes namespace.
    run_id: Prepull run identifier.

  Returns:
    Number of pods whose schedulingGates were stripped.
  """
  label_selector = f"tunix.google.com/prepull-run={run_id}"
  try:
    pods = core_api.list_namespaced_pod(
        namespace=namespace, label_selector=label_selector
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.debug("Listing prepull pods for Kueue gate check note: %s", e)
    return 0

  gated_names = [
      p.metadata.name
      for p in (pods.items or ())
      if p.spec is not None and p.spec.scheduling_gates
  ]
  if not gated_names:
    return 0

  patch_body = {
      "metadata": {
          "labels": {"kueue.x-k8s.io/managed": None},
          "finalizers": None,
      },
      "spec": {"schedulingGates": None},
  }

  def _patch_one(pod_name: str) -> bool:
    try:
      core_api.patch_namespaced_pod(
          name=pod_name, namespace=namespace, body=patch_body
      )
      return True
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.debug("Patching Kueue gate on %s note: %s", pod_name, e)
      return False

  with futures.ThreadPoolExecutor(
      max_workers=min(32, len(gated_names))
  ) as pool:
    results = list(pool.map(_patch_one, gated_names))
  unblocked = sum(1 for ok in results if ok)
  if unblocked:
    logging.info(
        "[Prepull] Stripped Kueue schedulingGates from %d/%d prepull pods.",
        unblocked,
        len(gated_names),
    )
  return unblocked


def _cleanup_prepull_resources(
    apps_api: Any,
    core_api: Any,
    *,
    namespace: str,
    run_id: str,
    timeout_s: float = 120.0,
    poll_s: float = 2.0,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
  """Deletes all DaemonSets and Pods for `run_id` and waits until 0 pods remain."""
  label_selector = f"tunix.google.com/prepull-run={run_id}"
  try:
    apps_api.delete_collection_namespaced_daemon_set(
        namespace=namespace,
        label_selector=label_selector,
        grace_period_seconds=0,
        propagation_policy="Background",
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.debug("DaemonSet collection delete note: %s", e)

  try:
    ds_list = apps_api.list_namespaced_daemon_set(
        namespace=namespace, label_selector=label_selector
    )
    for ds in ds_list.items or ():
      try:
        apps_api.delete_namespaced_daemon_set(
            name=ds.metadata.name,
            namespace=namespace,
            grace_period_seconds=0,
            propagation_policy="Background",
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.debug("DaemonSet delete note: %s", e)
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.debug("DaemonSet list during cleanup note: %s", e)

  try:
    core_api.delete_collection_namespaced_pod(
        namespace=namespace,
        label_selector=label_selector,
        grace_period_seconds=0,
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.debug("Pod collection delete note: %s", e)

  deadline = monotonic() + timeout_s
  while monotonic() < deadline:
    try:
      pods = core_api.list_namespaced_pod(
          namespace=namespace, label_selector=label_selector
      )
      if not pods.items:
        return
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.debug("Listing prepull pods during cleanup note: %s", e)
      return
    sleep(poll_s)


def prepull_sandbox_images(
    images: Sequence[str],
    *,
    namespace: str | None = None,
    run_id: str | None = None,
    node_selector: Mapping[str, str] | None = None,
    tolerations: Sequence[Mapping[str, Any]] | None = None,
    images_per_pod: int = DEFAULT_IMAGES_PER_POD,
    max_parallel_daemonsets: int = DEFAULT_MAX_PARALLEL_DAEMONSETS,
    drop_caches: bool = True,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    poll_s: float = DEFAULT_POLL_INTERVAL_S,
    apps_api: Any = None,
    core_api: Any = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
  """Stages `images` on all sandbox nodes via DaemonSets and drops caches.

  Args:
    images: Container image URIs to pre-pull.
    namespace: Kubernetes namespace for temporary DaemonSets.
    run_id: Identifier for labeling temporary DaemonSets/Pods.
    node_selector: Node selector targeting the sandbox nodepool.
    tolerations: Tolerations targeting the sandbox nodepool.
    images_per_pod: Number of initContainers per DaemonSet pod.
    max_parallel_daemonsets: Maximum concurrent DaemonSets in flight.
    drop_caches: Whether to run ``sync && sysctl -w vm.drop_caches=3`` after
      all images are pulled.
    timeout_s: Overall timeout in seconds.
    poll_s: Poll interval in seconds.
    apps_api: Optional injectable Kubernetes AppsV1Api.
    core_api: Optional injectable Kubernetes CoreV1Api.
    monotonic: Injectable clock function.
    sleep: Injectable sleep function.

  Returns:
    Number of unique images staged.
  """
  unique_images = sorted(set(img.strip() for img in images if img.strip()))
  if not unique_images:
    logging.info("[Prepull] No sandbox images to pre-pull.")
    return 0

  ns = (
      namespace
      or os.environ.get("SANDBOX_NAMESPACE")
      or os.environ.get("NAMESPACE")
      or "default"
  ).strip()
  prefix = (
      os.environ.get("JOB_PREFIX")
      or os.environ.get("ORCHESTRATOR_ID")
      or "tunix"
  ).strip()
  if not run_id:
    digest = hashlib.sha1(f"{prefix}-{len(unique_images)}".encode()).hexdigest()[
        :8
    ]
    run_id = f"prepull-{digest}"

  resolved_selector = (
      dict(node_selector)
      if node_selector is not None
      else _resolve_node_selector()
  )
  resolved_tolerations = (
      list(tolerations)
      if tolerations is not None
      else _resolve_tolerations()
  )

  if apps_api is None or core_api is None:
    apps_api, core_api = _load_k8s_apis()

  chunks = [
      unique_images[i : i + images_per_pod]
      for i in range(0, len(unique_images), images_per_pod)
  ]
  logging.info(
      "[Prepull] Pre-pulling %d static sandbox images across %d DaemonSet"
      " wave(s) (images_per_pod=%d, max_parallel=%d, namespace=%s,"
      " node_selector=%s)...",
      len(unique_images),
      len(chunks),
      images_per_pod,
      max_parallel_daemonsets,
      ns,
      resolved_selector,
  )

  start_time = monotonic()
  deadline = start_time + timeout_s
  _cleanup_prepull_resources(
      apps_api,
      core_api,
      namespace=ns,
      run_id=run_id,
      monotonic=monotonic,
      sleep=sleep,
  )

  try:
    next_wave_idx = 0
    active_ds: dict[str, int] = {}
    completed_waves = 0

    while completed_waves < len(chunks):
      created_any = False
      while (
          len(active_ds) < max_parallel_daemonsets
          and next_wave_idx < len(chunks)
      ):
        wave_idx = next_wave_idx
        next_wave_idx += 1
        ds_name = f"{run_id}-w{wave_idx}"
        manifest = build_prepull_daemonset_manifest(
            name=ds_name,
            namespace=ns,
            images=chunks[wave_idx],
            run_id=run_id,
            node_selector=resolved_selector,
            tolerations=resolved_tolerations,
        )
        apps_api.create_namespaced_daemon_set(namespace=ns, body=manifest)
        active_ds[ds_name] = wave_idx
        created_any = True

      if created_any:
        sleep(min(1.0, poll_s))
      _unblock_kueue_gated_pods(core_api, namespace=ns, run_id=run_id)

      if monotonic() >= deadline:
        raise TimeoutError(
            f"[Prepull] Timed out after {timeout_s:.0f}s with "
            f"{completed_waves}/{len(chunks)} waves completed."
        )

      finished_names: list[str] = []
      for ds_name, wave_idx in list(active_ds.items()):
        ds_obj = apps_api.read_namespaced_daemon_set(
            name=ds_name, namespace=ns
        )
        is_ready, ready_nodes, desired_nodes = _daemonset_ready(ds_obj)
        if is_ready:
          logging.info(
              "[Prepull] Wave %d/%d (%s) ready on %d/%d sandbox nodes.",
              wave_idx + 1,
              len(chunks),
              ds_name,
              ready_nodes,
              desired_nodes,
          )
          finished_names.append(ds_name)

      for ds_name in finished_names:
        active_ds.pop(ds_name, None)
        completed_waves += 1
        try:
          apps_api.delete_namespaced_daemon_set(
              name=ds_name,
              namespace=ns,
              grace_period_seconds=0,
              propagation_policy="Background",
          )
        except Exception as e:  # pylint: disable=broad-exception-caught
          logging.debug("Deleting wave DaemonSet %s note: %s", ds_name, e)

      if completed_waves < len(chunks) and not finished_names:
        sleep(poll_s)

    if drop_caches:
      drop_ds_name = f"{run_id}-drop-caches"
      logging.info(
          "[Prepull] Running sync && sysctl vm.drop_caches=3 across sandbox"
          " nodes via %s...",
          drop_ds_name,
      )
      drop_manifest = build_drop_caches_daemonset_manifest(
          name=drop_ds_name,
          namespace=ns,
          image=unique_images[0],
          run_id=run_id,
          node_selector=resolved_selector,
          tolerations=resolved_tolerations,
      )
      apps_api.create_namespaced_daemon_set(namespace=ns, body=drop_manifest)
      sleep(min(1.0, poll_s))
      while True:
        _unblock_kueue_gated_pods(core_api, namespace=ns, run_id=run_id)
        if monotonic() >= deadline:
          raise TimeoutError(
              f"[Prepull] Timed out waiting for {drop_ds_name} to complete."
          )
        ds_obj = apps_api.read_namespaced_daemon_set(
            name=drop_ds_name, namespace=ns
        )
        is_ready, ready_nodes, desired_nodes = _daemonset_ready(ds_obj)
        if is_ready:
          logging.info(
              "[Prepull] Dropped page/dentry/inode caches on %d/%d sandbox"
              " nodes.",
              ready_nodes,
              desired_nodes,
          )
          break
        sleep(poll_s)
      try:
        apps_api.delete_namespaced_daemon_set(
            name=drop_ds_name,
            namespace=ns,
            grace_period_seconds=0,
            propagation_policy="Background",
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.debug("Deleting drop-caches DaemonSet note: %s", e)

  finally:
    _cleanup_prepull_resources(
        apps_api,
        core_api,
        namespace=ns,
        run_id=run_id,
        monotonic=monotonic,
        sleep=sleep,
    )

  elapsed = monotonic() - start_time
  logging.info(
      "[Prepull] Completed staging %d static sandbox images and cleaned up all"
      " pre-pull containers in %.1fs.",
      len(unique_images),
      elapsed,
  )
  return len(unique_images)


def _load_static_dataset(
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_split: str = "train",
    dataset_path: str = "",
    cache_dir: str | None = None,
) -> Any:
  """Loads the R2E-Gym dataset with shuffle=False."""
  if dataset_path:
    from datasets import DatasetDict  # pylint: disable=g-import-not-at-top
    from datasets import load_from_disk  # pylint: disable=g-import-not-at-top

    logging.info("[Prepull] Loading dataset from disk: %s", dataset_path)
    dataset = load_from_disk(dataset_path)
    if isinstance(dataset, DatasetDict):
      dataset = dataset[dataset_split]
    return dataset

  from examples.deepswe import deepswe_data  # pylint: disable=g-import-not-at-top

  return deepswe_data.create_dataset(
      dataset_name=dataset_name,
      dataset_split=dataset_split,
      cache_dir=cache_dir,
      shuffle=False,
  )


def prepull_from_dataset(
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    dataset_split: str = "train",
    dataset_path: str = "",
    cache_dir: str | None = None,
    scaffold: str = "r2egym",
    namespace: str | None = None,
    images_per_pod: int | None = None,
    max_parallel_daemonsets: int | None = None,
    max_images: int | None = None,
    drop_caches: bool = True,
    timeout_s: float | None = None,
) -> int:
  """Loads the static dataset (unshuffled) and stages all task images."""
  resolved_images_per_pod = (
      images_per_pod
      if images_per_pod is not None
      else int(
          os.environ.get("PREPULL_IMAGES_PER_POD", str(DEFAULT_IMAGES_PER_POD))
      )
  )
  resolved_max_parallel = (
      max_parallel_daemonsets
      if max_parallel_daemonsets is not None
      else int(
          os.environ.get(
              "PREPULL_MAX_PARALLEL_DAEMONSETS",
              str(DEFAULT_MAX_PARALLEL_DAEMONSETS),
          )
      )
  )
  resolved_max_images = (
      max_images
      if max_images is not None
      else int(os.environ.get("PREPULL_MAX_IMAGES", "0"))
  )
  resolved_timeout_s = (
      timeout_s
      if timeout_s is not None
      else float(
          os.environ.get("PREPULL_TIMEOUT_S", str(DEFAULT_TIMEOUT_S))
      )
  )
  dataset = _load_static_dataset(
      dataset_name=dataset_name,
      dataset_split=dataset_split,
      dataset_path=dataset_path,
      cache_dir=cache_dir,
  )
  images = collect_static_sandbox_images(dataset, scaffold=scaffold)
  if resolved_max_images > 0:
    images = images[:resolved_max_images]
  return prepull_sandbox_images(
      images,
      namespace=namespace,
      images_per_pod=resolved_images_per_pod,
      max_parallel_daemonsets=resolved_max_parallel,
      drop_caches=drop_caches,
      timeout_s=resolved_timeout_s,
  )


def prepull_from_manifest(
    manifest_path: str,
    *,
    scaffold: str = "r2egym",
    namespace: str | None = None,
    images_per_pod: int | None = None,
    max_parallel_daemonsets: int | None = None,
    max_images: int | None = None,
    drop_caches: bool = True,
    timeout_s: float | None = None,
) -> int:
  """Loads task images from a static manifest file and stages them on sandbox nodes."""
  resolved_images_per_pod = (
      images_per_pod
      if images_per_pod is not None
      else int(
          os.environ.get("PREPULL_IMAGES_PER_POD", str(DEFAULT_IMAGES_PER_POD))
      )
  )
  resolved_max_parallel = (
      max_parallel_daemonsets
      if max_parallel_daemonsets is not None
      else int(
          os.environ.get(
              "PREPULL_MAX_PARALLEL_DAEMONSETS",
              str(DEFAULT_MAX_PARALLEL_DAEMONSETS),
          )
      )
  )
  resolved_max_images = (
      max_images
      if max_images is not None
      else int(os.environ.get("PREPULL_MAX_IMAGES", "0"))
  )
  resolved_timeout_s = (
      timeout_s
      if timeout_s is not None
      else float(
          os.environ.get("PREPULL_TIMEOUT_S", str(DEFAULT_TIMEOUT_S))
      )
  )
  logging.info("[Prepull] Loading sandbox images from manifest: %s", manifest_path)
  images = load_images_from_manifest(manifest_path, scaffold=scaffold)
  if resolved_max_images > 0:
    images = images[:resolved_max_images]
  return prepull_sandbox_images(
      images,
      namespace=namespace,
      images_per_pod=resolved_images_per_pod,
      max_parallel_daemonsets=resolved_max_parallel,
      drop_caches=drop_caches,
      timeout_s=resolved_timeout_s,
  )


def _parse_cli_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          "Pre-pull all static R2E-Gym task container images onto sandbox"
          " nodes and drop OS page caches before run_start."
      )
  )
  parser.add_argument(
      "--images_file",
      type=str,
      default=os.environ.get("PREPULL_IMAGES_FILE", ""),
      help=(
          "Path (local or gs://) to a manifest file listing container image"
          " URIs to pre-pull instead of loading from the dataset."
      ),
  )
  parser.add_argument(
      "--dataset_name",
      type=str,
      default=os.environ.get("DATASET_NAME", DEFAULT_DATASET_NAME),
  )
  parser.add_argument(
      "--dataset_split",
      type=str,
      default=os.environ.get("DATASET_SPLIT", "train"),
  )
  parser.add_argument(
      "--dataset_path",
      type=str,
      default=os.environ.get("DATASET_PATH", ""),
  )
  parser.add_argument(
      "--dataset_cache_dir",
      type=str,
      default=os.environ.get("DATASET_CACHE_DIR", None),
  )
  parser.add_argument(
      "--scaffold",
      type=str,
      default=os.environ.get("SCAFFOLD", "r2egym"),
  )
  parser.add_argument(
      "--namespace",
      type=str,
      default=os.environ.get(
          "SANDBOX_NAMESPACE", os.environ.get("NAMESPACE", "default")
      ),
  )
  parser.add_argument(
      "--images_per_pod",
      type=int,
      default=int(
          os.environ.get("PREPULL_IMAGES_PER_POD", str(DEFAULT_IMAGES_PER_POD))
      ),
  )
  parser.add_argument(
      "--max_parallel_daemonsets",
      type=int,
      default=int(
          os.environ.get(
              "PREPULL_MAX_PARALLEL_DAEMONSETS",
              str(DEFAULT_MAX_PARALLEL_DAEMONSETS),
          )
      ),
  )
  parser.add_argument(
      "--max_images",
      type=int,
      default=int(os.environ.get("PREPULL_MAX_IMAGES", "0")),
  )
  parser.add_argument(
      "--drop_caches",
      action=argparse.BooleanOptionalAction,
      default=True,
  )
  parser.add_argument(
      "--timeout_s",
      type=float,
      default=float(
          os.environ.get("PREPULL_TIMEOUT_S", str(DEFAULT_TIMEOUT_S))
      ),
  )
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - [SandboxPrepull] %(message)s",
      force=True,
  )
  args = _parse_cli_args(argv)
  if args.images_file:
    prepull_from_manifest(
        manifest_path=args.images_file,
        scaffold=args.scaffold,
        namespace=args.namespace,
        images_per_pod=args.images_per_pod,
        max_parallel_daemonsets=args.max_parallel_daemonsets,
        max_images=args.max_images,
        drop_caches=args.drop_caches,
        timeout_s=args.timeout_s,
    )
  else:
    prepull_from_dataset(
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        dataset_path=args.dataset_path,
        cache_dir=args.dataset_cache_dir,
        scaffold=args.scaffold,
        namespace=args.namespace,
        images_per_pod=args.images_per_pod,
        max_parallel_daemonsets=args.max_parallel_daemonsets,
        max_images=args.max_images,
        drop_caches=args.drop_caches,
        timeout_s=args.timeout_s,
    )


if __name__ == "__main__":
  main()

"""Bounded Kubernetes runtime patch for installed R2E-Gym environments."""

from __future__ import annotations

import logging
import os
import re
import time


def _kubernetes_label(value: str, *, fallback: str) -> str:
  """Returns a bounded Kubernetes label value without leaking raw metadata."""
  normalized = re.sub(r"[^a-z0-9_.-]+", "-", value.lower()).strip("-_.")
  return (normalized[:63].rstrip("-_.") or fallback)


def _cleanup_orphaned_kubernetes_pods(
    core,
    *,
    namespace: str,
    resume_tag: str,
    api_exception_type: type[BaseException],
) -> int:
  """Deletes and confirms a preconfigured client's same-tag sandboxes."""
  normalized = _kubernetes_label(resume_tag, fallback="unknown")
  selector = (
      "app.kubernetes.io/managed-by=tunix-deepswe,"
      f"canon.zero-tim/resume-tag={normalized}"
  )
  pods = core.list_namespaced_pod(
      namespace=namespace,
      label_selector=selector,
      _request_timeout=60,
  ).items
  names = sorted(
      str(getattr(getattr(pod, "metadata", None), "name", "") or "")
      for pod in pods
  )
  names = [name for name in names if name]
  for name in names:
    try:
      core.delete_namespaced_pod(
          name=name,
          namespace=namespace,
          grace_period_seconds=0,
          propagation_policy="Background",
          _request_timeout=60,
      )
    except api_exception_type as error:
      if getattr(error, "status", None) != 404:
        raise
  timeout = int(os.environ.get("R2E_POD_DELETE_TIMEOUT_SECONDS", "300"))
  deadline = time.monotonic() + timeout
  while names and time.monotonic() < deadline:
    remaining = core.list_namespaced_pod(
        namespace=namespace,
        label_selector=selector,
        _request_timeout=60,
    ).items
    names = [
        str(getattr(getattr(pod, "metadata", None), "name", "") or "")
        for pod in remaining
        if getattr(getattr(pod, "metadata", None), "name", "")
    ]
    if names:
      time.sleep(2)
  if names:
    raise TimeoutError(
        "stale R2E sandboxes survived resume cleanup: "
        + ",".join(sorted(names))
    )
  print(
      "[P46.RESUME] ORPHAN_SANDBOX_CLEANUP_PASS "
      f"resume_tag={normalized} deleted={len(pods)} remaining=0",
      flush=True,
  )
  return len(pods)


def cleanup_orphaned_kubernetes_pods(resume_tag: str) -> int:
  """Deletes only stale R2E sandboxes owned by one resume lineage."""
  from kubernetes import client
  from kubernetes import config as k8s_config
  from r2egym.agenthub.runtime import docker as docker_mod

  try:
    k8s_config.load_incluster_config()
  except k8s_config.config_exception.ConfigException:
    k8s_config.load_kube_config()
  return _cleanup_orphaned_kubernetes_pods(
      client.CoreV1Api(),
      namespace=docker_mod.DEFAULT_NAMESPACE,
      resume_tag=resume_tag,
      api_exception_type=client.ApiException,
  )


def ensure_huggingface_hub_compat() -> bool:
  """Provides the removed HfFolder API required by older R2E-Gym builds."""
  import huggingface_hub

  if hasattr(huggingface_hub, "HfFolder"):
    return False

  class HfFolderCompat:

    @staticmethod
    def get_token():
      return huggingface_hub.get_token()

    @staticmethod
    def save_token(token: str) -> None:
      huggingface_hub.login(token=token, add_to_git_credential=False)

    @staticmethod
    def delete_token() -> None:
      huggingface_hub.logout()

  huggingface_hub.HfFolder = HfFolderCompat
  return True


def apply_repoenv_kubernetes_poll_patch() -> str:
  """Replaces an unbounded pod watch with bounded direct status polling."""
  ensure_huggingface_hub_compat()
  try:
    from r2egym.agenthub.runtime import docker as docker_mod
  except ImportError:
    logging.warning(
        "r2egym is not importable; skipping the RepoEnv Kubernetes poll"
        " patch. Interactive SWE environments cannot start in this process."
    )
    return ""

  if getattr(docker_mod, "_tunix_repoenv_poll_patch_applied", False):
    return str(getattr(docker_mod, "__file__", ""))

  namespace = docker_mod.DEFAULT_NAMESPACE

  def pod_name_for(runtime) -> str:
    name = getattr(runtime, "_tunix_kubernetes_pod_name", "")
    if name:
      return str(name)
    metadata = getattr(getattr(runtime, "container", None), "metadata", None)
    return str(getattr(metadata, "name", "") or "")

  def delete_and_confirm(runtime, pod_name: str) -> None:
    """Deletes one known R2E pod and waits until the API reports 404."""
    if not pod_name:
      return
    try:
      runtime.client.delete_namespaced_pod(
          name=pod_name,
          namespace=namespace,
          grace_period_seconds=0,
          propagation_policy="Background",
          _request_timeout=60,
      )
      runtime.logger.info("Requested deletion of Kubernetes pod: %s", pod_name)
    except docker_mod.client.ApiException as error:
      if error.status != 404:
        raise
    timeout = int(os.environ.get("R2E_POD_DELETE_TIMEOUT_SECONDS", "300"))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
      try:
        runtime.client.read_namespaced_pod(
            name=pod_name,
            namespace=namespace,
            _request_timeout=60,
        )
      except docker_mod.client.ApiException as error:
        if error.status == 404:
          runtime.logger.info(
              "Confirmed deletion of Kubernetes pod: %s", pod_name
          )
          return
        raise
      time.sleep(2)
    raise TimeoutError(
        f"Kubernetes pod {pod_name!r} still exists {timeout}s after deletion"
    )

  def start_kubernetes_pod(
      self, docker_image: str, command: str, pod_name: str, **docker_kwargs
  ):
    self._tunix_kubernetes_pod_name = pod_name
    try:
      existing = self.client.read_namespaced_pod(
          name=pod_name,
          namespace=namespace,
          _request_timeout=60,
      )
      phase = str(existing.status.phase)
      if phase == "Running":
        self.container = existing
        self.logger.info("Found existing Running Kubernetes pod: %s", pod_name)
        return
      self.logger.warning(
          "Deleting stale Kubernetes pod %s in phase %s", pod_name, phase
      )
      delete_and_confirm(self, pod_name)
    except docker_mod.client.ApiException as error:
      if error.status != 404:
        raise

    environment = {
        "PATH": docker_mod.DOCKER_PATH,
        **docker_kwargs.get("environment", {}),
    }
    node_key = os.environ.get(
        "NODE_SELECTOR_KEY", "cloud.google.com/gke-nodepool"
    )
    node_value = os.environ.get("NODE_SELECTOR_VAL", "deepswe-cpu-pool")
    pull_secret = os.environ.get("IMAGE_PULL_SECRET", "dockerhub-pro")
    cpu_request = os.environ.get("R2E_K8S_CPU", "2")
    memory_request = os.environ.get("R2E_K8S_MEM", "4Gi")
    cpu_limit = os.environ.get("R2E_K8S_CPU_LIMIT", "4")
    memory_limit = os.environ.get("R2E_K8S_MEM_LIMIT", "8Gi")
    labels = {
        "app.kubernetes.io/name": "r2egym",
        "app.kubernetes.io/managed-by": "tunix-deepswe",
        "canon.zero-tim/run-id": _kubernetes_label(
            os.environ.get("CANON_RUN_ID", ""), fallback="unknown"
        ),
        "canon.zero-tim/resume-tag": _kubernetes_label(
            os.environ.get(
                "CANON_P46_RESUME_TAG",
                os.environ.get("CANON_RUN_ID", ""),
            ),
            fallback="unknown",
        ),
    }
    pod_spec = {
        "restartPolicy": "Never",
        "activeDeadlineSeconds": int(
            os.environ.get("R2E_ACTIVE_DEADLINE_SECONDS", "5100")
        ),
        "containers": [{
            "name": pod_name,
            "image": docker_image,
            "command": ["/bin/sh", "-c"],
            "args": [command] if isinstance(command, str) else command,
            "stdin": True,
            "tty": True,
            "env": [
                {"name": key, "value": str(value)}
                for key, value in environment.items()
            ],
            "resources": {
                "requests": {
                    "cpu": cpu_request,
                    "memory": memory_request,
                },
                "limits": {"cpu": cpu_limit, "memory": memory_limit},
            },
        }],
        "tolerations": [{
            "key": "node.kubernetes.io/disk-pressure",
            "operator": "Exists",
            "effect": "NoExecute",
            "tolerationSeconds": 10800,
        }],
    }
    if node_key and node_value:
      pod_spec["nodeSelector"] = {node_key: node_value}
    if pull_secret:
      pod_spec["imagePullSecrets"] = [{"name": pull_secret}]
    body = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": pod_name, "labels": labels},
        "spec": pod_spec,
    }

    delay = 5
    for attempt in range(1, 6):
      try:
        self.client.create_namespaced_pod(
            namespace=namespace,
            body=body,
            _request_timeout=120,
        )
        break
      except docker_mod.client.ApiException as error:
        if error.status not in (409, 429, 500, 503) or attempt == 5:
          raise
        self.logger.warning(
            "Transient pod-create error %s for %s, attempt %s/5",
            error.status,
            pod_name,
            attempt,
        )
        time.sleep(delay)
        delay = min(delay * 2, 60)

    timeout = int(os.environ.get("R2E_POD_START_TIMEOUT_SECONDS", "1200"))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
      pod = self.client.read_namespaced_pod(
          name=pod_name,
          namespace=namespace,
          _request_timeout=60,
      )
      phase = str(pod.status.phase)
      if phase == "Running":
        self.container = pod
        self.logger.info("Kubernetes pod %s is Running", pod_name)
        return
      if phase in ("Failed", "Succeeded", "Unknown"):
        delete_and_confirm(self, pod_name)
        raise RuntimeError(
            f"Kubernetes pod {pod_name!r} entered terminal phase {phase!r}"
        )
      self.logger.info(
          "Waiting for Kubernetes pod %s; phase=%s remaining=%ss",
          pod_name,
          phase,
          max(0, int(deadline - time.monotonic())),
      )
      time.sleep(5)
    delete_and_confirm(self, pod_name)
    raise TimeoutError(
        f"Kubernetes pod {pod_name!r} did not start within {timeout}s"
    )

  cleanup_method_name = (
      "stop"
      if hasattr(docker_mod.DockerRuntime, "stop")
      else "close"
      if hasattr(docker_mod.DockerRuntime, "close")
      else ""
  )
  if not cleanup_method_name:
    raise RuntimeError("R2E-Gym DockerRuntime has no stop/close cleanup method")
  original_cleanup = getattr(docker_mod.DockerRuntime, cleanup_method_name)

  def cleanup_with_delete_confirmation(self, *args, **kwargs):
    pod_name = pod_name_for(self)
    if pod_name:
      # Delete first: if the upstream cleanup later hangs, the sandbox is
      # already gone and cannot continue consuming the shared CPU node pool.
      delete_and_confirm(self, pod_name)
    cleanup_error = None
    try:
      original_cleanup(self, *args, **kwargs)
    except BaseException as error:  # Cleanup must still attempt pod deletion.
      if not (
          isinstance(error, docker_mod.client.ApiException)
          and error.status == 404
          and pod_name
      ):
        cleanup_error = error
    if cleanup_error is not None:
      raise cleanup_error

  docker_mod.DockerRuntime._start_kubernetes_pod = start_kubernetes_pod
  setattr(
      docker_mod.DockerRuntime,
      cleanup_method_name,
      cleanup_with_delete_confirmation,
  )
  docker_mod._tunix_repoenv_poll_patch_applied = True
  path = str(getattr(docker_mod, "__file__", ""))
  logging.info("Applied bounded R2E-Gym Kubernetes patch at %s", path)
  print(
      "[P34.R2E] BOUNDED_KUBERNETES_PATCH_PASS "
      f"path={path} cleanup_method={cleanup_method_name}",
      flush=True,
  )
  return path

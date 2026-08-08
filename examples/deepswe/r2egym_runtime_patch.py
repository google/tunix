"""Bounded Kubernetes runtime patch for installed R2E-Gym environments."""

from __future__ import annotations

import logging
import os
import time


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
  from r2egym.agenthub.runtime import docker as docker_mod

  if getattr(docker_mod, "_tunix_repoenv_poll_patch_applied", False):
    return str(getattr(docker_mod, "__file__", ""))

  def start_kubernetes_pod(
      self, docker_image: str, command: str, pod_name: str, **docker_kwargs
  ):
    try:
      self.container = self.client.read_namespaced_pod(
          name=pod_name,
          namespace=docker_mod.DEFAULT_NAMESPACE,
          _request_timeout=60,
      )
      self.logger.info("Found existing Kubernetes pod: %s", pod_name)
      return
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
    pod_spec = {
        "restartPolicy": "Never",
        "activeDeadlineSeconds": int(
            os.environ.get("R2E_ACTIVE_DEADLINE_SECONDS", "10800")
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
            "resources": {"requests": {"cpu": "1", "memory": "1Gi"}},
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
        "metadata": {"name": pod_name},
        "spec": pod_spec,
    }

    delay = 5
    for attempt in range(1, 6):
      try:
        self.client.create_namespaced_pod(
            namespace=docker_mod.DEFAULT_NAMESPACE,
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
          namespace=docker_mod.DEFAULT_NAMESPACE,
          _request_timeout=60,
      )
      phase = str(pod.status.phase)
      if phase == "Running":
        self.container = pod
        self.logger.info("Kubernetes pod %s is Running", pod_name)
        return
      if phase in ("Failed", "Succeeded", "Unknown"):
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
    raise TimeoutError(
        f"Kubernetes pod {pod_name!r} did not start within {timeout}s"
    )

  docker_mod.DockerRuntime._start_kubernetes_pod = start_kubernetes_pod
  docker_mod._tunix_repoenv_poll_patch_applied = True
  path = str(getattr(docker_mod, "__file__", ""))
  logging.info("Applied bounded R2E-Gym Kubernetes patch at %s", path)
  print(f"[P34.R2E] BOUNDED_KUBERNETES_PATCH_PASS path={path}", flush=True)
  return path

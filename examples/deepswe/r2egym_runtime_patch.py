"""Runtime patches for installed r2egym environments used by DeepSWE."""

from __future__ import annotations

import logging
import os
import time


def apply_repoenv_kubernetes_watch_patch() -> str | None:
  """Monkey-patch installed r2egym to avoid hanging forever on pod watches."""
  try:
    from r2egym.agenthub.runtime import docker as docker_mod
  except Exception as exc:  # pragma: no cover - debug helper path
    logging.warning("r2egym runtime patch skipped: import failed: %r", exc)
    return None

  if getattr(docker_mod, "_tunix_repoenv_watch_patch_applied", False):
    return getattr(docker_mod, "__file__", None)

  def _patched_start_kubernetes_pod(
      self, docker_image: str, command: str, pod_name: str, **docker_kwargs
  ):
    not_found_error = None
    try:
      self.container = self.client.read_namespaced_pod(
          name=pod_name,
          namespace=docker_mod.DEFAULT_NAMESPACE,
          _request_timeout=60,
      )
      self.logger.info("Found existing Kubernetes pod: %s", pod_name)
      return
    except docker_mod.client.ApiException as e:
      not_found_error = e

    if not_found_error.status != 404:
      self.logger.error(
          "Error checking Kubernetes pod '%s' status: %s. "
          "Check Kubernetes configuration and permissions.",
          pod_name,
          not_found_error,
      )
      raise not_found_error

    env_vars = {
        "PATH": docker_mod.DOCKER_PATH,
        **docker_kwargs.get("environment", {}),
    }
    env_spec = [{"name": k, "value": str(v)} for k, v in env_vars.items()]
    node_selector_key = os.getenv(
        "NODE_SELECTOR_KEY", "karpenter.sh/nodepool"
    )
    node_selector_val = os.getenv("NODE_SELECTOR_VAL", "bigcpu-standby")
    image_pull_secret = os.getenv("IMAGE_PULL_SECRET", "dockerhub-pro")
    self.logger.info(
        "Kubernetes scheduling config: nodeSelector=%s=%s imagePullSecret=%s",
        node_selector_key or "<none>",
        node_selector_val or "<none>",
        image_pull_secret or "<none>",
    )
    pod_spec = {
        "restartPolicy": "Never",
        "containers": [
            {
                "name": pod_name,
                "image": docker_image,
                "command": ["/bin/sh", "-c"],
                "args": [command] if isinstance(command, str) else command,
                "stdin": True,
                "tty": True,
                "env": env_spec,
                "resources": {
                    "requests": {"cpu": "1", "memory": "1Gi"},
                },
            }
        ],
        "tolerations": [
            {
                "key": "node.kubernetes.io/disk-pressure",
                "operator": "Exists",
                "effect": "NoExecute",
                "tolerationSeconds": 10800,
            }
        ],
    }
    if node_selector_key and node_selector_val:
      pod_spec["nodeSelector"] = {node_selector_key: node_selector_val}
    if image_pull_secret:
      pod_spec["imagePullSecrets"] = [{"name": image_pull_secret}]
    pod_body = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": pod_name},
        "spec": pod_spec,
    }

    max_retries = 5
    backoff = 5
    pod = None
    for attempt in range(1, max_retries + 1):
      try:
        pod = self.client.create_namespaced_pod(
            namespace=docker_mod.DEFAULT_NAMESPACE,
            body=pod_body,
            _request_timeout=120,
        )
        break
      except docker_mod.client.ApiException as e:
        if e.status in (409, 429, 500, 503):
          self.logger.warning(
              "Transient Kubernetes error %s while creating pod '%s' "
              "(attempt %s/%s); retrying in %ss",
              e.status,
              pod_name,
              attempt,
              max_retries,
              backoff,
          )
          time.sleep(backoff)
          backoff = min(backoff * 2, 60)
          continue
        self.logger.error("Failed to create Kubernetes pod '%s': %s", pod_name, e)
        raise
    else:
      raise RuntimeError(
          f"Exceeded retry limit ({max_retries}) while creating pod '{pod_name}'."
      )

    try:
      start_time = time.time()
      deadline = start_time + 1200
      rv = pod.metadata.resource_version

      while time.time() < deadline:
        pod_status = self.client.read_namespaced_pod(
            name=pod_name,
            namespace=docker_mod.DEFAULT_NAMESPACE,
            _request_timeout=60,
        )
        phase = pod_status.status.phase
        if phase == "Running":
          self.logger.info(
              "Kubernetes pod '%s' is Running (direct status check).",
              pod_name,
          )
          self.container = pod_status
          return
        if phase in ["Failed", "Succeeded", "Unknown"]:
          raise RuntimeError(
              f"Kubernetes pod '{pod_name}' entered terminal phase '{phase}'."
          )

        remaining = max(1, int(deadline - time.time()))
        watch_timeout = min(30, remaining)
        self.logger.info(
            "Waiting for Kubernetes pod '%s' to start; current phase=%s, remaining=%ss",
            pod_name,
            phase,
            remaining,
        )

        w = docker_mod.watch.Watch()
        try:
          stream = w.stream(
              self.client.list_namespaced_pod,
              namespace=docker_mod.DEFAULT_NAMESPACE,
              field_selector=f"metadata.name={pod_name}",
              resource_version=rv,
              timeout_seconds=watch_timeout,
          )
          for event in stream:
            obj = event["object"]
            rv = obj.metadata.resource_version
            phase = obj.status.phase
            self.logger.info(
                "Pod '%s' watch event=%s phase=%s",
                pod_name,
                event.get("type"),
                phase,
            )
            if phase == "Running":
              self.logger.info(
                  "Kubernetes pod '%s' is Running (watch event).", pod_name
              )
              self.container = obj
              return
            if phase in ["Failed", "Succeeded", "Unknown"]:
              raise RuntimeError(
                  f"Kubernetes pod '{pod_name}' entered terminal phase '{phase}'."
              )
        finally:
          w.stop()

      raise RuntimeError(
          f"Kubernetes pod '{pod_name}' timed out after 1200 seconds."
      )
    except docker_mod.client.ApiException as create_error:
      self.logger.error(
          "Failed to create Kubernetes pod '%s': %s", pod_name, create_error
      )
      raise create_error
    except Exception as e:
      self.logger.error("Error waiting for pod to start: %s", e)
      try:
        pod_status = self.client.read_namespaced_pod(
            name=pod_name,
            namespace=docker_mod.DEFAULT_NAMESPACE,
            _request_timeout=60,
        )
        if pod_status.status.phase == "Running":
          self.logger.info(
              "Pod '%s' is running (verified after watch error)", pod_name
          )
          self.container = pod_status
        else:
          self.logger.warning(
              "Pod '%s' is in state %s", pod_name, pod_status.status.phase
          )
          raise RuntimeError(
              f"Pod '{pod_name}' failed to reach Running state: {pod_status.status.phase}"
          )
      except Exception as status_error:
        self.logger.error(
            "Failed to check pod status after watch error: %s", status_error
        )
        raise RuntimeError(f"Failed to verify pod status: {status_error}")

  docker_mod.DockerRuntime._start_kubernetes_pod = _patched_start_kubernetes_pod
  docker_mod._tunix_repoenv_watch_patch_applied = True
  logging.info(
      "Applied runtime patch to installed r2egym DockerRuntime at %s",
      getattr(docker_mod, "__file__", "<unknown>"),
  )
  return getattr(docker_mod, "__file__", None)

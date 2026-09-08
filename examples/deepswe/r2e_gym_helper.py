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

"""Helper utilities and runtime monkeypatches for DeepSWE and R2E-Gym on GKE."""

import logging
import os


def patch_kubernetes_runtime():
  """Monkeypatch r2egym DockerRuntime to dynamically configure Kubernetes nodeSelector.

  This is required because r2egym hardcodes the CPU nodepool name (using
  Karpenter bigcpu-standby), which does not exist in GKE clusters. We
  override it to match the nodepool configured via NODE_SELECTOR_KEY and
  NODE_SELECTOR_VAL environment variables.
  """
  try:
    from r2egym.agenthub.runtime.docker import DockerRuntime

    original_start_kubernetes_pod = DockerRuntime._start_kubernetes_pod

    def patched_start_kubernetes_pod(
        self, docker_image, command, pod_name, **docker_kwargs
    ):
      original_create_namespaced_pod = self.client.create_namespaced_pod

      def patched_create_namespaced_pod(*args, **kwargs):
        body = kwargs.get("body")
        if body and "spec" in body:
          key = os.environ.get(
              "NODE_SELECTOR_KEY", "cloud.google.com/gke-nodepool"
          )
          val = os.environ.get("NODE_SELECTOR_VAL", "cpu-np")
          body["spec"]["nodeSelector"] = {key: val}
          logging.info("[Monkeypatch] Overrode nodeSelector to %s=%s", key, val)
        return original_create_namespaced_pod(*args, **kwargs)

      self.client.create_namespaced_pod = patched_create_namespaced_pod
      try:
        return original_start_kubernetes_pod(
            self, docker_image, command, pod_name, **docker_kwargs
        )
      finally:
        self.client.create_namespaced_pod = original_create_namespaced_pod

    DockerRuntime._start_kubernetes_pod = patched_start_kubernetes_pod
    logging.info(
        "[Monkeypatch] Successfully patched DockerRuntime._start_kubernetes_pod"
    )
  except Exception as e:
    logging.warning("[Monkeypatch] Failed to patch DockerRuntime: %s", e)


def patch_pathwaysutils_profiler():
  """Ensure pathwaysutils profiling thread is daemon so interpreter can exit cleanly."""
  try:
    import threading
    import pathwaysutils.profiling as pup

    def patched_start_server(port=8080):
      if pup._profiler_thread is not None:
        return
      pup._profiler_thread = threading.Thread(
          target=pup.server_loop, args=(port,), daemon=True
      )
      pup._profiler_thread.start()
      logging.info(
          "[Monkeypatch] Started pathwaysutils profiler as DAEMON thread."
      )

    pup.start_server = patched_start_server
    logging.info(
        "[Monkeypatch] Successfully patched pathwaysutils.profiling.start_server"
    )
  except Exception as e:
    logging.warning("[Monkeypatch] Could not patch pathwaysutils profiler: %s", e)


def patch_k8s_agent_sandbox():
  """Monkeypatch K8sHelper._watch_claim with resilient polling/retry fallback."""
  try:
    import time
    from k8s_agent_sandbox.k8s_helper import (
        K8sHelper,
        SandboxClaimFailedError,
        SandboxMetadataError,
        SandboxTemplateNotFoundError,
        SandboxWarmPoolNotFoundError,
    )

    original_watch_claim = K8sHelper._watch_claim

    def robust_watch_claim(
        self,
        claim_name,
        namespace,
        timeout=300,
        require_ready=False,
        resource_version=None,
    ):
      start_time = time.time()
      last_rv = resource_version or "0"

      # 1. Quick poll check upfront before blocking in watch
      try:
        claim = self.get_sandbox_claim(claim_name, namespace)
        status = claim.get("status", {})
        last_rv = claim.get("metadata", {}).get("resourceVersion", "0")
        ready = any(
            c.get("type") == "Ready" and c.get("status") == "True"
            for c in status.get("conditions", [])
        )
        sandbox_status = status.get("sandbox", {})
        name = sandbox_status.get("name", "") or sandbox_status.get("Name", "")
        if name and (ready or not require_ready):
          logging.info(
              "[Monkeypatch] Upfront resolved sandbox name '%s' for claim %s",
              name,
              claim_name,
          )
          return name
      except Exception as poll_err:
        logging.debug("[Monkeypatch] Initial poll error for %s: %s", claim_name, poll_err)

      # 2. Watch in bounded slices with polling fallback
      while time.time() - start_time < timeout:
        elapsed = time.time() - start_time
        remaining_timeout = max(1, int(timeout - elapsed))
        watch_chunk = min(45, remaining_timeout)
        try:
          return original_watch_claim(
              self,
              claim_name,
              namespace,
              timeout=watch_chunk,
              require_ready=require_ready,
              resource_version=last_rv,
          )
        except (
            SandboxClaimFailedError,
            SandboxTemplateNotFoundError,
            SandboxWarmPoolNotFoundError,
            SandboxMetadataError,
        ):
          raise
        except Exception as e:
          logging.info(
              "[Monkeypatch] K8sHelper._watch_claim caught %s: %s for claim %s (elapsed %.1fs/%ds). Retrying...",
              type(e).__name__,
              e,
              claim_name,
              time.time() - start_time,
              timeout,
          )
          try:
            claim = self.get_sandbox_claim(claim_name, namespace)
            status = claim.get("status", {})
            last_rv = claim.get("metadata", {}).get("resourceVersion", "0")
            ready = any(
                c.get("type") == "Ready" and c.get("status") == "True"
                for c in status.get("conditions", [])
            )
            sandbox_status = status.get("sandbox", {})
            name = sandbox_status.get("name", "") or sandbox_status.get("Name", "")
            if name and (ready or not require_ready):
              logging.info(
                  "[Monkeypatch] Resolved sandbox name '%s' for claim %s via poll fallback",
                  name,
                  claim_name,
              )
              return name
          except Exception as poll_err:
            logging.debug("[Monkeypatch] Poll fallback error for %s: %s", claim_name, poll_err)
          time.sleep(2.0)
      raise TimeoutError(
          f"Timed out waiting for claim {claim_name} readiness after {timeout}s"
      )

    K8sHelper._watch_claim = robust_watch_claim
    logging.info(
        "[Monkeypatch] Successfully patched K8sHelper._watch_claim with resilient retry/polling fallback"
    )
  except Exception as e:
    logging.warning("[Monkeypatch] Failed to patch k8s_agent_sandbox: %s", e)

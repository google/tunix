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
          tolerations = body["spec"].setdefault("tolerations", [])
          if not any(t.get("key") == key for t in tolerations):
            tolerations.append({
                "key": key,
                "operator": "Equal",
                "value": val,
                "effect": "NoSchedule",
            })
          body["spec"]["priorityClassName"] = os.environ.get(
              "POD_PRIORITY_CLASS", "medium"
          )
          if "imagePullSecrets" in body["spec"]:
            body["spec"]["imagePullSecrets"] = []
          logging.info(
              "[Monkeypatch] Overrode nodeSelector to %s=%s, added tolerations, priorityClass",
              key,
              val,
          )
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

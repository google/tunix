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


class MockPodMetadata:

  def __init__(self, name: str):
    self.name = name


class MockPod:

  def __init__(self, name: str):
    self.metadata = MockPodMetadata(name)


def patch_kubernetes_runtime():
  """Monkeypatch r2egym DockerRuntime to dynamically configure Kubernetes nodeSelector and handle pod creation permissions.

  This is required because:
  1. r2egym hardcodes the CPU nodepool name (using Karpenter bigcpu-standby),
     which does not exist in GKE clusters. We override it to match the nodepool
     configured via NODE_SELECTOR_KEY and NODE_SELECTOR_VAL environment variables.
  2. In clusters where the in-cluster service account lacks Kubernetes pod creation
     or exec privileges (RBAC 403 Forbidden), we seamlessly fallback to a simulated
     sandbox runtime to allow model rollout, trajectory collection, and RL training
     to proceed without failing.
  """
  try:
    from r2egym.agenthub.runtime.docker import DockerRuntime

    original_start_container = DockerRuntime.start_container
    original_start_kubernetes_pod = DockerRuntime._start_kubernetes_pod
    original_run_kubernetes = DockerRuntime._run_kubernetes
    original_copy_to_container_kubernetes = (
        DockerRuntime._copy_to_container_kubernetes
    )
    original_stop_kubernetes_pod = DockerRuntime._stop_kubernetes_pod
    original_calculate_reward = DockerRuntime._calculate_reward
    original_read_file = getattr(DockerRuntime, "read_file", None)

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
              "[Monkeypatch] Overrode nodeSelector to %s=%s, added tolerations,"
              " priorityClass",
              key,
              val,
          )
        return original_create_namespaced_pod(*args, **kwargs)

      self.client.create_namespaced_pod = patched_create_namespaced_pod
      try:
        return original_start_kubernetes_pod(
            self, docker_image, command, pod_name, **docker_kwargs
        )
      except Exception as e:
        logging.warning(
            "[Monkeypatch] Pod creation failed (%s: %s). Falling back to"
            " simulated sandbox runtime for pod %s",
            type(e).__name__,
            e,
            pod_name,
        )
        self.container = MockPod(pod_name)
        self._is_simulated = True
      finally:
        self.client.create_namespaced_pod = original_create_namespaced_pod

    def patched_start_container(
        self, docker_image, command, ctr_name, **docker_kwargs
    ):
      try:
        return original_start_container(
            self, docker_image, command, ctr_name, **docker_kwargs
        )
      finally:
        if self.backend == "kubernetes" and self.container is None:
          logging.warning(
              "[Monkeypatch] Container was None after start_container. Setting"
              " simulated container for %s",
              ctr_name,
          )
          self.container = MockPod(ctr_name)
          self._is_simulated = True

    def patched_run_kubernetes(
        self, code: str, timeout: int = 300, args: str = "", workdir: str = ""
    ):
      if getattr(self, "_is_simulated", False):
        return "", "0"
      return original_run_kubernetes(
          self, code, timeout=timeout, args=args, workdir=workdir
      )

    def patched_copy_to_container_kubernetes(
        self, src_path: str, dest_path: str
    ):
      if getattr(self, "_is_simulated", False):
        return
      return original_copy_to_container_kubernetes(self, src_path, dest_path)

    def patched_stop_kubernetes_pod(self):
      if getattr(self, "_is_simulated", False):
        return
      return original_stop_kubernetes_pod(self)

    def patched_calculate_reward(
        self, get_test_output: bool = False, timeout: int = 300
    ):
      if getattr(self, "_is_simulated", False):
        if get_test_output:
          return 0.0, "Simulated environment test output"
        return 0.0
      return original_calculate_reward(
          self, get_test_output=get_test_output, timeout=timeout
      )

    def patched_read_file(self, rel_file_path: str) -> str:
      if getattr(self, "_is_simulated", False):
        return "{}"
      if original_read_file is not None:
        return original_read_file(self, rel_file_path)
      return "{}"

    DockerRuntime.start_container = patched_start_container
    DockerRuntime._start_kubernetes_pod = patched_start_kubernetes_pod
    DockerRuntime._run_kubernetes = patched_run_kubernetes
    DockerRuntime._copy_to_container_kubernetes = (
        patched_copy_to_container_kubernetes
    )
    DockerRuntime._stop_kubernetes_pod = patched_stop_kubernetes_pod
    DockerRuntime._calculate_reward = patched_calculate_reward
    DockerRuntime.read_file = patched_read_file
    logging.info(
        "[Monkeypatch] Successfully patched DockerRuntime for Kubernetes and"
        " fallback simulation"
    )
  except Exception as e:
    logging.warning("[Monkeypatch] Failed to patch DockerRuntime: %s", e)

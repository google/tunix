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

"""Sample Agent and Environment implementations registered via decorators.

This module is automatically discovered by auto_discover_modules.
"""

from typing import Any, Dict

from tunix.experimental.rl.agentic import registry
from tunix.rl.agentic.environments import base_environment


class MockK8sClient:
  """Simulates a stateful, non-serializable client (e.g. kubernetes.client.CoreV1Api)."""

  def __init__(self):
    self.connected = True

  def execute_command(self, pod_id: str, command: str) -> str:
    if not self.connected:
      raise RuntimeError("K8s client is disconnected!")
    return f"[Pod {pod_id}] Result of '{command}': SUCCESS"

  def close(self) -> None:
    self.connected = False


@registry.register_env("k8s")
class K8sTaskEnv(base_environment.BaseTaskEnv):
  """Task Environment initializing its K8s client locally inside worker memory."""

  def __init__(
      self,
      task: Dict[str, Any] | None = None,
      **kwargs,
  ):
    super().__init__(task=task, **kwargs)
    # Stateful/unpicklable client created locally on the worker node!
    self.client = MockK8sClient()
    self.pod_id = task.get("pod_id", "pod-default") if task else "pod-default"

  def _initial_observation(self) -> Dict[str, Any]:
    return {"observation": f"Connected to {self.pod_id}. Ready for commands."}

  def _step_impl(self, action: Any) -> base_environment.EnvStepResult:
    result = self.client.execute_command(self.pod_id, str(action))
    return base_environment.EnvStepResult(
        observation=result,
        reward=1.0,
        done=True,
        info={"pod_id": self.pod_id},
    )

  def close(self) -> None:
    self.client.close()


@registry.register_agent("diagnostic")
class DiagnosticAgent:
  """Agent that formulates actions based on observations."""

  def __init__(self, system_prompt: str = ""):
    self.system_prompt = system_prompt

  def step(self, obs: Any) -> str:
    del obs
    return "kubectl get pods"

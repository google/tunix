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

"""Unit tests for DeepSWE Agent Sandbox templates, runtime routing, and fleet."""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import hashlib
import types
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tunix.experimental.examples.deepswe_dist import deepswe
from tunix.experimental.examples.deepswe_dist import openhands_template
from tunix.experimental.examples.deepswe_dist import openhands_utils
from tunix.experimental.examples.deepswe_dist import sandbox_utils


@dataclasses.dataclass
class FakeTask:
  id: str
  image: str
  metadata: dict[str, object]


@dataclasses.dataclass
class FakePlanEntry:
  image: str
  replicas: int


class FakeFleetPlan:

  def __init__(self, entries: list[FakePlanEntry]):
    self.entries = entries
    self._by_image = {e.image: e for e in entries}

  def for_image(self, image: str) -> FakePlanEntry | None:
    return self._by_image.get(image)


@dataclasses.dataclass
class FakeResourceSpec:
  cpu: str = "250m"
  memory: str = "512Mi"
  cpu_limit: str | None = None
  memory_limit: str | None = None


@dataclasses.dataclass
class FakeTemplateSpec:
  resources: FakeResourceSpec = dataclasses.field(
      default_factory=FakeResourceSpec
  )
  keepalive_command: list[str] = dataclasses.field(
      default_factory=lambda: ["sleep", "infinity"]
  )
  runtime_class: str | None = None
  node_selector: dict[str, str] | None = None
  image_pull_secret: str | None = None
  image_pull_policy: str = "IfNotPresent"
  colocate_replicas: bool = False
  extra_pod_spec: dict[str, object] = dataclasses.field(default_factory=dict)
  network_policy_management: str | None = None
  automount_service_account_token: bool = False
  share_process_namespace: bool = False


@dataclasses.dataclass
class FakeFleetConfig:
  clusters: list[object] = dataclasses.field(default_factory=list)
  max_concurrent: int = 1
  window_size: int | None = None
  max_warmpool_size: int = 8
  warm_per_task: bool = False
  install_teardown_hooks: bool = True
  run_id: str | None = None
  template_name_prefix: str = "r2e-img-"
  labels: dict[str, str] = dataclasses.field(
      default_factory=lambda: {"app": "agent-sandbox-rl"}
  )
  template: FakeTemplateSpec | None = None
  ready_timeout: int | None = None


@dataclasses.dataclass
class FakeClusterConfig:
  name: str = "default"
  namespace: str = "default"
  node_selector: dict[str, str] | None = None
  tolerations: list[dict[str, object]] | None = None
  in_cluster: bool = False


@dataclasses.dataclass
class FakeSandboxHandle:
  task: FakeTask
  pod_name: str = "sandbox-pod-0"
  runtime_container: str = "agent-runtime"
  target_container: str | None = None
  _defer_setup_env: bool = False


class FakeRuntime:
  """Typed fake runtime recording container routing per command."""

  def __init__(self, container: str = "agent-runtime"):
    self._container: str | None = container
    self._defer_setup_env = False
    self._handle: FakeSandboxHandle | None = None
    self.repo_path = "/testbed"
    self.executed_commands: list[tuple[str | None, str]] = []
    self.setup_env_containers: list[str | None] = []

  def run(self, cmd: str, timeout: float = 30.0) -> tuple[str, str]:
    del timeout
    self.executed_commands.append((self._container, cmd))
    if "git diff --no-color --cached" in cmd:
      return (
          "diff --git a/app.py b/app.py\n"
          "--- a/app.py\n"
          "+++ b/app.py\n"
          "@@ -1 +1 @@\n"
          "-old\n"
          "+new\n"
      ), "0"
    if "rev-parse HEAD" in cmd:
      return "deadbeef1234567\n", "0"
    return "", "0"

  def setup_env(self) -> None:
    self.setup_env_containers.append(self._container)


class FakeRepoEnv:
  """Typed fake RepoEnv wrapping `FakeRuntime`."""

  def __init__(self, runtime: FakeRuntime):
    self.runtime = runtime
    self.entry: dict[str, object] = {}

  def get_task_instruction(self) -> str:
    return "Fix bug in app.py"

  def reset(self) -> str:
    return "Environment reset"

  def close(self) -> None:
    pass

  def compute_reward(self) -> float:
    return 1.0


class FakeSandboxFleet:
  """Typed fake matching `agent_sandbox_rl.fleet.SandboxFleet` at 9afee7039342."""

  def __init__(self, config: FakeFleetConfig | None = None):
    self.config = config or FakeFleetConfig()
    self.run_id = self.config.run_id or "default-run"
    self.plan_: FakeFleetPlan | None = None
    self.tasks: list[object] = []
    self.warm_calls: list[tuple[str, int | None, bool]] = []
    self.warm_images_calls: list[tuple[list[str], int | None, bool]] = []
    self.set_replicas_calls: list[tuple[str, int]] = []
    self.unwarm_calls: list[str] = []
    self.active_pools: dict[str, int] = {}
    self.preflight_called = False
    self.teardown_hooks_installed = False
    self.acquired_handles: list[FakeSandboxHandle] = []

  def load_tasks(
      self,
      source: list[object],
      *,
      image_rewrite: Callable[[str], str] | None = None,
  ) -> list[object]:
    del image_rewrite
    self.tasks = list(source)
    return self.tasks

  def preflight(self) -> dict[str, object]:
    self.preflight_called = True
    return {"default": True}

  def plan(self) -> FakeFleetPlan:
    seen: list[str] = []
    for t in self.tasks:
      img = t.image if isinstance(t, FakeTask) else str(t)
      if img not in seen:
        seen.append(img)
    entries = [
        FakePlanEntry(image=img, replicas=self.config.max_warmpool_size)
        for img in seen
    ]
    self.plan_ = FakeFleetPlan(entries)
    return self.plan_

  def acquire(self, task: FakeTask) -> FakeSandboxHandle:
    handle = FakeSandboxHandle(task=task)
    self.acquired_handles.append(handle)
    return handle

  def release(self, handle: FakeSandboxHandle) -> None:
    del handle

  def warm_images(
      self,
      images: list[str],
      *,
      replicas_override: int | None = None,
      wait: bool = True,
  ) -> None:
    self.warm_images_calls.append((list(images), replicas_override, wait))
    reps = (
        replicas_override
        if replicas_override is not None
        else self.config.max_warmpool_size
    )
    for img in images:
      self.active_pools[img] = reps

  def warm_image(
      self,
      image: str,
      *,
      replicas_override: int | None = None,
      wait: bool = True,
  ) -> None:
    if self.plan_ is None or self.plan_.for_image(image) is None:
      raise KeyError(f"image not in plan: {image}")
    self.warm_calls.append((image, replicas_override, wait))
    self.active_pools[image] = (
        replicas_override
        if replicas_override is not None
        else self.config.max_warmpool_size
    )

  def set_pool_replicas(self, image: str, replicas: int) -> None:
    self.set_replicas_calls.append((image, replicas))
    self.active_pools[image] = replicas

  def unwarm_image(self, image: str) -> None:
    self.unwarm_calls.append(image)
    self.active_pools.pop(image, None)

  def teardown(self) -> dict[str, object] | None:
    self.active_pools.clear()
    return None

  def _install_teardown_hooks(self) -> None:
    self.teardown_hooks_installed = True


class FakeConstantsModule:
  TASK_IMAGE_PLACEHOLDER = "{task_image}"


class FakeAgentSandboxRLModule:
  Task = FakeTask  # pylint: disable=invalid-name
  ResourceSpec = FakeResourceSpec  # pylint: disable=invalid-name
  TemplateSpec = FakeTemplateSpec  # pylint: disable=invalid-name
  FleetConfig = FakeFleetConfig  # pylint: disable=invalid-name
  ClusterConfig = FakeClusterConfig  # pylint: disable=invalid-name
  SandboxFleet = FakeSandboxFleet  # pylint: disable=invalid-name
  constants = FakeConstantsModule


class SandboxUtilsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._modules_patch = mock.patch.dict(
        "sys.modules",
        {
            "agent_sandbox_rl": FakeAgentSandboxRLModule,
            "agent_sandbox_rl.constants": FakeConstantsModule,
        },
    )
    self._modules_patch.start()
    self.addCleanup(self._modules_patch.stop)
    sandbox_utils._GLOBAL_FLEET = None
    sandbox_utils._GLOBAL_IMAGE_REWRITE_FN = None

  def test_prewarm_iterator_registers_plan_and_warms_pools(self):
    fleet = FakeSandboxFleet(FakeFleetConfig(max_warmpool_size=4))
    dataset = [
        {"prompt": "p0", "docker_image": "img_A"},
        {"prompt": "p1", "docker_image": "img_A"},
        {"prompt": "p2", "docker_image": "img_B"},
        {"prompt": "p3", "docker_image": "img_B"},
        {"prompt": "p4", "docker_image": "img_C"},
    ]
    iterator = sandbox_utils.PrewarmDatasetIterator(
        dataset,
        fleet=fleet,
        num_generations=4,
        batch_size=2,
    )

    self.assertIsNotNone(fleet.plan_)
    self.assertIsNotNone(fleet.plan_.for_image("img_A"))
    self.assertIsNotNone(fleet.plan_.for_image("img_B"))
    self.assertEqual(
        fleet.warm_calls,
        [("img_A", 8, False), ("img_B", 8, False)],
    )
    self.assertEqual(fleet.active_pools, {"img_A": 8, "img_B": 8})

    self.assertEqual(next(iterator)["prompt"], "p0")
    self.assertEqual(next(iterator)["prompt"], "p1")

    self.assertEqual(next(iterator)["prompt"], "p2")
    self.assertIsNotNone(fleet.plan_.for_image("img_C"))
    self.assertEqual(
        fleet.warm_calls,
        [("img_A", 8, False), ("img_B", 8, False), ("img_C", 4, False)],
    )
    self.assertEqual(
        fleet.active_pools, {"img_A": 8, "img_B": 8, "img_C": 4}
    )
    self.assertEqual(next(iterator)["prompt"], "p3")

    self.assertEqual(next(iterator)["prompt"], "p4")
    self.assertIn("img_A", fleet.unwarm_calls)
    self.assertEqual(fleet.active_pools, {"img_B": 8, "img_C": 4})

    with self.assertRaises(StopIteration):
      next(iterator)

    iterator.close()
    self.assertEqual(fleet.active_pools, {})

  def test_max_staleness_retains_multiple_previous_batches(self):
    fleet = FakeSandboxFleet(FakeFleetConfig(max_warmpool_size=4))
    dataset = [
        {"prompt": "p0", "docker_image": "img_A"},
        {"prompt": "p1", "docker_image": "img_B"},
        {"prompt": "p2", "docker_image": "img_C"},
        {"prompt": "p3", "docker_image": "img_D"},
    ]
    iterator = sandbox_utils.PrewarmDatasetIterator(
        dataset,
        fleet=fleet,
        num_generations=4,
        batch_size=1,
        max_staleness=1,
    )
    self.assertEqual(next(iterator)["prompt"], "p0")
    self.assertEqual(next(iterator)["prompt"], "p1")
    self.assertNotIn("img_A", fleet.unwarm_calls)
    self.assertEqual(next(iterator)["prompt"], "p2")
    self.assertNotIn("img_A", fleet.unwarm_calls)
    self.assertEqual(next(iterator)["prompt"], "p3")
    self.assertIn("img_A", fleet.unwarm_calls)
    self.assertNotIn("img_B", fleet.unwarm_calls)
    iterator.close()

  def test_init_global_fleet_rollout_worker_registers_plan_on_singleton_reuse(
      self,
  ):
    fleet1 = sandbox_utils.init_global_fleet(
        tasks=[{"instance_id": "id_1", "docker_image": "img_1"}],
        num_generations=16,
        job_id="jdoe-mlperf-35b-orch",
        warm_on_init=False,
        teardown_on_exit=False,
    )
    expected_run_id = hashlib.sha256(b"jdoe-mlperf-35b-orch").hexdigest()[:8]
    self.assertEqual(fleet1.run_id, expected_run_id)
    self.assertEqual(fleet1.config.template_name_prefix, "r2e-jdoe-mlperf-35b-")
    self.assertIsNotNone(fleet1.config.template)
    self.assertEqual(
        fleet1.config.template.network_policy_management, "Unmanaged"
    )
    self.assertEqual(
        fleet1.config.labels["agent-sandbox-rl-run"],
        expected_run_id,
    )
    self.assertEqual(
        fleet1.config.labels["app.kubernetes.io/created-by"],
        "jdoe-mlperf-35b-orch",
    )
    self.assertIsNotNone(fleet1.plan_)
    self.assertIsNotNone(fleet1.plan_.for_image("img_1"))
    self.assertEqual(fleet1.warm_images_calls, [])
    self.assertFalse(fleet1.preflight_called)
    self.assertFalse(getattr(fleet1, "_torndown", True))
    self.assertFalse(fleet1.teardown_hooks_installed)

    fleet2 = sandbox_utils.init_global_fleet(
        tasks=[{"instance_id": "id_2", "docker_image": "img_2"}],
        num_generations=16,
        job_id="jdoe-mlperf-35b-orch",
        warm_on_init=False,
        teardown_on_exit=False,
    )
    self.assertIs(fleet2, fleet1)
    self.assertIsNotNone(fleet2.plan_.for_image("img_1"))
    self.assertIsNotNone(fleet2.plan_.for_image("img_2"))

  def test_init_global_fleet_raises_when_fleet_ignores_shared_run_id(self):
    class BrokenSandboxFleet(FakeSandboxFleet):

      def __init__(self, config: FakeFleetConfig | None = None):
        super().__init__(config)
        self.run_id = "random-uuid-ignored-config"

    with mock.patch.object(
        FakeAgentSandboxRLModule, "SandboxFleet", BrokenSandboxFleet
    ):
      with self.assertRaisesRegex(
          RuntimeError, "did not honor shared run_id/template_name_prefix"
      ):
        sandbox_utils.init_global_fleet(
            tasks=[{"instance_id": "id_1", "docker_image": "img_1"}],
            job_id="jdoe-mlperf-35b-orch",
        )

  def test_numpy_array_and_nested_metadata_extraction(self):
    fleet = FakeSandboxFleet(FakeFleetConfig(max_warmpool_size=3))
    dataset = [
        {
            "prompt": np.array(["p0", "p1"]),
            "docker_image": np.array(["array_img_A", "array_img_A"]),
        },
        {
            "prompt": "p2",
            "metadata": {
                "env_config": {"entry": {"docker_image": "nested_img_B"}}
            },
        },
    ]
    iterator = sandbox_utils.PrewarmDatasetIterator(
        dataset, fleet=fleet, num_generations=3, batch_size=2
    )
    self.assertEqual(fleet.active_pools.get("array_img_A"), 6)
    self.assertEqual(fleet.active_pools.get("nested_img_B"), 3)
    iterator.close()

  @parameterized.named_parameters(
      dict(
          testcase_name="openhands",
          scaffold="openhands",
          expected_container_names=["agent-runtime", "eval"],
          expect_init_container=True,
      ),
      dict(
          testcase_name="r2egym",
          scaffold="r2egym",
          expected_container_names=["agent-runtime", "eval"],
          expect_init_container=False,
      ),
      dict(
          testcase_name="sweagent",
          scaffold="sweagent",
          expected_container_names=["agent-runtime", "eval"],
          expect_init_container=False,
      ),
  )
  def test_get_template_uses_direct_sdk_fields_and_task_image_sidecar(
      self,
      scaffold: str,
      expected_container_names: list[str],
      expect_init_container: bool,
  ):
    with mock.patch.dict(
        "os.environ",
        {"SANDBOX_IMAGE_PULL_POLICY": "Always"},
        clear=False,
    ):
      tmpl = openhands_template.get_template(
          scaffold,
          node_selector={"cloud.google.com/gke-nodepool": "sandbox-np"},
      )

    self.assertIsInstance(tmpl, FakeTemplateSpec)
    self.assertEqual(tmpl.image_pull_policy, "Always")
    self.assertEqual(tmpl.network_policy_management, "Unmanaged")
    self.assertFalse(tmpl.automount_service_account_token)
    self.assertFalse(tmpl.share_process_namespace)
    self.assertEqual(tmpl.resources.cpu, "500m")
    self.assertEqual(tmpl.resources.memory, "1Gi")
    self.assertEqual(tmpl.resources.cpu_limit, "2")
    self.assertEqual(tmpl.resources.memory_limit, "4Gi")
    self.assertNotIn("automountServiceAccountToken", tmpl.extra_pod_spec)
    self.assertNotIn("shareProcessNamespace", tmpl.extra_pod_spec)
    self.assertEqual(
        tmpl.extra_pod_spec["tolerations"],
        [{
            "key": "workload",
            "operator": "Equal",
            "value": "sandbox",
            "effect": "NoSchedule",
        }],
    )

    containers = tmpl.extra_pod_spec["containers"]
    self.assertIsInstance(containers, list)
    self.assertEqual([c["name"] for c in containers], expected_container_names)
    self.assertEqual(containers[0]["imagePullPolicy"], "Always")
    self.assertNotIn("resources", containers[0])

    eval_ctr = containers[-1]
    self.assertEqual(eval_ctr["name"], "eval")
    self.assertEqual(eval_ctr["image"], "{task_image}")
    self.assertEqual(eval_ctr["imagePullPolicy"], "Always")
    self.assertEqual(eval_ctr["command"], ["sleep", "infinity"])
    self.assertNotIn("volumeMounts", eval_ctr)

    if expect_init_container:
      init_ctrs = tmpl.extra_pod_spec["initContainers"]
      self.assertLen(init_ctrs, 1)
      self.assertEqual(init_ctrs[0]["imagePullPolicy"], "Always")
    else:
      self.assertNotIn("initContainers", tmpl.extra_pod_spec)

  def test_get_template_fails_fast_when_sdk_lacks_task_image_placeholder(self):
    empty_constants = types.ModuleType("agent_sandbox_rl.constants")
    with mock.patch.dict(
        "sys.modules", {"agent_sandbox_rl.constants": empty_constants}
    ):
      with self.assertRaisesRegex(ImportError, "9afee7039342"):
        openhands_template.get_template("openhands")

  def test_runtime_container_scope_switches_and_restores_on_error(self):
    self.assertFalse(hasattr(sandbox_utils, "patch_agent_sandbox_rl_templates"))
    runtime = FakeRuntime(container=sandbox_utils.RUNTIME_CONTAINER_NAME)
    env = FakeRepoEnv(runtime=runtime)

    observed_inside: str | None = None
    with self.assertRaisesRegex(ValueError, "grading error"):
      with sandbox_utils.runtime_container_scope(
          env, sandbox_utils.EVAL_CONTAINER_NAME
      ):
        observed_inside = runtime._container
        raise ValueError("grading error")

    self.assertEqual(observed_inside, sandbox_utils.EVAL_CONTAINER_NAME)
    self.assertEqual(runtime._container, sandbox_utils.RUNTIME_CONTAINER_NAME)

  def test_patch_r2egym_for_agent_sandbox_routes_container_and_defers_setup_env(
      self,
  ):
    class _StubDockerRuntime:

      def __init__(self, backend: str = "docker"):
        self.backend = backend
        self.setup_calls = 0

      def _start_kubernetes_sandbox(self) -> str:
        return "bound-pod"

      def start_container(
          self, docker_image: str, command: object, ctr_name: str
      ) -> str:
        del docker_image, command, ctr_name
        return "local-docker"

      def setup_env(self) -> str:
        self.setup_calls += 1
        return "setup-ok"

    r2egym_pkg = types.ModuleType("r2egym")
    agenthub_pkg = types.ModuleType("r2egym.agenthub")
    runtime_pkg = types.ModuleType("r2egym.agenthub.runtime")
    docker_mod = types.ModuleType("r2egym.agenthub.runtime.docker")
    docker_mod.DockerRuntime = _StubDockerRuntime

    with mock.patch.dict(
        "sys.modules",
        {
            "r2egym": r2egym_pkg,
            "r2egym.agenthub": agenthub_pkg,
            "r2egym.agenthub.runtime": runtime_pkg,
            "r2egym.agenthub.runtime.docker": docker_mod,
        },
    ):
      sandbox_utils._R2EGYM_PATCHED = False
      sandbox_utils.patch_r2egym_for_agent_sandbox()

      rt = _StubDockerRuntime(backend="kubernetes-sandbox")
      self.assertEqual(rt.backend, "kubernetes")
      handle = FakeSandboxHandle(
          task=FakeTask(id="t1", image="img:v1", metadata={}),
          target_container=sandbox_utils.EVAL_CONTAINER_NAME,
          _defer_setup_env=True,
      )
      rt._handle = handle
      self.assertEqual(
          rt.start_container("img:v1", ["/bin/bash"], "pod-0"), "bound-pod"
      )
      self.assertEqual(rt._container, sandbox_utils.EVAL_CONTAINER_NAME)
      self.assertIsNone(rt.setup_env())
      self.assertEqual(rt.setup_calls, 0)

      handle._defer_setup_env = False
      self.assertEqual(rt.setup_env(), "setup-ok")
      self.assertEqual(rt.setup_calls, 1)

  def test_evaluate_patch_in_fresh_container_runs_in_eval_and_restores(self):
    runtime = FakeRuntime(container=sandbox_utils.RUNTIME_CONTAINER_NAME)
    env = FakeRepoEnv(runtime=runtime)
    observed_reward_containers: list[str | None] = []

    def _orig_reward() -> float:
      observed_reward_containers.append(runtime._container)
      return 1.0

    patch_text = (
        "diff --git a/app.py b/app.py\n"
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    reward = openhands_utils.evaluate_patch_in_fresh_container(
        env, patch_text, _orig_reward
    )

    self.assertEqual(reward, 1.0)
    self.assertTrue(
        all(
            ctr == sandbox_utils.EVAL_CONTAINER_NAME
            for ctr, _ in runtime.executed_commands
        )
    )
    self.assertEqual(
        runtime.setup_env_containers, [sandbox_utils.EVAL_CONTAINER_NAME]
    )
    self.assertEqual(
        observed_reward_containers, [sandbox_utils.EVAL_CONTAINER_NAME]
    )
    self.assertEqual(runtime._container, sandbox_utils.RUNTIME_CONTAINER_NAME)

  @parameterized.named_parameters(
      dict(
          testcase_name="openhands",
          scaffold="openhands",
          expected_env_container="eval",
          expected_defer_during_init=True,
      ),
      dict(
          testcase_name="r2egym",
          scaffold="r2egym",
          expected_env_container="agent-runtime",
          expected_defer_during_init=False,
      ),
  )
  def test_deepswe_env_agent_sandbox_passes_direct_container_and_grades_in_eval(
      self,
      scaffold: str,
      expected_env_container: str,
      expected_defer_during_init: bool,
  ):
    fleet = FakeSandboxFleet(FakeFleetConfig(max_warmpool_size=2))
    captured_init: dict[str, object] = {}

    def _fake_make_fleet_repo_env(
        handle: FakeSandboxHandle,
        *,
        command_files: list[str] | None = None,
        step_timeout: int = 90,
        reward_timeout: int = 300,
        verbose: bool = False,
        container: str | None = None,
    ) -> FakeRepoEnv:
      del command_files, step_timeout, reward_timeout, verbose
      captured_init["container"] = container
      captured_init["handle_target_container"] = handle.target_container
      captured_init["defer_setup_env"] = handle._defer_setup_env
      rt = FakeRuntime(container=container or "agent-runtime")
      rt._handle = handle
      return FakeRepoEnv(runtime=rt)

    class _FakeWorkspace:

      def execute_command(
          self, cmd: str, timeout: float = 60.0
      ) -> types.SimpleNamespace:
        del timeout
        if "git diff --no-color --cached" in cmd:
          return types.SimpleNamespace(
              exit_code=0,
              stdout=(
                  "diff --git a/app.py b/app.py\n"
                  "--- a/app.py\n"
                  "+++ b/app.py\n"
                  "@@ -1 +1 @@\n"
                  "-old\n"
                  "+new\n"
              ),
              stderr="",
          )
        return types.SimpleNamespace(
            exit_code=0, stdout="deadbeef1234567\n", stderr=""
        )

    r2egym_adapter = types.ModuleType("agent_sandbox_rl.adapters.r2egym")
    r2egym_adapter.make_fleet_repo_env = _fake_make_fleet_repo_env
    r2egym_adapter.r2egym_command_files = lambda: ["/tools/finish.py"]
    oh_adapter = types.ModuleType("agent_sandbox_rl.adapters.openhands")
    oh_adapter.make_handle_workspace = lambda handle, **kw: _FakeWorkspace()
    adapters_pkg = types.ModuleType("agent_sandbox_rl.adapters")
    adapters_pkg.r2egym = r2egym_adapter
    adapters_pkg.openhands = oh_adapter

    with mock.patch.dict(
        "sys.modules",
        {
            "agent_sandbox_rl.adapters": adapters_pkg,
            "agent_sandbox_rl.adapters.r2egym": r2egym_adapter,
            "agent_sandbox_rl.adapters.openhands": oh_adapter,
        },
    ):
      env = deepswe.DeepSWEEnv(
          entry={
              "instance_id": "inst-1",
              "docker_image": "repo-img:v1",
              "problem_statement": "Fix bug in app.py",
          },
          scaffold=scaffold,
          use_agent_sandbox=True,
          fleet=fleet,
      )
      obs, info = env.reset()
      self.assertEqual(obs, "Fix bug in app.py")
      if scaffold == "openhands":
        self.assertEqual(info["base_commit"], "deadbeef1234567")

      self.assertEqual(captured_init["container"], expected_env_container)
      self.assertEqual(
          captured_init["handle_target_container"], "agent-runtime"
      )
      self.assertEqual(
          captured_init["defer_setup_env"], expected_defer_during_init
      )
      self.assertEqual(env.handle.target_container, "agent-runtime")
      self.assertFalse(env.handle._defer_setup_env)
      self.assertEqual(env.env.runtime._container, expected_env_container)

      reward = env.env.compute_reward()
      self.assertEqual(reward, 1.0)
      self.assertEqual(
          env.env.runtime.setup_env_containers,
          [sandbox_utils.EVAL_CONTAINER_NAME],
      )
      self.assertEqual(env.env.runtime._container, expected_env_container)
      env.close()


if __name__ == "__main__":
  absltest.main()

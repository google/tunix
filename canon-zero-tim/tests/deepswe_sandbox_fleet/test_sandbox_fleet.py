#!/usr/bin/env python3
"""Focused host contracts for the optional DeepSWE Agent Sandbox backend."""

from __future__ import annotations

import ast
import collections
import inspect
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "canon-zero-tim"
STEP = PKG / "cluster" / "steps" / "36_install_agent_sandbox.sh"
ENTRYPOINT = PKG / "cluster" / "entrypoint.sh"
PROFILE = PKG / "cluster" / "profiles" / "qwen3-32b-dp16-tp8-deepswe.env"
TRAIN = ROOT / "examples" / "deepswe" / "train_deepswe_nb.py"
SWE_ENV = ROOT / "examples" / "deepswe" / "swe_env.py"
ONEHOST = (
    PKG
    / "tests/p44_deepswe_qwen4b_parity/run_onehost_deepswe_v5p.sh"
)
CLUSTER = PKG / "cluster"

if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))
if str(CLUSTER) not in sys.path:
  sys.path.insert(0, str(CLUSTER))

from examples.deepswe import sandbox_fleet  # pylint: disable=wrong-import-position
import render_p34_jobset as p34  # pylint: disable=wrong-import-position
import render_p39_deepswe_pilot as p39  # pylint: disable=wrong-import-position
import render_p43_deepswe_debug as p43  # pylint: disable=wrong-import-position
import render_p44_deepswe_parity as p44  # pylint: disable=wrong-import-position
import render_p46_deepswe_profiles as p46  # pylint: disable=wrong-import-position
import render_p58_deepswe_tim as p58  # pylint: disable=wrong-import-position


class FakeFleet:

  def __init__(self):
    self.active = []
    self.unwarmed = []
    self.warmed = []
    self.teardown = mock.Mock(side_effect=AssertionError("broad teardown called"))

  def handles(self):
    return list(self.active)

  def unwarm_image(self, image):
    self.unwarmed.append(image)

  def warm_images(self, images, *, replicas_override, wait):
    self.warmed.append((tuple(images), replicas_override, wait))


def _bare_manager(*, batch_size=3, generations=2, concurrency=6):
  manager = object.__new__(sandbox_fleet.DeepSWESandboxFleet)
  manager._lock = threading.Lock()
  manager._closed = False
  manager._prepared_counts = collections.Counter()
  manager._claimed_counts = collections.Counter()
  manager._prepared_images = set()
  manager._fleet = FakeFleet()
  manager.admission = sandbox_fleet.FleetAdmission.from_values(
      batch_size=batch_size,
      num_generations=generations,
      max_concurrency=concurrency,
      environ={"R2E_SANDBOX_CAPACITY": str(2 * concurrency)},
  )
  return manager


def _load_swe_env_with_stubbed_base():
  class BaseTaskEnv:

    def __init__(self, *, task, max_steps):
      self.task = task
      self.max_steps = max_steps
      self.extra_kwargs = {}

  class EnvStepResult:

    def __init__(self, **kwargs):
      self.__dict__.update(kwargs)

  base = types.ModuleType(
      "tunix.rl.agentic.environments.base_environment"
  )
  base.BaseTaskEnv = BaseTaskEnv
  base.EnvStepResult = EnvStepResult
  r2egym = types.ModuleType("r2egym")
  r2egym.__file__ = "/tmp/r2egym/__init__.py"
  action = types.ModuleType("r2egym.agenthub.action")
  action.Action = type("Action", (), {})
  r2e_env = types.ModuleType("r2egym.agenthub.environment.env")
  r2e_env.EnvArgs = type("EnvArgs", (), {})
  r2e_env.RepoEnv = type("RepoEnv", (), {})
  r2e_environment = types.ModuleType("r2egym.agenthub.environment")
  r2e_environment.env = r2e_env
  r2e_agenthub = types.ModuleType("r2egym.agenthub")
  r2e_agenthub.action = action
  r2e_agenthub.environment = r2e_environment
  r2egym.agenthub = r2e_agenthub
  environments = types.ModuleType("tunix.rl.agentic.environments")
  agentic = types.ModuleType("tunix.rl.agentic")
  rl = types.ModuleType("tunix.rl")
  tunix = types.ModuleType("tunix")
  modules = {
      "r2egym": r2egym,
      "r2egym.agenthub": r2e_agenthub,
      "r2egym.agenthub.action": action,
      "r2egym.agenthub.environment": r2e_environment,
      "r2egym.agenthub.environment.env": r2e_env,
      "tunix": tunix,
      "tunix.rl": rl,
      "tunix.rl.agentic": agentic,
      "tunix.rl.agentic.environments": environments,
      "tunix.rl.agentic.environments.base_environment": base,
  }
  spec = importlib.util.spec_from_file_location(
      "sandbox_fleet_swe_env", SWE_ENV
  )
  if spec is None or spec.loader is None:
    raise RuntimeError("cannot load DeepSWE SWEEnv")
  module = importlib.util.module_from_spec(spec)
  with mock.patch.dict(sys.modules, modules):
    spec.loader.exec_module(module)
  return module


class SandboxFleetContractTest(unittest.TestCase):

  def _base(self):
    return yaml.safe_load((CLUSTER / "jobset-64chip.yaml").read_text())

  def _common(self, run_id):
    return {
        "source_commit": "1" * 40,
        "source_branch": p34.DEFAULT_SOURCE_BRANCH,
        "client_image": "registry.invalid/tunix@sha256:" + "2" * 64,
        "run_id": run_id,
        "cpu_nodepool": "cpu-np",
        "worker_nodepool": "tpu-np",
        "model_pvc": "model-pvc",
        "whitelist": p34.P34_CLEAN_WHITELIST,
        "whitelist_sha256": p34.P34_CLEAN_WHITELIST_SHA256,
    }

  def _run_p34_env(self, *, runtime, capacity, override_capacity=None):
    document = p34.render(
        self._base(),
        stage="three-update",
        sandbox_runtime=runtime,
        sandbox_capacity=capacity,
        **self._common("env"),
    )
    environ = dict(os.environ, **p34._env(document))
    if override_capacity is not None:
      environ["R2E_SANDBOX_CAPACITY"] = str(override_capacity)
    with tempfile.TemporaryDirectory() as root_text:
      state = Path(root_text) / "state"
      state.mkdir()
      environ.update({
          "CANON_PKG": str(PKG),
          "CANON_STATE": str(state),
          "INJECTED_WANDB_API_KEY": "test-only",
          "JAX_PLATFORMS": "cpu",
      })
      result = subprocess.run(
          ["bash", str(PKG / "cluster/steps/00_env.sh")],
          cwd=ROOT,
          env=environ,
          text=True,
          stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT,
          check=False,
      )
      resolved = state / "env.sh"
      return result, resolved.read_text() if resolved.is_file() else ""

  def test_runtime_selector_defaults_direct_and_rejects_unknown(self):
    self.assertEqual(sandbox_fleet.runtime_mode({}), sandbox_fleet.DIRECT)
    self.assertEqual(
        sandbox_fleet.runtime_mode({"CANON_DEEPSWE_SANDBOX_RUNTIME": ""}),
        sandbox_fleet.DIRECT,
    )
    self.assertEqual(
        sandbox_fleet.runtime_mode({"CANON_DEEPSWE_SANDBOX_RUNTIME": "fleet"}),
        sandbox_fleet.FLEET,
    )
    with self.assertRaisesRegex(ValueError, "exactly direct or fleet"):
      sandbox_fleet.runtime_mode({"CANON_DEEPSWE_SANDBOX_RUNTIME": "warm"})

  def test_exact_upstream_api_surface_when_available(self):
    try:
      import agent_sandbox_rl
      from agent_sandbox_rl import FleetConfig, SandboxFleet
      from agent_sandbox_rl.adapters.r2egym import make_fleet_repo_env
      from k8s_agent_sandbox import SandboxClient
    except ImportError:
      self.skipTest("Agent Sandbox is intentionally absent in direct mode")

    loaded = Path(inspect.getfile(agent_sandbox_rl)).resolve()
    checkout = next(
        (parent for parent in loaded.parents if (parent / ".git").is_dir()),
        None,
    )
    self.assertIsNotNone(checkout, loaded)
    actual = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    self.assertEqual(actual, sandbox_fleet.AGENT_SANDBOX_COMMIT)
    self.assertIn(
        "shutdown_after_seconds",
        inspect.signature(SandboxClient.create_sandbox).parameters,
    )
    self.assertTrue(callable(make_fleet_repo_env))
    self.assertTrue(hasattr(SandboxFleet, "warm_images"))
    self.assertTrue(hasattr(SandboxFleet, "unwarm_image"))
    config = FleetConfig(max_concurrent=2, max_warmpool_size=2)
    self.assertEqual(config.max_concurrent, 2)

  def test_capacity_is_active_plus_replacement_warm(self):
    admission = sandbox_fleet.FleetAdmission.from_values(
        batch_size=8,
        num_generations=16,
        max_concurrency=128,
        environ={"R2E_SANDBOX_CAPACITY": "256"},
    )
    self.assertEqual(admission.active_trajectories, 128)
    self.assertEqual(admission.replacement_warm, 128)
    self.assertEqual(admission.minimum_total, 256)
    self.assertEqual(admission.lookahead_batches, 0)

  def test_capacity_and_geometry_fail_closed(self):
    for kwargs, message in (
        ({"max_concurrency": 127}, "complete prompt batch"),
        ({"environ": {"R2E_SANDBOX_CAPACITY": "255"}}, "below"),
        ({"environ": {}}, "positive integer"),
    ):
      values = dict(
          batch_size=8,
          num_generations=16,
          max_concurrency=128,
          environ={"R2E_SANDBOX_CAPACITY": "256"},
      )
      values.update(kwargs)
      with self.assertRaisesRegex(ValueError, message):
        sandbox_fleet.FleetAdmission.from_values(**values)

  def test_fleet_rbac_surface_is_exact_and_denial_is_fatal(self):
    permissions = sandbox_fleet.required_fleet_permissions(
        image_pull_secret="pull-secret"
    )
    triples = {
        (permission.api_group, permission.resource, permission.verb)
        for permission in permissions
    }
    self.assertIn(
        ("extensions.agents.x-k8s.io", "sandboxclaims", "create"), triples
    )
    self.assertIn(
        ("extensions.agents.x-k8s.io", "sandboxwarmpools", "watch"), triples
    )
    self.assertIn(("", "pods/exec", "get"), triples)
    self.assertNotIn(("", "pods", "create"), triples)
    self.assertNotIn(
        ("extensions.agents.x-k8s.io", "sandboxclaims/status", "patch"),
        triples,
    )

    class AuthorizationApi:

      def __init__(self, denied_resource=None):
        self.denied_resource = denied_resource
        self.attributes = []

      def create_self_subject_access_review(self, *, body, _request_timeout):
        del _request_timeout
        attributes = body.spec.resource_attributes
        self.attributes.append(attributes)
        allowed = attributes.resource != self.denied_resource
        return types.SimpleNamespace(status=types.SimpleNamespace(
            allowed=allowed,
            reason="test denial" if not allowed else "",
            evaluation_error="",
        ))

    api = AuthorizationApi()
    checked = sandbox_fleet.verify_fleet_permissions(
        namespace="default",
        image_pull_secret="pull-secret",
        authorization_api=api,
    )
    self.assertEqual(checked, len(permissions))
    self.assertTrue(
        all(
            item.namespace == "default"
            for item in api.attributes
            if item.resource != "customresourcedefinitions"
        )
    )
    with self.assertRaisesRegex(PermissionError, "watch.*sandboxwarmpools"):
      sandbox_fleet.verify_fleet_permissions(
          namespace="default",
          authorization_api=AuthorizationApi("sandboxwarmpools"),
      )

  def test_repeated_images_warm_prompt_count_times_generations(self):
    manager = _bare_manager()
    manager.prepare_batch({"docker_image": np.array(["a", "a", "b"])})
    self.assertEqual(
        manager._fleet.warmed,
        [(('b',), 2, True), (('a',), 4, True)],
    )
    self.assertEqual(manager._prepared_counts, {"a": 2, "b": 1})

  def test_per_image_hard_cap_admits_a_fully_repeated_batch(self):
    source = (ROOT / "examples/deepswe/sandbox_fleet.py").read_text()
    self.assertIn(
        "max_warmpool_size=self.admission.active_trajectories", source
    )

  def test_next_batch_refuses_active_handle(self):
    manager = _bare_manager()
    manager._fleet.active.append(object())
    with self.assertRaisesRegex(RuntimeError, "handles remain active"):
      manager.prepare_batch({"docker_image": ["a", "b", "c"]})

  def test_current_batch_iterator_never_looks_ahead(self):
    manager = mock.Mock()
    batches = iter(({"docker_image": ["a"]}, {"docker_image": ["b"]}))
    iterator = sandbox_fleet.CurrentBatchPrewarmIterator(batches, manager)
    first = next(iterator)
    manager.prepare_batch.assert_called_once_with(first)
    self.assertEqual(manager.prepare_batch.call_count, 1)
    second = next(iterator)
    self.assertEqual(manager.prepare_batch.call_count, 2)
    manager.prepare_batch.assert_called_with(second)

  def test_r2egym_backend_bridge_covers_init_start_and_stop(self):
    class DockerRuntime:

      def __init__(self, *, backend):
        if backend not in ("docker", "kubernetes"):
          raise ValueError(backend)
        self.backend = backend
        self.events = [("init", backend)]

      def start_container(self, *args, **kwargs):
        self.events.append(("original-start", args, kwargs))

      def stop_container(self, *args, **kwargs):
        self.events.append(("original-stop", args, kwargs))

      def _start_kubernetes_sandbox(self):
        self.events.append(("fleet-start",))

      def _stop_kubernetes_sandbox(self):
        self.events.append(("fleet-stop",))

    docker = types.ModuleType("r2egym.agenthub.runtime.docker")
    docker.DockerRuntime = DockerRuntime
    runtime = types.ModuleType("r2egym.agenthub.runtime")
    runtime.docker = docker
    agenthub = types.ModuleType("r2egym.agenthub")
    agenthub.runtime = runtime
    r2egym = types.ModuleType("r2egym")
    r2egym.agenthub = agenthub
    modules = {
        "r2egym": r2egym,
        "r2egym.agenthub": agenthub,
        "r2egym.agenthub.runtime": runtime,
        "r2egym.agenthub.runtime.docker": docker,
    }
    with mock.patch.dict(sys.modules, modules):
      sandbox_fleet.patch_r2egym_fleet_backend()
      fleet_runtime = DockerRuntime(backend="kubernetes-sandbox")
      fleet_runtime.start_container("i", "c", "n")
      fleet_runtime.stop_container()
      self.assertEqual(fleet_runtime.backend, "kubernetes-sandbox")
      self.assertEqual(
          fleet_runtime.events,
          [("init", "kubernetes"), ("fleet-start",), ("fleet-stop",)],
      )
      direct_runtime = DockerRuntime(backend="docker")
      direct_runtime.start_container("i", "c", "n")
      direct_runtime.stop_container()
      self.assertEqual(direct_runtime.events[0], ("init", "docker"))
      self.assertEqual(direct_runtime.events[1][0], "original-start")
      self.assertEqual(direct_runtime.events[2][0], "original-stop")

  def test_release_failure_is_not_downgraded_after_cleanup(self):
    manager = _bare_manager(batch_size=1, generations=1, concurrency=1)
    resources = mock.Mock()
    cluster = types.SimpleNamespace(resources=resources)
    manager._fleet.release = mock.Mock(side_effect=RuntimeError("release red"))
    manager._fleet.registry = mock.Mock()
    manager._fleet.registry.get.return_value = cluster
    manager._confirm_released = mock.Mock()
    handle = types.SimpleNamespace(cluster_name="deepswe", claim_name="claim")
    with self.assertRaisesRegex(RuntimeError, "release red"):
      manager.release(handle)
    resources.delete_claim.assert_called_once_with("claim")
    manager._confirm_released.assert_called_once_with(handle)

  def test_close_never_calls_upstream_broad_teardown(self):
    manager = _bare_manager(batch_size=1, generations=1, concurrency=1)
    manager._delete_run_owned_resources = mock.Mock()
    manager.close()
    manager._delete_run_owned_resources.assert_called_once_with()
    manager._fleet.teardown.assert_not_called()
    manager.close()
    manager._delete_run_owned_resources.assert_called_once_with()

  def test_swe_env_fleet_bind_and_close_release_exactly_once(self):
    swe_env = _load_swe_env_with_stubbed_base()
    manager = mock.Mock()
    manager.release.return_value = 0.0
    handle = object()
    manager.acquire.return_value = (handle, 0.25)
    repo_env = mock.Mock()
    repo_env.compute_reward = mock.Mock()
    repo_env.get_task_instruction.return_value = "real instruction"
    manager.make_repo_env.return_value = repo_env
    env = swe_env.SWEEnv(
        {"prompts": "fix it", "docker_image": "image"},
        sandbox_manager=manager,
    )
    env.extra_kwargs["_trajectory_batch_started_monotonic"] = 1.0
    self.assertEqual(env._initial_observation(), "real instruction")
    manager.acquire.assert_called_once()
    manager.make_repo_env.assert_called_once()
    env.close()
    manager.release.assert_called_once_with(handle)
    env.close()
    manager.release.assert_called_once_with(handle)

  def test_swe_env_bind_failure_releases_and_dual_close_error_is_fatal(self):
    swe_env = _load_swe_env_with_stubbed_base()
    handle = object()
    manager = mock.Mock()
    manager.acquire.return_value = (handle, 0.1)
    manager.make_repo_env.side_effect = RuntimeError("bind red")
    env = swe_env.SWEEnv(
        {"prompts": "fix it", "docker_image": "image"},
        sandbox_manager=manager,
    )
    with self.assertRaisesRegex(RuntimeError, "bind red"):
      env._initial_observation()
    manager.release.assert_called_once_with(handle)
    self.assertIsNone(env.sandbox_handle)

    manager = mock.Mock()
    manager.acquire.return_value = (handle, 0.1)
    repo_env = mock.Mock()
    repo_env.compute_reward = mock.Mock()
    repo_env.get_task_instruction.return_value = "instruction"
    repo_env.close.side_effect = RuntimeError("adapter close red")
    manager.make_repo_env.return_value = repo_env
    manager.release.side_effect = RuntimeError("release red")
    env = swe_env.SWEEnv(
        {"prompts": "fix it", "docker_image": "image"},
        sandbox_manager=manager,
    )
    env._initial_observation()
    with self.assertRaisesRegex(ExceptionGroup, "both failed"):
      env.close()
    manager.release.assert_called_once_with(handle)

  def test_run_scoped_sweep_uses_exact_lineage_selector(self):
    manager = _bare_manager(batch_size=1, generations=1, concurrency=1)
    manager._lineage = "run-abc"
    manager._delete_timeout = 1
    resources = mock.Mock()
    resources.list_claims.side_effect = [["claim"], []]
    resources.list_warmpools.side_effect = [["pool"], []]
    resources.list_templates.side_effect = [["template"], []]
    core_api = mock.Mock()
    core_api.list_namespaced_pod.return_value.items = []
    cluster = types.SimpleNamespace(
        resources=resources,
        core_api=core_api,
        namespace="default",
    )
    manager._fleet.registry = [cluster]
    manager._delete_run_owned_resources()
    selector = "canon.zero-tim/run-id=run-abc"
    resources.delete_claim.assert_called_once_with("claim")
    resources.delete_warmpool.assert_called_once_with("pool")
    resources.delete_template.assert_called_once_with("template")
    for method in (
        resources.list_claims,
        resources.list_warmpools,
        resources.list_templates,
    ):
      self.assertTrue(
          all(
              call.kwargs.get("label_selector") == selector
              for call in method.call_args_list
          )
      )
  def test_direct_path_has_no_top_level_agent_sandbox_import(self):
    tree = ast.parse((ROOT / "examples/deepswe/sandbox_fleet.py").read_text())
    imports = []
    for node in tree.body:
      if isinstance(node, ast.Import):
        imports.extend(alias.name for alias in node.names)
      elif isinstance(node, ast.ImportFrom):
        imports.append(node.module or "")
    self.assertFalse(
        any(name.startswith(("agent_sandbox_rl", "k8s_agent_sandbox")) for name in imports)
    )
    train_text = TRAIN.read_text()
    self.assertIn("SANDBOX_RUNTIME == sandbox_fleet_lib.FLEET", train_text)
    swe_text = SWE_ENV.read_text()
    self.assertIn("if self.sandbox_manager is None:", swe_text)
    self.assertIn("RepoEnv(", swe_text)

  def test_install_step_is_pinned_and_direct_mode_skips(self):
    step_text = STEP.read_text()
    self.assertIn(sandbox_fleet.AGENT_SANDBOX_COMMIT, step_text)
    self.assertIn("shutdown_after_seconds", step_text)
    self.assertIn("verify_fleet_permissions", step_text)
    self.assertIn("examples/agent-sandbox-rl", step_text)
    self.assertNotIn('rm -rf "$DEST"', step_text)
    profile_text = PROFILE.read_text()
    self.assertIn("CANON_DEEPSWE_SANDBOX_RUNTIME", profile_text)
    self.assertIn(sandbox_fleet.AGENT_SANDBOX_COMMIT, profile_text)
    self.assertEqual(ENTRYPOINT.read_text().count("36_install_agent_sandbox.sh"), 4)
    with tempfile.TemporaryDirectory() as state_text:
      Path(state_text, "env.sh").write_text(
          "export CANON_DEEPSWE_SANDBOX_RUNTIME=direct\n"
      )
      result = subprocess.run(
          ["bash", str(STEP)],
          env=dict(os.environ, CANON_STATE=state_text, CANON_PKG=str(PKG)),
          text=True,
          capture_output=True,
          check=False,
      )
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn("runtime=direct -- skipped", result.stdout)

  def test_all_training_renderers_deliver_exact_fleet_capacity(self):
    common = self._common("flt")
    p58_common = {
        key: value
        for key, value in common.items()
        if key not in ("whitelist", "whitelist_sha256")
    }
    p58_common["cpu_nodepool"] = "canon-cpu-pool"
    documents = {
        "p34": p34.render(
            self._base(),
            stage="three-update",
            sandbox_runtime="fleet",
            sandbox_capacity=128,
            **common,
        ),
        "p39": p39.render(
            self._base(),
            stage="three-update",
            sandbox_runtime="fleet",
            sandbox_capacity=128,
            **common,
        ),
        "p43": p43.render(
            self._base(),
            stage="three-update",
            sandbox_runtime="fleet",
            sandbox_capacity=32,
            **common,
        ),
        "p44": p44.render(
            self._base(),
            stage="three-update",
            topology="64",
            sandbox_runtime="fleet",
            sandbox_capacity=32,
            **common,
        ),
        "p46-q4": p46.render(
            self._base(),
            workload="q4-debug",
            topology="64",
            sandbox_runtime="fleet",
            sandbox_capacity=32,
            **common,
        ),
        "p46-q32": p46.render(
            self._base(),
            workload="q32-train",
            topology="64",
            sandbox_runtime="fleet",
            sandbox_capacity=128,
            **common,
        ),
        "p58": p58.render(
            self._base(),
            stage="three-update",
            arm="native",
            sandbox_runtime="fleet",
            sandbox_capacity=256,
            **p58_common,
        ),
    }
    expected_capacity = {
        "p34": "128",
        "p39": "128",
        "p43": "32",
        "p44": "32",
        "p46-q4": "32",
        "p46-q32": "128",
        "p58": "256",
    }
    for name, document in documents.items():
      with self.subTest(name=name):
        env = p34._env(document)
        self.assertEqual(env["CANON_DEEPSWE_SANDBOX_RUNTIME"], "fleet")
        self.assertEqual(env["R2E_SANDBOX_CAPACITY"], expected_capacity[name])

  def test_renderer_capacity_underflow_and_eval_fleet_fail_closed(self):
    common = self._common("neg")
    cases = (
        (p34.render, dict(stage="three-update"), 127),
        (p39.render, dict(stage="three-update"), 127),
        (p43.render, dict(stage="three-update"), 31),
        (
            p44.render,
            dict(stage="three-update", topology="64"),
            31,
        ),
        (
            p46.render,
            dict(workload="q4-debug", topology="64"),
            31,
        ),
        (
            p46.render,
            dict(workload="q32-train", topology="64"),
            127,
        ),
    )
    for renderer, extra, capacity in cases:
      with self.subTest(renderer=renderer.__module__, capacity=capacity):
        with self.assertRaisesRegex(ValueError, "below active"):
          renderer(
              self._base(),
              sandbox_runtime="fleet",
              sandbox_capacity=capacity,
              **extra,
              **common,
          )
    p58_common = {
        key: value
        for key, value in common.items()
        if key not in ("whitelist", "whitelist_sha256")
    }
    p58_common["cpu_nodepool"] = "canon-cpu-pool"
    with self.assertRaisesRegex(ValueError, "below active"):
      p58.render(
          self._base(),
          stage="three-update",
          arm="native",
          sandbox_runtime="fleet",
          sandbox_capacity=255,
          **p58_common,
      )
    with self.assertRaisesRegex(ValueError, "does not admit"):
      p46.render(
          self._base(),
          workload="q4-clean-eval",
          topology="64",
          sandbox_runtime="fleet",
          sandbox_capacity=128,
          **common,
      )

  def test_environment_gate_admits_fleet_and_rejects_capacity_drift(self):
    result, resolved = self._run_p34_env(runtime="fleet", capacity=128)
    self.assertEqual(result.returncode, 0, result.stdout)
    self.assertIn("DeepSWE Fleet contract OK", result.stdout)
    self.assertIn("export CANON_DEEPSWE_SANDBOX_RUNTIME=fleet", resolved)
    self.assertIn("export R2E_SANDBOX_CAPACITY=128", resolved)

    result, _ = self._run_p34_env(
        runtime="fleet", capacity=128, override_capacity=127
    )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("Fleet capacity too small", result.stdout)

    result, _ = self._run_p34_env(
        runtime="direct", capacity=None, override_capacity=128
    )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("valid only with", result.stdout)

  def test_all_training_renderer_clis_expose_one_shared_selector_pair(self):
    renderers = (
        "render_p34_jobset.py",
        "render_p39_deepswe_pilot.py",
        "render_p43_deepswe_debug.py",
        "render_p44_deepswe_parity.py",
        "render_p46_deepswe_profiles.py",
        "render_p58_deepswe_tim.py",
    )
    for filename in renderers:
      with self.subTest(renderer=filename):
        result = subprocess.run(
            [sys.executable, str(CLUSTER / filename), "--help"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("--sandbox-runtime {direct,fleet}", result.stdout)
        self.assertIn("--sandbox-capacity", result.stdout)

  def test_onehost_fleet_selector_is_explicit_and_fail_closed(self):
    common = {
        "DEEPSWE_ONEHOST_SANDBOX_RUNTIME": "fleet",
        "DEEPSWE_ONEHOST_SANDBOX_NODEPOOL": "sandbox-pool",
        "DEEPSWE_AGENT_SANDBOX_ROOT": "/missing/agent-sandbox",
    }
    result = subprocess.run(
        ["bash", str(ONEHOST), "rollout-only"],
        cwd=ROOT,
        env=dict(os.environ, **common, DEEPSWE_ONEHOST_SANDBOX_CAPACITY="7"),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    self.assertEqual(result.returncode, 2, result.stdout)
    self.assertIn("capacity must be an explicit integer >=8", result.stdout)

    result = subprocess.run(
        ["bash", str(ONEHOST), "rollout-only"],
        cwd=ROOT,
        env=dict(
            os.environ,
            DEEPSWE_ONEHOST_SANDBOX_RUNTIME="direct",
            DEEPSWE_ONEHOST_SANDBOX_CAPACITY="8",
        ),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    self.assertEqual(result.returncode, 2, result.stdout)
    self.assertIn("invalid with sandbox runtime direct", result.stdout)

    source = ONEHOST.read_text()
    self.assertIn('export CANON_DEEPSWE_SANDBOX_RUNTIME="$SANDBOX_RUNTIME"', source)
    self.assertIn('export R2E_SANDBOX_CAPACITY="$SANDBOX_CAPACITY"', source)
    self.assertIn('export NODE_SELECTOR_VAL="$SANDBOX_NODEPOOL"', source)
    self.assertIn("export DEEPSWE_ONEHOST_REAL_B2=1", source)
    self.assertIn("--batch_size 2", source)
    self.assertIn("--max_concurrency 4", source)
    self.assertIn("DEEPSWE_ONEHOST_SANDBOX_FLEET_PASS", source)
    self.assertIn("ThreadPoolExecutor(max_workers=4)", source)
    self.assertIn(sandbox_fleet.AGENT_SANDBOX_COMMIT, source)


if __name__ == "__main__":
  unittest.main()

#!/usr/bin/env python3
"""Locks the r2egym availability contract for the DeepSWE example modules.

The p39d4 Stage 1 attempt died at module initialization because
``examples/deepswe/swe_agent.py`` raised at import time when ``r2egym`` was
absent, even though offline gold-whitelist replay never parses interactive
SWE actions.  These tests pin the repaired contract:

  * both example modules import without r2egym;
  * the interactive parser fails closed, at use, with the exact remedy;
  * the RepoEnv poll patch reports a skip instead of crashing;
  * when r2egym is present the parser binds to it unchanged.

A green import is not evidence by itself (KNOWN_FOOTGUNS #1); the negative
controls here are the point.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

_R2EGYM_PREFIX = "r2egym"
_AGENT_MODULE = "examples.deepswe.swe_agent"
_PATCH_MODULE = "examples.deepswe.r2egym_runtime_patch"


class _R2egymBlocker:
  """Meta-path finder that makes every r2egym import raise ImportError."""

  def find_spec(self, fullname, path=None, target=None):  # pylint: disable=unused-argument
    if fullname == _R2EGYM_PREFIX or fullname.startswith(_R2EGYM_PREFIX + "."):
      raise ImportError(f"{fullname} blocked by the offline-contract test")
    return None


def _purge(prefix: str) -> None:
  for name in list(sys.modules):
    if name == prefix or name.startswith(prefix + "."):
      del sys.modules[name]


class R2egymOptionalContractTest(unittest.TestCase):

  def _require_agent_import_chain(self):
    """Skips loudly when the host python cannot import the tunix agent chain.

    ``swe_agent`` pulls ``tunix.rl.agentic.agents.base_agent`` whose transitive
    imports (for example ``metrax``) exist in the pinned image but not on every
    development host.  The skip names the missing module so an absent run is
    visible, never silently green; ``run_exact_image.sh`` executes the full
    suite inside the image.
    """
    try:
      importlib.import_module("tunix.rl.agentic.agents.base_agent")
    except ModuleNotFoundError as error:
      raise unittest.SkipTest(
          f"host python lacks {error.name!r}; run inside the pinned image"
      )

  def setUp(self):
    self._blocker = _R2egymBlocker()
    self._saved_modules = {
        name: sys.modules[name]
        for name in list(sys.modules)
        if name.startswith(_R2EGYM_PREFIX) or name.startswith("examples.deepswe")
    }
    sys.meta_path.insert(0, self._blocker)
    _purge(_R2EGYM_PREFIX)
    _purge(_AGENT_MODULE)
    _purge(_PATCH_MODULE)

  def tearDown(self):
    sys.meta_path.remove(self._blocker)
    _purge(_R2EGYM_PREFIX)
    _purge(_AGENT_MODULE)
    _purge(_PATCH_MODULE)
    sys.modules.update(self._saved_modules)

  def test_swe_agent_imports_and_parser_fails_closed_without_r2egym(self):
    self._require_agent_import_chain()
    module = importlib.import_module(_AGENT_MODULE)
    self.assertIsNone(module.SWEAction)
    with self.assertRaises(ImportError) as caught:
      module.parse_xml_response("<function=execute_bash></function>")
    self.assertIn("r2egym", str(caught.exception))
    with self.assertRaises(ImportError):
      module.parse_oai_response(object())

  def test_poll_patch_skips_without_r2egym(self):
    module = importlib.import_module(_PATCH_MODULE)
    self.assertEqual(module.apply_repoenv_kubernetes_poll_patch(), "")

  def test_poll_patch_labels_bounds_and_confirms_pod_deletion(self):
    class FakeApiException(Exception):

      def __init__(self, status):
        super().__init__(status)
        self.status = status

    class FakeDockerRuntime:

      def stop(self):
        self.original_stop_calls += 1

    docker_mod = types.ModuleType("r2egym.agenthub.runtime.docker")
    docker_mod.__file__ = "/fake/r2egym/docker.py"
    docker_mod.DEFAULT_NAMESPACE = "default"
    docker_mod.DOCKER_PATH = "/usr/local/bin:/usr/bin:/bin"
    docker_mod.client = types.SimpleNamespace(ApiException=FakeApiException)
    docker_mod.DockerRuntime = FakeDockerRuntime
    runtime_pkg = types.ModuleType("r2egym.agenthub.runtime")
    runtime_pkg.docker = docker_mod
    agenthub = types.ModuleType("r2egym.agenthub")
    agenthub.runtime = runtime_pkg
    package = types.ModuleType("r2egym")
    package.agenthub = agenthub
    sys.modules.update({
        "r2egym": package,
        "r2egym.agenthub": agenthub,
        "r2egym.agenthub.runtime": runtime_pkg,
        "r2egym.agenthub.runtime.docker": docker_mod,
    })

    module = importlib.import_module(_PATCH_MODULE)
    with mock.patch.dict(os.environ, {
        "CANON_RUN_ID": "Test_Run/01",
        "R2E_ACTIVE_DEADLINE_SECONDS": "3300",
        "R2E_POD_DELETE_TIMEOUT_SECONDS": "1",
    }, clear=False):
      self.assertEqual(
          module.apply_repoenv_kubernetes_poll_patch(),
          "/fake/r2egym/docker.py",
      )

      class FakeClient:

        def __init__(self):
          self.created = None
          self.deleted = False
          self.reads = 0

        def read_namespaced_pod(self, **kwargs):
          del kwargs
          self.reads += 1
          if self.reads == 1 or self.deleted:
            raise FakeApiException(404)
          return types.SimpleNamespace(
              metadata=types.SimpleNamespace(name="pod-1"),
              status=types.SimpleNamespace(phase="Running"),
          )

        def create_namespaced_pod(self, **kwargs):
          self.created = kwargs["body"]

        def delete_namespaced_pod(self, **kwargs):
          del kwargs
          self.deleted = True

      runtime = FakeDockerRuntime()
      runtime.client = FakeClient()
      runtime.logger = mock.Mock()
      runtime.container = None
      runtime.original_stop_calls = 0
      runtime._start_kubernetes_pod("image", "command", "pod-1")
      body = runtime.client.created
      self.assertEqual(body["spec"]["activeDeadlineSeconds"], 3300)
      self.assertEqual(
          body["metadata"]["labels"]["canon.zero-tim/run-id"],
          "test_run-01",
      )
      self.assertEqual(
          body["spec"]["containers"][0]["resources"]["requests"],
          {"cpu": "2", "memory": "4Gi"},
      )
      runtime.stop()
      self.assertTrue(runtime.client.deleted)
      self.assertEqual(runtime.original_stop_calls, 1)

  def test_swe_agent_binds_action_when_r2egym_is_present(self):
    self._require_agent_import_chain()
    parsed = []

    class _FakeAction:

      @staticmethod
      def from_string(text):
        parsed.append(text)
        return ("fake-action", text)

    package = types.ModuleType("r2egym")
    agenthub = types.ModuleType("r2egym.agenthub")
    action_mod = types.ModuleType("r2egym.agenthub.action")
    action_mod.Action = _FakeAction
    agenthub.action = action_mod
    package.agenthub = agenthub
    sys.modules["r2egym"] = package
    sys.modules["r2egym.agenthub"] = agenthub
    sys.modules["r2egym.agenthub.action"] = action_mod

    module = importlib.import_module(_AGENT_MODULE)
    self.assertIs(module.SWEAction, _FakeAction)
    thought, action = module.parse_xml_response(
        "plan<function=file_editor></function>"
    )
    self.assertEqual(thought, "plan")
    self.assertEqual(action, ("fake-action", "<function=file_editor></function>"))
    self.assertEqual(parsed, ["<function=file_editor></function>"])


if __name__ == "__main__":
  unittest.main()

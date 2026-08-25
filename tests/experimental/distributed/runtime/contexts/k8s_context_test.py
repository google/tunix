# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for Kubernetes distributed runtime contexts."""

import argparse
import os
import sys
from unittest import mock

from absl.testing import absltest
import portpicker
from tunix.experimental.distributed.runtime import context
from tunix.experimental.distributed.runtime.contexts import k8s_context


class K8sContextTest(absltest.TestCase):

  def test_resolve_discovery_address(self):
    resolved = k8s_context.resolve_discovery_address("door:12345")
    self.assertEqual(resolved, "door-proc-0-0.door:12345")

  def test_resolve_self_hostname_success(self):
    envs = {
        "JOBSET_NAME": "myjobset",
        "REPLICATED_JOB_NAME": "worker",
        "JOB_INDEX": "0",
        "POD_INDEX": "2",
    }
    with mock.patch.dict(os.environ, envs):
      fqdn = k8s_context.resolve_self_hostname()
      self.assertEqual(fqdn, "myjobset-worker-0-2.myjobset")

  def test_resolve_self_hostname_missing_env(self):
    envs = {
        "JOBSET_NAME": "myjobset",
    }
    with mock.patch.dict(os.environ, envs, clear=True):
      with self.assertRaises(ValueError):
        k8s_context.resolve_self_hostname()

  def test_k8s_jax_context_pathways(self):
    envs = {
        "JAX_PLATFORMS": "proxy",
        "JAX_BACKEND_TARGET": "0.0.0.0:8000",
    }
    mock_pw = mock.MagicMock()
    with mock.patch.dict(os.environ, envs):
      with mock.patch.dict(sys.modules, {"pathwaysutils": mock_pw}):
        k8s_context.K8sJaxContext().initialize()
        mock_pw.initialize.assert_called_once()

  def test_k8s_jax_context_mcjax(self):
    envs = {}
    mock_jax = mock.MagicMock()
    with mock.patch.dict(os.environ, envs, clear=True):
      with mock.patch.dict(sys.modules, {"jax": mock_jax}):
        k8s_context.K8sJaxContext().initialize()
        mock_jax.distributed.initialize.assert_called_once()

  def test_k8s_discovery_context_register(self):
    envs = {
        "JOBSET_NAME": "myjobset",
        "REPLICATED_JOB_NAME": "worker",
        "JOB_INDEX": "0",
        "POD_INDEX": "1",
    }
    port = portpicker.pick_unused_port()
    args = argparse.Namespace(
        discovery_port=port,
        discovery_addrs="door:8888",
    )

    with mock.patch.dict(os.environ, envs):
      with k8s_context.K8sDiscoveryContext(args) as disc_ctx:
        with mock.patch(
            "tunix.experimental.distributed.runtime.contexts.k8s_context.discovery.register"
        ) as mock_reg:
          disc_ctx.register(b"pod-meta")
          mock_reg.assert_called_once_with(
              "door-proc-0-0.door:8888",
              "myjobset-worker-0-1.myjobset",
              port,
              b"pod-meta",
          )

  def test_k8s_process_context(self):
    args = argparse.Namespace(
        discovery_port=portpicker.pick_unused_port(),
        discovery_addrs="door:8888",
    )
    proc_ctx = k8s_context.K8sProcessContext(args)
    with proc_ctx as entered:
      self.assertIs(entered, proc_ctx)
      self.assertIsInstance(proc_ctx.jax, k8s_context.K8sJaxContext)
      self.assertIsInstance(proc_ctx.ipc, k8s_context.K8sIpcContext)


if __name__ == "__main__":
  absltest.main()

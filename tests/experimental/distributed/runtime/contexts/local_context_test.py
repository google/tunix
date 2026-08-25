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

"""Unit tests for local distributed runtime contexts."""

import argparse
from unittest import mock

from absl.testing import absltest
import portpicker
from tunix.experimental.distributed.runtime import context
from tunix.experimental.distributed.runtime.contexts import local_context


class LocalContextTest(absltest.TestCase):

  def test_resolve_discovery_address(self):
    resolved = local_context.resolve_discovery_address("worker-id:12345")
    self.assertEqual(resolved, "localhost:12345")

  @mock.patch(
      "tunix.experimental.distributed.runtime.discovery.discovery.grpc.server"
  )
  def test_local_discovery_context_lifecycle_and_registration(
      self, mock_grpc_server
  ):
    port = portpicker.pick_unused_port()
    args = argparse.Namespace(
        discovery_port=port,
        discovery_addrs="leader:9999",
    )

    with local_context.LocalDiscoveryContext(args) as disc_ctx:
      cb = mock.MagicMock()
      disc_ctx.on_register(cb)
      self.assertTrue(disc_ctx._server.is_started())

      with mock.patch(
          "tunix.experimental.distributed.runtime.contexts.local_context.discovery.register"
      ) as mock_reg:
        disc_ctx.register(b"my-metadata")
        mock_reg.assert_called_once_with(
            "localhost:9999", "localhost", port, b"my-metadata"
        )

    self.assertFalse(disc_ctx._server.is_started())

  def test_local_process_context(self):
    args = argparse.Namespace(
        discovery_port=portpicker.pick_unused_port(),
        discovery_addrs="leader:9999",
    )
    proc_ctx = local_context.LocalProcessContext(args)

    with proc_ctx as entered:
      self.assertIs(entered, proc_ctx)
      self.assertIsInstance(proc_ctx.jax, context.JaxContext)
      self.assertIsInstance(proc_ctx.ipc, local_context.LocalIpcContext)
      self.assertIsInstance(
          proc_ctx.ipc.discovery, local_context.LocalDiscoveryContext
      )
      self.assertTrue(proc_ctx.ipc.discovery._server.is_started() is False)


if __name__ == "__main__":
  absltest.main()

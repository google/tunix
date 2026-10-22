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

"""Unit tests for base distributed runtime context interfaces."""

from absl.testing import absltest
from tunix.experimental.distributed.runtime import context


class ContextTest(absltest.TestCase):

  def test_jax_context_defaults(self):
    jax_ctx = context.JaxContext()
    self.assertIsNone(jax_ctx.initialize())

  def test_discovery_context_defaults(self):
    disc_ctx = context.DiscoveryContext()
    self.assertIsNone(disc_ctx.on_register(lambda host, port, meta: None))
    self.assertIsNone(disc_ctx.register(b"metadata"))

  def test_ipc_context_defaults(self):
    ipc_ctx = context.IpcContext()
    self.assertIsInstance(ipc_ctx.discovery, context.DiscoveryContext)

  def test_process_context_defaults(self):
    proc_ctx = context.ProcessContext()
    with proc_ctx as entered:
      self.assertIs(entered, proc_ctx)
    self.assertIsInstance(proc_ctx.jax, context.JaxContext)
    self.assertIsInstance(proc_ctx.ipc, context.IpcContext)
    self.assertIsInstance(proc_ctx.ipc.discovery, context.DiscoveryContext)


if __name__ == "__main__":
  absltest.main()

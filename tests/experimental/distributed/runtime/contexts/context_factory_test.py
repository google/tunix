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

"""Unit tests for context_factory."""

import argparse
import os
from unittest import mock

from absl.testing import absltest
from tunix.experimental.distributed.runtime.contexts import borg_context
from tunix.experimental.distributed.runtime.contexts import context_factory
from tunix.experimental.distributed.runtime.contexts import k8s_context
from tunix.experimental.distributed.runtime.contexts import local_context


class ContextFactoryTest(absltest.TestCase):

  def test_get_default_process_context_local(self):
    args = argparse.Namespace(discovery_port=12345, discovery_addrs="")
    with mock.patch.dict(os.environ, {}, clear=True):
      ctx = context_factory.get_default_process_context(args)
      self.assertIsInstance(ctx, local_context.LocalProcessContext)

  def test_get_default_process_context_borg(self):
    args = argparse.Namespace(discovery_port=12345, discovery_addrs="")
    with mock.patch.dict(os.environ, {"BORG_TASK_HANDLE": "12345"}, clear=True):
      ctx = context_factory.get_default_process_context(args)
      self.assertIsInstance(ctx, borg_context.BorgProcessContext)

  def test_get_default_process_context_k8s(self):
    args = argparse.Namespace(discovery_port=12345, discovery_addrs="")
    with mock.patch.dict(
        os.environ, {"KUBERNETES_SERVICE_HOST": "10.0.0.1"}, clear=True
    ):
      ctx = context_factory.get_default_process_context(args)
      self.assertIsInstance(ctx, k8s_context.K8sProcessContext)


if __name__ == "__main__":
  absltest.main()

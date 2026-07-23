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

"""Unit tests for local and Kubernetes process executors."""

import argparse
from unittest import mock

from absl.testing import absltest
import portpicker
from tunix.experimental.distributed.runtime.contexts import k8s_context
from tunix.experimental.distributed.runtime.contexts import local_context
from tunix.experimental.distributed.runtime.executors import k8s_executor
from tunix.experimental.distributed.runtime.executors import local_executor


class ExecutorTest(absltest.TestCase):

  def test_local_executor_run(self):
    executor = local_executor.LocalExecutor()
    args = argparse.Namespace(
        discovery_port=portpicker.pick_unused_port(),
        discovery_addrs="door:1234",
    )

    received = {}

    def main_fn(argv, ctx):
      received["argv"] = argv
      received["ctx"] = ctx

    executor.run(main_fn, ["--foo=bar"], args)
    self.assertEqual(received["argv"], ["--foo=bar"])
    self.assertIsInstance(received["ctx"], local_context.LocalProcessContext)

  def test_k8s_executor_run(self):
    executor = k8s_executor.K8sExecutor()
    args = argparse.Namespace(
        discovery_port=portpicker.pick_unused_port(),
        discovery_addrs="door:1234",
    )

    received = {}

    def main_fn(argv, ctx):
      received["argv"] = argv
      received["ctx"] = ctx

    executor.run(main_fn, ["--bar=baz"], args)
    self.assertEqual(received["argv"], ["--bar=baz"])
    self.assertIsInstance(received["ctx"], k8s_context.K8sProcessContext)


if __name__ == "__main__":
  absltest.main()

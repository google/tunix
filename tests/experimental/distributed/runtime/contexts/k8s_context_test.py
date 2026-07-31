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

"""Choosing a runtime must not depend on a variable happening to be set."""

import os
from unittest import mock

from absl.testing import absltest

try:
  from tunix.experimental.distributed.runtime.contexts import k8s_context
except ImportError as e:
  # The runtime package needs generated protobuf stubs, which are a build
  # step rather than a checkout. Skip rather than fail: this file is about one
  # environment check, not about proto generation.
  k8s_context = None
  _IMPORT_ERROR = e


@absltest.skipIf(
    k8s_context is None, "distributed runtime stubs are not generated here"
)
class ShouldUsePathwaysTest(absltest.TestCase):

  def test_an_unset_platform_variable_means_no(self):
    """It used to raise instead of answering, which killed startup."""
    with mock.patch.dict(os.environ, {}, clear=True):
      self.assertFalse(k8s_context.should_use_pathways())

  def test_a_proxy_platform_with_a_backend_target_means_yes(self):
    with mock.patch.dict(
        os.environ,
        {"JAX_PLATFORMS": "proxy", "JAX_BACKEND_TARGET": "grpc://x"},
        clear=True,
    ):
      self.assertTrue(k8s_context.should_use_pathways())

  def test_a_proxy_platform_without_a_target_means_no(self):
    with mock.patch.dict(
        os.environ, {"JAX_PLATFORMS": "proxy"}, clear=True
    ):
      self.assertFalse(k8s_context.should_use_pathways())

  def test_another_platform_means_no(self):
    with mock.patch.dict(
        os.environ,
        {"JAX_PLATFORMS": "cpu", "JAX_BACKEND_TARGET": "grpc://x"},
        clear=True,
    ):
      self.assertFalse(k8s_context.should_use_pathways())


if __name__ == "__main__":
  absltest.main()

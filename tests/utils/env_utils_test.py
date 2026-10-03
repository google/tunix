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

"""Tests for environment utilities."""

import os
from unittest import mock
from absl.testing import absltest
from tunix.utils import env_utils


class EnvUtilsTest(absltest.TestCase):

  def test_is_debug_inference_logs_enabled_default_false(self):
    with mock.patch.dict(os.environ, {}, clear=True):
      self.assertFalse(env_utils.is_debug_inference_logs_enabled())

  def test_is_debug_inference_logs_enabled_tunix_flag(self):
    for val in ("1", "true", "True", "yes", "YES"):
      with mock.patch.dict(
          os.environ, {"TUNIX_DEBUG_INFERENCE_LOGS": val}, clear=True
      ):
        self.assertTrue(env_utils.is_debug_inference_logs_enabled())

    for val in ("0", "false", "no", "", "2"):
      with mock.patch.dict(
          os.environ, {"TUNIX_DEBUG_INFERENCE_LOGS": val}, clear=True
      ):
        self.assertFalse(env_utils.is_debug_inference_logs_enabled())

  def test_is_debug_inference_logs_unprefixed_flag_ignored(self):
    with mock.patch.dict(
        os.environ, {"DEBUG_INFERENCE_LOGS": "1"}, clear=True
    ):
      self.assertFalse(env_utils.is_debug_inference_logs_enabled())


if __name__ == "__main__":
  absltest.main()

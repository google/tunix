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

"""Unit tests for distributed RL environment variable parsing utilities."""

import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from tunix.dist_rl.common import env_utils


class EnvUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("unset_false", {}, False, False),
      ("unset_true", {}, True, True),
      ("blank_uses_default", {"FLAG": "  "}, True, True),
      ("truthy_one", {"FLAG": "1"}, False, True),
      ("truthy_true_padded", {"FLAG": " TRUE "}, False, True),
      ("truthy_yes", {"FLAG": "YeS"}, False, True),
      ("falsy_zero", {"FLAG": "0"}, True, False),
      ("falsy_no", {"FLAG": "no"}, True, False),
  )
  def test_parse_env_bool(
      self, env: dict[str, str], default: bool, expected: bool
  ):
    with mock.patch.dict(os.environ, env, clear=True):
      self.assertEqual(
          env_utils.parse_env_bool("FLAG", default=default, strict=True),
          expected,
      )

  def test_parse_env_bool_strict_rejects_invalid_token(self):
    with mock.patch.dict(os.environ, {"FLAG": "maybe"}, clear=True):
      self.assertFalse(env_utils.parse_env_bool("FLAG", strict=False))
      with self.assertRaisesRegex(ValueError, "Unsupported FLAG='maybe'"):
        env_utils.parse_env_bool("FLAG", strict=True)


if __name__ == "__main__":
  absltest.main()

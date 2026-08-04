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

"""Tests for the canonical engine-module forward registry."""

import os
from unittest import mock

from absl.testing import absltest

from tunix.rl import canonical_forward


class _Adapter:
  implementation_id = "test.tpu_inference.qwen3"
  is_engine_module = True
  supports_value_and_grad = True

  def compute_per_token_logps(self, **kwargs):
    return kwargs["sentinel"]


class CanonicalForwardTest(absltest.TestCase):

  def tearDown(self):
    canonical_forward._clear_for_test()  # pylint: disable=protected-access
    super().tearDown()

  def test_env_without_adapter_rejects_native_fallback(self):
    with mock.patch.dict(os.environ, {canonical_forward.ENV: "1"}, clear=False):
      with self.assertRaisesRegex(
          canonical_forward.CanonicalForwardError, "no tpu_inference"
      ):
        canonical_forward.compute_per_token_logps(sentinel=3)

  def test_registered_engine_adapter_is_used_and_attested(self):
    adapter = _Adapter()
    with mock.patch.dict(os.environ, {canonical_forward.ENV: "1"}, clear=False):
      canonical_forward.register(adapter)
      self.assertEqual(
          canonical_forward.compute_per_token_logps(sentinel=7), 7
      )
      self.assertEqual(
          canonical_forward.attestation()["implementation_id"],
          adapter.implementation_id,
      )

  def test_bad_adapter_is_rejected(self):
    adapter = _Adapter()
    adapter.is_engine_module = False
    with mock.patch.dict(os.environ, {canonical_forward.ENV: "1"}, clear=False):
      with self.assertRaisesRegex(
          canonical_forward.CanonicalForwardError, "is_engine_module"
      ):
        canonical_forward.register(adapter)


if __name__ == "__main__":
  absltest.main()

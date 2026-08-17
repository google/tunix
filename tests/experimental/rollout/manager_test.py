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

"""Tests for RolloutManager.get_weight_sync_metadata."""

import unittest

from absl.testing import absltest
from tunix.experimental.rollout import manager as manager_lib
from tunix.experimental.rollout import sampler as sampler_lib


class _FakeSampler(sampler_lib.Sampler):

  def __init__(self, metadata):
    self._metadata = metadata
    self.calls = []

  async def get_weight_sync_metadata(self, **kwargs):
    self.calls.append(kwargs)
    return self._metadata


class GetWeightSyncMetadataTest(unittest.IsolatedAsyncioTestCase):

  async def test_delegates_to_sampler(self):
    sampler = _FakeSampler([{"unit": "sampler0"}])
    manager = manager_lib.RolloutManager(
        sampler=sampler, tokenizer="mock", chat_parser="mock"
    )
    result = await manager.get_weight_sync_metadata()
    self.assertEqual(result, [{"unit": "sampler0"}])

  async def test_forwards_kwargs(self):
    sampler = _FakeSampler([])
    manager = manager_lib.RolloutManager(
        sampler=sampler, tokenizer="mock", chat_parser="mock"
    )
    await manager.get_weight_sync_metadata(timeout_s=5)
    self.assertEqual(sampler.calls, [{"timeout_s": 5}])

  async def test_default_sampler_raises_not_implemented(self):
    manager = manager_lib.RolloutManager(tokenizer="mock", chat_parser="mock")
    with self.assertRaises(NotImplementedError):
      await manager.get_weight_sync_metadata()


if __name__ == "__main__":
  absltest.main()

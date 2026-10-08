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

import dataclasses
from typing import Any
from absl.testing import absltest
from tunix.experimental.common import tagging


@dataclasses.dataclass(frozen=True)
class _FrozenPrompt:
  prompt: str
  prompt_id: str
  metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


class _ObjectPrompt:

  def __init__(self, prompt: str, metadata: dict[str, Any] | None = None):
    self.prompt = prompt
    self.prompt_id = "p0"
    self.metadata = metadata or {}


class _UntaggablePrompt:
  __slots__ = ("prompt", "prompt_id")

  def __init__(self, prompt: str):
    self.prompt = prompt
    self.prompt_id = "p0"


class TaggingTest(absltest.TestCase):

  def test_prompt_coordinates(self):
    self.assertEqual(
        tagging.prompt_coordinates(prompt_idx=5, full_batch_size=2),
        {"prompt_idx": 5, "batch_idx": 2, "intra_batch_idx": 1},
    )

  def test_tag_prompt_dict_merges_and_overrides_without_mutation(self):
    original = {"prompt": "q", "metadata": {"keep": 1, "batch_idx": 99}}
    coords = tagging.prompt_coordinates(0, full_batch_size=2)
    tagged = tagging.tag_prompt(original, coords)

    self.assertEqual(
        tagged["metadata"],
        {"keep": 1, "prompt_idx": 0, "batch_idx": 0, "intra_batch_idx": 0},
    )
    self.assertEqual(original["metadata"], {"keep": 1, "batch_idx": 99})

  def test_tag_prompt_frozen_dataclass(self):
    original = _FrozenPrompt(prompt="q", prompt_id="p0", metadata={"keep": 1})
    coords = tagging.prompt_coordinates(3, full_batch_size=2)
    tagged = tagging.tag_prompt(original, coords)

    self.assertEqual(
        tagged.metadata,
        {"keep": 1, "prompt_idx": 3, "batch_idx": 1, "intra_batch_idx": 1},
    )
    self.assertEqual(original.metadata, {"keep": 1})

  def test_tag_prompt_object_and_untaggable_fallback(self):
    obj = _ObjectPrompt("q", metadata={"keep": 1})
    coords = tagging.prompt_coordinates(1, full_batch_size=2)
    tagged = tagging.tag_prompt(obj, coords)
    self.assertEqual(
        tagged.metadata,
        {"keep": 1, "prompt_idx": 1, "batch_idx": 0, "intra_batch_idx": 1},
    )
    self.assertEqual(obj.metadata, {"keep": 1})

    untaggable = _UntaggablePrompt("q")
    self.assertIs(tagging.tag_prompt(untaggable, coords), untaggable)


if __name__ == "__main__":
  absltest.main()

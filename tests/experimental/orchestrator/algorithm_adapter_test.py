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

"""Tests for the AgenticGRPOAdapter."""

import numpy as np

from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import algorithm_adapter
from tunix.experimental.orchestrator import group_assembler
from tunix.experimental.orchestrator import request_ledger


_TOKENIZER = datatypes.TokenizerInfo(pad_id=0, eos_id=1)
_SHAPE = datatypes.ShapeConfig(max_prompt_length=3, max_response_tokens=4)


def _record(group_id, sample_index, request_id):
  return request_ledger.RequestRecord(
      request=datatypes.RolloutRequest(
          request_id=request_id,
          prompt_id="p",
          prompt_text="hi",
          sampling_params=datatypes.SamplingParams(max_tokens=4),
      ),
      group_id=group_id,
      sample_index=sample_index,
  )


def _result(request_id, env_reward, completion, *, policy_version=0):
  completion = np.asarray(completion, np.int32)
  segment = datatypes.TokenSegment(
      source="assistant",
      tokens=completion,
      loss_mask=np.ones_like(completion),
  )
  return datatypes.RolloutResult(
      request_id=request_id,
      prompt_id="p",
      status="COMPLETED",
      prompt_tokens=np.array([1, 2], np.int32),
      segments=[segment],
      env_reward=env_reward,
      policy_version=policy_version,
  )


class AgenticGRPOAdapterTest(absltest.TestCase):

  def test_make_trajectory_requests_makes_g_per_row(self):
    adapter = algorithm_adapter.AgenticGRPOAdapter(group_size=3)
    groups = adapter.make_trajectory_requests(
        [{"prompt_text": "a"}, {"prompt_text": "b"}], step=0
    )
    self.assertLen(groups, 2)
    for group in groups:
      self.assertLen(group, 3)
      self.assertEqual([r.sample_index for r in group], [0, 1, 2])
    # group_ids are distinct and request_ids unique.
    group_ids = {group[0].group_id for group in groups}
    self.assertLen(group_ids, 2)
    request_ids = {r.request_id for group in groups for r in group}
    self.assertLen(request_ids, 6)

  def test_group_size_below_two_rejected(self):
    with self.assertRaises(ValueError):
      algorithm_adapter.AgenticGRPOAdapter(group_size=1)

  def test_postprocess_group_assembles_train_example(self):
    adapter = algorithm_adapter.AgenticGRPOAdapter(group_size=2)
    group = group_assembler.AssembledGroup(
        group_id="g0",
        members=[
            (_record("g0", 0, "g0:0"), _result("g0:0", 0.0, [3, 4, 5])),
            (_record("g0", 1, "g0:1"), _result("g0:1", 2.0, [3, 4])),
        ],
    )
    example = adapter.postprocess_group(
        group, tokenizer_info=_TOKENIZER, shape_config=_SHAPE
    )
    np.testing.assert_array_equal(
        example.completion_ids, [[3, 4, 5, 0], [3, 4, 0, 0]]
    )
    np.testing.assert_array_equal(
        example.loss_mask, [[1, 1, 1, 0], [1, 1, 0, 0]]
    )
    # Group-relative advantages of rewards [0, 2]: mean 1, std 1 -> [-1, 1].
    np.testing.assert_allclose(example.advantages, [-1.0, 1.0], atol=1e-4)

  def test_group_relative_advantages_sum_to_zero(self):
    adapter = algorithm_adapter.AgenticGRPOAdapter(group_size=3)
    group = group_assembler.AssembledGroup(
        group_id="g0",
        members=[
            (_record("g0", 0, "g0:0"), _result("g0:0", 1.0, [3])),
            (_record("g0", 1, "g0:1"), _result("g0:1", 4.0, [3])),
            (_record("g0", 2, "g0:2"), _result("g0:2", 7.0, [3])),
        ],
    )
    example = adapter.postprocess_group(
        group, tokenizer_info=_TOKENIZER, shape_config=_SHAPE
    )
    self.assertAlmostEqual(float(np.sum(example.advantages)), 0.0, places=4)

  def test_loss_spec_carries_name_and_config(self):
    adapter = algorithm_adapter.AgenticGRPOAdapter(
        group_size=2, loss_fn_name="grpo", loss_config={"beta": 0.0}
    )
    spec = adapter.loss_spec()
    self.assertEqual(spec.name, "grpo")
    self.assertEqual(spec.config, {"beta": 0.0})


if __name__ == "__main__":
  absltest.main()

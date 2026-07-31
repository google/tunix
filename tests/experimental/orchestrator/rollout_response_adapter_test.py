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

"""Tests converting a worker RolloutResponse into postprocess-shaped items."""

import types

from absl.testing import absltest
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import algorithm_adapter
from tunix.experimental.orchestrator import rollout_response_adapter as adapter
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.agentic import agentic_grpo_learner


class _Tokenizer:
  """Decodes token ids as their string values, joined by spaces."""

  def decode(self, ids):
    return " ".join(str(i) for i in ids)


def _response(request_id="r0", with_logps=True, metadata=None):
  return datatypes.RolloutResponse(
      request_id=request_id,
      status="COMPLETED",
      prompt_tokens=np.array([7, 8], dtype=np.int32),
      segments=[
          datatypes.TokenSegment(
              source="assistant",
              tokens=np.array([1, 2], dtype=np.int32),
              loss_mask=np.array([1, 1], dtype=np.int32),
              logps=np.array([-0.1, -0.2], dtype=np.float32)
              if with_logps
              else None,
          ),
          datatypes.TokenSegment(
              source="env",
              tokens=np.array([3], dtype=np.int32),
              loss_mask=np.array([0], dtype=np.int32),
              logps=None,
          ),
      ],
      env_reward=1.5,
      policy_version=4,
      metadata=metadata or {},
  )


def _request(request_id="r0", prompt="hello"):
  return datatypes.RolloutRequest(
      request_id=request_id,
      prompt=prompt,
      prompt_id="p9",
      group_id="g3",
  )


class RolloutResponseAdapterTest(absltest.TestCase):

  def test_concatenates_segments_in_emission_order(self):
    item = adapter.to_trajectory_item(_response(), _request())
    np.testing.assert_array_equal(
        item.traj["conversation_tokens"], [1, 2, 3]
    )
    np.testing.assert_array_equal(item.traj["conversation_masks"], [1, 1, 0])

  def test_env_spans_are_zero_filled_in_logps(self):
    item = adapter.to_trajectory_item(_response(), _request())
    np.testing.assert_allclose(
        item.traj["old_logprobs"], [-0.1, -0.2, 0.0], atol=1e-6
    )

  def test_logps_are_none_when_the_sampler_reported_none(self):
    item = adapter.to_trajectory_item(
        _response(with_logps=False), _request()
    )
    # None is distinct from all-zeros: it means "not reported".
    self.assertIsNone(item.traj["old_logprobs"])

  def test_carries_scalars_and_prompt_tokens(self):
    item = adapter.to_trajectory_item(_response(), _request())
    np.testing.assert_array_equal(item.traj["prompt_tokens"], [7, 8])
    self.assertEqual(item.traj["policy_version"], 4)
    self.assertEqual(item.traj["trajectory_reward"], 1.5)
    self.assertEqual(item.traj["status"], "COMPLETED")

  def test_identifiers_come_from_the_request_not_the_response(self):
    item = adapter.to_trajectory_item(_response(), _request())
    self.assertEqual(item.traj["prompt_id"], "p9")
    self.assertEqual(item.traj["group_id"], "g3")
    self.assertEqual(item.traj["request_id"], "r0")

  def test_original_input_comes_from_the_request(self):
    # The worker never echoes the dataset row; the orchestrator supplies it.
    item = adapter.to_trajectory_item(
        _response(), _request(prompt={"prompts": "hi", "answer": "42"})
    )
    self.assertEqual(item.traj["original_input"]["answer"], "42")

    plain = adapter.to_trajectory_item(_response(), _request(prompt="hi"))
    self.assertEqual(plain.traj["original_input"], {"prompts": "hi"})

  def test_completion_text_is_detokenized_from_assistant_spans(self):
    item = adapter.to_trajectory_item(
        _response(), _request(), tokenizer=_Tokenizer()
    )
    conversation = item.traj["conversation_text"]
    self.assertEqual(conversation[0]["role"], "assistant")
    # Only the assistant span (1, 2) - not the env token (3).
    self.assertEqual(conversation[0]["content"], "1 2")

  def test_worker_supplied_text_wins_over_detokenizing(self):
    item = adapter.to_trajectory_item(
        _response(metadata={"completion_text": "from worker"}),
        _request(),
        tokenizer=_Tokenizer(),
    )
    self.assertEqual(
        item.traj["conversation_text"][0]["content"], "from worker"
    )

  def test_text_is_empty_without_a_tokenizer(self):
    item = adapter.to_trajectory_item(_response(), _request())
    self.assertEqual(item.traj["conversation_text"][0]["content"], "")

  def test_payload_less_response_yields_empty_arrays(self):
    # The metadata-only shape: no segments, payload parked elsewhere.
    response = datatypes.RolloutResponse(
        request_id="r0", status="COMPLETED", segments=[]
    )
    item = adapter.to_trajectory_item(response, _request())
    self.assertEqual(item.traj["conversation_tokens"].size, 0)
    self.assertIsNone(item.traj["old_logprobs"])

  def test_group_conversion_pairs_by_request_id_not_position(self):
    responses = [_response("r1"), _response("r0")]  # completion order
    requests = [_request("r0", "first"), _request("r1", "second")]
    items = adapter.to_trajectory_items(responses, requests)
    self.assertEqual(items[0].traj["original_input"], {"prompts": "second"})
    self.assertEqual(items[1].traj["original_input"], {"prompts": "first"})


class _FakeOrchestrator:
  """Minimal orchestrator surface `postprocess_group` reads."""

  def __init__(self):
    self.rollout = types.SimpleNamespace(
        pad_id=lambda: 0, eos_id=lambda: 2
    )
    self.cluster_config = types.SimpleNamespace(
        training_config=types.SimpleNamespace(
            compute_logps_micro_batch_size=2,
            max_seq_token_per_tpu=None,
        )
    )
    # No models here, so no actor mesh: the sampler-vs-trainer diagnostic is
    # skipped, exactly as it is on a machine without a device topology.
    self.r2m = {rl_cluster_lib.Role.ACTOR: None}
    self.buffered = []

  def get_rollout_config(self, mode):
    del mode
    return types.SimpleNamespace(max_prompt_length=8)

  def buffer_metrics_async(self, metrics, **kwargs):
    del kwargs
    self.buffered.append(metrics)


class AdapterFeedsExistingPostprocessTest(absltest.TestCase):
  """The point of the adapter: one postprocess serves local and remote rollouts."""

  def test_postprocess_group_consumes_adapter_items(self):
    items = adapter.to_trajectory_items(
        [_response("r0"), _response("r1")],
        [_request("r0", "a"), _request("r1", "b")],
        tokenizer=_Tokenizer(),
    )
    grpo = algorithm_adapter.GRPOAdapter(
        agentic_grpo_learner.GRPOConfig(
            num_generations=2,
            num_iterations=1,
            beta=0.0,
            max_response_length=4,
        )
    )

    def compute_rewards(prompts, completions, **kwargs):
      del prompts, kwargs
      return np.array([float(i) for i in range(len(completions))])

    [example] = grpo.postprocess_group(
        _FakeOrchestrator(),
        items,
        compute_rewards=compute_rewards,
        mode="train",
        expected_step=0,
    )

    # A real TrainExample came out the other side, shaped by the group.
    self.assertEqual(example.prompt_ids.shape[0], 2)
    self.assertEqual(example.completion_ids.shape, (2, 4))
    self.assertEqual(example.completion_mask.shape, (2, 4))
    self.assertEqual(example.advantages.shape, (2,))


if __name__ == "__main__":
  absltest.main()

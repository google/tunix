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

"""Tests for mock sampler and fake trainer utilities in test_utils."""

from __future__ import annotations

import asyncio
import time

from absl.testing import absltest
import numpy as np
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.common import test_utils
from tunix.dist_rl.rollout import sampler as sampler_lib
from tunix.dist_rl.train import abstract_trainer


class MockBaseSamplerImplTest(absltest.TestCase):

  def test_routed_experts_shape_defaults_to_none_and_returns_configured_spec(
      self,
  ):
    dense_sampler = test_utils.MockBaseSamplerImpl()
    self.assertIsNone(dense_sampler.routed_experts_shape())

    moe_sampler = test_utils.MockBaseSamplerImpl(
        routed_experts_shape_spec=(61, 8),
        num_experts=384,
    )
    self.assertEqual(moe_sampler.routed_experts_shape(), (61, 8))

  def test_sample_generates_configured_length_and_multi_turn_routed_experts(
      self,
  ):
    sampler = test_utils.MockBaseSamplerImpl(
        default_delay=0.0,
        response_tokens_per_turn=16,
        routed_experts_shape_spec=(4, 8),
        num_experts=384,
        eos_token_id=151645,
    )
    prompt_turn0 = np.arange(10, 22, dtype=np.int32)  # 12 prompt tokens
    req0 = sampler_lib.SamplingRequest(
        request_id="traj-1",
        prompt=prompt_turn0,
        metadata={"min_turns": 1, "return_routed_experts": True},
        max_tokens=32,
        routed_experts_prompt_start=0,
    )
    resp0 = asyncio.run(sampler.sample(req0))
    np.testing.assert_array_equal(resp0.prompt_token_ids, prompt_turn0)
    self.assertEqual(resp0.token_ids.shape, (16,))
    self.assertIsNotNone(resp0.routed_experts)
    routed0 = np.asarray(resp0.routed_experts)
    self.assertEqual(routed0.shape, (12 + 16, 4, 8))
    self.assertTrue(np.all((routed0 >= 0) & (routed0 < 384)))

    # Turn 1 echoes full continuation history (prompt + turn0 + 5 env tokens)
    # and slices routed_experts starting from routed_experts_prompt_start=28.
    prompt_turn1 = np.arange(10, 10 + 12 + 16 + 5, dtype=np.int32)
    req1 = sampler_lib.SamplingRequest(
        request_id="traj-1",
        prompt=prompt_turn1,
        metadata={"min_turns": 1, "return_routed_experts": True},
        max_tokens=10,
        routed_experts_prompt_start=28,
    )
    resp1 = asyncio.run(sampler.sample(req1))
    np.testing.assert_array_equal(resp1.prompt_token_ids, prompt_turn1)
    self.assertEqual(resp1.token_ids.shape, (10,))
    self.assertEqual(int(resp1.token_ids[-1]), 151645)
    routed1 = np.asarray(resp1.routed_experts)
    self.assertEqual(routed1.shape, (5 + 10, 4, 8))

  def test_routed_experts_tiles_deterministically_beyond_slab_rows(self):
    sampler = test_utils.MockBaseSamplerImpl(
        default_delay=0.0,
        response_tokens_per_turn=4500,
        routed_experts_shape_spec=(2, 4),
        num_experts=128,
    )
    prompt = np.arange(100, dtype=np.int32)
    resp_a = asyncio.run(
        sampler.sample(
            sampler_lib.SamplingRequest(
                request_id="long-traj",
                prompt=prompt,
                max_tokens=4500,
                metadata={"return_routed_experts": True},
            )
        )
    )
    sampler_b = test_utils.MockBaseSamplerImpl(
        default_delay=0.0,
        response_tokens_per_turn=4500,
        routed_experts_shape_spec=(2, 4),
        num_experts=128,
    )
    resp_b = asyncio.run(
        sampler_b.sample(
            sampler_lib.SamplingRequest(
                request_id="long-traj",
                prompt=prompt,
                max_tokens=4500,
                metadata={"return_routed_experts": True},
            )
        )
    )
    self.assertIsNotNone(resp_a.routed_experts)
    assert resp_a.routed_experts is not None
    self.assertIsNone(resp_a.routed_experts.base)
    routed_a = np.asarray(resp_a.routed_experts)
    self.assertEqual(routed_a.shape, (4600, 2, 4))
    np.testing.assert_array_equal(routed_a, np.asarray(resp_b.routed_experts))

  def test_rejects_invalid_routed_experts_prompt_start_and_bounds_turn_cache(
      self,
  ):
    sampler = test_utils.MockBaseSamplerImpl(
        default_delay=0.0,
        response_tokens_per_turn=4,
        routed_experts_shape_spec=(2, 4),
    )
    with self.assertRaisesRegex(ValueError, "routed_experts_prompt_start"):
      asyncio.run(
          sampler.sample(
              sampler_lib.SamplingRequest(
                  request_id="bad-start",
                  prompt=np.arange(4, dtype=np.int32),
                  max_tokens=4,
                  routed_experts_prompt_start=5,
                  metadata={"return_routed_experts": True},
              )
          )
      )

    # Verify LRU eviction through observable multi-turn output: `traj-0` runs
    # turn 0 (`TOOL_CALL`), 4096 distinct trajectories evict `traj-0` while
    # retaining `traj-4096`, so `traj-4096` advances to turn 1 (`FINAL_ANSWER`)
    # whereas `traj-0` resets to turn 0 (`TOOL_CALL`).
    first_resp = asyncio.run(
        sampler.sample(
            sampler_lib.SamplingRequest(
                request_id="traj-0",
                prompt=np.array([1, 2], dtype=np.int32),
                metadata={"min_turns": 1},
                max_tokens=2,
            )
        )
    )
    self.assertStartsWith(first_resp.text, "TOOL_CALL:")

    for i in range(1, 4097):
      asyncio.run(
          sampler.sample(
              sampler_lib.SamplingRequest(
                  request_id=f"traj-{i}",
                  prompt=np.array([1, 2], dtype=np.int32),
                  metadata={"min_turns": 1},
                  max_tokens=2,
              )
          )
      )

    retained_resp = asyncio.run(
        sampler.sample(
            sampler_lib.SamplingRequest(
                request_id="traj-4096",
                prompt=np.array([1, 2], dtype=np.int32),
                metadata={"min_turns": 1},
                max_tokens=2,
            )
        )
    )
    self.assertStartsWith(retained_resp.text, "FINAL_ANSWER:")

    # Completed (`is_final=True`) trajectories — including omitted `request_id`
    # (`"default"`) — are evicted immediately instead of poisoning later calls.
    reused_resp = asyncio.run(
        sampler.sample(
            sampler_lib.SamplingRequest(
                request_id="traj-4096",
                prompt=np.array([1, 2], dtype=np.int32),
                metadata={"min_turns": 1},
                max_tokens=2,
            )
        )
    )
    self.assertStartsWith(reused_resp.text, "TOOL_CALL:")

    for _ in range(2):
      default_turn0 = asyncio.run(
          sampler.sample(np.array([1, 2], dtype=np.int32))
      )
      self.assertStartsWith(default_turn0.text, "TOOL_CALL:")
      default_turn1 = asyncio.run(
          sampler.sample(np.array([1, 2, 3, 4], dtype=np.int32))
      )
      self.assertStartsWith(default_turn1.text, "FINAL_ANSWER:")

    evicted_resp = asyncio.run(
        sampler.sample(
            sampler_lib.SamplingRequest(
                request_id="traj-0",
                prompt=np.array([1, 2], dtype=np.int32),
                metadata={"min_turns": 1},
                max_tokens=2,
            )
        )
    )
    self.assertStartsWith(evicted_resp.text, "TOOL_CALL:")

  def test_custom_tokenizer_controls_turn0_string_prompt_token_count(self):
    class _WordTokenizer:

      def encode(self, text: str) -> list[int]:
        return [len(w) for w in text.split()]

    sampler = test_utils.MockBaseSamplerImpl(
        default_delay=0.0,
        response_tokens_per_turn=4,
        routed_experts_shape_spec=(2, 4),
        tokenizer=_WordTokenizer(),
    )
    req = sampler_lib.SamplingRequest(
        request_id="traj-words",
        prompt="prompt 1 tok tok tok",
        max_tokens=4,
        metadata={"return_routed_experts": True},
    )
    resp = asyncio.run(sampler.sample(req))
    self.assertEqual(resp.prompt_token_ids.shape, (5,))
    self.assertEqual(np.asarray(resp.routed_experts).shape, (5 + 4, 2, 4))


class FakeTrainerTest(absltest.TestCase):

  def _make_payload(
      self, routed_experts: np.ndarray | None
  ) -> datatypes.RLTrainerPayload:
    tokens = np.ones((2, 8), dtype=np.int32)
    mask = np.ones((2, 8), dtype=np.bool_)
    zeros_f = np.zeros((2, 8), dtype=np.float32)
    return datatypes.RLTrainerPayload(
        prompt_ids=np.zeros((2, 0), dtype=np.int32),
        prompt_mask=np.zeros((2, 0), dtype=np.bool_),
        completion_ids=tokens,
        completion_mask=mask,
        segment_ids=np.ones((2, 8), dtype=np.int32),
        segment_positions=np.broadcast_to(np.arange(8, dtype=np.int32), (2, 8)),
        advantages=zeros_f,
        old_per_token_logps=zeros_f,
        ref_per_token_logps=None,
        routed_experts=routed_experts,
    )

  def test_fwd_bwd_tracks_routed_experts_telemetry_and_sleep(self):
    trainer = test_utils.FakeTrainer(
        fwd_bwd_sleep_s=0.02,
        expected_routed_experts_ndim=5,
    )
    routed = np.ones((2, 8, 4, 2, 5), dtype=np.uint8)
    payload = self._make_payload(routed)
    # Packed payload: `segment_ids > 0` marks all non-padding tokens (prompt +
    # assistant + env = 14), even when loss `completion_mask` is smaller (10).
    packed_segments = np.array(
        [[1] * 8, [1] * 6 + [0] * 2], dtype=np.int32
    )
    loss_mask = np.array(
        [[True] * 6 + [False] * 2, [True] * 4 + [False] * 4], dtype=np.bool_
    )
    payload = datatypes.RLTrainerPayload(
        prompt_ids=payload.prompt_ids,
        prompt_mask=payload.prompt_mask,
        completion_ids=payload.completion_ids,
        completion_mask=loss_mask,
        segment_ids=packed_segments,
        segment_positions=payload.segment_positions,
        advantages=payload.advantages,
        old_per_token_logps=payload.old_per_token_logps,
        ref_per_token_logps=None,
        routed_experts=routed,
    )

    t0 = time.monotonic()
    trainer.fwd_bwd(payload)
    elapsed = time.monotonic() - t0

    self.assertGreaterEqual(elapsed, 0.015)
    self.assertEqual(trainer.fwd_bwd_calls, 1)
    self.assertEqual(trainer.total_tokens, 14)
    self.assertEqual(trainer.total_token_slots, 16)
    self.assertEqual(trainer.total_routed_experts_bytes, routed.nbytes)
    self.assertEqual(trainer.last_batch_shape, (2, 8))
    self.assertEqual(trainer.last_routed_experts_shape, (2, 8, 4, 2, 5))
    metrics = trainer.get_metrics()
    self.assertEqual(metrics["mock_trainer/fwd_bwd_calls"], 1)
    self.assertEqual(
        metrics["mock_trainer/routed_experts_bytes"], routed.nbytes
    )

    # Unpacked (`segment_ids=None`) sums `prompt_mask` + `completion_mask`.
    unpacked_trainer = test_utils.FakeTrainer()
    unpacked_trainer.fwd_bwd(
        datatypes.RLTrainerPayload(
            prompt_ids=np.ones((2, 3), dtype=np.int32),
            prompt_mask=np.array([[1, 1, 1], [0, 1, 1]], dtype=np.int32),
            completion_ids=np.ones((2, 4), dtype=np.int32),
            completion_mask=np.array(
                [[1, 1, 1, 0], [1, 1, 0, 0]], dtype=np.int32
            ),
            segment_ids=None,
            segment_positions=None,
            advantages=np.zeros((2, 4), dtype=np.float32),
            old_per_token_logps=np.zeros((2, 4), dtype=np.float32),
            ref_per_token_logps=None,
        )
    )
    self.assertEqual(unpacked_trainer.total_tokens, 5 + 5)
    self.assertEqual(unpacked_trainer.total_token_slots, 2 * 7)

  def test_fwd_bwd_rejects_missing_or_wrong_ndim_routed_experts(self):
    trainer = test_utils.FakeTrainer(expected_routed_experts_ndim=5)
    with self.assertRaisesRegex(ValueError, "routed_experts"):
      trainer.fwd_bwd(self._make_payload(None))
    with self.assertRaisesRegex(ValueError, "ndim"):
      trainer.fwd_bwd(self._make_payload(np.ones((2, 8, 4, 8), dtype=np.uint8)))

  def test_resolve_dummy_routed_experts_expands_placeholder_on_fake_trainer(
      self,
  ):
    trainer = test_utils.FakeTrainer(
        routed_experts_shape_spec=(6, 8),
        num_experts=384,
    )
    placeholder = np.zeros((2, 8, 0, 0), dtype=np.int32)
    payload = self._make_payload(placeholder)
    expanded = abstract_trainer.resolve_dummy_routed_experts(trainer, payload)
    trainer.compile(expanded)
    self.assertEqual(trainer.compiled_routed_experts_shape, (2, 8, 6, 8))

  def test_synthetic_fwd_only_scores_logprobs_and_records_routed_experts(self):
    routed = np.ones((2, 8, 4, 2, 5), dtype=np.uint8)
    scoring_payload = datatypes.LogprobsScoringPayload(
        prompt_ids=np.zeros((2, 0), dtype=np.int32),
        completion_ids=np.ones((2, 8), dtype=np.int32),
        pad_id=0,
        eos_id=1,
        temperature=1.0,
        routed_experts=routed,
    )
    synth_trainer = test_utils.FakeTrainer(
        expected_routed_experts_ndim=5,
        synthetic_fwd_only=True,
    )
    synth_trainer.with_fwd_only_fn(
        lambda model, _: (_ for _ in ()).throw(AssertionError(model))
    )
    out = synth_trainer.fwd_only(scoring_payload)
    self.assertEqual(out.shape, (2, 8))
    self.assertEqual(out.dtype, np.float32)
    self.assertEqual(synth_trainer.fwd_only_calls, 1)
    self.assertEqual(synth_trainer.total_routed_experts_bytes, routed.nbytes)

    callback_trainer = test_utils.FakeTrainer()
    with self.assertRaisesRegex(ValueError, "with_fwd_only_fn"):
      callback_trainer.fwd_only(scoring_payload)


if __name__ == "__main__":
  absltest.main()

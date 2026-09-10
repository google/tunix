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

"""Unit tests for TrainerWorker.

Verifies TrainerWorker RPC request unpacking (TrainRequest), delegation to
AbstractTrainer (fwd_bwd, eval_step, update), and response metadata stamping.
"""

from typing import Any

from absl.testing import absltest
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.train import abstract_trainer
from tunix.experimental.worker import trainer_worker


class FakeTrainer(abstract_trainer.AbstractTrainer):

  def __init__(self):
    self.fwd_bwd_calls = []
    self.eval_step_calls = []
    self.per_token_logps_calls = []
    self.policy_version = 3
    self.step_count = 10
    self.target_state = None

  def compile(self, dummy_data=None):
    pass

  def with_loss_fn(self, loss_fn, has_aux=False):
    pass

  def with_gen_model_input_fn(self, gen_model_input_fn):
    pass

  def fwd_bwd(self, payload, **kwargs):
    self.fwd_bwd_calls.append((payload, kwargs))

  def update(self, **kwargs):
    self.step_count += 1
    return self.step_count

  def eval_step(self, payload, **kwargs):
    self.eval_step_calls.append((payload, kwargs))

  def per_token_logps(
      self,
      *,
      prompt_tokens,
      completion_tokens,
      pad_id,
      eos_id,
      temperature=None,
      segment_ids=None,
      segment_positions=None,
      micro_batch_size=None,
  ):
    self.per_token_logps_calls.append({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "pad_id": pad_id,
        "eos_id": eos_id,
        "temperature": temperature,
        "segment_ids": segment_ids,
        "segment_positions": segment_positions,
        "micro_batch_size": micro_batch_size,
    })
    return np.zeros(np.asarray(completion_tokens).shape, dtype=np.float32)

  def save_checkpoint(self, metadata, **kwargs):
    pass

  def restore_checkpoint(self, **kwargs):
    return {}

  def get_metrics(self):
    return {"loss": 0.25}

  def prepare_weight_sync(self, **kwargs):
    pass

  def set_target_state(self, target_state: Any) -> None:
    self.target_state = target_state

  def close(self):
    pass


class TrainerWorkerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.fake_trainer = FakeTrainer()
    self.worker = trainer_worker.TrainerWorker(
        trainer_factory=lambda: self.fake_trainer,
        worker_id="trainer_0",
    )
    self.worker.initialize()

  def test_fwd_bwd_with_train_request(self):
    payload = datatypes.RLTrainerPayload(
        prompt_ids=np.array([[1, 2], [1, 2]], dtype=np.int32),
        prompt_mask=np.ones((2, 2), dtype=np.float32),
        completion_ids=np.array([[3, 4], [3, 4]], dtype=np.int32),
        completion_mask=np.array([[1, 1], [1, 0]], dtype=np.float32),
        advantages=np.array([1.0, 2.0], dtype=np.float32),
        metadata={"step": 1},
    )
    request = datatypes.TrainRequest(
        request_id="req-train-123",
        payload=payload,
        metadata={"batch_id": "b0"},
    )

    resp = self.worker.fwd_bwd(request=request)

    self.assertIsInstance(resp, datatypes.Response)
    self.assertEqual(resp.request_id, "req-train-123")
    self.assertEqual(resp.metadata["worker_id"], "trainer_0")
    self.assertEqual(resp.metadata["batch_id"], "b0")
    self.assertEqual(resp.metadata["policy_version"], 3)
    self.assertTrue(resp.metadata["queued"])
    self.assertLen(self.fake_trainer.fwd_bwd_calls, 1)
    self.assertIs(self.fake_trainer.fwd_bwd_calls[0][0], payload)

  def test_eval_step_with_train_request(self):
    payload = datatypes.RLTrainerPayload(
        prompt_ids=np.array([[1]], dtype=np.int32),
        prompt_mask=np.ones((1, 1), dtype=np.float32),
        completion_ids=np.array([[2]], dtype=np.int32),
        completion_mask=np.array([[1]], dtype=np.float32),
        advantages=np.array([1.0], dtype=np.float32),
    )
    request = datatypes.TrainRequest(
        request_id="req-eval-456",
        payload=payload,
        metadata={"eval_split": "val"},
    )

    resp = self.worker.eval_step(request=request)

    self.assertIsInstance(resp, datatypes.Response)
    self.assertEqual(resp.request_id, "req-eval-456")
    self.assertEqual(resp.metadata["eval_split"], "val")
    self.assertTrue(resp.metadata["evaluated"])
    self.assertLen(self.fake_trainer.eval_step_calls, 1)
    self.assertIs(self.fake_trainer.eval_step_calls[0][0], payload)

  def test_update_returns_step_count(self):
    step = self.worker.update()
    self.assertEqual(step, 11)

  def test_set_target_state_configures_trainer(self):
    target_state = {"params": np.zeros((4, 4))}
    resp = self.worker.set_target_state(target_state=target_state)

    self.assertIsInstance(resp, datatypes.Response)
    self.assertTrue(resp.metadata["target_state_configured"])
    self.assertEqual(self.fake_trainer.target_state, target_state)

  def test_set_target_state_raises_when_trainer_unsupported(self):
    self.fake_trainer.set_target_state = None
    with self.assertRaises(AttributeError):
      self.worker.set_target_state(target_state={"params": 1})

  def test_per_token_logps_unpacks_request_and_delegates(self):
    request = datatypes.LogprobsRequest(
        prompt_tokens=np.zeros((2, 0), dtype=np.int32),
        completion_tokens=np.array([[3, 4, 5], [3, 4, 0]], dtype=np.int32),
        temperature=0.7,
        pad_id=0,
        eos_id=1,
        segment_ids=np.array([[1, 1, 1], [1, 1, 0]], dtype=np.int32),
        segment_positions=np.array([[0, 1, 2], [0, 1, 0]], dtype=np.int32),
    )

    result = self.worker.per_token_logps(items=request)

    self.assertIsInstance(result, datatypes.LogprobsResponse)
    self.assertEqual(result.request_id, request.request_id)
    self.assertEqual(result.model_version, self.fake_trainer.policy_version)
    self.assertIsInstance(result.per_token_logps, np.ndarray)
    self.assertEqual(result.per_token_logps.dtype, np.float32)
    self.assertEqual(result.per_token_logps.shape, (2, 3))
    self.assertLen(self.fake_trainer.per_token_logps_calls, 1)
    call = self.fake_trainer.per_token_logps_calls[0]
    self.assertEqual(call["pad_id"], 0)
    self.assertEqual(call["eos_id"], 1)
    self.assertEqual(call["temperature"], 0.7)
    np.testing.assert_array_equal(
        call["completion_tokens"], request.completion_tokens
    )
    np.testing.assert_array_equal(call["segment_ids"], request.segment_ids)
    np.testing.assert_array_equal(
        call["segment_positions"], request.segment_positions
    )

  def test_per_token_logps_requires_pad_and_eos(self):
    request = datatypes.LogprobsRequest(
        prompt_tokens=np.zeros((1, 0), dtype=np.int32),
        completion_tokens=np.array([[3, 4]], dtype=np.int32),
        temperature=1.0,
    )
    with self.assertRaises(ValueError):
      self.worker.per_token_logps(items=request)
    self.assertEmpty(self.fake_trainer.per_token_logps_calls)


if __name__ == "__main__":
  absltest.main()

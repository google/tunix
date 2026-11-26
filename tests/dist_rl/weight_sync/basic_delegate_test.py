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

"""Unit tests for framework-neutral BasicDestinationDelegate weight sync."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

from absl.testing import absltest
import numpy as np
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.weight_sync import basic_delegate
from tunix.dist_rl.weight_sync import basic_handler
from tunix.dist_rl.weight_sync import weight_sync


def _tensor_meta(
    name: str, shape: tuple[int, ...], *, item_size: int = 4
) -> weight_sync.TensorMetadata:
  return weight_sync.TensorMetadata(
      name=name,
      shape=shape,
      mesh_shape=(1,) * len(shape),
      layout=tuple(range(len(shape) - 1, -1, -1)),
      item_size=item_size,
  )


class _FakeHfSampler:
  """Typed test double implementing `HfWeightUpdatableSampler`."""

  def __init__(
      self,
      variables: tuple[weight_sync.TensorMetadata, ...],
      *,
      error: Exception | None = None,
  ):
    self._variables = variables
    self.error = error
    self.received: list[
        dict[str, basic_handler.RawTensorPayload | np.ndarray]
    ] = []
    self.preserve_kv_calls: list[bool | None] = []
    self.delete_cache_calls = 0
    self.reinit_cache_calls = 0

  def expected_hf_variables(self) -> tuple[weight_sync.TensorMetadata, ...]:
    return self._variables

  def update_params_from_hf(
      self,
      hf_weights: Mapping[str, basic_handler.RawTensorPayload | np.ndarray],
      *,
      preserve_active_kv_cache: bool | None = None,
  ) -> None:
    if self.error is not None:
      raise self.error
    self.preserve_kv_calls.append(preserve_active_kv_cache)
    self.received.append(dict(hf_weights))

  def delete_cache(self) -> None:
    self.delete_cache_calls += 1

  def reinitialize_cache(self) -> None:
    self.reinit_cache_calls += 1


class BasicDestinationDelegateTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.delegate = basic_delegate.BasicDestinationDelegate(
        server_id="rollout_0",
        worker_index=0,
    )

  def tearDown(self):
    self.delegate.close()
    super().tearDown()

  def test_bind_and_metadata(self):
    sampler = _FakeHfSampler((_tensor_meta("model.layers.0.weight", (4, 4)),))

    res = asyncio.run(self.delegate.bind_weight_sync(sampler=sampler))
    self.assertTrue(res)
    self.assertTrue(self.delegate.is_bounded())

    meta = asyncio.run(self.delegate.get_weight_sync_metadata())
    self.assertLen(meta, 1)
    self.assertEqual(meta[0].unit.job_name, "rollout_0")
    self.assertEqual(meta[0].transport_mode, "basic")
    self.assertIsNotNone(meta[0].artifact_uri)
    self.assertTrue(meta[0].artifact_uri.startswith("basic-token:"))
    self.assertEqual(meta[0].variables[0].name, "model.layers.0.weight")
    self.assertEqual(meta[0].variables[0].shape, (4, 4))

  def test_bind_rejects_invalid_sampler_and_ignores_state(self):
    with self.assertRaises(TypeError):
      asyncio.run(self.delegate.bind_weight_sync(sampler=None))
    with self.assertRaises(TypeError):
      asyncio.run(
          self.delegate.bind_weight_sync(sampler=object())  # pyrefly: ignore[bad-argument-type]
      )
    sampler = _FakeHfSampler((_tensor_meta("w", (1,)),))
    self.assertTrue(
        asyncio.run(self.delegate.bind_weight_sync(sampler=sampler, state=None))
    )

  def test_basic_destination_delegate_does_not_touch_kv_cache(self):
    sampler = _FakeHfSampler((_tensor_meta("w", (2, 2), item_size=2),))
    self.assertIsInstance(sampler, basic_delegate.HfWeightUpdatableSampler)

    asyncio.run(self.delegate.bind_weight_sync(sampler=sampler))

    req = datatypes.WeightSyncRequest(
        extra_config={"req_id": "r1", "uuid": 1},
        policy_version=1,
    )
    asyncio.run(self.delegate.pre_weight_sync(req))
    assert self.delegate._receiver is not None
    self.delegate._receiver._staged_rounds[("r1", 1)] = {
        "w": basic_handler.RawTensorPayload(
            buffer=np.ones((2, 2), dtype=np.uint16), shape=(2, 2), dtype="bf16"
        )
    }
    asyncio.run(self.delegate.weight_sync(req))
    asyncio.run(self.delegate.post_weight_sync(req))

    self.assertLen(sampler.received, 1)
    self.assertEqual(sampler.delete_cache_calls, 0)
    self.assertEqual(sampler.reinit_cache_calls, 0)

  def test_full_lifecycle_and_corruption_invariant(self):
    sampler = _FakeHfSampler((_tensor_meta("w", (2, 2), item_size=2),))

    asyncio.run(self.delegate.bind_weight_sync(sampler=sampler))

    # 1. Pre weight sync
    req = datatypes.WeightSyncRequest(
        extra_config={"req_id": "r1", "uuid": 1},
        policy_version=10,
    )
    asyncio.run(self.delegate.pre_weight_sync(req))

    # Manually inject round into receiver
    assert self.delegate._receiver is not None
    payload = basic_handler.RawTensorPayload(
        buffer=np.ones((2, 2), dtype=np.uint16), shape=(2, 2), dtype="bf16"
    )
    self.delegate._receiver._staged_rounds[("r1", 1)] = {"w": payload}

    # 2. Weight sync
    v = asyncio.run(self.delegate.weight_sync(req))
    self.assertEqual(v, 10)
    self.assertLen(sampler.received, 1)
    self.assertIs(sampler.received[0]["w"], payload)

    # 3. Post weight sync
    asyncio.run(self.delegate.post_weight_sync(req))
    status = self.delegate.get_weight_sync_status()
    self.assertEqual(status["phase"], "committed")

    # Invariant 5: Fail during update_params_from_hf sets serving_corrupted
    req2 = datatypes.WeightSyncRequest(
        extra_config={"req_id": "r2", "uuid": 2},
        policy_version=11,
    )
    asyncio.run(self.delegate.pre_weight_sync(req2))
    assert self.delegate._receiver is not None
    self.delegate._receiver._staged_rounds[("r2", 2)] = {
        "w": basic_handler.RawTensorPayload(
            buffer=np.ones((2, 2), dtype=np.uint16), shape=(2, 2), dtype="bf16"
        )
    }
    sampler.error = RuntimeError("TPU HBM corrupted!")

    with self.assertRaises(RuntimeError):
      asyncio.run(self.delegate.weight_sync(req2))

    self.assertTrue(self.delegate.serving_corrupted)

    # Abort and post must now fail and refuse to clear corruption
    with self.assertRaises(RuntimeError):
      asyncio.run(self.delegate.abort_weight_sync(req2))
    with self.assertRaises(RuntimeError):
      asyncio.run(self.delegate.post_weight_sync(req2))

    # New round must also be refused
    req3 = datatypes.WeightSyncRequest(
        extra_config={"req_id": "r3", "uuid": 3},
        policy_version=12,
    )
    with self.assertRaises(RuntimeError):
      asyncio.run(self.delegate.pre_weight_sync(req3))

  def test_phase_methods_reject_invalid_sync_request_without_tracker_pollution(
      self,
  ):
    initial_status = self.delegate.get_weight_sync_status()
    invalid_requests = [
        None,
        datatypes.WeightSyncRequest(request_id="123", extra_config={}),
        datatypes.WeightSyncRequest(extra_config={"req_id": "r_no_uuid"}),
        datatypes.WeightSyncRequest(
            extra_config={"req_id": "r_zero_uuid", "uuid": 0}
        ),
    ]
    for bad_req in invalid_requests:
      for phase_fn in (
          self.delegate.pre_weight_sync,
          self.delegate.weight_sync,
          self.delegate.post_weight_sync,
          self.delegate.abort_weight_sync,
      ):
        with self.assertRaises(ValueError):
          asyncio.run(phase_fn(bad_req))
        self.assertEqual(self.delegate.get_weight_sync_status(), initial_status)

  def test_weight_sync_forwards_raw_uint16_and_bf16_payloads_verbatim(self):
    sampler = _FakeHfSampler((
        _tensor_meta("u16", (1,), item_size=2),
        _tensor_meta("bf16", (1,), item_size=2),
        _tensor_meta("bfloat16", (1,), item_size=2),
    ))
    asyncio.run(self.delegate.bind_weight_sync(sampler=sampler))

    req = datatypes.WeightSyncRequest(
        extra_config={"req_id": "r_dtypes", "uuid": 1},
        policy_version=1,
    )
    asyncio.run(self.delegate.pre_weight_sync(req))
    assert self.delegate._receiver is not None
    self.delegate._receiver._staged_rounds[("r_dtypes", 1)] = {
        "u16": basic_handler.RawTensorPayload(
            buffer=np.array([0x3F80], dtype=np.uint16),
            shape=(1,),
            dtype="uint16",
        ),
        "bf16": basic_handler.RawTensorPayload(
            buffer=np.array([0x3F80], dtype=np.uint16),
            shape=(1,),
            dtype="bf16",
        ),
        "bfloat16": basic_handler.RawTensorPayload(
            buffer=np.array([0x4000], dtype=np.uint16),
            shape=(1,),
            dtype="bfloat16",
        ),
    }
    asyncio.run(self.delegate.weight_sync(req))
    asyncio.run(self.delegate.post_weight_sync(req))

    self.assertLen(sampler.received, 1)
    received = sampler.received[0]
    self.assertIsInstance(received["u16"], basic_handler.RawTensorPayload)
    assert isinstance(received["u16"], basic_handler.RawTensorPayload)
    self.assertEqual(received["u16"].dtype, "uint16")
    self.assertEqual(int(received["u16"].buffer[0]), 0x3F80)

    assert isinstance(received["bf16"], basic_handler.RawTensorPayload)
    self.assertEqual(received["bf16"].dtype, "bf16")
    self.assertEqual(int(received["bf16"].buffer[0]), 0x3F80)

    assert isinstance(received["bfloat16"], basic_handler.RawTensorPayload)
    self.assertEqual(received["bfloat16"].dtype, "bfloat16")
    self.assertEqual(int(received["bfloat16"].buffer[0]), 0x4000)

  def test_abort_after_in_place_weight_sync_latches_corruption(self):
    # (a) pre -> weight_sync -> abort latches corruption and leaves h2d_done.
    sampler = _FakeHfSampler((_tensor_meta("w", (2, 2)),))
    asyncio.run(self.delegate.bind_weight_sync(sampler=sampler))
    req = datatypes.WeightSyncRequest(
        extra_config={"req_id": "r_abort_post_h2d", "uuid": 1},
        policy_version=1,
    )
    asyncio.run(self.delegate.pre_weight_sync(req))
    assert self.delegate._receiver is not None
    self.delegate._receiver._staged_rounds[("r_abort_post_h2d", 1)] = {
        "w": basic_handler.RawTensorPayload(
            buffer=np.ones((2, 2), dtype=np.float32), shape=(2, 2), dtype="f32"
        )
    }
    asyncio.run(self.delegate.weight_sync(req))

    with self.assertRaisesRegex(RuntimeError, "Cannot roll back uncommitted"):
      asyncio.run(self.delegate.abort_weight_sync(req))
    self.assertTrue(self.delegate.serving_corrupted)
    self.assertEqual(
        self.delegate.get_weight_sync_status()["phase"], "h2d_done"
    )

    # (b) pre -> weight_sync -> post -> late abort returns False cleanly.
    clean_delegate = basic_delegate.BasicDestinationDelegate(
        server_id="rollout_clean", worker_index=1
    )
    try:
      clean_sampler = _FakeHfSampler((_tensor_meta("w", (2, 2)),))
      asyncio.run(clean_delegate.bind_weight_sync(sampler=clean_sampler))
      req_ok = datatypes.WeightSyncRequest(
          extra_config={"req_id": "r_committed", "uuid": 2},
          policy_version=2,
      )
      asyncio.run(clean_delegate.pre_weight_sync(req_ok))
      assert clean_delegate._receiver is not None
      clean_delegate._receiver._staged_rounds[("r_committed", 2)] = {
          "w": basic_handler.RawTensorPayload(
              buffer=np.ones((2, 2), dtype=np.float32),
              shape=(2, 2),
              dtype="f32",
          )
      }
      asyncio.run(clean_delegate.weight_sync(req_ok))
      asyncio.run(clean_delegate.post_weight_sync(req_ok))

      self.assertFalse(asyncio.run(clean_delegate.abort_weight_sync(req_ok)))
      self.assertFalse(clean_delegate.serving_corrupted)
      self.assertFalse(clean_delegate.has_uncommitted_mutation)
      self.assertEqual(
          clean_delegate.get_weight_sync_status()["phase"], "committed"
      )
    finally:
      clean_delegate.close()

  def test_abort_newer_round_while_uncommitted_h2d_latches_corruption(self):
    sampler = _FakeHfSampler((_tensor_meta("w", (2, 2)),))
    asyncio.run(self.delegate.bind_weight_sync(sampler=sampler))
    req1 = datatypes.WeightSyncRequest(
        extra_config={"req_id": "r1_uncommitted", "uuid": 1},
        policy_version=1,
    )
    asyncio.run(self.delegate.pre_weight_sync(req1))
    assert self.delegate._receiver is not None
    self.delegate._receiver._staged_rounds[("r1_uncommitted", 1)] = {
        "w": basic_handler.RawTensorPayload(
            buffer=np.ones((2, 2), dtype=np.float32), shape=(2, 2), dtype="f32"
        )
    }
    asyncio.run(self.delegate.weight_sync(req1))
    self.assertTrue(self.delegate.has_uncommitted_mutation)

    req2 = datatypes.WeightSyncRequest(
        extra_config={"req_id": "r2_newer", "uuid": 2},
        policy_version=2,
    )
    with self.assertRaisesRegex(RuntimeError, "Cannot roll back uncommitted"):
      asyncio.run(self.delegate.abort_weight_sync(req2))
    self.assertTrue(self.delegate.serving_corrupted)

  def test_abort_before_h2d_evicts_staged_round_and_marks_aborted(self):
    sampler = _FakeHfSampler((_tensor_meta("w", (2, 2)),))
    asyncio.run(self.delegate.bind_weight_sync(sampler=sampler))
    req = datatypes.WeightSyncRequest(
        extra_config={"req_id": "r_pre_h2d_abort", "uuid": 3},
        policy_version=3,
    )
    asyncio.run(self.delegate.pre_weight_sync(req))
    assert self.delegate._receiver is not None
    self.delegate._receiver._staged_rounds[("r_pre_h2d_abort", 3)] = {
        "w": basic_handler.RawTensorPayload(
            buffer=np.ones((2, 2), dtype=np.float32), shape=(2, 2), dtype="f32"
        )
    }

    self.assertTrue(asyncio.run(self.delegate.abort_weight_sync(req)))
    self.assertFalse(self.delegate.serving_corrupted)
    self.assertFalse(self.delegate.has_uncommitted_mutation)
    self.assertEqual(self.delegate.get_weight_sync_status()["phase"], "aborted")
    with self.assertRaises(RuntimeError):
      self.delegate._receiver.pop_round("r_pre_h2d_abort", 3)
    self.assertIn(
        ("r_pre_h2d_abort", 3), self.delegate._receiver._discarded_rounds
    )

  def test_weight_sync_refused_when_serving_corrupted(self):
    failing_sampler = _FakeHfSampler(
        (_tensor_meta("w", (2, 2)),), error=RuntimeError("boom")
    )
    asyncio.run(self.delegate.bind_weight_sync(sampler=failing_sampler))
    req1 = datatypes.WeightSyncRequest(
        extra_config={"req_id": "r_corrupt", "uuid": 1},
        policy_version=1,
    )
    asyncio.run(self.delegate.pre_weight_sync(req1))
    assert self.delegate._receiver is not None
    self.delegate._receiver._staged_rounds[("r_corrupt", 1)] = {
        "w": basic_handler.RawTensorPayload(
            buffer=np.ones((2, 2), dtype=np.float32), shape=(2, 2), dtype="f32"
        )
    }
    with self.assertRaisesRegex(RuntimeError, "boom"):
      asyncio.run(self.delegate.weight_sync(req1))
    self.assertTrue(self.delegate.serving_corrupted)

    req2 = datatypes.WeightSyncRequest(
        extra_config={"req_id": "r_after_corrupt", "uuid": 2},
        policy_version=2,
    )
    with self.assertRaisesRegex(RuntimeError, "Serving weights are corrupted"):
      asyncio.run(self.delegate.weight_sync(req2))
    with self.assertRaisesRegex(RuntimeError, "Serving weights are corrupted"):
      asyncio.run(self.delegate.pre_weight_sync(req2))

  def test_close_latches_closed_and_rejects_subsequent_calls(self):
    sampler = _FakeHfSampler((_tensor_meta("w", (2, 2)),))
    asyncio.run(self.delegate.bind_weight_sync(sampler=sampler))
    self.delegate.close()
    self.delegate.close()  # idempotent
    self.assertFalse(self.delegate.is_bounded())

    req = datatypes.WeightSyncRequest(
        extra_config={"req_id": "r_after_close", "uuid": 1},
        policy_version=1,
    )
    with self.assertRaisesRegex(RuntimeError, "closed"):
      asyncio.run(self.delegate.bind_weight_sync(sampler=sampler))
    with self.assertRaisesRegex(RuntimeError, "closed"):
      asyncio.run(self.delegate.get_weight_sync_metadata())
    with self.assertRaisesRegex(RuntimeError, "closed"):
      asyncio.run(self.delegate.pre_weight_sync(req))
    with self.assertRaisesRegex(RuntimeError, "closed"):
      asyncio.run(self.delegate.weight_sync(req))
    with self.assertRaisesRegex(RuntimeError, "closed"):
      asyncio.run(self.delegate.post_weight_sync(req))
    with self.assertRaisesRegex(RuntimeError, "closed"):
      asyncio.run(self.delegate.abort_weight_sync(req))

  def test_partial_rollout_forwards_preserve_active_kv_cache(self):
    # Default partial_rollout=None -> preserve_active_kv_cache=None.
    sampler_default = _FakeHfSampler((_tensor_meta("w", (2, 2)),))
    asyncio.run(self.delegate.bind_weight_sync(sampler=sampler_default))
    req_default = datatypes.WeightSyncRequest(
        extra_config={"req_id": "r_default", "uuid": 1}, policy_version=1
    )
    asyncio.run(self.delegate.pre_weight_sync(req_default))
    assert self.delegate._receiver is not None
    self.delegate._receiver._staged_rounds[("r_default", 1)] = {
        "w": basic_handler.RawTensorPayload(
            buffer=np.ones((2, 2), dtype=np.float32), shape=(2, 2), dtype="f32"
        )
    }
    asyncio.run(self.delegate.weight_sync(req_default))
    asyncio.run(self.delegate.post_weight_sync(req_default))
    self.assertEqual(sampler_default.preserve_kv_calls, [None])

    # Explicit partial_rollout=True and per-round partial_rollout=False.
    delegate_partial = basic_delegate.BasicDestinationDelegate(
        server_id="rollout_partial",
        worker_index=2,
        partial_rollout=True,
    )
    try:
      sampler_partial = _FakeHfSampler((_tensor_meta("w", (2, 2)),))
      asyncio.run(delegate_partial.bind_weight_sync(sampler=sampler_partial))
      req1 = datatypes.WeightSyncRequest(
          extra_config={"req_id": "r_partial_1", "uuid": 1}, policy_version=1
      )
      asyncio.run(delegate_partial.pre_weight_sync(req1))
      assert delegate_partial._receiver is not None
      delegate_partial._receiver._staged_rounds[("r_partial_1", 1)] = {
          "w": basic_handler.RawTensorPayload(
              buffer=np.ones((2, 2), dtype=np.float32),
              shape=(2, 2),
              dtype="f32",
          )
      }
      asyncio.run(delegate_partial.weight_sync(req1))
      asyncio.run(delegate_partial.post_weight_sync(req1))

      req2 = datatypes.WeightSyncRequest(
          extra_config={"req_id": "r_partial_2", "uuid": 2}, policy_version=2
      )
      asyncio.run(delegate_partial.pre_weight_sync(req2))
      delegate_partial._receiver._staged_rounds[("r_partial_2", 2)] = {
          "w": basic_handler.RawTensorPayload(
              buffer=np.ones((2, 2), dtype=np.float32),
              shape=(2, 2),
              dtype="f32",
          )
      }
      asyncio.run(delegate_partial.weight_sync(req2, partial_rollout=False))
      asyncio.run(delegate_partial.post_weight_sync(req2))

      self.assertEqual(sampler_partial.preserve_kv_calls, [True, False])
    finally:
      delegate_partial.close()

  def test_end_to_end_loopback_basic_transfer_via_handler(self):
    sampler = _FakeHfSampler((
        _tensor_meta("w_f32", (2, 3), item_size=4),
        _tensor_meta("w_bf16", (4,), item_size=2),
    ))

    exporter = basic_handler.BasicSourceExporter(
        job_name="trainer_e2e", bind_port=0, advertised_host="127.0.0.1"
    )
    delegate = basic_delegate.BasicDestinationDelegate(
        server_id="rollout_e2e",
        worker_index=0,
        bind_port=0,
        advertised_host="127.0.0.1",
    )
    handler = basic_handler.BasicWeightSyncHandler(chunk_bytes=16)
    try:
      f32_arr = np.arange(6, dtype=np.float32).reshape(2, 3)
      bf16_u16_arr = np.array(
          [0x3FC0, 0xC010, 0x4040, 0x3F00], dtype=np.uint16
      )
      req = datatypes.WeightSyncRequest(
          extra_config={"req_id": "r_e2e", "uuid": 5},
          policy_version=5,
      )
      src_metas = exporter.prepare_weight_sync(
          req,
          {
              "w_f32": basic_handler.RawTensorPayload(
                  buffer=f32_arr, shape=(2, 3), dtype="f32"
              ),
              "w_bf16": basic_handler.RawTensorPayload(
                  buffer=bf16_u16_arr, shape=(4,), dtype="bf16"
              ),
          },
      )
      self.assertLen(src_metas, 1)
      handler.register_work_unit(src_metas[0])

      asyncio.run(delegate.bind_weight_sync(sampler=sampler))
      dst_metas = asyncio.run(delegate.get_weight_sync_metadata())
      self.assertLen(dst_metas, 1)
      handler.register_work_unit(dst_metas[0])

      asyncio.run(delegate.pre_weight_sync(req))
      res = handler.transfer(
          src_units=[src_metas[0].unit],
          dst_units=[dst_metas[0].unit],
          req_id="r_e2e",
          generation=5,
      )
      self.assertTrue(res.success, msg=res.message)

      installed_ver = asyncio.run(delegate.weight_sync(req))
      self.assertEqual(installed_ver, 5)
      asyncio.run(delegate.post_weight_sync(req))
      exporter.release_weight_sync(req)

      self.assertLen(sampler.received, 1)
      got_f32 = sampler.received[0]["w_f32"]
      assert isinstance(got_f32, basic_handler.RawTensorPayload)
      self.assertEqual(got_f32.dtype, "f32")
      self.assertEqual(got_f32.shape, (2, 3))
      np.testing.assert_array_equal(got_f32.buffer, f32_arr)

      got_bf16 = sampler.received[0]["w_bf16"]
      assert isinstance(got_bf16, basic_handler.RawTensorPayload)
      self.assertEqual(got_bf16.dtype, "bf16")
      self.assertEqual(got_bf16.shape, (4,))
      np.testing.assert_array_equal(got_bf16.buffer, bf16_u16_arr)
      self.assertEqual(delegate.get_weight_sync_status()["phase"], "committed")
    finally:
      exporter.close()
      delegate.close()


if __name__ == "__main__":
  absltest.main()

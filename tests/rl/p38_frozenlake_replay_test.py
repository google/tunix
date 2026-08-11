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

"""CPU contracts for the P38 FrozenLake causal replay schedules."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

from absl.testing import absltest
import numpy as np


_MODULE = Path(__file__).parents[2] / "tunix/rl/p38_frozenlake_replay.py"
_SPEC = importlib.util.spec_from_file_location("p38_frozenlake_replay", _MODULE)
replay = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = replay
_SPEC.loader.exec_module(replay)


def _sha(value):
  return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


class P38FrozenLakeReplayTest(absltest.TestCase):

  def _write_capsule(self, root: Path, *, mutate=None) -> Path:
    arrays = {
        "prompt_ids": np.asarray([[0, 0, 11, 12, 13]], np.int32),
        "prompt_mask": np.asarray([[False, False, True, True, True]]),
        "completion_ids": np.asarray([[21, 22, 31, 32, 23, 0]], np.int32),
        "completion_valid_mask": np.asarray(
            [[True, True, True, True, True, False]]
        ),
        "action_mask": np.asarray(
            [[True, True, False, False, True, False]]
        ),
        "s_decode": np.asarray([[1, 2, 3, 4, 5, 0]], np.float32),
        "s_prefill": np.asarray([[1, 2, 3, 4, 5, 0]], np.float32),
        "t_old": np.asarray([[1, 2, 3, 4, 5, 0]], np.float32),
        "policy_version": np.asarray([[7]], np.int32),
        "sampling_values": np.asarray([[0.7, 1.0, 0.0]], np.float32),
    }
    if mutate is not None:
      mutate(arrays)
    metadata = {
        "schema": replay.SCHEMA,
        "selected_rows": [238],
        "arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _sha(value),
            }
            for name, value in arrays.items()
        },
    }
    path = root / "capsule.npz"
    np.savez_compressed(
        path,
        selected_rows=np.asarray([238], np.int32),
        metadata_json=np.frombuffer(
            json.dumps(metadata, sort_keys=True).encode(), dtype=np.uint8
        ),
        **arrays,
    )
    return path

  def test_loads_hash_verified_compact_row(self):
    with tempfile.TemporaryDirectory() as tmp:
      capsule = replay.load_verified_capsule(
          self._write_capsule(Path(tmp))
      )
    self.assertLen(capsule.rows, 1)
    row = capsule.rows[0]
    self.assertEqual(row.source_row, 238)
    np.testing.assert_array_equal(row.prompt_ids, [11, 12, 13])
    np.testing.assert_array_equal(row.completion_ids, [21, 22, 31, 32, 23])
    np.testing.assert_array_equal(row.action_mask, [True, True, False, False, True])

  def test_rejects_one_bit_array_change_after_metadata_is_written(self):
    with tempfile.TemporaryDirectory() as tmp:
      path = self._write_capsule(Path(tmp))
      with np.load(path, allow_pickle=False) as archive:
        values = {name: archive[name].copy() for name in archive.files}
      values["completion_ids"][0, 0] ^= np.int32(1)
      np.savez_compressed(path, **values)
      with self.assertRaisesRegex(replay.P38ReplayError, "array hash mismatch"):
        replay.load_verified_capsule(path)

  def test_rejects_action_outside_valid_completion(self):
    def mutate(arrays):
      arrays["action_mask"][0, -1] = True

    with tempfile.TemporaryDirectory() as tmp:
      with self.assertRaisesRegex(replay.P38ReplayError, "invalid completion"):
        replay.load_verified_capsule(
            self._write_capsule(Path(tmp), mutate=mutate)
        )

  def test_rejects_completion_validity_hole(self):
    def mutate(arrays):
      arrays["completion_valid_mask"][0, 2] = False
      arrays["action_mask"][0, 2] = False

    with tempfile.TemporaryDirectory() as tmp:
      with self.assertRaisesRegex(replay.P38ReplayError, "contiguous prefix"):
        replay.load_verified_capsule(
            self._write_capsule(Path(tmp), mutate=mutate)
        )

  def test_r0_groups_environment_tokens_and_covers_actions_once(self):
    with tempfile.TemporaryDirectory() as tmp:
      row = replay.load_verified_capsule(
          self._write_capsule(Path(tmp))
      ).rows[0]
    schedule = replay.build_r0_mask_derived_schedule(row, local_m=256)
    self.assertEqual(schedule.provenance, "mask-derived-v1")
    self.assertEqual(
        [call.kind for call in schedule.calls],
        ["initial_prefill", "decode", "decode", "environment_prefill"],
    )
    self.assertEqual(schedule.calls[-1].query_length, 2)
    self.assertEqual(schedule.calls[-1].distribution, (0, 0, 1))
    self.assertEqual(
        sorted(target for call in schedule.calls for target in call.action_targets),
        [0, 1, 4],
    )

  def test_r1_decodes_every_post_prompt_input_token(self):
    with tempfile.TemporaryDirectory() as tmp:
      row = replay.load_verified_capsule(
          self._write_capsule(Path(tmp))
      ).rows[0]
    schedule = replay.build_r1_continuous_decode_schedule(row, local_m=256)
    self.assertEqual(schedule.calls[0].kind, "initial_prefill")
    self.assertTrue(
        all(call.query_length == 1 for call in schedule.calls[1:])
    )
    self.assertTrue(
        all(call.distribution == (1, 1, 1) for call in schedule.calls[1:])
    )
    self.assertEqual(schedule.logical_input_ids.size, 7)

  def test_reference_uses_full_sequence_and_mixed_fixed_chunks(self):
    with tempfile.TemporaryDirectory() as tmp:
      row = replay.load_verified_capsule(
          self._write_capsule(Path(tmp))
      ).rows[0]
    schedule = replay.build_fixed_chunk_reference_schedule(row, local_m=4)
    self.assertEqual(schedule.logical_input_ids.size, 8)
    self.assertTrue(
        all(call.distribution == (0, 0, 1) for call in schedule.calls)
    )
    self.assertTrue(
        all(call.query_length <= 4 for call in schedule.calls)
    )

  def test_long_prompt_is_chunked_without_losing_first_action_predictor(self):
    row = replay.CapsuleRow(
        source_row=7,
        prompt_ids=np.arange(300, dtype=np.int32),
        completion_ids=np.asarray([901, 902], np.int32),
        action_mask=np.asarray([True, True]),
        s_decode=np.zeros(2, np.float32),
        s_prefill=np.zeros(2, np.float32),
        t_old=np.zeros(2, np.float32),
        policy_version=np.asarray([0], np.int32),
        sampling_values=np.asarray([0.7], np.float32),
    )
    schedule = replay.build_r0_mask_derived_schedule(row, local_m=256)
    self.assertEqual(schedule.calls[0].query_length, 256)
    self.assertEqual(schedule.calls[0].distribution, (0, 1, 1))
    self.assertEqual(schedule.calls[1].query_length, 44)
    self.assertEqual(schedule.calls[1].distribution, (0, 0, 1))
    self.assertEqual(schedule.calls[1].action_targets, (0,))

  def test_schedule_report_preserves_not_run_claim_ceiling(self):
    with tempfile.TemporaryDirectory() as tmp:
      capsule = replay.load_verified_capsule(
          self._write_capsule(Path(tmp))
      )
      schedules = (
          replay.build_r0_mask_derived_schedule(capsule.rows[0]),
          replay.build_r1_continuous_decode_schedule(capsule.rows[0]),
      )
      report = replay.schedules_report(capsule, schedules)
    self.assertEqual(report["verdict"], "LOCALLY_ADMITTED")
    self.assertEqual(report["tpu_status"], "NOT_RUN")
    self.assertIn("derived", report["claim_ceiling"])

  def test_engine_records_preserve_fixed_m_and_call_distribution(self):
    with tempfile.TemporaryDirectory() as tmp:
      row = replay.load_verified_capsule(
          self._write_capsule(Path(tmp))
      ).rows[0]
    schedule = replay.build_r0_mask_derived_schedule(row, local_m=256)
    records = replay.build_engine_records(
        schedule,
        max_num_reqs=2,
        blocks_per_request=8,
        cache_blocks=8,
    )
    self.assertLen(records, len(schedule.calls))
    for record, call in zip(records, schedule.calls, strict=True):
      arrays = record["arrays"]
      self.assertEqual(arrays["input_ids"].shape, (256,))
      self.assertEqual(
          arrays["md_query_start_loc"].tolist(),
          [0, call.query_length, call.query_length],
      )
      self.assertEqual(
          arrays["md_request_distribution"].tolist(),
          list(call.distribution),
      )
      self.assertEqual(record["meta"]["schedule_provenance"], "mask-derived-v1")

  def test_engine_records_refuse_page_table_larger_than_cache(self):
    with tempfile.TemporaryDirectory() as tmp:
      row = replay.load_verified_capsule(
          self._write_capsule(Path(tmp))
      ).rows[0]
    schedule = replay.build_r0_mask_derived_schedule(row)
    with self.assertRaisesRegex(replay.P38ReplayError, "exceeds"):
      replay.build_engine_records(
          schedule,
          max_num_reqs=2,
          blocks_per_request=9,
          cache_blocks=8,
      )


if __name__ == "__main__":
  absltest.main()

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""CPU contracts for the P35 rank-strided envelope producer."""

import json
import tempfile
from pathlib import Path

from absl.testing import absltest
import numpy as np

from tunix.rl import envelope_probe


class EnvelopeProbeTest(absltest.TestCase):

  def _write_metadata_record(
      self,
      root,
      index,
      arm,
      *,
      data_size=2,
      local_m=256,
      sequences=((11, 12), (21, 22, 23)),
      request_ids=("a", "b"),
      starts=None,
      query_lengths=None,
  ):
    starts = tuple(starts or (0,) * data_size)
    if query_lengths is None:
      query_lengths = tuple(
          min(local_m, len(sequence) - start)
          for sequence, start in zip(sequences, starts)
      )
    input_ids = np.zeros((data_size, local_m), np.int32)
    positions = np.zeros_like(input_ids)
    for rank, (sequence, start, query_length) in enumerate(
        zip(sequences, starts, query_lengths)
    ):
      stop = start + query_length
      input_ids[rank, :query_length] = sequence[start:stop]
      positions[rank, :query_length] = np.arange(start, stop)
    lengths = np.asarray(
        [
            start + query_length if query_length else 0
            for start, query_length in zip(starts, query_lengths)
        ],
        np.int32,
    )
    active = np.asarray([int(length > 0) for length in query_lengths], np.int32)
    arrays = {
        "input_ids": input_ids.reshape(-1),
        "input_positions": positions.reshape(-1),
        "md_input_positions": positions.reshape(-1),
        "md_seq_lens": lengths,
        "md_query_start_loc": np.stack(
            (np.zeros_like(lengths), np.asarray(query_lengths, np.int32)), axis=1
        ).reshape(-1),
        "md_request_distribution": np.stack(
            (np.zeros_like(active), np.zeros_like(active), active), axis=1
        ).reshape(-1),
        "md_block_tables": np.where(
            active > 0, np.arange(data_size, dtype=np.int32), -1
        ),
    }
    base = Path(root) / f"p35_metadata_{index:04d}"
    np.savez(str(base) + ".npz", **arrays)
    (base.with_suffix(".json")).write_text(
        json.dumps({
            "schema_version": 1,
            "arm": arm,
            "meta": {
                "md_padded_num_reqs": data_size,
                "num_prompt_logprobs": {key: 1 for key in request_ids},
                "mesh": {
                    "shape": {"data": data_size, "model": 1},
                    "device_ids": list(range(data_size)),
                },
            },
        }),
        encoding="utf-8",
    )

  def test_rank_strided_groups_match_adapter_order(self):
    np.testing.assert_array_equal(
        envelope_probe.rank_strided_row_groups(12, 3),
        np.asarray(
            [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]],
            dtype=np.int64,
        ),
    )

  def test_selects_group_containing_current_first_red(self):
    a = np.zeros((12, 2), np.float32)
    c = a.copy()
    c[5, 1] = np.nextafter(c[5, 1], np.float32(np.inf))
    rows, coordinate = envelope_probe.select_reproducing_group(
        a, c, np.ones_like(a, dtype=np.bool_), data_size=3
    )
    np.testing.assert_array_equal(rows, np.asarray([1, 5, 9]))
    self.assertEqual(coordinate, (5, 1))

  def test_refuses_batch_that_does_not_reproduce_known_red(self):
    value = np.zeros((4, 2), np.float32)
    with self.assertRaisesRegex(
        envelope_probe.EnvelopeProbeError, "known A-C red"
    ):
      envelope_probe.select_reproducing_group(
          value, value.copy(), np.ones_like(value, dtype=np.bool_), data_size=2
      )

  def test_report_contains_three_pairs_and_effective_negative_control(self):
    a = np.zeros((2, 3), np.float32)
    b = a.copy()
    c = a.copy()
    c[0, 0] = np.nextafter(c[0, 0], np.float32(np.inf))
    mask = np.ones_like(a, dtype=np.bool_)
    report = envelope_probe.build_report(
        a=a,
        b=b,
        c=c,
        action_mask=mask,
        selected_row_indices=[0, 2],
        first_full_ac_mismatch=(0, 0),
        attestations={"weights_equal": True},
        metadata={"source": "test"},
    )
    self.assertEqual(report["schema_version"], 2)
    self.assertEqual(
        set(report["pairs"]),
        {envelope_probe.PAIR_AB, envelope_probe.PAIR_BC, envelope_probe.PAIR_AC},
    )
    self.assertEqual(report["negative_control"]["differing_elements"], 1)
    self.assertFalse(report["negative_control"]["masked_hashes_equal"])

  def test_exact_replay_report_contains_six_anchored_pairs(self):
    b = np.zeros((2, 3), np.float32)
    r0 = b.copy()
    r1 = b.copy()
    r2 = b.copy()
    c = b.copy()
    c[0, 0] = np.nextafter(c[0, 0], np.float32(np.inf))
    r3 = c.copy()
    stage = {
        "logps": {
            "valid": True,
            "differing_elements": 0,
            "total_elements": 6,
            "exact": True,
        }
    }
    report = envelope_probe.build_exact_replay_report(
        b=b,
        c=c,
        r0_live=r0,
        r1_mapped=r1,
        r2_adapter_direct=r2,
        r3_adapter_envelope=r3,
        action_mask=np.ones_like(b, dtype=np.bool_),
        stage_comparisons={"R0_live_vs_R1_mapped": stage},
        repeat_comparisons={
            "R0_live_repeat": stage,
            "R1_mapped_repeat": stage,
            "R2_adapter_direct_repeat": stage,
        },
        attestations={"repeat_exact": True},
        metadata={"source": "test"},
    )
    self.assertEqual(report["schema_version"], 1)
    self.assertEqual(
        set(report["pairs"]),
        {
            envelope_probe.REPLAY_PAIR_B0,
            envelope_probe.REPLAY_PAIR_01,
            envelope_probe.REPLAY_PAIR_12,
            envelope_probe.REPLAY_PAIR_23,
            envelope_probe.REPLAY_PAIR_3C,
            envelope_probe.REPLAY_PAIR_BC,
        },
    )
    self.assertEqual(report["negative_control"]["differing_elements"], 1)

  def test_report_refuses_evidence_collision(self):
    with tempfile.TemporaryDirectory() as temporary:
      path = f"{temporary}/report.json"
      envelope_probe.write_report({"value": 1}, path)
      with open(path, encoding="utf-8") as stream:
        self.assertEqual(json.load(stream), {"value": 1})
      with self.assertRaises(FileExistsError):
        envelope_probe.write_report({"value": 2}, path)

  def test_pre_replay_report_path_is_distinct_and_stable(self):
    self.assertEqual(
        envelope_probe.pre_replay_report_path("/tmp/p35.json"),
        Path("/tmp/p35.pre_replay.json"),
    )
    self.assertEqual(
        envelope_probe.pre_replay_report_path("/tmp/p35"),
        Path("/tmp/p35.pre_replay.json"),
    )

  def test_metadata_attestation_accepts_one_request_per_rank(self):
    with tempfile.TemporaryDirectory() as temporary:
      self._write_metadata_record(temporary, 0, "A")
      self._write_metadata_record(temporary, 1, "B")
      attestations, summary = envelope_probe.attest_metadata(
          directory=temporary,
          expected_b_sequences=((11, 12), (21, 22, 23)),
          expected_a_rows=2,
          data_size=2,
          local_m=256,
      )
      self.assertTrue(all(attestations.values()), attestations)
      self.assertEqual(summary["B_local_m"], 256)

  def test_metadata_attestation_accepts_multiple_fixed_m_chunks(self):
    sequences = (
        tuple(range(300)),
        tuple(range(1000, 1513)),
    )
    with tempfile.TemporaryDirectory() as temporary:
      self._write_metadata_record(temporary, 0, "A")
      self._write_metadata_record(
          temporary,
          1,
          "B",
          sequences=sequences,
          query_lengths=(256, 256),
      )
      self._write_metadata_record(
          temporary,
          2,
          "B",
          sequences=sequences,
          starts=(256, 256),
          query_lengths=(44, 256),
      )
      self._write_metadata_record(
          temporary,
          3,
          "B",
          sequences=sequences,
          starts=(300, 512),
          query_lengths=(0, 1),
      )
      attestations, summary = envelope_probe.attest_metadata(
          directory=temporary,
          expected_b_sequences=sequences,
          expected_a_rows=2,
          data_size=2,
          local_m=256,
      )
      self.assertTrue(all(attestations.values()), attestations)
      self.assertEqual(summary["B_records"], 3)
      self.assertEqual(summary["B_consumed_lengths"], [300, 513])
      self.assertEqual(
          summary["B_query_lengths_by_record"],
          [[256, 256], [44, 256], [0, 1]],
      )

  def test_metadata_attestation_rejects_incomplete_chunk_sequence(self):
    sequences = (tuple(range(300)), tuple(range(1000, 1300)))
    with tempfile.TemporaryDirectory() as temporary:
      self._write_metadata_record(temporary, 0, "A")
      self._write_metadata_record(
          temporary,
          1,
          "B",
          sequences=sequences,
          query_lengths=(256, 256),
      )
      attestations, summary = envelope_probe.attest_metadata(
          directory=temporary,
          expected_b_sequences=sequences,
          expected_a_rows=2,
          data_size=2,
          local_m=256,
      )
      self.assertFalse(attestations["grouped_B_observed"])
      self.assertFalse(attestations["metadata_B_matches_C"])
      self.assertEqual(summary["B_consumed_lengths"], [256, 256])

  def test_metadata_attestation_rejects_missing_rank_request(self):
    with tempfile.TemporaryDirectory() as temporary:
      self._write_metadata_record(temporary, 0, "A")
      self._write_metadata_record(temporary, 1, "B")
      path = Path(temporary) / "p35_metadata_0001.npz"
      with np.load(path) as original:
        arrays = {key: original[key].copy() for key in original.files}
      arrays["md_seq_lens"][1] = 0
      np.savez(path, **arrays)
      attestations, _ = envelope_probe.attest_metadata(
          directory=temporary,
          expected_b_sequences=((11, 12), (21, 22, 23)),
          expected_a_rows=2,
          data_size=2,
          local_m=256,
      )
      self.assertFalse(attestations["request_distribution_B_one_per_rank"])
      self.assertFalse(attestations["metadata_B_matches_C"])

  def test_metadata_attestation_rejects_bad_active_page_id(self):
    with tempfile.TemporaryDirectory() as temporary:
      self._write_metadata_record(temporary, 0, "A")
      self._write_metadata_record(temporary, 1, "B")
      path = Path(temporary) / "p35_metadata_0001.npz"
      with np.load(path) as original:
        arrays = {key: original[key].copy() for key in original.files}
      arrays["md_block_tables"][0] = -1
      np.savez(path, **arrays)
      attestations, _ = envelope_probe.attest_metadata(
          directory=temporary,
          expected_b_sequences=((11, 12), (21, 22, 23)),
          expected_a_rows=2,
          data_size=2,
          local_m=256,
      )
      self.assertFalse(attestations["block_tables_B_observed"])
      self.assertFalse(attestations["metadata_B_matches_C"])

  def test_metadata_attestation_rejects_position_channel_disagreement(self):
    with tempfile.TemporaryDirectory() as temporary:
      self._write_metadata_record(temporary, 0, "A")
      self._write_metadata_record(temporary, 1, "B")
      path = Path(temporary) / "p35_metadata_0001.npz"
      with np.load(path) as original:
        arrays = {key: original[key].copy() for key in original.files}
      arrays["md_input_positions"][0] = 9
      np.savez(path, **arrays)
      attestations, _ = envelope_probe.attest_metadata(
          directory=temporary,
          expected_b_sequences=((11, 12), (21, 22, 23)),
          expected_a_rows=2,
          data_size=2,
          local_m=256,
      )
      self.assertTrue(attestations["local_m256_B"])
      self.assertFalse(attestations["positions_equal"])
      self.assertFalse(attestations["metadata_B_matches_C"])

  def test_metadata_attestation_rejects_page_change_between_chunks(self):
    sequences = (tuple(range(300)), tuple(range(1000, 1300)))
    with tempfile.TemporaryDirectory() as temporary:
      self._write_metadata_record(temporary, 0, "A")
      self._write_metadata_record(
          temporary,
          1,
          "B",
          sequences=sequences,
          query_lengths=(256, 256),
      )
      self._write_metadata_record(
          temporary,
          2,
          "B",
          sequences=sequences,
          starts=(256, 256),
          query_lengths=(44, 44),
      )
      path = Path(temporary) / "p35_metadata_0002.npz"
      with np.load(path) as original:
        arrays = {key: original[key].copy() for key in original.files}
      arrays["md_block_tables"][0] = 99
      np.savez(path, **arrays)
      attestations, _ = envelope_probe.attest_metadata(
          directory=temporary,
          expected_b_sequences=sequences,
          expected_a_rows=2,
          data_size=2,
          local_m=256,
      )
      self.assertFalse(attestations["block_tables_B_observed"])
      self.assertFalse(attestations["metadata_B_matches_C"])


if __name__ == "__main__":
  absltest.main()

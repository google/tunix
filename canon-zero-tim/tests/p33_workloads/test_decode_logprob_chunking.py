"""Exact-image tests for canonical decode logprob chunking."""

from __future__ import annotations

import argparse
import ast
from contextlib import nullcontext
import importlib.util
import inspect
from pathlib import Path
import sys
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np


def _load_runner(path: Path):
  # The production overlay directory is prepended through PYTHONPATH. Mirror
  # that import contract when this test loads the runner by absolute filename,
  # so sibling shims such as p38_kv_fingerprint resolve identically.
  sys.path.insert(0, str(path.parent))
  spec = importlib.util.spec_from_file_location("canon_p33_test_runner", path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load canonical runner from {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


class DecodeLogprobChunkingTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.runner = _load_runner(Path(cls.overlay) / "tpu_runner_p21_l30.py")

  def _run(self, rows: int):
    calls = []

    def fake_compute(logits, token_ids, max_logprobs):
      calls.append((np.asarray(logits), np.asarray(token_ids), max_logprobs))
      ids = jnp.asarray(token_ids, dtype=jnp.int32)
      return self.runner.LogprobsTensors(
          logprob_token_ids=ids[:, None],
          logprobs=ids.astype(jnp.float32)[:, None],
          selected_token_ranks=ids,
      )

    logits = jnp.arange(rows * 3, dtype=jnp.float32).reshape(rows, 3)
    token_ids = jnp.arange(rows, dtype=jnp.int32)
    with mock.patch.object(
        self.runner, "compute_and_gather_logprobs", side_effect=fake_compute
    ):
      tensors, original_rows, chunk_count = (
          self.runner._canon_compute_decode_logprobs(
              logits, token_ids, max_logprobs=1, target_rows=256
          )
      )
    return tensors, original_rows, chunk_count, calls

  def test_two_full_chunks_reuse_m256_and_preserve_order(self):
    tensors, rows, chunks, calls = self._run(512)
    self.assertEqual((rows, chunks), (512, 2))
    self.assertEqual([call[0].shape for call in calls], [(256, 3), (256, 3)])
    np.testing.assert_array_equal(
        np.asarray(tensors.selected_token_ranks), np.arange(512)
    )

  def test_partial_tail_is_padded_then_removed(self):
    tensors, rows, chunks, calls = self._run(513)
    self.assertEqual((rows, chunks), (513, 3))
    self.assertEqual([call[0].shape for call in calls], [(256, 3)] * 3)
    self.assertEqual(calls[-1][1][0], 512)
    np.testing.assert_array_equal(calls[-1][1][1:], np.zeros(255, np.int32))
    np.testing.assert_array_equal(
        np.asarray(tensors.selected_token_ranks), np.arange(513)
    )

  def test_small_batch_still_runs_one_m256_chunk(self):
    tensors, rows, chunks, calls = self._run(17)
    self.assertEqual((rows, chunks), (17, 1))
    self.assertEqual(calls[0][0].shape, (256, 3))
    self.assertEqual(tensors.selected_token_ranks.shape, (17,))

  def test_p38_capture_serializes_typed_prng_key_without_changing_bits(self):
    key = jax.random.key(7)
    captured = self.runner._p38_capture_leaf(key)
    np.testing.assert_array_equal(
        captured["array"], np.asarray(jax.random.key_data(key))
    )
    self.assertEqual(captured["dtype"], "uint32")
    self.assertEqual(captured["shape"], (2,))
    self.assertEqual(captured["logical_dtype"], "key<fry>")
    self.assertEqual(
        captured["prng_key_impl"], str(jax.random.key_impl(key))
    )

  def test_p38_capture_preserves_legacy_array_capture(self):
    value = jnp.array([3, 5], dtype=jnp.uint32)
    captured = self.runner._p38_capture_leaf(value)
    np.testing.assert_array_equal(captured["array"], np.array([3, 5]))
    self.assertNotIn("logical_dtype", captured)

  def test_mismatched_rows_fail_closed(self):
    with self.assertRaisesRegex(ValueError, "row mismatch"):
      self.runner._canon_compute_decode_logprobs(
          jnp.zeros((512, 3)),
          jnp.zeros((511,), dtype=jnp.int32),
          max_logprobs=1,
          target_rows=256,
      )

  def test_empty_rows_fail_closed(self):
    with self.assertRaisesRegex(ValueError, "must be positive"):
      self.runner._canon_compute_decode_logprobs(
          jnp.zeros((0, 3)),
          jnp.zeros((0,), dtype=jnp.int32),
          max_logprobs=1,
          target_rows=256,
      )

  def _run_prompt(
      self, *, rows_per_dp: int, target_rows: int, dp_size: int = 2
  ):
    rows = dp_size * rows_per_dp
    calls = []

    def fake_sample(rng, mesh, logits, metadata):
      del rng, mesh
      calls.append({
          "logits": np.asarray(logits),
          "temperature": np.asarray(metadata.temperature),
          "top_k": np.asarray(metadata.top_k),
          "top_p": np.asarray(metadata.top_p),
      })
      return jnp.zeros((logits.shape[0],), dtype=jnp.int32), logits

    def fake_compute(logits, token_ids, max_logprobs):
      del logits, max_logprobs
      ids = jnp.asarray(token_ids, dtype=jnp.int32)
      return self.runner.LogprobsTensors(
          logprob_token_ids=ids[:, None],
          logprobs=ids.astype(jnp.float32)[:, None],
          selected_token_ranks=ids,
      )

    logits = jnp.arange(rows, dtype=jnp.float32)[:, None]
    temperatures = jnp.arange(rows, dtype=jnp.float32) + 100.0
    top_ks = jnp.arange(rows, dtype=jnp.int32)
    top_ps = jnp.arange(rows, dtype=jnp.float32) + 200.0
    target_ids = jnp.arange(rows, dtype=jnp.int32)
    with (
        mock.patch.object(self.runner, "sample", side_effect=fake_sample),
        mock.patch.object(
            self.runner,
            "compute_and_gather_logprobs",
            side_effect=fake_compute,
        ),
    ):
      tensors, observed_rows_per_dp, chunks = (
          self.runner._canon_compute_prompt_logprobs(
              None,
              None,
              logits,
              temperatures,
              top_ks,
              top_ps,
              target_ids,
              max_logprobs=1,
              dp_size=dp_size,
              target_rows=target_rows,
          )
      )
    return tensors, observed_rows_per_dp, chunks, calls

  def test_prompt_chunks_each_dp_rank_at_canonical_local_m(self):
    tensors, rows_per_dp, chunks, calls = self._run_prompt(
        rows_per_dp=6, target_rows=3
    )
    self.assertEqual((rows_per_dp, chunks), (6, 2))
    self.assertEqual([call["logits"].shape for call in calls], [(6, 1)] * 2)
    np.testing.assert_array_equal(
        calls[0]["logits"][:, 0], np.array([0, 1, 2, 6, 7, 8])
    )
    np.testing.assert_array_equal(
        calls[1]["logits"][:, 0], np.array([3, 4, 5, 9, 10, 11])
    )
    np.testing.assert_array_equal(
        np.asarray(tensors.selected_token_ranks), np.arange(12)
    )

  def test_prompt_single_chunk_preserves_dp_major_order(self):
    tensors, rows_per_dp, chunks, calls = self._run_prompt(
        rows_per_dp=3, target_rows=3
    )
    self.assertEqual((rows_per_dp, chunks), (3, 1))
    np.testing.assert_array_equal(calls[0]["logits"][:, 0], np.arange(6))
    np.testing.assert_array_equal(
        np.asarray(tensors.selected_token_ranks), np.arange(6)
    )

  def test_prompt_partial_tail_is_padded_per_dp_then_removed(self):
    tensors, rows_per_dp, chunks, calls = self._run_prompt(
        rows_per_dp=5, target_rows=3
    )
    self.assertEqual((rows_per_dp, chunks), (5, 2))
    np.testing.assert_array_equal(
        calls[1]["logits"][:, 0], np.array([3, 4, 0, 8, 9, 0])
    )
    np.testing.assert_array_equal(
        calls[1]["temperature"], np.array([103, 104, 1, 108, 109, 1])
    )
    np.testing.assert_array_equal(
        np.asarray(tensors.selected_token_ranks), np.arange(10)
    )

  def test_prompt_r13_shape_reuses_eight_local_m256_chunks(self):
    tensors, rows_per_dp, chunks, calls = self._run_prompt(
        rows_per_dp=2048, target_rows=256, dp_size=16
    )
    self.assertEqual((rows_per_dp, chunks), (2048, 8))
    self.assertEqual([call["logits"].shape for call in calls], [(4096, 1)] * 8)
    np.testing.assert_array_equal(
        np.asarray(tensors.selected_token_ranks), np.arange(32768)
    )

  def test_prompt_metadata_row_mismatch_fails_closed(self):
    with self.assertRaisesRegex(ValueError, "temperature rows"):
      self.runner._canon_compute_prompt_logprobs(
          None,
          None,
          jnp.zeros((8, 3)),
          jnp.ones((7,)),
          jnp.ones((8,), dtype=jnp.int32),
          jnp.ones((8,)),
          jnp.zeros((8,), dtype=jnp.int32),
          max_logprobs=1,
          dp_size=2,
          target_rows=2,
      )

  def _serving_mapping_case(self):
    input_batch = SimpleNamespace(
        num_reqs=3,
        req_ids=["request-a", "request-idle", "request-b"],
        req_id_to_index={"request-a": 0, "request-idle": 1, "request-b": 2},
        num_computed_tokens_cpu=np.array([2, 0, 4], dtype=np.int32),
        num_prompt_tokens=np.array([2, 0, 3], dtype=np.int32),
        num_tokens=np.array([3, 1, 5], dtype=np.int32),
        token_ids_cpu=np.array([
            [101, 102, 103, 0, 0, 0],
            [201, 0, 0, 0, 0, 0],
            [301, 302, 303, 304, 305, 0],
        ], dtype=np.int32),
    )
    runner = SimpleNamespace(
        input_batch=input_batch,
        dp_size=2,
        max_num_reqs=8,
        block_size=256,
        requests={
            "request-a": SimpleNamespace(block_ids=[[7]]),
            "request-idle": SimpleNamespace(block_ids=[[8]]),
            "request-b": SimpleNamespace(block_ids=[[9]]),
        },
    )
    scheduler = SimpleNamespace(
        num_scheduled_tokens={
            "request-a": 1,
            "request-idle": 0,
            "request-b": 1,
        },
        assigned_dp_rank={
            "request-a": 0,
            "request-idle": 1,
            "request-b": 1,
        },
    )
    req_ids_dp = {
        0: ["request-a"],
        1: ["request-idle", "request-b"],
    }
    selector = np.array([0, 2, 3], dtype=np.int32)
    positions = np.array([2, 0, 0, 4], dtype=np.int32)
    active = np.array([True, False, False, True])
    block_tables = np.zeros((8, 4), dtype=np.int32)
    block_tables[0, 0] = 7
    block_tables[5, 0] = 9
    seq_lens = np.array([3, 0, 0, 0, 0, 5, 0, 0], dtype=np.int32)
    query_start = np.array(
        [0, 1, 1, 1, 1, 0, 0, 1, 1, 1], dtype=np.int32
    )
    return (
        runner,
        scheduler,
        req_ids_dp,
        selector,
        positions,
        active,
        block_tables,
        seq_lens,
        query_start,
    )

  def test_serving_capture_keeps_physical_slot_while_filtering_idle_request(self):
    args = self._serving_mapping_case()
    result = self.runner._p38_serving_request_meta(
        *args[:3], args[3], 2, *args[4:], "continue_decode"
    )
    self.assertEqual(result["request_ids"], ["request-a", "request-b"])
    self.assertEqual(result["request_ids_by_dp"], {"0": ["request-a"], "1": ["request-b"]})
    request_b = result["requests"][1]
    self.assertEqual(request_b["local_scheduler_slot"], 1)
    self.assertEqual(request_b["global_row"], 3)
    self.assertEqual(request_b["attention_row"], 5)

  def test_standard_capture_maps_decode_after_packed_prefill_tokens(self):
    token_ids_cpu = np.zeros((2, 1801), dtype=np.int32)
    token_ids_cpu[0, :20] = 11
    token_ids_cpu[1, :] = 22
    input_batch = SimpleNamespace(
        num_reqs=2,
        req_ids=["prefill", "decode"],
        req_id_to_index={"prefill": 0, "decode": 1},
        num_computed_tokens_cpu=np.array([10, 1800], dtype=np.int32),
        num_prompt_tokens=np.array([20, 100], dtype=np.int32),
        num_tokens=np.array([20, 1801], dtype=np.int32),
        token_ids_cpu=token_ids_cpu,
    )
    runner = SimpleNamespace(
        input_batch=input_batch,
        dp_size=1,
        max_num_reqs=4,
        block_size=256,
        requests={
            "prefill": SimpleNamespace(block_ids=[[3]]),
            "decode": SimpleNamespace(block_ids=[list(range(7, 15))]),
        },
    )
    scheduler = SimpleNamespace(
        num_scheduled_tokens={"prefill": 3, "decode": 1},
        assigned_dp_rank={"prefill": 0, "decode": 0},
    )
    positions = np.array([10, 11, 12, 1800, 0, 0, 0, 0], dtype=np.int32)
    active = np.array([True, True, True, True, False, False, False, False])
    block_tables = np.zeros((4, 16), dtype=np.int32)
    block_tables[1, :8] = np.arange(7, 15)
    seq_lens = np.array([13, 1801, 0, 0], dtype=np.int32)
    query_start = np.array([0, 3, 4, 4, 4], dtype=np.int32)
    result = self.runner._p38_serving_request_meta(
        runner,
        scheduler,
        {0: ["prefill", "decode"]},
        None,
        8,
        positions,
        active,
        block_tables,
        seq_lens,
        query_start,
        "standard",
    )
    self.assertEqual(result["request_ids"], ["decode"])
    decode = result["requests"][0]
    self.assertEqual(decode["local_scheduler_slot"], 1)
    self.assertEqual(decode["packed_token_offset"], 3)
    self.assertEqual(decode["global_row"], 3)
    self.assertEqual(decode["attention_row"], 1)

  def test_serving_capture_rejects_empty_scheduled_selection(self):
    args = list(self._serving_mapping_case())
    args[1].num_scheduled_tokens = {
        "request-a": 0,
        "request-idle": 0,
        "request-b": 0,
    }
    with self.assertRaisesRegex(RuntimeError, "selected no scheduled requests"):
      self.runner._p38_serving_request_meta(
          *args[:3], args[3], 2, *args[4:], "continue_decode"
      )

  def test_serving_capture_rejects_selector_mapping_drift(self):
    args = list(self._serving_mapping_case())
    args[3] = np.array([1, 2, 3], dtype=np.int32)
    with self.assertRaisesRegex(RuntimeError, "selector mapping mismatch"):
      self.runner._p38_serving_request_meta(
          *args[:3], args[3], 2, *args[4:], "continue_decode"
      )

  def test_serving_capture_selects_each_prefix_stratum_once(self):
    self.assertEqual(
        [self.runner._p38_capture_stratum(value) for value in (
            1535, 1536, 1663, 1664, 1791, 1792, 1919, 1920, 2047,
            2048,
        )],
        [
            None,
            (0, 1536, 1664),
            (0, 1536, 1664),
            (1, 1664, 1792),
            (1, 1664, 1792),
            (2, 1792, 1920),
            (2, 1792, 1920),
            (3, 1920, 2048),
            (3, 1920, 2048),
            None,
        ],
    )

  def test_request_journal_uses_only_host_metadata(self):
    token_ids = np.arange(1601, dtype=np.int32)[None, :]
    runner = SimpleNamespace(
        block_size=256,
        input_batch=SimpleNamespace(
            num_tokens=np.array([1601], dtype=np.int32),
            num_prompt_tokens=np.array([100], dtype=np.int32),
            token_ids_cpu=token_ids,
        ),
        requests={
            "request-a": SimpleNamespace(
                block_ids=[list(range(7, 14))]
            )
        },
    )
    original = {
        "path": self.runner._P38_REQUEST_JOURNAL,
        "bands": set(self.runner._P38_REQUEST_JOURNALED_BANDS),
        "state": dict(self.runner._P38_REQUEST_JOURNAL_STATE),
        "ownership": dict(self.runner._P38_PAGE_OWNERSHIP),
    }
    try:
      with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "p38_request_journal.jsonl"
        self.runner._P38_REQUEST_JOURNAL = str(path)
        self.runner._P38_REQUEST_JOURNALED_BANDS.clear()
        self.runner._P38_REQUEST_JOURNAL_STATE.update(
            {"calls": 0, "records": 0}
        )
        self.runner._P38_PAGE_OWNERSHIP.clear()
        with mock.patch.object(
            self.runner.jax, "device_get",
            side_effect=AssertionError("journal touched a device buffer"),
        ):
          self.runner._p38_request_journal(
              runner,
              SimpleNamespace(
                  num_scheduled_tokens={"request-a": 1}
              ),
              {0: ["request-a"]},
              [{
                  "request_id": "request-a",
                  "request_index": 0,
                  "num_computed_tokens": 1600,
              }],
              "standard",
          )
        record = __import__("json").loads(path.read_text())
        self.assertEqual(record["physical_pages"], list(range(7, 14)))
        self.assertEqual(record["stratum"], [1536, 1664])
        self.assertEqual(record["scheduled_request_count"], 1)
        self.assertEqual(
            record["page_generations"][0]["observation_generation"], 0
        )
    finally:
      self.runner._P38_REQUEST_JOURNAL = original["path"]
      self.runner._P38_REQUEST_JOURNALED_BANDS.clear()
      self.runner._P38_REQUEST_JOURNALED_BANDS.update(original["bands"])
      self.runner._P38_REQUEST_JOURNAL_STATE.clear()
      self.runner._P38_REQUEST_JOURNAL_STATE.update(original["state"])
      self.runner._P38_PAGE_OWNERSHIP.clear()
      self.runner._P38_PAGE_OWNERSHIP.update(original["ownership"])

  def _fixed_m_incident_case(self):
    token_ids = np.arange(1601, dtype=np.int32)[None, :]
    runner = SimpleNamespace(
        dp_size=2,
        max_num_reqs=8,
        block_size=256,
        input_batch=SimpleNamespace(
            num_tokens=np.array([1601], dtype=np.int32),
            token_ids_cpu=token_ids,
        ),
        requests={
            "request-a": SimpleNamespace(
                block_ids=[list(range(7, 14))]
            )
        },
    )
    scheduler = SimpleNamespace(
        num_scheduled_tokens={"request-a": 1}
    )
    prefixes = [{
        "request_id": "request-a",
        "request_index": 0,
        "num_computed_tokens": 1600,
    }]
    attention = SimpleNamespace(
        block_tables=jnp.zeros((8, 8), dtype=jnp.int32),
        seq_lens=jnp.zeros((8,), dtype=jnp.int32),
        query_start_loc=jnp.zeros((10,), dtype=jnp.int32),
        request_distribution=jnp.zeros((3,), dtype=jnp.int32),
    )
    return runner, scheduler, prefixes, attention

  def test_incident_attests_fixed_m_and_embeds_single_active_tokens(self):
    runner, scheduler, prefixes, attention = self._fixed_m_incident_case()
    original = {
        "path": self.runner._P38_INCIDENT_LEDGER,
        "round": self.runner._P38_DIAGNOSTIC_ROUND_FILE,
        "state": dict(self.runner._P38_INCIDENT_STATE),
        "ownership": dict(self.runner._P38_PAGE_OWNERSHIP),
    }
    try:
      with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ledger = tmp_path / "p38_incident_ledger.jsonl"
        round_file = tmp_path / "p38_diagnostic_round"
        round_file.write_text("0\n", encoding="utf-8")
        self.runner._P38_INCIDENT_LEDGER = str(ledger)
        self.runner._P38_DIAGNOSTIC_ROUND_FILE = str(round_file)
        self.runner._P38_INCIDENT_STATE.update({"records": 0, "bytes": 0})
        self.runner._P38_PAGE_OWNERSHIP.clear()
        with mock.patch.object(
            self.runner.jax, "device_get",
            side_effect=AssertionError("incident attestation fetched a device buffer"),
        ):
          self.runner._p38_incident_ledger(
              runner,
              scheduler,
              {0: ["request-a"], 1: []},
              prefixes,
              "standard",
              1,
              jnp.zeros((256,), dtype=jnp.int32),
              jnp.zeros((256,), dtype=jnp.int32),
              attention,
              128,
          )
        record = __import__("json").loads(ledger.read_text())
        geometry = record["compile_geometry"]
        self.assertEqual(geometry["dp_size"], 2)
        self.assertEqual(geometry["padded_rows_per_dp"], 128)
        self.assertEqual(geometry["global_padded_rows"], 256)
        self.assertEqual(geometry["canonical_logprob_rows"], 256)
        self.assertEqual(geometry["input_ids"]["shape"], [256])
        self.assertEqual(
            record["requests"][0]["single_active_token_ids"],
            list(range(1601)),
        )
    finally:
      self.runner._P38_INCIDENT_LEDGER = original["path"]
      self.runner._P38_DIAGNOSTIC_ROUND_FILE = original["round"]
      self.runner._P38_INCIDENT_STATE.clear()
      self.runner._P38_INCIDENT_STATE.update(original["state"])
      self.runner._P38_PAGE_OWNERSHIP.clear()
      self.runner._P38_PAGE_OWNERSHIP.update(original["ownership"])

  def test_incident_rejects_shape_one_program_substitution(self):
    runner, scheduler, prefixes, attention = self._fixed_m_incident_case()
    original = {
        "path": self.runner._P38_INCIDENT_LEDGER,
        "round": self.runner._P38_DIAGNOSTIC_ROUND_FILE,
    }
    try:
      with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        round_file = tmp_path / "p38_diagnostic_round"
        round_file.write_text("0\n", encoding="utf-8")
        self.runner._P38_INCIDENT_LEDGER = str(
            tmp_path / "p38_incident_ledger.jsonl"
        )
        self.runner._P38_DIAGNOSTIC_ROUND_FILE = str(round_file)
        with self.assertRaisesRegex(RuntimeError, "fixed-M geometry drifted"):
          self.runner._p38_incident_ledger(
              runner,
              scheduler,
              {0: ["request-a"], 1: []},
              prefixes,
              "standard",
              1,
              jnp.zeros((1,), dtype=jnp.int32),
              jnp.zeros((1,), dtype=jnp.int32),
              attention,
              128,
          )
    finally:
      self.runner._P38_INCIDENT_LEDGER = original["path"]
      self.runner._P38_DIAGNOSTIC_ROUND_FILE = original["round"]

  def test_serving_capture_triggers_from_host_scheduler_prefix(self):
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            req_id_to_index={"prefill": 0, "decode": 1},
            num_computed_tokens_cpu=np.array([100, 1800], dtype=np.int32),
            num_prompt_tokens=np.array([200, 100], dtype=np.int32),
        )
    )
    scheduler = SimpleNamespace(
        num_scheduled_tokens={"prefill": 32, "decode": 1}
    )
    prefixes = self.runner._p38_scheduled_decode_prefixes(runner, scheduler)
    self.assertEqual(prefixes, [{
        "request_id": "decode",
        "request_index": 1,
        "num_computed_tokens": 1800,
    }])
    self.assertEqual(
        self.runner._p38_capture_stratum(
            prefixes[0]["num_computed_tokens"]
        ),
        (2, 1792, 1920),
    )

  def test_serving_capture_observation_is_bounded_by_prefix_band(self):
    state = self.runner._P38_SERVING_CAPTURE_OBSERVATION
    original = {
        "calls": state["calls"],
        "lines": state["lines"],
        "prefix_bands": set(state["prefix_bands"]),
    }
    try:
      state.update({"calls": 0, "lines": 0, "prefix_bands": set()})
      with mock.patch("builtins.print") as print_mock:
        self.runner._p38_observe_scheduled_prefixes([
            {"num_computed_tokens": 1600}
        ], "standard")
        self.runner._p38_observe_scheduled_prefixes([
            {"num_computed_tokens": 1610}
        ], "standard")
        self.runner._p38_observe_scheduled_prefixes([
            {"num_computed_tokens": 1800}
        ], "standard")
      self.assertEqual(print_mock.call_count, 2)
      self.assertEqual(state["calls"], 3)
      self.assertEqual(state["lines"], 2)
    finally:
      state.update(original)

  def _fake_execute_runner(self, *, enable_continue_decode: bool):
    input_ids = jnp.array([101, 102], dtype=jnp.int32)
    positions = jnp.array([10, 20], dtype=jnp.int32)
    attention = SimpleNamespace(
        input_positions=positions,
        block_tables=jnp.zeros((2, 4), dtype=jnp.int32),
        seq_lens=jnp.array([11, 21], dtype=jnp.int32),
        query_start_loc=jnp.array([0, 1, 2], dtype=jnp.int32),
        request_distribution=jnp.array([2, 2, 2], dtype=jnp.int32),
    )
    sampling = SimpleNamespace()
    prepared = (
        input_ids,
        positions,
        attention,
        sampling,
        jnp.array([0, 1], dtype=jnp.int32),
        None,
        None,
        2,
        {0: ["a", "b"]},
        2,
        None,
    )
    hidden = jnp.ones((2, 2), dtype=jnp.float32)
    runner = SimpleNamespace(
        persistent_batch_manager=SimpleNamespace(
            update_states=mock.Mock()
        ),
        get_mrope_input_positions_fn=None,
        input_batch=SimpleNamespace(
            num_reqs=2,
            request_distribution=np.array([2, 2, 2], dtype=np.int32),
            num_prompt_logprobs={},
        ),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        _pre_async_results=None,
        enable_continue_decode=enable_continue_decode,
        _execute_continue_decode=mock.Mock(return_value="continue-output"),
        _prepare_inputs=mock.Mock(return_value=prepared),
        is_multimodal_model=False,
        speculative_config=None,
        _get_input_ids_embeds=mock.Mock(return_value=(input_ids, None)),
        lora_utils=SimpleNamespace(
            extract_lora_metadata=mock.Mock(return_value=None)
        ),
        maybe_forbid_compile=nullcontext(),
        vllm_config=SimpleNamespace(),
        maybe_get_kv_connector_output=mock.Mock(
            return_value=nullcontext(None)
        ),
        state_leaves=(),
        mesh=None,
        kv_caches=(),
        model_fn=mock.Mock(return_value=((), hidden, None, None)),
        layer_name_to_kvcache_index={},
        is_first_rank=True,
        is_last_rank=True,
        is_pooling_model=False,
        _select_from_array_fn=mock.Mock(return_value=hidden),
        compute_logits_fn=mock.Mock(
            return_value=jnp.ones((2, 3), dtype=jnp.float32)
        ),
        execute_model_state=None,
    )
    scheduler = SimpleNamespace(
        total_num_scheduled_tokens=2,
        finished_req_ids=[],
    )
    return runner, scheduler

  def test_standard_execute_path_reaches_capture_when_continue_is_disabled(self):
    runner, scheduler = self._fake_execute_runner(
        enable_continue_decode=False
    )
    with (
        mock.patch.object(
            self.runner, "_p38_serving_begin", return_value=17
        ) as begin,
        mock.patch.object(
            self.runner, "set_forward_context", return_value=nullcontext()
        ),
    ):
      result = self.runner.TPUModelRunner._execute_model(runner, scheduler)
    self.assertIsNone(result)
    runner._execute_continue_decode.assert_not_called()
    self.assertEqual(begin.call_args.kwargs["program_path"], "standard")
    self.assertEqual(runner.execute_model_state.p38_serving_seq, 17)

  def test_continue_path_does_not_masquerade_as_standard_capture(self):
    runner, scheduler = self._fake_execute_runner(
        enable_continue_decode=True
    )
    with mock.patch.object(self.runner, "_p38_serving_begin") as begin:
      result = self.runner.TPUModelRunner._execute_model(runner, scheduler)
    self.assertEqual(result, "continue-output")
    runner._execute_continue_decode.assert_called_once_with(scheduler)
    begin.assert_not_called()

  def test_standard_capture_finish_writes_step_major_numeric_arrays(self):
    attention = SimpleNamespace(
        input_positions=jnp.array([10, 20], dtype=jnp.int32),
        seq_lens=jnp.array([11, 21], dtype=jnp.int32),
    )
    output = SimpleNamespace(
        sampled_token_ids=[[101], [202]],
        logprobs=SimpleNamespace(
            logprob_token_ids=np.array([[101], [202]], dtype=np.int32),
            logprobs=np.array([[-0.1], [-0.2]], dtype=np.float32),
            sampled_token_ranks=np.array([0, 0], dtype=np.int32),
        ),
    )
    with mock.patch.object(self.runner, "_p38_serving_dump") as dump:
      self.runner._p38_serving_finish_standard(3, attention, output)
    stage, seq, payload, meta = dump.call_args.args
    self.assertEqual((stage, seq), ("post", 3))
    self.assertEqual(payload["generated_tokens"].shape, (1, 2))
    self.assertEqual(payload["logprob_values"].shape, (1, 2, 1))
    self.assertEqual(meta["program_path"], "standard")

  def test_kv_observer_helpers_fail_closed_and_match_exact_prefix(self):
    finished = SimpleNamespace(
        sampling_params=SimpleNamespace(
            all_stop_token_ids={99}, max_tokens=4),
        output_token_ids=[7, 8, 9, 10],
    )
    self.assertTrue(self.runner._p38_kv_observer_request_finished(
        finished, [10], 12, 4096))
    finished.output_token_ids = []
    self.assertTrue(self.runner._p38_kv_observer_request_finished(
        finished, [99], 12, 4096))
    self.assertFalse(self.runner._p38_kv_observer_request_finished(
        finished, [10], 12, 4096))
    self.assertTrue(self.runner._p38_kv_observer_prefix_matches(
        [1, 2, 3, 4], [1, 2, 3]))
    self.assertFalse(self.runner._p38_kv_observer_prefix_matches(
        [1, 9, 3, 4], [1, 2, 3]))
    np.testing.assert_array_equal(
        self.runner._p38_kv_observer_padded_pages([7, 8], 4),
        np.array([7, 8, 8, 8], dtype=np.int32),
    )
    with self.assertRaisesRegex(RuntimeError, "page bound"):
      self.runner._p38_kv_observer_padded_pages([1, 2, 3], 2)

  def test_kv_observer_selects_one_single_active_candidate_per_round(self):
    original = {
        "dir": self.runner._P38_KV_OBSERVER_DIR,
        "candidates": dict(self.runner._P38_KV_OBSERVER_CANDIDATES),
        "rounds": set(self.runner._P38_KV_OBSERVER_CANDIDATE_ROUNDS),
    }
    try:
      self.runner._P38_KV_OBSERVER_DIR = "/tmp/p38-test"
      self.runner._P38_KV_OBSERVER_CANDIDATES.clear()
      self.runner._P38_KV_OBSERVER_CANDIDATE_ROUNDS.clear()
      base = {
          "scheduled_request_count": 1,
          "diagnostic_round": 2,
          "call_index": 4223,
          "requests": [{
              "request_id": "candidate-a",
              "num_computed_tokens": 2209,
              "dp_rank": 0,
              "physical_pages": [36, 9],
          }],
      }
      self.runner._p38_kv_observer_note_candidate(base)
      duplicate_round = {
          **base,
          "call_index": 4224,
          "requests": [{
              **base["requests"][0],
              "request_id": "candidate-b",
          }],
      }
      self.runner._p38_kv_observer_note_candidate(duplicate_round)
      self.assertEqual(
          list(self.runner._P38_KV_OBSERVER_CANDIDATES), ["candidate-a"])
    finally:
      self.runner._P38_KV_OBSERVER_DIR = original["dir"]
      self.runner._P38_KV_OBSERVER_CANDIDATES.clear()
      self.runner._P38_KV_OBSERVER_CANDIDATES.update(original["candidates"])
      self.runner._P38_KV_OBSERVER_CANDIDATE_ROUNDS.clear()
      self.runner._P38_KV_OBSERVER_CANDIDATE_ROUNDS.update(original["rounds"])

  def test_kv_observer_captures_finished_a_then_exact_prefix_b(self):
    original = {
        "dir": self.runner._P38_KV_OBSERVER_DIR,
        "candidates": dict(self.runner._P38_KV_OBSERVER_CANDIDATES),
        "a_records": list(self.runner._P38_KV_OBSERVER_A_RECORDS),
        "matched": set(self.runner._P38_KV_OBSERVER_MATCHED_A),
    }
    candidate = {
        "request_id": "decode-a",
        "diagnostic_round": 0,
        "tag_call_index": 17,
        "tag_prefix": 1600,
        "tag_dp_rank": 0,
        "tag_physical_pages": [4, 5],
    }
    try:
      self.runner._P38_KV_OBSERVER_DIR = "/tmp/p38-test"
      self.runner._P38_KV_OBSERVER_CANDIDATES.clear()
      self.runner._P38_KV_OBSERVER_CANDIDATES["decode-a"] = candidate
      self.runner._P38_KV_OBSERVER_A_RECORDS.clear()
      self.runner._P38_KV_OBSERVER_MATCHED_A.clear()
      decode_state = SimpleNamespace(
          num_computed_tokens=2,
          output_token_ids=[42],
          sampling_params=SimpleNamespace(
              all_stop_token_ids=set(), max_tokens=1),
          block_ids=[[4]],
      )
      decode_runner = SimpleNamespace(
          input_batch=SimpleNamespace(
              num_prompt_logprobs={},
              req_id_to_index={"decode-a": 0},
              token_ids_cpu=np.array([[11, 12, 13, 42]], dtype=np.int32),
          ),
          requests={"decode-a": decode_state},
          max_model_len=4096,
      )
      output = SimpleNamespace(
          req_ids=["decode-a"], sampled_token_ids=[[42]])
      scheduler = SimpleNamespace(
          num_scheduled_tokens={"decode-a": 1})
      a_record = {
          **candidate,
          "record_index": 0,
          "target_token_ids": np.array([11, 12, 13], dtype=np.int32),
      }
      with mock.patch.object(
          self.runner, "_p38_kv_observer_capture", return_value=a_record
      ) as capture:
        self.runner._p38_kv_observer_after_standard(
            decode_runner, scheduler, {0: ["decode-a"]}, set(), output)
      self.assertEqual(capture.call_args.args[1], "A")
      np.testing.assert_array_equal(
          capture.call_args.args[4], np.array([11, 12, 13]))
      self.assertEqual(len(self.runner._P38_KV_OBSERVER_A_RECORDS), 1)

      clean_state = SimpleNamespace(
          num_computed_tokens=0,
          num_tokens=4,
          block_ids=[[8]],
      )
      clean_runner = SimpleNamespace(
          input_batch=SimpleNamespace(
              num_prompt_logprobs={"clean-b": 1},
              req_id_to_index={"clean-b": 0},
              token_ids_cpu=np.array([[11, 12, 13, 42]], dtype=np.int32),
          ),
          requests={"clean-b": clean_state},
      )
      # Prompt-logprob-only work is captured immediately after model_fn, before
      # sample_tokens can consume its identity or return through a prompt-only
      # path.  B therefore has no sampled output object by construction.
      clean_scheduler = SimpleNamespace(
          num_scheduled_tokens={"clean-b": 4})
      with mock.patch.object(
          self.runner, "_p38_kv_observer_capture", return_value={}
      ) as capture:
        self.runner._p38_kv_observer_after_standard(
            clean_runner, clean_scheduler, {0: ["clean-b"]},
            {"clean-b"}, None)
      self.assertEqual(capture.call_args.args[1], "B")
      self.assertEqual(capture.call_args.args[6]["source_a_record_index"], 0)
      self.assertEqual(self.runner._P38_KV_OBSERVER_MATCHED_A, {0})
    finally:
      self.runner._P38_KV_OBSERVER_DIR = original["dir"]
      self.runner._P38_KV_OBSERVER_CANDIDATES.clear()
      self.runner._P38_KV_OBSERVER_CANDIDATES.update(original["candidates"])
      self.runner._P38_KV_OBSERVER_A_RECORDS.clear()
      self.runner._P38_KV_OBSERVER_A_RECORDS.extend(original["a_records"])
      self.runner._P38_KV_OBSERVER_MATCHED_A.clear()
      self.runner._P38_KV_OBSERVER_MATCHED_A.update(original["matched"])

  def test_kv_observer_clean_b_hook_is_outside_compile_forbidden_context(self):
    source = textwrap.dedent(inspect.getsource(
        self.runner.TPUModelRunner._execute_model))
    tree = ast.parse(source)
    parents = {}
    for parent in ast.walk(tree):
      for child in ast.iter_child_nodes(parent):
        parents[child] = parent
    hooks = []
    for node in ast.walk(tree):
      if not isinstance(node, ast.Call):
        continue
      if not isinstance(node.func, ast.Name):
        continue
      if node.func.id != "_p38_kv_observer_after_standard":
        continue
      if len(node.args) != 5 or not isinstance(node.args[-1], ast.Constant):
        continue
      if node.args[-1].value is None:
        hooks.append(node)
    self.assertEqual(len(hooks), 1)
    ancestor = parents.get(hooks[0])
    while ancestor is not None and not isinstance(
        ancestor, (ast.FunctionDef, ast.AsyncFunctionDef)):
      self.assertNotIsInstance(
          ancestor, (ast.With, ast.AsyncWith),
          "clean-B observer must compile outside maybe_forbid_compile")
      ancestor = parents.get(ancestor)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--overlay", required=True)
  args, remaining = parser.parse_known_args()
  DecodeLogprobChunkingTest.overlay = args.overlay
  unittest.main(argv=[__file__, *remaining])

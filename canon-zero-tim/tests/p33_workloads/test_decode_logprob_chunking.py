"""Exact-image tests for canonical decode logprob chunking."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

import jax.numpy as jnp
import numpy as np


def _load_runner(path: Path):
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


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--overlay", required=True)
  args, remaining = parser.parse_known_args()
  DecodeLogprobChunkingTest.overlay = args.overlay
  unittest.main(argv=[__file__, *remaining])

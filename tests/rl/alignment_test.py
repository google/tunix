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

"""Tests for the fail-closed zero-TIM alignment sidecar."""

import json
import os
import tempfile
from unittest import mock

from absl.testing import absltest
import flax
import jax
import jax.numpy as jnp
import numpy as np

from tunix.rl import alignment
from tunix.rl import canonical_forward


@flax.struct.dataclass(frozen=True)
class _Example:
  prompt_ids: object
  completion_ids: object
  completion_mask: object
  advantages: object
  is_update_step: object


def _real_rescore():
  pass


_real_rescore.is_real_rescore = True


class _CanonicalAdapter:
  implementation_id = "test.tpu_inference.qwen3"
  is_engine_module = True
  supports_value_and_grad = True

  def compute_per_token_logps(self, **kwargs):
    raise AssertionError(f"unexpected compute in host-only test: {kwargs}")


class AlignmentTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.env = mock.patch.dict(
        os.environ,
        {
            canonical_forward.ENV: "1",
            alignment.ALIGN_ENV: "1",
            alignment.GATE_ONLY_ENV: "1",
        },
        clear=False,
    )
    self.env.start()
    canonical_forward.register(_CanonicalAdapter())

  def tearDown(self):
    canonical_forward._clear_for_test()  # pylint: disable=protected-access
    self.env.stop()
    super().tearDown()

  def _wrapped(self, rows=2):
    ids = np.arange(rows * 3, dtype=np.int32).reshape(rows, 3)
    mask = np.ones_like(ids, dtype=np.bool_)
    values = np.arange(rows * 3, dtype=np.float32).reshape(rows, 3) / 8
    example = _Example(
        prompt_ids=jnp.asarray(ids + 100),
        completion_ids=jnp.asarray(ids),
        completion_mask=jnp.asarray(mask),
        advantages=jnp.ones((rows,), dtype=jnp.float32),
        is_update_step=None,
    )
    return alignment.wrap_train_example(
        example,
        s_decode=values,
        s_prefill=values.copy(),
        t_old=values.copy(),
        action_mask=mask,
        tokens=ids,
        policy_version=np.zeros((rows,), dtype=np.int32),
        temperature=0.7,
        top_k=0,
        top_p=1.0,
        s_prefill_source=_real_rescore,
    )

  def test_observed_example_proxies_prompt_ids_before_unwrap(self):
    wrapped = self._wrapped()
    np.testing.assert_array_equal(
        np.asarray(wrapped.prompt_ids),
        np.asarray(wrapped.train_example.prompt_ids),
    )

  def test_exact_gate_passes_and_writes_report(self):
    wrapped = self._wrapped()
    with tempfile.TemporaryDirectory() as tmpdir:
      report = os.path.join(tmpdir, "report.jsonl")
      with mock.patch.dict(
          os.environ,
          {
              alignment.ALIGN_ENV: "1",
              alignment.GATE_ONLY_ENV: "1",
              alignment.REPORT_ENV: report,
          },
          clear=False,
      ):
        result = alignment.check_batch(
            wrapped,
            t_current=wrapped.t_old.copy(),
            gradient_norm=np.asarray(2.0, np.float32),
            optimizer_skipped=np.asarray(1, np.int32),
            step=0,
        )
      self.assertEqual(result["verdict"], "PASS")
      self.assertTrue(result["exact"]["wr_all_exactly_1"])
      with open(report, encoding="utf-8") as report_file:
        self.assertEqual(json.loads(report_file.readline())["verdict"], "PASS")

  def test_one_ulp_drift_fails_closed(self):
    wrapped = self._wrapped()
    drift = wrapped.t_old.copy()
    drift[0, 0] = np.nextafter(drift[0, 0], np.float32(np.inf))
    with tempfile.TemporaryDirectory() as tmpdir:
      debug_path = os.path.join(tmpdir, "debug.npz")
      with mock.patch.dict(
          os.environ,
          {
              alignment.ALIGN_ENV: "1",
              alignment.GATE_ONLY_ENV: "1",
              alignment.REPORT_ENV: os.path.join(tmpdir, "report.jsonl"),
              alignment.DEBUG_ARRAYS_ENV: debug_path,
          },
          clear=False,
      ):
        with self.assertRaisesRegex(
            alignment.AlignmentGateError, "T_old_vs_T_current"
        ):
          alignment.check_batch(
              wrapped,
              t_current=drift,
              gradient_norm=np.asarray(2.0, np.float32),
              optimizer_skipped=np.asarray(1, np.int32),
              step=0,
          )
      with np.load(debug_path) as debug:
        np.testing.assert_array_equal(debug["t_old"], wrapped.t_old)
        np.testing.assert_array_equal(debug["t_current"], drift)
        np.testing.assert_array_equal(debug["action_mask"], wrapped.action_mask)
        np.testing.assert_array_equal(debug["tokens"], wrapped.tokens)

  def test_rejects_cached_alias_and_missing_skip_attestation(self):
    wrapped = self._wrapped()
    ids = np.asarray(wrapped.tokens)
    with self.assertRaisesRegex(alignment.AlignmentGateError, "does not declare"):
      alignment.wrap_train_example(
          wrapped.train_example,
          s_decode=wrapped.s_decode,
          s_prefill=wrapped.s_prefill,
          t_old=wrapped.t_old,
          action_mask=wrapped.action_mask,
          tokens=ids,
          policy_version=wrapped.policy_version,
          temperature=0.7,
          top_k=0,
          top_p=1.0,
          s_prefill_source=lambda: None,
      )
    with mock.patch.dict(
        os.environ,
        {alignment.GATE_ONLY_ENV: "1"},
        clear=False,
    ):
      with self.assertRaisesRegex(alignment.AlignmentGateError, "optimizer_skipped"):
        alignment.check_batch(
            wrapped,
            t_current=wrapped.t_old,
            gradient_norm=np.asarray(2.0, np.float32),
            optimizer_skipped=np.asarray(0, np.int32),
            step=0,
        )

  def test_update_canary_requires_optimizer_execution(self):
    wrapped = self._wrapped()
    with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
        os.environ,
        {
            alignment.GATE_ONLY_ENV: "0",
            alignment.UPDATE_CANARY_ENV: "1",
            alignment.REPORT_ENV: os.path.join(tmpdir, "report.jsonl"),
        },
        clear=False,
    ):
      result = alignment.check_batch(
          wrapped,
          t_current=wrapped.t_old,
          gradient_norm=np.asarray(2.0, np.float32),
          optimizer_skipped=np.asarray(0, np.int32),
          step=0,
      )
      self.assertEqual(result["execution_mode"], "update-canary")
      with self.assertRaisesRegex(
          alignment.AlignmentGateError, "optimizer attestation mismatch"
      ):
        alignment.check_batch(
            wrapped,
            t_current=wrapped.t_old,
            gradient_norm=np.asarray(2.0, np.float32),
            optimizer_skipped=np.asarray(1, np.int32),
            step=0,
        )

  def test_forward_only_requires_no_backward_or_optimizer(self):
    wrapped = self._wrapped()
    with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
        os.environ,
        {
            alignment.FORWARD_ONLY_ENV: "1",
            alignment.GATE_ONLY_ENV: "0",
            alignment.UPDATE_CANARY_ENV: "0",
            alignment.TRAIN_ENV: "0",
            alignment.REPORT_ENV: os.path.join(tmpdir, "report.jsonl"),
        },
        clear=False,
    ):
      result = alignment.check_batch(
          wrapped,
          t_current=wrapped.t_old,
          gradient_norm=np.asarray(0.0, np.float32),
          optimizer_skipped=np.asarray(1, np.int32),
          backward_executed=np.asarray(0, np.int32),
          step=0,
      )
      self.assertEqual(result["execution_mode"], "forward-only")
      self.assertFalse(result["gradient"]["executed"])
      with self.assertRaisesRegex(
          alignment.AlignmentGateError, "backward attestation mismatch"
      ):
        alignment.check_batch(
            wrapped,
            t_current=wrapped.t_old,
            gradient_norm=np.asarray(0.0, np.float32),
            optimizer_skipped=np.asarray(1, np.int32),
            backward_executed=np.asarray(1, np.int32),
            step=0,
        )

  def test_train_mode_allows_real_zero_signal_but_not_numerical_drift(self):
    wrapped = self._wrapped()
    with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
        os.environ,
        {
            alignment.GATE_ONLY_ENV: "0",
            alignment.UPDATE_CANARY_ENV: "0",
            alignment.TRAIN_ENV: "1",
            alignment.REPORT_ENV: os.path.join(tmpdir, "report.jsonl"),
        },
        clear=False,
    ):
      result = alignment.check_batch(
          wrapped,
          t_current=wrapped.t_old,
          gradient_norm=np.asarray(0.0, np.float32),
          optimizer_skipped=np.asarray(0, np.int32),
          step=0,
      )
      self.assertEqual(result["execution_mode"], "train")
      self.assertEqual(result["verdict"], "PASS")
      self.assertFalse(result["gradient"]["nonzero"])

      drift = wrapped.t_old.copy()
      drift[0, 0] = np.nextafter(drift[0, 0], np.float32(np.inf))
      with self.assertRaisesRegex(
          alignment.AlignmentGateError, "T_old_vs_T_current"
      ):
        alignment.check_batch(
            wrapped,
            t_current=drift,
            gradient_norm=np.asarray(0.0, np.float32),
            optimizer_skipped=np.asarray(0, np.int32),
            step=1,
        )

  def test_execution_mode_rejects_multiple_modes(self):
    with mock.patch.dict(
        os.environ,
        {
            alignment.GATE_ONLY_ENV: "1",
            alignment.UPDATE_CANARY_ENV: "0",
            alignment.TRAIN_ENV: "1",
        },
        clear=False,
    ):
      with self.assertRaisesRegex(
          alignment.AlignmentGateError, "exactly one alignment execution mode"
      ):
        alignment.execution_mode()

  def test_merge_and_slice_keep_sampling_metadata_batch_aligned(self):
    a = self._wrapped(rows=1)
    b = self._wrapped(rows=2)
    merged = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), a, b)
    self.assertEqual(merged.sampling_values.shape, (3, 3))
    sliced = jax.tree.map(
        lambda x: x[1:] if hasattr(x, "shape") and x.shape else x, merged
    )
    self.assertEqual(sliced.sampling_values.shape, (2, 3))
    core, sidecar = alignment.unwrap_train_example(sliced)
    self.assertIs(sidecar, sliced)
    self.assertEqual(core.completion_ids.shape, (2, 3))

  def test_signed_zero_reports_bitwise_mismatch_without_index_error(self):
    count, first = alignment._masked_bytes_differ(  # pylint: disable=protected-access
        np.asarray([[0.0]], np.float32),
        np.asarray([[-0.0]], np.float32),
        np.asarray([[True]]),
    )
    self.assertGreater(count, 0)
    self.assertEqual(first["masked_index"], 0)

  def _d2b_inputs(self):
    vocab = 8
    raw = np.arange(2 * vocab, dtype=np.float32).reshape(2, vocab) / 8
    processed = raw / np.float32(0.7)
    target_logps = np.asarray([-1.0, -2.0], np.float32)
    engine = {
        "generated_tokens": np.asarray([[1, 2], [3, 4]], np.int32),
        "decode_target_logps": target_logps.copy(),
        "prefill_target_logps": target_logps.copy(),
        "decode": {
            "raw_rows": raw.copy(),
            "processed_rows": processed.copy(),
            "dp_ranks": (0, 1),
        },
        "prefill": {
            "raw_rows": raw.copy(),
            "processed_rows": processed.copy(),
            "dp_ranks": (0, 1),
        },
        "sampling": {"temperature": 0.7, "top_k": 0, "top_p": 1.0},
    }
    diagnostics = {
        "raw_rows": np.stack((np.zeros_like(raw), raw), axis=1),
        "processed_rows": np.stack(
            (np.zeros_like(processed), processed), axis=1
        ),
        "target_ids": np.asarray([[0, 2], [0, 4]], np.int32),
    }
    logps = np.stack((np.zeros_like(target_logps), target_logps), axis=1)
    return engine, logps, diagnostics

  def test_p32_d2b_full_distribution_positive_and_negative(self):
    engine, logps, diagnostics = self._d2b_inputs()
    with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
        os.environ,
        {
            "CANON_P32_D2B_FULL_DISTRIBUTION": "1",
            alignment.D2B_REPORT_ENV: os.path.join(tmpdir, "positive.json"),
            alignment.D2B_ARRAYS_ENV: os.path.join(tmpdir, "positive.npz"),
        },
        clear=False,
    ):
      report = alignment.check_p32_d2b_full_distribution(
          engine_result=engine,
          t_old_logps=logps,
          t_old_diagnostics=diagnostics,
          t_current_logps=logps.copy(),
          t_current_diagnostics=jax.tree.map(np.copy, diagnostics),
      )
      self.assertEqual(report["verdict"], "P32_DP2TP2_D2B_PASS")
      self.assertEqual(report["comparison_count"], 15)
      self.assertTrue(os.path.isfile(report["artifact_npz"]))

    engine, logps, diagnostics = self._d2b_inputs()
    bad = jax.tree.map(np.copy, diagnostics)
    bad["processed_rows"][0, 1, 0] = np.nextafter(
        bad["processed_rows"][0, 1, 0], np.float32(np.inf)
    )
    with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
        os.environ,
        {
            "CANON_P32_D2B_FULL_DISTRIBUTION": "1",
            alignment.D2B_REPORT_ENV: os.path.join(tmpdir, "negative.json"),
            alignment.D2B_ARRAYS_ENV: os.path.join(tmpdir, "negative.npz"),
        },
        clear=False,
    ):
      with self.assertRaisesRegex(
          alignment.AlignmentGateError, "T_current.processed"
      ):
        alignment.check_p32_d2b_full_distribution(
            engine_result=engine,
            t_old_logps=logps,
            t_old_diagnostics=diagnostics,
            t_current_logps=logps.copy(),
            t_current_diagnostics=bad,
        )


if __name__ == "__main__":
  absltest.main()

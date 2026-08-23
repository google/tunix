from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
CLASSIFIER = (
    ROOT
    / "canon-zero-tim/tasks/v1-phase3-prefix-cache/scripts"
    / "classify_p3_boundary.py"
)
SPEC = importlib.util.spec_from_file_location("classify_p3_boundary", CLASSIFIER)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _report(
    *, apc: bool, cached: int, differing_prefix: int | None = None,
    dirty: bool = False,
):
  cases = []
  for prefix in MODULE.PREFIX_LENGTHS:
    differing = 4 if prefix == differing_prefix else 0
    cases.append({
        "prefix_length": prefix,
        "target_length": 16,
        "target_tokens": list(range(16)),
        "target_sha256": "t" * 64,
        "input_sha256": "i" * 64,
        "prime_num_cached_tokens": 0,
        "a_num_cached_tokens": cached,
        "b_num_cached_tokens": 0,
        "b_reset_prefix_cache": True,
        "finite": True,
        "differing_bytes": differing,
        "differing_elements": 1 if differing else 0,
        "first_mismatch": (
            {"target_index": 0, "a": -1.0, "b": -2.0}
            if differing
            else None
        ),
        "a_sha256": "a" * 64,
        "b_sha256": "b" * 64,
    })
  return {
      "schema": "phase3-apc-boundary-probe-v2",
      "apc_enabled": apc,
      "topology": "DP1xTP4",
      "canonical_m": 256,
      "prefix_lengths": MODULE.PREFIX_LENGTHS,
      "cases": cases,
      "backward": 0,
      "optimizer_commits": 0,
      "token_source": "fixed-arange-prefix-v1:a-decode-completion-v1",
      "a_request_contract": {
          "max_tokens": 16,
          "sampled_logprobs": 1,
          "prompt_logprobs": None,
          "skip_reading_prefix_cache": False,
          "ignore_eos": True,
      },
      "weight_attestation": {"equal": True, "mismatch_indices": []},
      "dirty_page_control": (
          {
              "enabled": True,
              "target_prefix_length": MODULE.PREFIX_LENGTHS[0],
              "page": {
                  "layer_index": 0,
                  "physical_block_id": 7,
                  "logical_token_extent": 256,
                  "page_shape": [256, 2, 2, 128],
                  "page_dtype": "bfloat16",
                  "mutation": "fill-zero",
                  "before_sha256": "c" * 64,
                  "after_sha256": "d" * 64,
                  "differing_bytes": 128,
                  "differing_elements": 64,
              },
          }
          if dirty
          else {"enabled": False, "target_prefix_length": None, "page": None}
      ),
  }


class BoundaryClassifierTest(unittest.TestCase):

  def classify(self, report, expect_apc, expect_dirty_page=False):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "report.json"
      path.write_text(json.dumps(report), encoding="utf-8")
      return MODULE.classify(path, expect_apc, expect_dirty_page)

  def test_off_control_is_green(self):
    result = self.classify(_report(apc=False, cached=0), False)
    self.assertEqual(result["status"], "BOUNDARY_CONTROL_GREEN")

  def test_on_exact_is_preserved_as_deep_negative(self):
    result = self.classify(_report(apc=True, cached=1536), True)
    self.assertEqual(result["status"], "BOUNDARY_DEEP_EXACT_NO_RED")
    self.assertEqual(result["preceding_green_prefix"], 2049)

  def test_on_red_names_first_interval(self):
    result = self.classify(
        _report(apc=True, cached=1536, differing_prefix=1686), True
    )
    self.assertEqual(result["status"], "BOUNDARY_REPRODUCED_RED")
    self.assertEqual(result["preceding_green_prefix"], 1685)
    self.assertEqual(result["first_red_prefix"], 1686)

  def test_dirty_page_control_requires_the_authoritative_gate_to_fire(self):
    report = _report(
        apc=True,
        cached=1536,
        differing_prefix=MODULE.PREFIX_LENGTHS[0],
        dirty=True,
    )
    result = self.classify(report, True, expect_dirty_page=True)
    self.assertEqual(result["status"], "DIRTY_PAGE_GATE_CAUGHT")

  def test_dirty_page_control_rejects_an_ineffective_gate(self):
    report = _report(apc=True, cached=1536, dirty=True)
    with self.assertRaisesRegex(
        MODULE.ClassificationError, "gate did not catch"
    ):
      self.classify(report, True, expect_dirty_page=True)

  def test_on_without_hit_is_rejected(self):
    with self.assertRaisesRegex(MODULE.ClassificationError, "did not hit"):
      self.classify(_report(apc=True, cached=0), True)

  def test_b_cache_consumption_is_rejected(self):
    report = _report(apc=True, cached=1536)
    report["cases"][0]["b_num_cached_tokens"] = 256
    with self.assertRaisesRegex(MODULE.ClassificationError, "B consumed"):
      self.classify(report, True)

  def test_prompt_logprob_a_is_rejected(self):
    report = _report(apc=True, cached=1536)
    report["a_request_contract"]["prompt_logprobs"] = 0
    report["a_request_contract"]["skip_reading_prefix_cache"] = True
    with self.assertRaisesRegex(
        MODULE.ClassificationError, "not a cache-readable production decode"
    ):
      self.classify(report, True)


if __name__ == "__main__":
  unittest.main()

#!/usr/bin/env python3
"""Tests for the P38s12b single-variable manifest gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import yaml


_SCRIPT = Path(__file__).with_name("check_p38_intent_diff.py")
_SPEC = importlib.util.spec_from_file_location("check_p38_intent_diff", _SCRIPT)
checker = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = checker
_SPEC.loader.exec_module(checker)

_ROOT = Path(__file__).resolve().parents[4]
_RENDERER_PATH = _ROOT / "canon-zero-tim/cluster/render_p38_serving_jobsets.py"
_RENDERER_SPEC = importlib.util.spec_from_file_location(
    "p38_renderer_for_intent_test", _RENDERER_PATH
)
renderer = importlib.util.module_from_spec(_RENDERER_SPEC)
assert _RENDERER_SPEC.loader is not None
sys.modules[_RENDERER_SPEC.name] = renderer
_RENDERER_SPEC.loader.exec_module(renderer)


def _documents():
  base = renderer.p33.load_base(
      _ROOT / "canon-zero-tim/cluster/jobset-64chip.yaml"
  )
  spec, unified = renderer._SPECS[0]
  baseline = renderer.render_jobset(
      base, spec, "1" * 40, "p38s12b", unified=unified,
      max_concurrency=256,
  )
  candidate = renderer.render_jobset(
      base, spec, "1" * 40, "p38s12b", unified=unified,
      max_concurrency=32,
  )
  return baseline, candidate


class CheckP38IntentDiffTest(unittest.TestCase):

  def test_accepts_only_concurrency_and_attestation_label(self):
    baseline, candidate = _documents()
    self.assertEqual(checker.classify(baseline, candidate)["verdict"], "PASS")

  def test_rejects_an_unrelated_environment_change(self):
    baseline, candidate = _documents()
    checker._env_entry(candidate, "CANON_KV_UNIFIED")["value"] = "1"
    self.assertEqual(checker.classify(baseline, candidate)["verdict"], "FAIL")

  def test_rejects_a_wrong_candidate_concurrency(self):
    baseline, candidate = _documents()
    entry = checker._env_entry(candidate, "CANON_RUN_CMD")
    entry["value"] = entry["value"].replace(
        "--max_concurrency=32", "--max_concurrency=64"
    )
    self.assertEqual(checker.classify(baseline, candidate)["verdict"], "FAIL")


if __name__ == "__main__":
  unittest.main()

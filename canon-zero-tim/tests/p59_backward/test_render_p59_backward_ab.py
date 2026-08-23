#!/usr/bin/env python3
"""CPU gates for the immutable P59 A/B and profile carrier."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
CLUSTER = ROOT / "canon-zero-tim" / "cluster"
sys.path.insert(0, str(CLUSTER))
SPEC = importlib.util.spec_from_file_location(
    "render_p59_backward_ab", CLUSTER / "render_p59_backward_ab.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RenderP59BackwardABTest(unittest.TestCase):

  def test_three_immutable_kinds(self):
    source = "a" * 40
    base = CLUSTER / "jobset-64chip.yaml"
    documents = {
        kind: MODULE.render(
            base_path=base, source_commit=source, run_id="p59a", kind=kind
        )
        for kind in ("control", "candidate", "profile")
    }
    self.assertEqual(len({d["metadata"]["name"] for d in documents.values()}), 3)
    for kind, document in documents.items():
      env = MODULE.p33._env_values(document)
      self.assertIn("--max_steps=3", env["CANON_P59_INNER_RUN_CMD"])
      self.assertEqual(env["CANON_P33_RUN_STAGE"], "three-update")
      self.assertEqual(document["spec"]["failurePolicy"]["maxRestarts"], 0)
      self.assertEqual(
          env["CANON_P59_RANK_PARALLEL_BACKWARD"],
          "0" if kind == "control" else "1",
      )
      self.assertEqual(env["CANON_P59_REQUIRE_XPROF"], "1" if kind == "profile" else "0")
      self.assertEqual(env["CANON_XPROF_LABELS"], "1" if kind == "profile" else "0")
      self.assertTrue(env["CANON_P59_GCS_PREFIX"].endswith("/attempt-0"))

  def test_refuses_output_overwrite(self):
    with tempfile.TemporaryDirectory() as directory:
      output = Path(directory) / "jobset.yaml"
      output.write_text("owned\n", encoding="utf-8")
      with self.assertRaises(FileExistsError):
        if output.exists():
          raise FileExistsError(
              f"refusing to overwrite rendered JobSet: {output}"
          )


if __name__ == "__main__":
  unittest.main()

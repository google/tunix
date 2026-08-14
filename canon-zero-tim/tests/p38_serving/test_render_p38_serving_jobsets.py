#!/usr/bin/env python3
"""Tests for bounded P38 serving-capture JobSets."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import yaml
from tunix.rl import dp_workloads


_ROOT = Path(__file__).resolve().parents[3]
_RENDERER = _ROOT / "canon-zero-tim/cluster/render_p38_serving_jobsets.py"
_BASE = _ROOT / "canon-zero-tim/cluster/jobset-64chip.yaml"
_SOURCE = "1" * 40
_RUN_ID = "capture-a"
_SPEC = importlib.util.spec_from_file_location(
    "render_p38_serving_jobsets", _RENDERER
)
assert _SPEC and _SPEC.loader
renderer = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = renderer
_SPEC.loader.exec_module(renderer)


def _env(document):
  return renderer.p33._env_values(document)


class RenderP38ServingJobsetsTest(unittest.TestCase):

  def test_renders_stock_and_unified_attempt_zero_jobsets(self):
    with tempfile.TemporaryDirectory() as tmp:
      paths = renderer.render_all(
          base_path=_BASE,
          output_dir=Path(tmp),
          source_commit=_SOURCE,
          run_id=_RUN_ID,
      )
      self.assertEqual(len(paths), 2)
      documents = [yaml.safe_load(path.read_text()) for path in paths]
      self.assertEqual(len({doc["metadata"]["name"] for doc in documents}), 2)
      self.assertEqual(
          {doc["metadata"]["labels"]["canon.zero-tim/kv-unified"] for doc in documents},
          {"0", "1"},
      )
      for document in documents:
        env = _env(document)
        self.assertEqual(env["CANON_P33_RUN_STAGE"], "backward-no-commit")
        self.assertEqual(env["CANON_P33_NO_COMMIT"], "1")
        self.assertEqual(env["CANON_P38_PRECHECK_ONLY"], "1")
        self.assertEqual(env["CANON_P38_SERVING_CAPTURE_MAX_CALLS"], "4")
        self.assertEqual(env["CANON_P38_SERVING_CAPTURE_EXPECTED_RECORDS"], "4")
        self.assertEqual(env["CANON_P38_SERVING_CAPTURE_MIN_PREFIX"], "1536")
        self.assertEqual(
            env["CANON_P38_SERVING_CAPTURE_PREFIX_BOUNDS"],
            "1536,1664,1792,1920,2048",
        )
        self.assertEqual(
            env["CANON_P38_MISMATCH_CAPSULE_MAX_ROWS"], "256"
        )
        self.assertEqual(env["CANON_P38_CONTROLLED_EXIT"], "1")
        self.assertEqual(env["CANON_P38_DIAGNOSTIC_ROUNDS"], "3")
        self.assertEqual(
            env["CANON_P38_DIAGNOSTIC_ROUND_FILE"],
            f"{env['CANON_STATE']}/p38_diagnostic_round",
        )
        self.assertEqual(env["CANON_P38_MIN_ACTION_KV"], "1686")
        self.assertEqual(
            env["CANON_P38_SERVING_CAPTURE_FREE_SPACE_MULTIPLIER"], "5"
        )
        self.assertEqual(
            env["CANON_P38_SERVING_CAPTURE_EXPECTED_PATH"], "standard"
        )
        self.assertEqual(
            env["CANON_P38_REQUEST_JOURNAL"],
            env["CANON_P38_SERVING_CAPTURE_DIR"]
            + "/p38_request_journal.jsonl",
        )
        self.assertEqual(
            env["CANON_P38_INCIDENT_LEDGER"],
            env["CANON_P38_SERVING_CAPTURE_DIR"]
            + "/p38_incident_ledger.jsonl",
        )
        self.assertEqual(env["CANON_P38_INCIDENT_MIN_PREFIX"], "1400")
        self.assertEqual(env["CANON_P38_INCIDENT_MAX_PREFIX"], "3072")
        self.assertEqual(env["CANON_P38_INCIDENT_MAX_BYTES"], "134217728")
        self.assertEqual(
            env["CANON_P38_LIVE_SNAPSHOT_INTERVAL_SECONDS"], "30"
        )
        self.assertEqual(
            env["CANON_P38_LIVE_SNAPSHOT_STOP_FILE"],
            env["CANON_STATE"] + "/p38_live.stop",
        )
        self.assertEqual(
            env["CANON_P38_LIVE_SNAPSHOT_WORKER_LOG"],
            env["CANON_STATE"] + "/p38_live_worker.log",
        )
        self.assertEqual(
            env["CANON_P38_GCS_PREFIX"],
            "gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/"
            + document["metadata"]["name"]
            + "/attempt-0",
        )
        self.assertTrue(env["CANON_P38_MISMATCH_CAPSULE"].endswith(".npz"))
        self.assertEqual(document["spec"]["failurePolicy"]["maxRestarts"], 0)
        self.assertIn("--batch_size=32", env["CANON_RUN_CMD"])
        self.assertIn("--mini_batch_size=4", env["CANON_RUN_CMD"])
        self.assertIn("--num_generations=8", env["CANON_RUN_CMD"])
        self.assertIn("--mesh_dp=16", env["CANON_RUN_CMD"])
        self.assertIn("--max_concurrency=256", env["CANON_RUN_CMD"])
        self.assertNotIn("--mini_batch_size=32", env["CANON_RUN_CMD"])
        self.assertIn("--max_response_length=2048", env["CANON_RUN_CMD"])
        self.assertEqual(renderer._DIAGNOSTIC_UNITS, 8)
        self.assertEqual(renderer._COVERED_PROMPTS, 32)

  def test_renders_preregistered_concurrency_32_arm(self):
    with tempfile.TemporaryDirectory() as tmp:
      paths = renderer.render_all(
          base_path=_BASE,
          output_dir=Path(tmp),
          source_commit=_SOURCE,
          run_id="p38s12b",
          stock_only=True,
          max_concurrency=32,
      )
      document = yaml.safe_load(paths[0].read_text())
      env = _env(document)
      self.assertIn("--max_concurrency=32", env["CANON_RUN_CMD"])
      self.assertNotIn("--max_concurrency=256", env["CANON_RUN_CMD"])
      self.assertEqual(
          document["metadata"]["labels"]["canon.zero-tim/max-concurrency"],
          "32",
      )
      dp_workloads.validate_frozenlake_max_concurrency(
          dp_workloads.get_workload("frozenlake"), 32, env
      )

  def test_stock_only_omits_the_already_falsified_unified_arm(self):
    with tempfile.TemporaryDirectory() as tmp:
      paths = renderer.render_all(
          base_path=_BASE,
          output_dir=Path(tmp),
          source_commit=_SOURCE,
          run_id=_RUN_ID,
          stock_only=True,
      )
      self.assertEqual(
          [path.name for path in paths],
          ["jobset-p38-serving-stock.yaml"],
      )
      document = yaml.safe_load(paths[0].read_text())
      self.assertEqual(
          document["metadata"]["labels"]["canon.zero-tim/kv-unified"],
          "0",
      )

  def test_rejects_capture_contract_drift(self):
    base = renderer.p33.load_base(_BASE)
    spec, unified = renderer._SPECS[0]
    document = renderer.render_jobset(
        base, spec, _SOURCE, _RUN_ID, unified=unified
    )
    main = renderer._main_container(document)
    entry = next(
        item for item in main["env"]
        if item["name"] == "CANON_P38_SERVING_CAPTURE_MAX_CALLS"
    )
    entry["value"] = "2"
    with self.assertRaisesRegex(ValueError, "environment drifted"):
      renderer.validate_capture_jobset(document, unified=unified)

  def test_rejects_missing_mismatch_capsule_path(self):
    base = renderer.p33.load_base(_BASE)
    spec, unified = renderer._SPECS[0]
    document = renderer.render_jobset(
        base, spec, _SOURCE, _RUN_ID, unified=unified
    )
    main = renderer._main_container(document)
    entry = next(
        item for item in main["env"]
        if item["name"] == "CANON_P38_MISMATCH_CAPSULE"
    )
    entry["value"] = ""
    with self.assertRaisesRegex(ValueError, "requires a mismatch capsule"):
      renderer.validate_capture_jobset(document, unified=unified)

  def test_rejects_gcs_prefix_drift(self):
    base = renderer.p33.load_base(_BASE)
    spec, unified = renderer._SPECS[0]
    document = renderer.render_jobset(
        base, spec, _SOURCE, _RUN_ID, unified=unified
    )
    main = renderer._main_container(document)
    entry = next(
        item for item in main["env"]
        if item["name"] == "CANON_P38_GCS_PREFIX"
    )
    entry["value"] = "gs://wrong-bucket/p38"
    with self.assertRaisesRegex(ValueError, "GCS evidence prefix drifted"):
      renderer.validate_capture_jobset(document, unified=unified)

  def test_rejects_non_divisible_diagnostic_batch(self):
    base = renderer.p33.load_base(_BASE)
    spec, unified = renderer._SPECS[0]
    document = renderer.render_jobset(
        base, spec, _SOURCE, _RUN_ID, unified=unified
    )
    main = renderer._main_container(document)
    entry = next(
        item for item in main["env"] if item["name"] == "CANON_RUN_CMD"
    )
    entry["value"] = entry["value"].replace(
        "--mini_batch_size=4", "--mini_batch_size=5"
    )
    with self.assertRaisesRegex(ValueError, "batch geometry changed"):
      renderer.validate_capture_jobset(document, unified=unified)

  def test_quotes_numeric_looking_source_commit(self):
    source = "022893e2" + "0" * 32
    with tempfile.TemporaryDirectory() as tmp:
      paths = renderer.render_all(
          base_path=_BASE,
          output_dir=Path(tmp),
          source_commit=source,
          run_id=_RUN_ID,
      )
      for path in paths:
        self.assertEqual(_env(yaml.safe_load(path.read_text()))["CANON_EXPECT_COMMIT"], source)

  def test_refuses_overwrite(self):
    with tempfile.TemporaryDirectory() as tmp:
      output = Path(tmp)
      renderer.render_all(
          base_path=_BASE,
          output_dir=output,
          source_commit=_SOURCE,
          run_id=_RUN_ID,
      )
      with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
        renderer.render_all(
            base_path=_BASE,
            output_dir=output,
            source_commit=_SOURCE,
            run_id=_RUN_ID,
        )


if __name__ == "__main__":
  unittest.main()

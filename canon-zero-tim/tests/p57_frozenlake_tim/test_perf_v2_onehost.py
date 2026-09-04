#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import collections
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "canon-zero-tim/tasks/p57-frozenlake-tim-causal-study"
RUNNER = TASK / "scripts/run_perf_v2_onehost.sh"
CLASSIFIER = TASK / "scripts/classify_perf_v2_onehost.py"
CENSUS = TASK / "scripts/census_perf_v2_onehost.py"


def _load_classifier():
  spec = importlib.util.spec_from_file_location("p57_perf_v2_classifier", CLASSIFIER)
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


def _load_census():
  spec = importlib.util.spec_from_file_location("p57_perf_v2_census", CENSUS)
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


class PerfV2OnehostContractTest(unittest.TestCase):

  def test_runner_is_three_update_target_step_two_and_fail_closed(self):
    text = RUNNER.read_text(encoding="utf-8")
    self.assertIn("updates=3 concurrency=2", text)
    self.assertIn("-e CANON_PERF_TRACE_EXPORT_STEP=2", text)
    self.assertIn("--num_batches=3", text)
    self.assertIn("--mesh_dp=1 --mesh_tp=4", text)
    self.assertIn("--max_concurrency=2", text)
    self.assertIn("--beta=0", text)
    self.assertIn("CANON_P57_TITO_ONEHOST_NEUTRALITY", text)
    self.assertIn("CANON_P57_TOKEN_CONTINUITY_DEBUG=record-full", text)
    self.assertIn("--max_steps=3", text)
    self.assertIn("--reference-inference disabled", text)
    self.assertIn("refusing existing evidence root", text)
    self.assertIn("refusing a host with a non-system privileged container", text)
    self.assertIn("vbarcontrolagent|google-runtime-monitor", text)
    self.assertIn("{{.HostConfig.Privileged}}", text)
    self.assertIn("rpa_kernel_p66.py", text)
    self.assertIn('-e CANON_P29_LOG_DIR="$root/logs"', text)
    for redundant in (
        "CANON_" + "P27_TRAJECTORY_MICRO",
        "CANON_" + "P28_G3_ONLY",
        "CANON_" + "P28_G4_ONLY",
        "CANON_" + "P28_G5C_SHARED_LOGSOFTMAX",
        "CANON_" + "RPA_VJP=1",
    ):
      self.assertNotIn(f"-e {redundant}", text)
    recipe = (ROOT / "examples/frozenlake/train_frozenlake_qwen3.py").read_text(
        encoding="utf-8"
    )
    self.assertIn(
        "if CANON_P32_WORKLOAD or args.mesh_dp is not None", recipe
    )
    self.assertIn(
        "dp_workloads.configure_model_sharding_for_mesh(\n"
        "    config, SHARED_MESH_AXIS_NAMES\n"
        ")",
        recipe,
    )
    learner = (
        ROOT / "tunix/rl/agentic/agentic_rl_learner.py"
    ).read_text(encoding="utf-8")
    self.assertIn(
        'else workload.name\n        if p33_workload\n        else "legacy-segmented"',
        learner,
    )
    self.assertIn("P57_PERF_V2_ONEHOST_PASS", text)
    launch = text.split("sudo docker run --rm --privileged", 1)[1]
    launch = launch.split("docker_wait_pid=$!", 1)[0]
    self.assertNotIn(" | ", launch)
    pair = (
        ROOT
        / "canon-zero-tim/tasks/multiturn-tito-cross-workload/scripts/"
        "run_tito_onehost_neutrality_pair.sh"
    ).read_text(encoding="utf-8")
    self.assertIn('while [ "$idle_samples" -lt 12 ]', pair)
    self.assertIn("sleep 10", pair)
    self.assertNotIn("docker kill", pair)

  def test_classifier_accepts_complete_gate_and_rejects_tracer_red(self):
    classifier = _load_classifier()
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      raw = root / "raw.log"
      align = root / "alignment.jsonl"
      updates = root / "updates.jsonl"
      semantic = root / "semantic.json"
      raw.write_text(
          "\n".join(
              [
                  "Global step 0 completed in 1.0 seconds.",
                  "[CANON_FROZENLAKE_P27] update_step_committed",
                  "Global step 1 completed in 1.0 seconds.",
                  "[CANON_FROZENLAKE_P27] update_step_committed",
                  "Global step 2 completed in 1.0 seconds.",
                  "[CANON_FROZENLAKE_P27] update_step_committed",
                  "[V1.PERFETTO] captured training_step=2 timelines=7",
              ]
          )
          + "\n",
          encoding="utf-8",
      )
      boundary = {"differing_bytes": 0, "finite": True}
      align_row = {
          "verdict": "PASS",
          "blocking_reds": [],
          "N_action": 8,
          "boundaries": {
              "S_decode_vs_S_prefill": boundary,
              "S_prefill_vs_T_old": boundary,
              "T_old_vs_T_current": boundary,
          },
      }
      align.write_text(
          "".join(json.dumps(align_row) + "\n" for _ in range(12)),
          encoding="utf-8",
      )
      update_rows = []
      for index in range(3):
        update_rows.append({
            "verdict": "PASS",
            "commits": 1,
            "train_steps_before": index,
            "train_steps_after": index + 1,
            "gradient_finite": True,
            "commit_gradient_norm": 1.0,
            "optimizer_transaction_valid": True,
            "commit_evidence": {
                "gradient_nonzero_elements": 1,
                "parameter_changed_elements": 1,
                "parameter_delta_finite": True,
            },
        })
      updates.write_text(
          "".join(json.dumps(row) + "\n" for row in update_rows),
          encoding="utf-8",
      )
      semantic.write_text(
          json.dumps({
              "verdict": "PASS",
              "files": 1,
              "reference_inference_contract": "disabled",
              "event_counts": {
                  name: 2
                  for name in (
                      "data_loading",
                      "rollout",
                      "advantage_computation",
                      "peft_train",
                      "weight_sync",
                  )
              },
          }),
          encoding="utf-8",
      )
      result = classifier.classify(
          raw_path=raw,
          alignment_path=align,
          update_path=updates,
          semantic_path=semantic,
          docker_exit=0,
      )
      self.assertEqual(result["verdict"], "PASS")

      tito_state = root / "tito-off"
      tito_state.mkdir()
      raw.write_text(
          "neutrality_arm=tito-off\n" + raw.read_text(encoding="utf-8"),
          encoding="utf-8",
      )
      off = classifier.classify(
          raw_path=raw,
          alignment_path=align,
          update_path=updates,
          semantic_path=semantic,
          docker_exit=0,
          neutrality_arm="tito-off",
          tito_state=tito_state,
      )
      self.assertEqual(off["verdict"], "PASS")

      raw.write_text(
          raw.read_text(encoding="utf-8").replace(
              "neutrality_arm=tito-off", "neutrality_arm=tito-on"
          )
          + "[P57.TITO.FIRST_UPDATE_TOKEN_GATE] PASS step=0 rows=8 "
          "different_rows=0\n",
          encoding="utf-8",
      )
      tito_state = root / "tito-on"
      witness = tito_state / "p57_tito_witness"
      witness.mkdir(parents=True)
      writer = witness / "single-writer.json"
      writer.write_text(json.dumps({
          "schema": "canon.p57-tito-single-writer.v1",
          "status": "PASS",
          "workload": "p45",
          "dp": 1,
          "tp": 4,
          "neutrality_arm": "on",
      }))
      writer.chmod(0o600)
      summary = witness / "full-record-summary.json"
      summary.write_text(json.dumps({
          "schema": "canon.p57-tito-full-record.v1",
          "workload": "p45",
          "dp": 1,
          "tp": 4,
          "expected_updates": 3,
          "optimizer_commits": 3,
          "global_updates": 3,
          "checkpoint_writes": 0,
      }))
      sidecars = witness / "update-sidecars"
      sidecars.mkdir()
      for step in range(3):
        (sidecars / f"step-{step:06d}.npz").write_bytes(b"npz")
      on = classifier.classify(
          raw_path=raw,
          alignment_path=align,
          update_path=updates,
          semantic_path=semantic,
          docker_exit=0,
          neutrality_arm="tito-on",
          tito_state=tito_state,
      )
      self.assertEqual(on["verdict"], "PASS")

      raw.write_text(
          raw.read_text(encoding="utf-8")
          + "Purging uncompleted span 'rollout'\n",
          encoding="utf-8",
      )
      red = classifier.classify(
          raw_path=raw,
          alignment_path=align,
          update_path=updates,
          semantic_path=semantic,
          docker_exit=0,
      )
      self.assertEqual(red["verdict"], "FAIL")
      self.assertTrue(any(reason.startswith("tracer_red:") for reason in red["reasons"]))

  def test_beta_zero_semantic_contract_rejects_reference_inference(self):
    census = _load_census()
    counts = collections.Counter({
        "data_loading": 1,
        "rollout": 1,
        "advantage_computation": 1,
        "peft_train": 1,
        "weight_sync": 1,
    })
    self.assertEqual(
        census._event_contract_reasons(counts, "disabled"), []  # pylint: disable=protected-access
    )
    counts["reference_inference"] = 1
    self.assertEqual(
        census._event_contract_reasons(counts, "disabled"),  # pylint: disable=protected-access
        ["unexpected_event=reference_inference"],
    )


if __name__ == "__main__":
  unittest.main()

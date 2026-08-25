"""Contracts for the Attempt-7 P62 backward numerical carrier."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import yaml

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = (
    _REPO
    / "canon-zero-tim/tasks/v1-phase4-three-full-recipes/scripts"
    / "render_attempt7_numeric_debug.py"
)
_SPEC = importlib.util.spec_from_file_location("p62_renderer", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
renderer = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(renderer)
_CLASSIFIER_SCRIPT = (
    _REPO
    / "canon-zero-tim/tasks/v1-phase4-three-full-recipes/scripts"
    / "classify_attempt7_numeric_debug.py"
)
_CLASSIFIER_SPEC = importlib.util.spec_from_file_location(
    "p62_classifier", _CLASSIFIER_SCRIPT
)
assert (
    _CLASSIFIER_SPEC is not None and _CLASSIFIER_SPEC.loader is not None
)
classifier = importlib.util.module_from_spec(_CLASSIFIER_SPEC)
_CLASSIFIER_SPEC.loader.exec_module(classifier)


def _env(document: dict) -> dict[str, str]:
  pod = document["spec"]["replicatedJobs"][0]["template"]["spec"][
      "template"
  ]["spec"]
  main = next(
      item for item in pod["containers"] if item["name"] == "jax-tpu"
  )
  return {
      item["name"]: item["value"]
      for item in main["env"]
      if "value" in item
  }


def _p62_fixture(*, red_stage: str | None = None, overflow=False) -> str:
  pre = {
      "verdict": "PASS",
      "N_action": 128,
      "context": {"mesh": "16,4", "run_stage": "backward-no-commit"},
      "boundaries": {
          name: {
              "valid": True,
              "finite": True,
              "differing_elements": 0,
              "differing_bytes": 0,
          }
          for name in ("S_decode_vs_S_prefill", "S_prefill_vs_T_old")
      },
  }
  loss = {
      "schema": "canon-p62-loss-scale-v1",
      "stage": "loss_scale",
      "dp": 16,
      "tp": 4,
      "global_trajectories": 256,
      "local_trajectories": 16,
      "gradient_groups": 16,
      "global_M": 4096,
      "local_M": 256,
      "expected_accumulator_denominator": 16,
      "expected_streamed_multiplier": 0.0625,
      "loss_denominator": 256.0,
      "loss_scale": 0.00390625,
  }

  def tree(stage, group, *, final=False):
    nonfinite = stage == red_stage
    record = {
        "schema": "canon-p62-tree-numeric-v1",
        "stage": stage,
        "group": group,
        "groups": 16,
        "all_finite": not nonfinite,
        "naive_norm_finite": not (
            nonfinite or (overflow and stage == "engine_vjp")
        ),
        "first_nonfinite": (
            {"leaf": 0, "path": "['x']"} if nonfinite else None
        ),
        "first_nonfinite_rank": (
            {"rank": 3, "leaf": 0, "path": "['x']"}
            if nonfinite else None
        ),
        "max_abs": 2.0,
        "stable_norm": 3.0,
    }
    if final:
      record["accumulator_denominator"] = 16.0
    return record

  lines = [
      "[" "CANON" "_ALIGN_PRE_JSON] " + json.dumps(pre),
      "[P62.NUMERIC] profile_resolved workload=gsm8k dp=16 tp=4 "
      "stage=backward-no-commit optimizer_commits=0",
      "[P62.NUMERIC] admission workload=gsm8k dp=16 tp=4 "
      "global_trajectories=256 local_trajectories=16 "
      "global_M=4096 local_M=256 optimizer_commits=0",
      "[P62.NUMERIC] " + json.dumps(loss),
      "[P62.NUMERIC] "
      + json.dumps(tree("loss_cotangent", -1)),
  ]
  for group in (0, 15):
    for stage in (
        "engine_vjp",
        "trainer_rank_local",
        "fixed_dp_reduced",
        "scaled_microgradient",
    ):
      lines.append("[P62.NUMERIC] " + json.dumps(
          tree(stage, group)
      ))
      if stage == red_stage:
        return "\n".join(lines) + "\n"
  lines.extend(
      "[P33.DP16] reverse_group_done "
      f"group={group}/16 rank_contributions=16 "
      "pullback_invocations=1 replicas_exact=1 gradient_nonzero=1"
      for group in range(1, 17)
  )
  lines.append(
      "[P62.NUMERIC] "
      + json.dumps(tree("final_accumulator", 15, final=True))
  )
  lines.append(
      "[P62.NUMERIC] discard_complete optimizer_commits=0 "
      "microsteps=16 denominator=16.0"
  )
  return "\n".join(lines) + "\n"


class NumericDebugTest(unittest.TestCase):

  def _render(self, root: Path) -> Path:
    return renderer.render(
        source_commit="a" * 40,
        run_id="p62a",
        output_dir=root,
        base_path=_REPO / "canon-zero-tim/cluster/jobset-64chip.yaml",
    )

  def test_renderer_is_strict_zero_commit_and_default_full_identity_off(self):
    with tempfile.TemporaryDirectory() as tmp:
      output = self._render(Path(tmp) / "rendered")
      document = yaml.safe_load(output.read_text(encoding="utf-8"))
      values = _env(document)
      expected = {
          "CANON_P33_RUN_STAGE": "backward-no-commit",
          "CANON_P33_NO_COMMIT": "1",
          "CANON_P59_RANK_PARALLEL_BACKWARD": "1",
          "CANON_P38_FIXED_LM_HEAD": "1",
          "CANON_P62_BACKWARD_NUMERIC_DEBUG": "1",
          "CANON_V1_HP_FULL": "0",
          "CANON_GSM8K_ALIGNMENT_WARN_ONLY": "0",
      }
      self.assertEqual(
          {name: values.get(name) for name in expected}, expected
      )
      self.assertEqual(
          document["metadata"]["labels"][
              "canon.zero-tim/optimizer-commits"
          ],
          "0",
      )
      state = values["CANON_STATE"]
      self.assertEqual(values["CANON_RUN_LOG"], f"{state}/run.log")
      receipt = json.loads(
          (output.parent / "render-receipt.json").read_text(encoding="utf-8")
      )
      self.assertEqual(receipt["run_log"], f"{state}/run.log")
      self.assertEqual(
          receipt["classification"],
          f"{state}/p62_backward_numeric.classification.json",
      )

  def test_cluster_postflight_persists_and_classifies_full_p62_log(self):
    runner = (
        _REPO / "canon-zero-tim/cluster/steps/90_run.sh"
    ).read_text(encoding="utf-8")
    self.assertIn(
        "cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k-p62-debug.env)",
        runner,
    )
    self.assertIn(
        'p62_classification="$CANON_STATE/'
        'p62_backward_numeric.classification.json"',
        runner,
    )
    self.assertIn('p62_log_sha="$(sha256sum "$LOG"', runner)
    self.assertIn('printf \'%s\\n\' "$p62_profile_receipt" > "$LOG"', runner)
    self.assertIn('run_tee_args=(-a "$LOG")', runner)
    self.assertIn("[P62.NUMERIC.POSTFLIGHT] PASS", runner)
    self.assertIn("P62 full-log classification failed", runner)

  def test_profile_resolves_only_from_exact_renderer_tuple(self):
    with tempfile.TemporaryDirectory() as tmp:
      output = self._render(Path(tmp) / "rendered")
      values = _env(yaml.safe_load(output.read_text(encoding="utf-8")))
      profile = _REPO / "canon-zero-tim" / values["CANON_PROFILE_FILE"]
      command = (
          f"source {profile}; "
          "test \"$CANON_P62_BACKWARD_NUMERIC_DEBUG\" = 1; "
          "test \"$CANON_P59_RANK_PARALLEL_BACKWARD\" = 1; "
          "test \"$CANON_P38_FIXED_LM_HEAD\" = 1; "
          "test \"$CANON_V1_HP_FULL\" = 0; "
          "test \"${CANON_P63_OVERFLOW_SAFE_CLIP:-0}\" = 0; "
          "test \"$CANON_VLLM_ENABLE_PREFIX_CACHING\" = 0"
      )
      result = subprocess.run(
          ["bash", "-c", command],
          cwd=_REPO,
          env={**os.environ, **values},
          text=True,
          capture_output=True,
          check=False,
      )
      self.assertEqual(result.returncode, 0, result.stderr)

      wrong = dict(values)
      wrong["CANON_P62_BACKWARD_NUMERIC_DEBUG"] = "0"
      negative = subprocess.run(
          ["bash", "-c", f"source {profile}"],
          cwd=_REPO,
          env={**os.environ, **wrong},
          text=True,
          capture_output=True,
          check=False,
      )
      self.assertNotEqual(negative.returncode, 0)
      self.assertIn("P62 requires", negative.stderr)

  def test_classifier_accepts_only_complete_finite_no_commit_carrier(self):
    result = classifier.classify(_p62_fixture())
    self.assertEqual(result["verdict"], "ALL_BOUNDARIES_FINITE_NO_COMMIT")
    self.assertEqual(result["reverse_groups"], list(range(1, 17)))
    self.assertEqual(result["discard_count"], 1)

  def test_classifier_localizes_nonfinite_and_finite_naive_overflow(self):
    nonfinite = classifier.classify(
        _p62_fixture(red_stage="trainer_rank_local")
    )
    self.assertEqual(nonfinite["verdict"], "ROOT_LOCALIZED_NONFINITE")
    self.assertEqual(nonfinite["first_red"]["stage"], "trainer_rank_local")
    self.assertEqual(nonfinite["first_red"]["first_nonfinite_rank"]["rank"], 3)

    overflow = classifier.classify(_p62_fixture(overflow=True))
    self.assertEqual(overflow["verdict"], "FINITE_NAIVE_L2_OVERFLOW")
    self.assertEqual(overflow["first_red"]["stage"], "engine_vjp")

  def test_classifier_rejects_or_downgrades_truncated_evidence(self):
    complete = _p62_fixture(overflow=True)
    partial = "\n".join(complete.splitlines()[:6]) + "\n"
    result = classifier.classify(partial)
    self.assertIn(
        result["verdict"], {"FATAL_CONTRACT", "INCONCLUSIVE_INCOMPLETE"}
    )
    self.assertNotEqual(result["verdict"], "FINITE_NAIVE_L2_OVERFLOW")

    with tempfile.TemporaryDirectory() as tmp:
      raw = Path(tmp) / "partial.log"
      output = Path(tmp) / "classification.json"
      raw.write_text(partial, encoding="utf-8")
      process = subprocess.run(
          [
              "python3",
              str(_CLASSIFIER_SCRIPT),
              str(raw),
              "--output",
              str(output),
          ],
          cwd=_REPO,
          text=True,
          capture_output=True,
          check=False,
      )
      self.assertNotEqual(process.returncode, 0)
      self.assertTrue(output.is_file())

  def test_classifier_rejects_unknown_or_wrong_text_markers(self):
    unknown = _p62_fixture() + "[P62.NUMERIC] future_marker foo=1\n"
    self.assertEqual(
        classifier.classify(unknown)["verdict"], "FATAL_CONTRACT"
    )
    missing_profile = _p62_fixture().replace(
        "[P62.NUMERIC] profile_resolved workload=gsm8k dp=16 tp=4 "
        "stage=backward-no-commit optimizer_commits=0\n",
        "",
    )
    self.assertEqual(
        classifier.classify(missing_profile)["verdict"], "FATAL_CONTRACT"
    )
    wrong_profile = _p62_fixture().replace(
        "stage=backward-no-commit optimizer_commits=0",
        "stage=full optimizer_commits=0",
        1,
    )
    self.assertEqual(
        classifier.classify(wrong_profile)["verdict"], "FATAL_CONTRACT"
    )

  def test_classifier_requires_all_completion_seams(self):
    for missing in (
        '"stage": "fixed_dp_reduced", "group": 15',
        '"stage": "scaled_microgradient", "group": 15',
        '"stage": "final_accumulator", "group": 15',
        "reverse_group_done group=16/16",
        "discard_complete optimizer_commits=0",
    ):
      lines = _p62_fixture().splitlines()
      partial = "\n".join(line for line in lines if missing not in line) + "\n"
      self.assertEqual(
          classifier.classify(partial)["verdict"],
          "INCONCLUSIVE_INCOMPLETE",
          missing,
      )

  def test_classifier_rejects_alignment_scale_and_optimizer_violations(self):
    alignment = _p62_fixture().replace(
        '"differing_bytes": 0', '"differing_bytes": 4', 1
    )
    self.assertEqual(
        classifier.classify(alignment)["verdict"], "FATAL_CONTRACT"
    )
    explicit_fail = _p62_fixture() + "[CANON_ALIGN] verdict=FAIL\n"
    self.assertEqual(
        classifier.classify(explicit_fail)["verdict"], "FATAL_CONTRACT"
    )
    scale = _p62_fixture().replace(
        '"loss_denominator": 256.0', '"loss_denominator": 128.0'
    )
    self.assertEqual(
        classifier.classify(scale)["verdict"], "FATAL_CONTRACT"
    )
    committed = _p62_fixture() + "optimizer_commits=1\n"
    self.assertEqual(
        classifier.classify(committed)["verdict"], "FATAL_CONTRACT"
    )

if __name__ == "__main__":
  unittest.main()

"""Tests for the strict P33 queue renderer."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import shlex
import sys
import tempfile
import unittest

import yaml

from tunix.rl import dp_workloads


_ROOT = Path(__file__).resolve().parents[3]
_RENDERER_PATH = _ROOT / "canon-zero-tim/cluster/render_p33_jobsets.py"
_BASE_PATH = _ROOT / "canon-zero-tim/cluster/jobset-64chip.yaml"
_SOURCE_COMMIT = "1" * 40
_RUN_ID = "queue-a"

_MODULE_SPEC = importlib.util.spec_from_file_location(
    "render_p33_jobsets", _RENDERER_PATH
)
assert _MODULE_SPEC is not None and _MODULE_SPEC.loader is not None
renderer = importlib.util.module_from_spec(_MODULE_SPEC)
sys.modules[_MODULE_SPEC.name] = renderer
_MODULE_SPEC.loader.exec_module(renderer)


def _main_env(document):
  pod = document["spec"]["replicatedJobs"][0]["template"]["spec"][
      "template"
  ]["spec"]
  container = next(
      item for item in pod["containers"] if item["name"] == "jax-tpu"
  )
  return {
      entry["name"]: entry.get("value") for entry in container["env"]
  }


class RenderP33JobSetsTest(unittest.TestCase):

  def _render(self, output_dir: Path):
    return renderer.render_all(
        base_path=_BASE_PATH,
        output_dir=output_dir,
        source_commit=_SOURCE_COMMIT,
        run_id=_RUN_ID,
    )

  def test_renders_six_isolated_strict_jobsets(self):
    with tempfile.TemporaryDirectory() as tmp:
      outputs = self._render(Path(tmp))
      self.assertEqual(len(outputs), 6)
      documents = [yaml.safe_load(path.read_text()) for path in outputs]
      names = [document["metadata"]["name"] for document in documents]
      self.assertEqual(len(set(names)), 6)
      scratches = []
      states = []
      wandb_names = []
      workloads = set()
      for document in documents:
        labels = document["metadata"]["labels"]
        workloads.add(labels["canon.zero-tim/workload"])
        expected_max_restarts = (
            3
            if labels["canon.zero-tim/workload"] == "gsm8k"
            and labels["canon.zero-tim/stage"] == "full"
            else 0
        )
        self.assertEqual(
            document["spec"]["failurePolicy"]["maxRestarts"],
            expected_max_restarts,
        )
        self.assertEqual(
            document["spec"]["replicatedJobs"][0]["template"]["spec"][
                "backoffLimit"
            ],
            0,
        )
        self.assertEqual(
            document["spec"]["replicatedJobs"][1]["template"]["spec"][
                "backoffLimit"
            ],
            0,
        )
        env = _main_env(document)
        self.assertEqual(env["CANON_EXPECT_COMMIT"], _SOURCE_COMMIT)
        self.assertEqual(env["CANON_MODE"], "run")
        self.assertEqual(env["CANON_P32_TRAIN_ADMITTED"], "1")
        self.assertEqual(env["CANON_P32_DP_REDUCTION_ADMITTED"], "1")
        self.assertEqual(env["CANON_P33_WORKLOAD_LAUNCH_ADMITTED"], "1")
        self.assertEqual(env["CANON_P33_SHARED_MESH"], "16,4")
        self.assertEqual(env["CANON_OPT_STATE_RESIDENT"], "1")
        self.assertEqual(env["CANON_P30_OPT_STATE_OFFLOAD"], "0")
        self.assertEqual(env["CANON_PRE_ALIGN_GATE"], "1")
        self.assertTrue(env["CANON_PRE_ALIGN_REPORT"].endswith("pre_alignment.jsonl"))
        if (
            labels["canon.zero-tim/workload"] == "frozenlake"
            and labels["canon.zero-tim/stage"] == "backward-no-commit"
        ):
          self.assertTrue(
              env["CANON_P38_MISMATCH_CAPSULE"].endswith(
                  "p38_frozenlake_mismatch_capsule.npz"
              )
          )
        else:
          self.assertEqual(env["CANON_P38_MISMATCH_CAPSULE"], "")
        self.assertEqual(env["CANON_P38_MISMATCH_CAPSULE_MAX_ROWS"], "2")
        self.assertEqual(env["CANON_P32_EXPECT_MODEL_MESH_IDS"], "")
        self.assertNotIn("CANON_P32_RC_STAGE", env)
        states.append(env["CANON_STATE"])
        wandb_names.append(env["CANON_WANDB_RUN_NAME"])
        head = document["spec"]["replicatedJobs"][0]["template"]["spec"][
            "template"
        ]["spec"]
        worker = document["spec"]["replicatedJobs"][1]["template"]["spec"][
            "template"
        ]["spec"]
        self.assertEqual(head["priorityClassName"], "very-high")
        self.assertEqual(worker["priorityClassName"], "very-high")
        scratches.append(tuple(
            arg
            for container in head["initContainers"]
            for arg in container["args"]
            if arg.startswith("--gcs_scratch_location=")
        ))
      self.assertEqual(len(set(states)), 6)
      self.assertEqual(len(set(wandb_names)), 6)
      self.assertEqual(len(set(scratches)), 6)
      self.assertTrue(all(len(values) == 2 for values in scratches))
      self.assertEqual(workloads, {"gsm8k", "frozenlake"})

  def test_rendered_commands_equal_frozen_workload_commands(self):
    with tempfile.TemporaryDirectory() as tmp:
      outputs = self._render(Path(tmp))
      by_key = {}
      for path in outputs:
        document = yaml.safe_load(path.read_text())
        env = _main_env(document)
        key = next(
            spec.key
            for spec in renderer._SPECS
            if path.name == spec.filename
        )
        by_key[key] = env
      for key in (
          "gsm8k-alignment-short",
          "frozenlake-alignment-short",
          "frozenlake-backward-no-commit",
          "gsm8k-full",
          "frozenlake-full",
      ):
        spec = next(item for item in renderer._SPECS if item.key == key)
        expected = shlex.join(
            dp_workloads.get_workload(spec.workload).command(
                run_stage=spec.stage
            )
        )
        self.assertEqual(by_key[key]["CANON_RUN_CMD"], expected)

      frozenlake_command = by_key["frozenlake-full"]["CANON_RUN_CMD"]
      self.assertIn("--vllm_max_num_seqs=16", frozenlake_command)
      self.assertIn("--vllm_max_num_batched_tokens=256", frozenlake_command)
      self.assertNotIn("--vllm_max_num_seqs=256", frozenlake_command)
      self.assertNotIn("--vllm_max_num_batched_tokens=4096", frozenlake_command)
      gsm8k_command = by_key["gsm8k-full"]["CANON_RUN_CMD"]
      self.assertIn("--rollout_vllm_max_num_seqs=16", gsm8k_command)
      self.assertIn(
          "--rollout_vllm_max_num_batched_tokens=256", gsm8k_command
      )
      self.assertNotIn("--rollout_vllm_max_num_seqs=256", gsm8k_command)
      self.assertNotIn(
          "--rollout_vllm_max_num_batched_tokens=4096", gsm8k_command
      )
      gsm8k_short = by_key["gsm8k-alignment-short"]
      self.assertEqual(gsm8k_short["CANON_P33_SHORT_ALIGNMENT"], "1")
      self.assertEqual(gsm8k_short["CANON_P33_NO_COMMIT"], "1")
      self.assertIn("--max_steps=1", gsm8k_short["CANON_RUN_CMD"])
      self.assertIn("--max_response_length=1024", gsm8k_short["CANON_RUN_CMD"])
      short_env = by_key["frozenlake-alignment-short"]
      self.assertEqual(short_env["CANON_P33_SHORT_ALIGNMENT"], "1")
      self.assertIn("--max_response_length=512", short_env["CANON_RUN_CMD"])
      self.assertIn("--env_max_steps=2", short_env["CANON_RUN_CMD"])
      self.assertEqual(by_key["frozenlake-full"]["CANON_P33_SHORT_ALIGNMENT"], "0")
      self.assertEqual(
          by_key["gsm8k-full"]["CANON_GSM8K_AB_REPORT_ONLY"],
          "0",
      )
      self.assertEqual(
          by_key["gsm8k-full"]["CANON_GSM8K_ALIGNMENT_WARN_ONLY"],
          "1",
      )
      for key in (
          "gsm8k-alignment-short",
          "frozenlake-alignment-short",
          "frozenlake-backward-no-commit",
          "frozenlake-full",
      ):
        self.assertEqual(
            by_key[key]["CANON_GSM8K_AB_REPORT_ONLY"],
            "0",
        )
        self.assertEqual(
            by_key[key]["CANON_GSM8K_ALIGNMENT_WARN_ONLY"],
            "0",
        )
      eval_env = by_key["frozenlake-full-eval"]
      self.assertEqual(eval_env["CANON_P33_ENABLE_EVAL"], "1")
      self.assertEqual(eval_env["CANON_P33_DISABLE_EVAL"], "0")
      self.assertEqual(eval_env["CANON_P31_ENABLE_EVAL"], "1")
      self.assertIn("--num_test_batches=4", eval_env["CANON_RUN_CMD"])
      self.assertIn("--eval_every_n_steps=10", eval_env["CANON_RUN_CMD"])

  def test_rejects_warning_policy_outside_gsm8k_full(self):
    base = renderer.load_base(_BASE_PATH)
    spec = next(
        item for item in renderer._SPECS
        if item.key == "frozenlake-backward-no-commit"
    )
    document = renderer.render_jobset(base, spec, _SOURCE_COMMIT, _RUN_ID)
    pod = document["spec"]["replicatedJobs"][0]["template"]["spec"][
        "template"
    ]["spec"]
    main = next(
        item for item in pod["containers"] if item["name"] == "jax-tpu"
    )
    policy = next(
        item for item in main["env"]
        if item["name"] == "CANON_GSM8K_ALIGNMENT_WARN_ONLY"
    )
    policy["value"] = "1"
    with self.assertRaisesRegex(ValueError, "environment drifted"):
      renderer.validate_jobset(document, spec, _SOURCE_COMMIT, _RUN_ID)

  def test_rejects_unreviewed_offload_optimizer_render(self):
    base = renderer.load_base(_BASE_PATH)
    spec = renderer._SPECS[0]
    document = renderer.render_jobset(base, spec, _SOURCE_COMMIT, _RUN_ID)
    pod = document["spec"]["replicatedJobs"][0]["template"]["spec"][
        "template"
    ]["spec"]
    main = next(
        item for item in pod["containers"] if item["name"] == "jax-tpu"
    )
    resident = next(
        item
        for item in main["env"]
        if item["name"] == "CANON_OPT_STATE_RESIDENT"
    )
    resident["value"] = "0"
    with self.assertRaisesRegex(ValueError, "environment drifted"):
      renderer.validate_jobset(document, spec, _SOURCE_COMMIT, _RUN_ID)

  def test_frozenlake_jobs_select_exactly_one_evaluation_mode(self):
    with tempfile.TemporaryDirectory() as tmp:
      for path in self._render(Path(tmp)):
        env = _main_env(yaml.safe_load(path.read_text()))
        if "frozenlake" in env["CANON_PROFILE_FILE"]:
          self.assertIn(
              (
                  env["CANON_P33_ENABLE_EVAL"],
                  env["CANON_P33_DISABLE_EVAL"],
                  env["CANON_P31_ENABLE_EVAL"],
              ),
              (("0", "1", "0"), ("1", "0", "1")),
          )

  def test_rejects_invalid_source_commit_and_run_id(self):
    base = renderer.load_base(_BASE_PATH)
    spec = renderer._SPECS[0]
    with self.assertRaisesRegex(ValueError, "source commit"):
      renderer.render_jobset(base, spec, "abc", _RUN_ID)
    with self.assertRaisesRegex(ValueError, "run id"):
      renderer.render_jobset(base, spec, _SOURCE_COMMIT, "Bad_ID")

  def test_refuses_to_overwrite_existing_render(self):
    with tempfile.TemporaryDirectory() as tmp:
      output_dir = Path(tmp)
      self._render(output_dir)
      with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
        self._render(output_dir)


  def _proxy(self, document):
    pod = document["spec"]["replicatedJobs"][0]["template"]["spec"][
        "template"
    ]["spec"]
    return next(
        item
        for item in pod["initContainers"]
        if item["name"] == "pathways-proxy"
    )

  def test_proxy_delivers_excess_precision_through_env(self):
    with tempfile.TemporaryDirectory() as tmp:
      for path in self._render(Path(tmp)):
        document = yaml.safe_load(path.read_text())
        proxy = self._proxy(document)
        entries = [
            entry
            for entry in proxy["env"]
            if entry["name"] == renderer.PROXY_XLA_ENV
        ]
        self.assertEqual(
            entries,
            [{
                "name": renderer.PROXY_XLA_ENV,
                "value": renderer.PROXY_XLA_FLAG,
            }],
        )
        self.assertFalse(
            [a for a in proxy["args"] if "excess_precision" in a]
        )

  def test_rejects_base_with_raw_proxy_excess_precision_arg(self):
    base = yaml.safe_load(Path(_BASE_PATH).read_text())
    pod = base["spec"]["replicatedJobs"][0]["template"]["spec"]["template"][
        "spec"
    ]
    proxy = next(
        item
        for item in pod["initContainers"]
        if item["name"] == "pathways-proxy"
    )
    proxy["args"].append("--xla_allow_excess_precision=false")
    with tempfile.TemporaryDirectory() as tmp:
      bad = Path(tmp) / "bad_base.yaml"
      bad.write_text(yaml.safe_dump(base))
      with self.assertRaisesRegex(ValueError, "raw excess-precision"):
        renderer.render_all(
            base_path=bad,
            output_dir=Path(tmp) / "out",
            source_commit=_SOURCE_COMMIT,
            run_id=_RUN_ID,
        )

  def test_rejects_base_with_conflicting_proxy_xla_env(self):
    base = yaml.safe_load(Path(_BASE_PATH).read_text())
    pod = base["spec"]["replicatedJobs"][0]["template"]["spec"]["template"][
        "spec"
    ]
    proxy = next(
        item
        for item in pod["initContainers"]
        if item["name"] == "pathways-proxy"
    )
    proxy.setdefault("env", []).append(
        {"name": "XLA_FLAGS", "value": "--xla_allow_excess_precision=true"}
    )
    with tempfile.TemporaryDirectory() as tmp:
      bad = Path(tmp) / "bad_base.yaml"
      bad.write_text(yaml.safe_dump(base))
      with self.assertRaisesRegex(ValueError, "conflicting or duplicate"):
        renderer.render_all(
            base_path=bad,
            output_dir=Path(tmp) / "out",
            source_commit=_SOURCE_COMMIT,
            run_id=_RUN_ID,
        )

  def test_negative_control_rejects_shared_scratch(self):
    base = renderer.load_base(_BASE_PATH)
    spec = renderer._SPECS[0]
    document = renderer.render_jobset(base, spec, _SOURCE_COMMIT, _RUN_ID)
    head = document["spec"]["replicatedJobs"][0]["template"]["spec"][
        "template"
    ]["spec"]
    proxy = next(
        item for item in head["initContainers"] if item["name"] == "pathways-proxy"
    )
    scratch_index = next(
        index
        for index, value in enumerate(proxy["args"])
        if value.startswith("--gcs_scratch_location=")
    )
    proxy["args"][scratch_index] = "--gcs_scratch_location=gs://shared"
    with self.assertRaisesRegex(ValueError, "isolated scratch"):
      renderer.validate_jobset(
          document, spec, _SOURCE_COMMIT, _RUN_ID
      )

  def test_rejects_missing_or_mismatched_priority_class(self):
    spec = renderer._SPECS[0]
    for role, bad_value in (("head", None), ("worker", "low")):
      with self.subTest(role=role):
        base = renderer.load_base(_BASE_PATH)
        pod = (
            renderer._head_pod(base)
            if role == "head"
            else renderer._worker_pod(base)
        )
        if bad_value is None:
          pod.pop("priorityClassName")
        else:
          pod["priorityClassName"] = bad_value
        with self.assertRaisesRegex(ValueError, "priority class drifted"):
          renderer.render_jobset(base, spec, _SOURCE_COMMIT, _RUN_ID)


if __name__ == "__main__":
  unittest.main()

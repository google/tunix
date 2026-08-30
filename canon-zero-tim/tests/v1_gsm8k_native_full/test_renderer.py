"""Contracts for the GSM8K Native/mismatch full control."""

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
_PACKAGE = _REPO / "canon-zero-tim"
_TASK = _PACKAGE / "tasks/v1-gsm8k-native-full-control"
_SCRIPT = _TASK / "render_gsm8k_native_full.py"
_PREPARE = _TASK / "prepare_gsm8k_native_full.sh"
_ZERO_SCRIPT = (
    _PACKAGE
    / "tasks/v1-phase4-three-full-recipes/scripts"
    / "render_three_full_recipes.py"
)


def _load(name: str, path: Path):
  spec = importlib.util.spec_from_file_location(name, path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


native = _load("v1_gsm8k_native_full_renderer", _SCRIPT)
zero = _load("v1_gsm8k_zero_full_renderer", _ZERO_SCRIPT)


def _main(document: dict) -> dict:
  pod = document["spec"]["replicatedJobs"][0]["template"]["spec"][
      "template"
  ]["spec"]
  return next(item for item in pod["containers"] if item["name"] == "jax-tpu")


def _env(document: dict) -> dict[str, str]:
  return {
      item["name"]: item["value"]
      for item in _main(document)["env"]
      if "value" in item
  }


def _proxy_env(document: dict) -> dict[str, str]:
  pod = document["spec"]["replicatedJobs"][0]["template"]["spec"][
      "template"
  ]["spec"]
  proxy = next(
      item for item in pod["initContainers"] if item["name"] == "pathways-proxy"
  )
  return {
      item["name"]: item["value"]
      for item in proxy.get("env", [])
      if "value" in item
  }


def _resolve_profile(values: dict[str, str]) -> tuple[str, str, str]:
  profile = _PACKAGE / values["CANON_PROFILE_FILE"]
  environment = os.environ.copy()
  environment.update(values)
  completed = subprocess.run(
      [
          "bash",
          "-euo",
          "pipefail",
          "-c",
          'source "$1"; printf "RESULT:%s|%s|%s\\n" '
          '"$CANON_WANDB_PROJECT" "$CANON_WANDB_GROUP" "$CANON_PROFILE"',
          "profile-resolver",
          str(profile),
      ],
      cwd=_REPO,
      env=environment,
      text=True,
      capture_output=True,
      check=False,
  )
  if completed.returncode:
    raise AssertionError(completed.stderr or completed.stdout)
  receipt = next(
      line.removeprefix("RESULT:")
      for line in completed.stdout.splitlines()
      if line.startswith("RESULT:")
  )
  project, group, profile_name = receipt.split("|", 2)
  return project, group, profile_name


def _run_00_env(values: dict[str, str], state: Path) -> subprocess.CompletedProcess:
  environment = {
      key: value
      for key, value in os.environ.items()
      if not key.startswith("CANON_")
  }
  environment.update(values)
  environment.update({
      "CANON_PKG": str(_PACKAGE),
      "CANON_STATE": str(state),
      "CANON_MODE": "run",
      "INJECTED_HF_TOKEN": "test-token",
      "INJECTED_WANDB_API_KEY": "test-key",
  })
  state.mkdir(parents=True)
  return subprocess.run(
      ["bash", str(_PACKAGE / "cluster/steps/00_env.sh")],
      cwd=_REPO,
      env=environment,
      text=True,
      capture_output=True,
      check=False,
  )


class Gsm8kNativeFullRendererTest(unittest.TestCase):

  def _render_pair(self, root: Path):
    native_path = native.render_native_full(
        source_commit="a" * 40,
        output_dir=root / "native",
        run_id="native-a",
        base_path=_PACKAGE / "cluster/jobset-64chip.yaml",
    )
    zero_path = zero.render_gsm8k_full(
        source_commit="a" * 40,
        output_dir=root / "zero",
        run_id="zero-a",
        base_path=_PACKAGE / "cluster/jobset-64chip.yaml",
    )
    native_doc = yaml.safe_load(native_path.read_text(encoding="utf-8"))
    zero_doc = yaml.safe_load(zero_path.read_text(encoding="utf-8"))
    return native_path, zero_path, native_doc, zero_doc

  def test_native_reuses_registered_full_and_isolates_zero_selectors(self):
    with tempfile.TemporaryDirectory() as tmp:
      native_path, _, native_doc, _ = self._render_pair(Path(tmp))
      values = _env(native_doc)
      self.assertEqual(
          native_path.name, "jobset-v1-gsm8k-native-mismatch-full.yaml"
      )
      self.assertEqual(values["CANON_PROFILE_FILE"], native._NATIVE_PROFILE)
      self.assertEqual(values["CANON_EXPECT_COMMIT"], "a" * 40)
      self.assertEqual(values["CANON_P33_SHARED_MESH"], "16,4")
      self.assertEqual(values["CANON_P33_RUN_STAGE"], "full")
      self.assertEqual(values["CANON_P33_NO_COMMIT"], "0")
      self.assertEqual(values["CANON_OPT_STATE_RESIDENT"], "1")
      self.assertEqual(values["CANON_P30_OPT_STATE_OFFLOAD"], "0")
      self.assertEqual(values["CANON_P32_TRAIN_ADMITTED"], "0")
      self.assertEqual(values["CANON_P32_DP_REDUCTION_ADMITTED"], "0")
      self.assertEqual(values["CANON_P33_WORKLOAD_LAUNCH_ADMITTED"], "0")
      self.assertEqual(values["CANON_GSM8K_TRAIN"], "1")
      self.assertEqual(values["CANON_GSM8K_VANILLA"], "1")
      self.assertIn("--max_steps=200", values["CANON_RUN_CMD"])
      for name in native._FORBIDDEN_ZERO_SELECTORS:
        self.assertNotIn(name, values)
      self.assertNotIn("XLA_FLAGS", _proxy_env(native_doc))

      labels = native_doc["metadata"]["labels"]
      self.assertEqual(labels["canon.zero-tim/treatment"], "native-mismatch")
      self.assertEqual(labels["canon.zero-tim/control-for"], "v1-hp-zero")
      self.assertEqual(
          labels["canon.zero-tim/performance-profile"], "stock-native"
      )
      self.assertEqual(native_doc["spec"]["failurePolicy"]["maxRestarts"], 3)

      index = json.loads(
          (native_path.parent / "manifest-index.json").read_text(encoding="utf-8")
      )
      self.assertEqual(index["schema"], "v1-gsm8k-native-mismatch-full-v1")
      self.assertEqual(index["arm"], "native-mismatch")
      self.assertEqual(index["comparison_arm"], "v1-hp-zero")
      self.assertEqual(index["wandb_project"], native._WANDB_PROJECT)
      self.assertEqual(index["wandb_group"], native._WANDB_GROUP)
      self.assertFalse(index["launch_executed"])

  def test_native_and_zero_share_scientific_command_and_wandb_project(self):
    with tempfile.TemporaryDirectory() as tmp:
      _, _, native_doc, zero_doc = self._render_pair(Path(tmp))
      native_values = _env(native_doc)
      zero_values = _env(zero_doc)
      for name in (
          "CANON_EXPECT_COMMIT",
          "CANON_P33_SHARED_MESH",
          "CANON_P33_RUN_STAGE",
          "CANON_P33_NO_COMMIT",
          "CANON_OPT_STATE_RESIDENT",
          "CANON_P30_OPT_STATE_OFFLOAD",
          "CANON_RUN_CMD",
      ):
        self.assertEqual(native_values[name], zero_values[name], msg=name)
      self.assertNotEqual(
          native_values["CANON_WANDB_RUN_NAME"],
          zero_values["CANON_WANDB_RUN_NAME"],
      )
      native_project, native_group, native_profile = _resolve_profile(
          native_values
      )
      zero_project, zero_group, zero_profile = _resolve_profile(zero_values)
      self.assertEqual(native_project, zero_project)
      self.assertEqual(native_project, native._WANDB_PROJECT)
      self.assertEqual(native_group, zero_group)
      self.assertEqual(native_group, native._WANDB_GROUP)
      self.assertEqual(native_profile, "qwen3-1p7b-dp16-tp4-gsm8k-native")
      self.assertEqual(zero_profile, "qwen3-1p7b-dp16-tp4-gsm8k-v1-hp")

      driver = (
          _REPO / "examples/math_gsm8k/qwen3_grpo_demo.py"
      ).read_text(encoding="utf-8")
      self.assertIn("SEED = 42", driver)

  def test_real_env_reload_selects_stock_train_and_zero_tim_off(self):
    with tempfile.TemporaryDirectory() as tmp:
      native_path = native.render_native_full(
          source_commit="d" * 40,
          output_dir=Path(tmp) / "rendered",
          run_id="native-d",
          base_path=_PACKAGE / "cluster/jobset-64chip.yaml",
      )
      document = yaml.safe_load(native_path.read_text(encoding="utf-8"))
      state = Path(tmp) / "state"
      completed = _run_00_env(_env(document), state)
      self.assertEqual(
          completed.returncode,
          0,
          msg=f"stdout={completed.stdout}\nstderr={completed.stderr}",
      )
      self.assertIn(
          "[GSM8K.NATIVE] ZERO_TIM_OFF_PASS p32=absent "
          "canonical_engine=off alignment=off p59=off v1=off",
          completed.stdout,
      )
      resolved = subprocess.run(
          [
              "bash",
              "-euo",
              "pipefail",
              "-c",
              'source "$1"; '
              'test "$CANON_GSM8K_TRAIN" = 1; '
              'test "$CANON_GSM8K_VANILLA" = 1; '
              'test "${CANON_ENGINE_MODULE_C:-0}" = 0; '
              'test "${CANON_ALIGNMENT_GATE:-0}" = 0; '
              'test "${CANON_ALIGNMENT_TRAIN:-0}" = 0; '
              'test "${CANON_P59_RANK_PARALLEL_BACKWARD:-0}" = 0; '
              'test "${CANON_V1_HP_FULL:-0}" = 0; '
              'test -z "${CANON_P32_WORKLOAD:-}"; '
              'test "$CANON_WANDB_PROJECT" = zero-tim-gsm8k-dp16-tp4; '
              'test "$CANON_WANDB_GROUP" = qwen3-1p7b-dp16-tp4; '
              'case "$XLA_FLAGS" in *--xla_allow_excess_precision=false*) exit 9;; esac',
              "resolved-native",
              str(state / "env.sh"),
          ],
          cwd=_REPO,
          text=True,
          capture_output=True,
          check=False,
      )
      self.assertEqual(resolved.returncode, 0, msg=resolved.stderr)

  def test_native_profile_asks_for_auto_mesh_axis_types(self):
    """The untreated arm cannot run on Explicit axes; the Zero arm must not
    inherit the escape hatch."""
    native_profile = (
        _PACKAGE / "cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k-native.env"
    ).read_text(encoding="utf-8")
    self.assertIn("export CANON_GSM8K_MESH_AXIS_TYPES=auto", native_profile)
    self.assertIn("export FL_SHARED_MESH=16,4", native_profile)
    for zero_profile_name in (
        "qwen3-1p7b-dp16-tp4-gsm8k.env",
        "qwen3-1p7b-dp16-tp4-gsm8k-v1-hp.env",
    ):
      zero_profile = (
          _PACKAGE / "cluster/profiles" / zero_profile_name
      ).read_text(encoding="utf-8")
      self.assertNotIn("CANON_GSM8K_MESH_AXIS_TYPES", zero_profile)

  def test_demo_mesh_axis_type_selector_is_fail_closed(self):
    """Unset keeps the historical pairing; junk and orphaned values raise."""
    demo = (
        _PACKAGE.parent / "examples/math_gsm8k/qwen3_grpo_demo.py"
    ).read_text(encoding="utf-8")
    self.assertIn(
        'axis_type_choice = os.environ.get("CANON_GSM8K_MESH_AXIS_TYPES", "")',
        demo,
    )
    self.assertIn('if axis_type_choice not in ("", "auto", "explicit"):', demo)
    # Unset with a shape still means Explicit, exactly as before.
    self.assertIn('explicit_axes = axis_type_choice != "auto"', demo)
    # Asking for Explicit without a shape is refused rather than ignored.
    self.assertIn(
        'CANON_GSM8K_MESH_AXIS_TYPES=explicit requires FL_SHARED_MESH', demo
    )

  def test_real_env_rejects_mixed_p32_native_input(self):
    with tempfile.TemporaryDirectory() as tmp:
      native_path = native.render_native_full(
          source_commit="e" * 40,
          output_dir=Path(tmp) / "rendered",
          run_id="native-e",
          base_path=_PACKAGE / "cluster/jobset-64chip.yaml",
      )
      document = yaml.safe_load(native_path.read_text(encoding="utf-8"))
      values = _env(document)
      values["CANON_P32_WORKLOAD"] = "gsm8k"
      completed = _run_00_env(values, Path(tmp) / "state")
      self.assertNotEqual(completed.returncode, 0)
      self.assertIn(
          "GSM8K native caller contradictions: CANON_P32_WORKLOAD=gsm8k",
          completed.stderr,
      )

  def test_forbidden_zero_selector_is_a_live_negative_control(self):
    with tempfile.TemporaryDirectory() as tmp:
      native_path = native.render_native_full(
          source_commit="b" * 40,
          output_dir=Path(tmp) / "native",
          run_id="native-b",
          base_path=_PACKAGE / "cluster/jobset-64chip.yaml",
      )
      document = yaml.safe_load(native_path.read_text(encoding="utf-8"))
      _main(document)["env"].append({"name": "CANON_P71_SCAN", "value": "fwd"})
      with self.assertRaisesRegex(ValueError, "Zero selectors"):
        native._validate_document(
            document, native._registered_spec(), "b" * 40, "native-b"
        )

  def test_output_reuse_and_invalid_source_are_rejected(self):
    with tempfile.TemporaryDirectory() as tmp:
      output = Path(tmp) / "native"
      output.mkdir()
      with self.assertRaises(FileExistsError):
        native.render_native_full(
            source_commit="c" * 40,
            output_dir=output,
            run_id="native-c",
            base_path=_PACKAGE / "cluster/jobset-64chip.yaml",
        )
      with self.assertRaisesRegex(ValueError, "40 lowercase"):
        native.render_native_full(
            source_commit="not-a-sha",
            output_dir=Path(tmp) / "other",
            run_id="native-c",
            base_path=_PACKAGE / "cluster/jobset-64chip.yaml",
        )

  def test_prepare_wrapper_is_clean_sha_bound_and_render_only(self):
    script = _PREPARE.read_text(encoding="utf-8")
    self.assertIn('git -C "$REPO_ROOT" rev-parse HEAD', script)
    self.assertIn(
        'git -C "$REPO_ROOT" status --porcelain --untracked-files=all', script
    )
    self.assertIn("refusing to render from a dirty worktree", script)
    self.assertIn("refusing to reuse output directory", script)
    self.assertIn("V1_GSM8K_NATIVE_FULL_READY", script)
    self.assertIn("manifests=1", script)
    self.assertIn("launch=not-executed", script)
    self.assertEqual(script.count('"kubectl apply -f '), 1)
    self.assertFalse(
        any(line.strip().startswith("kubectl apply") for line in script.splitlines())
    )
    completed = subprocess.run(
        ["bash", "-n", str(_PREPARE)],
        cwd=_REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    self.assertEqual(completed.returncode, 0, msg=completed.stderr)

    invalid = subprocess.run(
        ["bash", str(_PREPARE), "bad", "/tmp/not-created", "native"],
        cwd=_REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    self.assertEqual(invalid.returncode, 2)
    self.assertIn("source SHA", invalid.stderr)

  def test_exact_image_gate_mounts_the_workspace_read_only(self):
    script = (
        _REPO / "canon-zero-tim/tests/v1_gsm8k_native_full/run_exact_image.sh"
    ).read_text(encoding="utf-8")
    self.assertIn('-v "$root:/workspace:ro"', script)
    self.assertIn("V1_GSM8K_NATIVE_FULL_EXACT_IMAGE_PASS", script)
    entrypoint = (_PACKAGE / "cluster/entrypoint.sh").read_text(encoding="utf-8")
    self.assertIn("step gsm8k_verify_stock_engine.sh", entrypoint)
    self.assertIn(
        "GSM8K_NATIVE_STOCK_PATH source=$CANON_EXPECT_COMMIT "
        "canonical_overlay=skipped alignment=off",
        entrypoint,
    )

  @unittest.skipUnless(
      os.environ.get("TEST_GSM8K_NATIVE_STOCK_PREFLIGHT") == "1",
      "pinned-image stock files are required",
  )
  def test_pinned_image_stock_engine_and_driver_import_preflight(self):
    with tempfile.TemporaryDirectory() as tmp:
      native_path = native.render_native_full(
          source_commit="f" * 40,
          output_dir=Path(tmp) / "rendered",
          run_id="native-f",
          base_path=_PACKAGE / "cluster/jobset-64chip.yaml",
      )
      document = yaml.safe_load(native_path.read_text(encoding="utf-8"))
      state = Path(tmp) / "state"
      completed = _run_00_env(_env(document), state)
      self.assertEqual(completed.returncode, 0, msg=completed.stderr)
      environment = os.environ.copy()
      environment.update({
          "CANON_PKG": str(_PACKAGE),
          "CANON_STATE": str(state),
      })
      probe = subprocess.run(
          ["bash", str(_PACKAGE / "cluster/steps/20_probe_image.sh")],
          cwd=_REPO,
          env=environment,
          text=True,
          capture_output=True,
          check=False,
      )
      self.assertEqual(
          probe.returncode, 0, msg=f"stdout={probe.stdout}\nstderr={probe.stderr}"
      )
      verifier = subprocess.run(
          [
              "bash",
              str(_PACKAGE / "cluster/steps/gsm8k_verify_stock_engine.sh"),
          ],
          cwd=_REPO,
          env=environment,
          text=True,
          capture_output=True,
          check=False,
      )
      self.assertEqual(
          verifier.returncode,
          0,
          msg=f"stdout={verifier.stdout}\nstderr={verifier.stderr}",
      )
      self.assertIn(
          "[GSM8K.NATIVE] STOCK_PREFLIGHT_PASS files=6 "
          "driver_import=pass canonical_overlay=absent alignment=off",
          verifier.stdout,
      )

  def test_handoffs_route_both_arms_to_one_wandb_project(self):
    task_handoff = (_TASK / "HANDOFF.md").read_text(encoding="utf-8")
    phase4_handoff = (
        _PACKAGE / "tasks/v1-phase4-three-full-recipes/HANDOFF.md"
    ).read_text(encoding="utf-8")
    for source in (task_handoff, phase4_handoff):
      self.assertIn("prepare_gsm8k_native_full.sh", source)
      self.assertIn("prepare_gsm8k_full_dp16tp4_p74.sh", source)
      self.assertIn(native._WANDB_PROJECT, source)
      self.assertIn(native._WANDB_GROUP, source)
    for selector in native._FORBIDDEN_ZERO_SELECTORS:
      self.assertIn(selector, task_handoff)
    self.assertIn("must all be absent from the raw Native manifest", task_handoff)
    self.assertIn("TARGET NOT RUN", task_handoff)


if __name__ == "__main__":
  unittest.main()

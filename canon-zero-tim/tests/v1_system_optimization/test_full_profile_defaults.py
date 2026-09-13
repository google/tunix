"""Presence, admission and real-process gates for full-profile defaults."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "canon-zero-tim"
CLUSTER = PACKAGE / "cluster"
sys.path.insert(0, str(CLUSTER))
import v1_full_system_optimization as policy


def _identity(workload="gsm8k", dp="8"):
  result = {
      "CANON_V1_HP_FULL": "1", "CANON_P33_RUN_STAGE": "full",
      "CANON_P33_NO_COMMIT": "0",
  }
  if workload == "gsm8k":
    profile = "qwen3-1p7b-dp16-tp4-gsm8k-v1-hp"
    result.update({
        "CANON_P32_WORKLOAD": "gsm8k", "CANON_MODEL_DIR_NAME": "qwen1p7b",
        "CANON_DP_SIZE": "16", "CANON_TP_SIZE": "4",
        "CANON_TOTAL_DEVICES": "64", "CANON_GSM8K_TRAIN": "1",
    })
  else:
    profile = f"qwen3-8b-dp{dp}-tp8-frozenlake-v1-hp"
    result.update({
        "CANON_P32_WORKLOAD": f"frozenlake-dp{dp}-tp8",
        "CANON_MODEL_DIR_NAME": "qwen8b_tp8",
        "CANON_DP_SIZE": dp, "CANON_TP_SIZE": "8",
        "CANON_TOTAL_DEVICES": str(int(dp) * 8),
        "CANON_P57_TIM_ARM": "zero", "CANON_P57_RUN_KIND": "train",
        "CANON_P57_WORKLOAD_CANDIDATE": "m15" if workload == "m15" else "",
        "CANON_P57_DATA_SPLIT": "main" if workload == "m15" else "",
    })
  result["CANON_PROFILE"] = profile
  result["CANON_PROFILE_FILE"] = f"cluster/profiles/{profile}.env"
  return result


class FullProfileDefaultsTest(unittest.TestCase):

  def test_exact_full_identities_resolve_the_registered_pair(self):
    for workload, dp in (("gsm8k", "8"), ("p45", "8"), ("m15", "8"),
                         ("p45", "4"), ("m15", "4")):
      with self.subTest(workload=workload, dp=dp):
        identity = _identity(workload, dp)
        before = dict(identity)
        result = policy.full_profile_defaults(identity)
        self.assertEqual(result, {
            "CANON_P32_KEEP_TAPE": "stream", "CANON_DP_REDUCE_ONCE": "1",
        })
        self.assertEqual(identity, before)
        result["CANON_DP_REDUCE_ONCE"] = "0"
        self.assertEqual(
            policy.full_profile_defaults(identity)["CANON_DP_REDUCE_ONCE"],
            "1",
        )

  def test_any_explicit_presence_preserves_the_old_pair(self):
    for workload in ("gsm8k", "p45", "m15"):
      for name in policy.FULL_PROFILE_DEFAULT_NAMES:
        for value in ("", "0", "1", "stream", "invalid"):
          with self.subTest(workload=workload, name=name, value=value):
            identity = {**_identity(workload), name: value}
            before = dict(identity)
            self.assertEqual(policy.full_profile_defaults(identity), {})
            self.assertEqual(identity, before)

  def test_every_required_identity_field_is_checked_even_with_explicit_values(self):
    for workload in ("gsm8k", "p45", "m15"):
      identity = _identity(workload)
      # P45's empty candidate/split are intentionally equivalent to absent.
      required = [key for key, value in identity.items() if value]
      for key in required:
        for explicit in (False, True):
          with self.subTest(workload=workload, key=key, explicit=explicit):
            broken = {**identity, key: "wrong"}
            if explicit:
              broken["CANON_DP_REDUCE_ONCE"] = "0"
            with self.assertRaises(ValueError):
              policy.full_profile_defaults(broken)

  def test_native_diagnostic_and_neighbor_profiles_do_not_inherit_defaults(self):
    for overrides in (
        {"CANON_GSM8K_VANILLA": "1"},
        {"CANON_PROFILE_FILE": "cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k-native.env"},
        {"CANON_PROFILE_FILE":
         "cluster/profiles/qwen3-8b-dp2-tp2-frozenlake-onehost.env"},
        {"CANON_PROFILE_FILE": "cluster/profiles/qwen3-4b-deepswe.env"},
        {"CANON_P33_RUN_STAGE": "backward-no-commit", "CANON_P33_NO_COMMIT": "1"},
    ):
      with self.subTest(overrides=overrides), self.assertRaises(ValueError):
        policy.full_profile_defaults({**_identity(), **overrides})

  def test_render_tuple_omits_only_profile_owned_names(self):
    for workload in policy.REGISTERED_FULL_WORKLOADS:
      full = policy.full_system_optimization_additions(workload)
      raw = policy.full_system_optimization_render_additions(workload)
      self.assertEqual(raw, {key: value for key, value in full.items()
                             if key not in policy.FULL_PROFILE_DEFAULT_NAMES})
    deepswe = policy.full_system_optimization_additions("deepswe-qwen4b")
    self.assertFalse(set(policy.FULL_PROFILE_DEFAULT_NAMES) & deepswe.keys())

  def test_cli_is_stdlib_only_and_exports_only_the_registered_pair(self):
    program = CLUSTER / "v1_full_system_optimization.py"
    environment = {"PATH": "/usr/bin:/bin", **_identity()}
    result = subprocess.run(
        [sys.executable, "-S", str(program), "--profile-defaults"],
        env=environment, capture_output=True, text=True,
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(
        result.stdout,
        "export CANON_P32_KEEP_TAPE=stream\nexport CANON_DP_REDUCE_ONCE=1\n",
    )
    # A site-disabled Python cannot import JAX from the environment. The CLI
    # also stays empty for an explicit off value without changing its sibling.
    result = subprocess.run(
        [sys.executable, "-S", str(program), "--profile-defaults"],
        env={**environment, "CANON_DP_REDUCE_ONCE": "0"},
        capture_output=True, text=True,
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(result.stdout, "")

  def test_real_three_recipe_profile_and_stale_parent_reload_match_explicit_bundle(self):
    script = (
        PACKAGE / "tasks/v1-phase4-three-full-recipes/scripts"
        / "render_three_full_recipes.py"
    )
    spec = importlib.util.spec_from_file_location("f1_delivery_renderer", script)
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      paths = renderer.render_three(
          source_commit="a" * 40, output_dir=root / "rendered",
          gsm8k_run_id="f1g", p45_run_id="f1p", m15_run_id="f1m",
          campaign_root="f1-test", base_path=CLUSTER / "jobset-64chip.yaml",
      )
      for index, path in enumerate(paths):
        raw = renderer._env(yaml.safe_load(path.read_text()))
        for name in policy.FULL_PROFILE_DEFAULT_NAMES:
          self.assertTrue(name not in raw, f"raw recipe still writes {name}")
        automatic = self._resolve(raw, root / f"default-{index}")
        explicit = self._resolve(
            {**raw, "CANON_P32_KEEP_TAPE": "stream",
             "CANON_DP_REDUCE_ONCE": "1"},
            root / f"explicit-{index}",
        )
        self.assertEqual(automatic, explicit)
        # The FrozenLake full profile exports the r3 length-sorted backward
        # grouping (tasks/zero_tim_perf2 phase C/D); it is not part of the
        # derived pair and the stale parent value must not survive either way.
        sort = "1" if "frozenlake" in raw.get("CANON_PROFILE_FILE", "") else None
        self.assertEqual(automatic, ["stream", True, True, sort])
        # An explicit off control cannot be promoted merely by selecting the
        # optimized profile. Preserve the old partial-pair semantics too.
        for suffix, overrides in (
            ("both-off", {"CANON_P32_KEEP_TAPE": "0",
                          "CANON_DP_REDUCE_ONCE": "0"}),
            ("reduce-off", {"CANON_DP_REDUCE_ONCE": "0"}),
            ("empty-tape", {"CANON_P32_KEEP_TAPE": ""}),
        ):
          self.assertEqual(
              self._resolve({**raw, **overrides}, root / f"{suffix}-{index}"),
              ["", False, True, sort],
          )

  def test_renderer_rejects_profile_defaults_in_a_stale_raw_template(self):
    script = (
        PACKAGE / "tasks/v1-phase4-three-full-recipes/scripts"
        / "render_three_full_recipes.py"
    )
    spec = importlib.util.spec_from_file_location("f1_stale_renderer", script)
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    for name in policy.FULL_PROFILE_DEFAULT_NAMES:
      for value in ("", "0", "1", "stream"):
        with self.subTest(name=name, value=value):
          self._assert_stale_template_rejected(renderer, name, value)

  def _assert_stale_template_rejected(self, renderer, name, value):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      base = yaml.safe_load((CLUSTER / "jobset-64chip.yaml").read_text())
      job = base["spec"]["replicatedJobs"][0]
      pod = job["template"]["spec"]["template"]["spec"]
      main = next(c for c in pod["containers"] if c["name"] == "jax-tpu")
      main["env"].append({"name": name, "value": value})
      source = root / "base.yaml"
      source.write_text(yaml.safe_dump(base, sort_keys=False))
      with self.assertRaisesRegex(ValueError, "full profile must own default"):
        renderer.render_three(
            source_commit="a" * 40, output_dir=root / "rendered",
            gsm8k_run_id="f1g", p45_run_id="f1p", m15_run_id="f1m",
            campaign_root="f1-test", base_path=source,
        )

  def _resolve(self, raw, state):
    state.mkdir()
    environment = {
        "PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1", "JAX_PLATFORMS": "cpu",
        **raw, "CANON_PKG": str(PACKAGE), "CANON_STATE": str(state),
        "INJECTED_HF_TOKEN": "test-only", "INJECTED_WANDB_API_KEY": "test-only",
    }
    result = subprocess.run(["bash", str(CLUSTER / "steps/00_env.sh")],
                            cwd=ROOT, env=environment, text=True, capture_output=True)
    self.assertEqual(result.returncode, 0, result.stderr)
    code = (
        "import importlib.util,json,os,sys; "
        "s=importlib.util.spec_from_file_location('config',sys.argv[1]); "
        "c=importlib.util.module_from_spec(s); s.loader.exec_module(c); "
        "print(json.dumps([c.keep_tape_mode(),c.reduce_once_enabled(),"
        "c.rank_parallel_backward(),os.environ.get('CANON_P32_LENGTH_SORT')]))"
    )
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c",
         'source "$1"; export JAX_PLATFORMS=cpu; exec "$2" -c "$3" "$4"',
         "f1-readback", str(state / "env.sh"), sys.executable, code,
         str(ROOT / "tunix/rl/canonical_training_config.py")],
        cwd=ROOT,
        env={**environment, "CANON_P32_KEEP_TAPE": "stale",
             "CANON_DP_REDUCE_ONCE": "stale", "CANON_P32_LENGTH_SORT": "stale"},
        text=True, capture_output=True,
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    return json.loads(result.stdout)

"""Real one-host shell/profile delivery, without assets, credentials or TPU."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "canon-zero-tim"
SCRIPTS = PACKAGE / "tasks/v1-gsm8k-onehost-xprof-pair/scripts"
COMMON = SCRIPTS / "run_onehost_gsm8k_xprof_common.sh"
INNER = SCRIPTS / "run_onehost_gsm8k_xprof_inner.sh"
POLICY = PACKAGE / "cluster/v1_full_system_optimization.py"
PAIR = ("CANON_P32_KEEP_TAPE", "CANON_DP_REDUCE_ONCE")
GEOMETRIES = ("dp4-tp1", "dp2-tp2", "dp2-tp2-long", "dp2-tp2-long8k")
ENVIRONMENT = {
    "PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin",
    "JAX_PLATFORMS": "cpu", "PYTHONDONTWRITEBYTECODE": "1",
}


def shell_probe(script, args, values, boundary, reader=False):
  """Runs the actual script, stopping at a verified pre-side-effect command.

  No launcher source is copied or rewritten. A test-only DEBUG trap observes
  a fixed command before it executes. The outer fallback uses an existing
  artifact root and impossible hostname, refusing before credentials/docker.
  The inner fallback remains on CPU and cannot initialize a TPU backend.
  """
  code = (
      "import json,os; "
      f"print(json.dumps({{k:os.environ.get(k) for k in {PAIR!r}}}))"
  )
  if reader:
    code = (
        "import importlib.util,json,os; "
        f"s=importlib.util.spec_from_file_location('config',"
        f"{str(ROOT / 'tunix/rl/canonical_training_config.py')!r}); "
        "c=importlib.util.module_from_spec(s); s.loader.exec_module(c); "
        "print(json.dumps([c.keep_tape_mode(),c.reduce_once_enabled(),"
        "c.rank_parallel_backward(),"
        "os.environ.get('CANON_P66_P59_CHECK_VMA')]))"
    )
  with tempfile.TemporaryDirectory() as temporary:
    startup = Path(temporary) / "probe.bash"
    startup.write_text(
        "trap 'if [[ $BASH_COMMAND == \"$F1C_TEST_BOUNDARY\" ]]; then\n"
        "  trap - DEBUG\n"
        "  printf \"F1C_TEST_BOUNDARY_REACHED\\n\" >&2\n"
        "  \"$F1C_TEST_PYTHON\" -S -c \"$F1C_TEST_CODE\"\n"
        "  exit $?\n"
        "fi' DEBUG\n"
    )
    result = subprocess.run(
        ["bash", str(script), *args], cwd=ROOT, text=True,
        capture_output=True, timeout=30,
        env={
            **ENVIRONMENT, **values, "BASH_ENV": str(startup),
            "F1C_TEST_BOUNDARY": boundary, "F1C_TEST_CODE": code,
            "F1C_TEST_PYTHON": sys.executable,
            "V1_GSM8K_XPROF_ARTIFACT_DIR": temporary,
            "V1_GSM8K_XPROF_EXPECT_HOSTNAME": "invalid-test-only-host",
        },
    )
  return result


def outer_probe(values=None, *, arm="zero-hp", geometry="dp2-tp2",
                stage="three-update", wrapper=None):
  return shell_probe(
      wrapper or COMMON, [arm, "f1c-test"] if wrapper is None else ["f1c-test"],
      {"V1_GSM8K_XPROF_GEOMETRY": geometry, "CANON_P33_RUN_STAGE": stage,
       **(values or {})},
      "canon_env=/mnt/disks/tunix-data/claude_work/canon_env.sh",
  )


def inner_probe(pair, *, arm="zero-hp", geometry="dp2-tp2",
                stage="three-update"):
  # Match the common runner's literal -e NAME="${NAME:-}" transport. A
  # separate source assertion below protects this missing-to-empty mapping.
  environment = {key: pair.get(key) or "" for key in PAIR}
  environment.update({
      "V1_GSM8K_XPROF_REPO": str(ROOT), "V1_GSM8K_XPROF_ARM": arm,
      "V1_GSM8K_XPROF_XLA_FLAGS": "test-only",
      "V1_GSM8K_XPROF_GEOMETRY": geometry,
      "CANON_V1_GSM8K_XPROF_GEOMETRY": geometry,
      "V1_GSM8K_XPROF_RUN_STAGE": stage, "CANON_P33_RUN_STAGE": stage,
      "CANON_P33_NO_COMMIT": "0", "CANON_P59_KIND": "v1",
      "CANON_P59_DP4_SERIAL_MESH_BRIDGE": "1" if geometry == "dp4-tp1" else "0",
      "CANON_P59_RANK_PARALLEL_BACKWARD": "1" if arm == "zero-hp" else "0",
      "CANON_P32_TRAIN_ADMITTED": "1", "CANON_P32_DP_REDUCTION_ADMITTED": "1",
      "CANON_P33_WORKLOAD_LAUNCH_ADMITTED": "1",
  })
  if arm == "native":
    environment["CANON_GSM8K_VANILLA"] = "1"
  return shell_probe(
      INNER, [], environment,
      'run_stage="${V1_GSM8K_XPROF_RUN_STAGE:-three-update}"', reader=True,
  )


class OnehostDefaultsTest(unittest.TestCase):

  def readback(self, result):
    self.assertIn("F1C_TEST_BOUNDARY_REACHED", result.stderr)
    self.assertEqual(result.returncode, 0, result.stderr)
    return json.loads(result.stdout)

  def test_default_matches_explicit_bundle_through_real_scripts_and_readers(self):
    for geometry in GEOMETRIES:
      for stage in ("three-update", "six-update"):
        with self.subTest(geometry=geometry, stage=stage):
          automatic = self.readback(outer_probe(geometry=geometry, stage=stage))
          explicit = self.readback(outer_probe(
              dict(zip(PAIR, ("stream", "1"))), geometry=geometry, stage=stage,
          ))
          self.assertEqual(automatic, explicit)
          self.assertEqual(self.readback(inner_probe(
              automatic, geometry=geometry, stage=stage,
          )), ["stream", True, True, None if geometry == "dp4-tp1" else "1"])

  def test_each_explicit_presence_preserves_both_raw_values(self):
    for name in PAIR:
      for value in ("", "0", "1", "stream", "invalid"):
        with self.subTest(name=name, value=value):
          raw = self.readback(outer_probe({name: value}))
          self.assertEqual(raw, {key: value if key == name else None for key in PAIR})
    off = self.readback(outer_probe(dict(zip(PAIR, ("0", "0")))))
    self.assertEqual(self.readback(inner_probe(off)), ["", False, True, "1"])

  def test_explicit_bundle_works_through_all_real_profiles(self):
    for geometry in GEOMETRIES:
      with self.subTest(geometry=geometry):
        raw = self.readback(outer_probe(
            dict(zip(PAIR, ("stream", "1"))), geometry=geometry,
        ))
        result = self.readback(inner_probe(raw, geometry=geometry))
        self.assertEqual(result[:3], ["stream", True, True])
        if geometry != "dp4-tp1":
          self.assertEqual(result[3], "1")

  def test_invalid_values_reach_unchanged_reader_refusals(self):
    for name in PAIR:
      with self.subTest(name=name):
        raw = self.readback(outer_probe({name: "invalid"}))
        result = inner_probe(raw)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(name + " must be", result.stderr)

  def test_native_and_shape_diagnostic_keep_the_original_absent_pair(self):
    native = self.readback(outer_probe(arm="native"))
    self.assertEqual(native, dict.fromkeys(PAIR))
    self.assertEqual(self.readback(inner_probe(native, arm="native")),
                     ["", False, False, None])
    shape = self.readback(outer_probe(geometry="dp2-tp2-p45"))
    self.assertEqual(shape, dict.fromkeys(PAIR))
    self.assertEqual(self.readback(inner_probe(shape, geometry="dp2-tp2-p45")),
                     ["", False, True, "1"])

  def test_capture_and_signed_negative_do_not_acquire_defaults(self):
    for values in (
        {"V2_P0_CAPTURE_FULL_TREE": "1"},
        {"V2_P0_NEGATIVE_CONTROL": "length-sort-no-inverse",
         "CANON_P32_LENGTH_SORT": "1"},
    ):
      with self.subTest(values=values):
        self.assertEqual(self.readback(outer_probe(values)), dict.fromkeys(PAIR))
    result = outer_probe({"V2_P0_NEGATIVE_CONTROL": "reduce-once-reassociate-tail"})
    self.assertEqual(result.returncode, 2)
    self.assertIn("requires CANON_DP_REDUCE_ONCE=1", result.stderr)
    self.assertNotIn("F1C_TEST_BOUNDARY_REACHED", result.stderr)

  def test_direct_backward_route_uses_the_same_default(self):
    wrapper = SCRIPTS / "run_onehost_xprof_backward_zero.sh"
    self.assertEqual(self.readback(outer_probe(wrapper=wrapper)),
                     dict(zip(PAIR, ("stream", "1"))))

  def test_p74_wrapper_retains_its_original_off_program(self):
    wrapper = SCRIPTS / "run_onehost_xprof_backward_p74_dp2tp2.sh"
    raw = self.readback(outer_probe(wrapper=wrapper))
    self.assertEqual(self.readback(inner_probe(raw)), ["", False, True, "1"])
    for name in PAIR:
      with self.subTest(name=name):
        explicit = {name: "0"}
        self.assertEqual(self.readback(outer_probe(explicit, wrapper=wrapper)),
                         {key: explicit.get(key) for key in PAIR})

  def test_unregistered_geometry_and_stage_refuse_before_external_resources(self):
    for geometry, stage in (("dp1-tp4", "three-update"), ("dp2-tp2", "full")):
      with self.subTest(geometry=geometry, stage=stage):
        result = outer_probe(geometry=geometry, stage=stage)
        self.assertEqual(result.returncode, 2)
        self.assertIn("unsupported", result.stderr)
        self.assertNotIn("F1C_TEST_BOUNDARY_REACHED", result.stderr)

  def test_census_and_docker_consume_the_same_pair(self):
    source = COMMON.read_text()
    consumers = ("modules", "hierarchy", "trace")
    for name, option in zip(PAIR, ("p32-keep-tape", "dp-reduce-once")):
      self.assertEqual(source.count(f'-e {name}="${{{name}:-}}"'), 1)
      argument = f'--{option} "${{{name}:-}}"'
      self.assertEqual(source.count(argument), len(consumers))
      for consumer in consumers:
        with self.subTest(consumer=consumer, name=name):
          command = f'python3 "$script_dir/census_gsm8k_xprof_{consumer}.py"'
          call = source.split(command, 1)[1].split("2>&1", 1)[0]
          self.assertEqual(call.count(argument), 1)

  def test_policy_is_in_both_runtime_hash_lists(self):
    source = COMMON.read_text()
    self.assertEqual(source.count('"$pkg/cluster/v1_full_system_optimization.py"'), 3)

  def test_cli_is_stdlib_only_and_does_not_accept_foreign_routes(self):
    result = subprocess.run(
        [sys.executable, "-S", str(POLICY), "--onehost-defaults",
         "zero-hp", "dp2-tp2", "three-update"],
        env=ENVIRONMENT, text=True, capture_output=True,
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(result.stdout,
                     "export CANON_P32_KEEP_TAPE=stream\nexport CANON_DP_REDUCE_ONCE=1\n")
    spec = importlib.util.spec_from_file_location("f1c_policy", POLICY)
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    for arm, geometry, stage in (
        ("zero", "dp2-tp2", "three-update"),
        ("zero-hp", "dp1-tp4", "three-update"),
        ("zero-hp", "dp2-tp2", "full"),
    ):
      with self.subTest(arm=arm, geometry=geometry, stage=stage):
        with self.assertRaises(ValueError):
          policy.onehost_defaults({}, arm=arm, geometry=geometry, run_stage=stage)
